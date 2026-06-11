"""
cameras/analyser.py

Thin orchestration layer. Per frame:
  1. Preprocess (dewarp / crop / CLAHE)
  2. Run YOLO Pose → body keypoints
  3. Run grlib Pipeline → hand landmarks (optional; NoHandDetectedException → None)
  4. ArmTracker.classify() → ArmReading
  5. GestureStateMachine.tick() → Gesture or None
  6. Debug overlay render
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from version import __version__
from cameras.arm_tracker import ArmState, ArmTracker, ArmTrackerConfig
from core.state_machine import (
    Gesture, GestureStateMachine, StateMachineConfig,
)

log = logging.getLogger(__name__)

# ── COCO → MediaPipe index remap ──────────────────────────────────────────────
# YOLO Pose outputs 17 COCO keypoints; ArmTracker uses MediaPipe PoseLandmark
# indices.  This map translates at read-time so ArmTracker needs no changes.
_COCO_TO_MP: dict[int, int] = {
    5:  11,  # left shoulder
    6:  12,  # right shoulder
    7:  13,  # left elbow
    8:  14,  # right elbow
    9:  15,  # left wrist
    10: 16,  # right wrist
    11: 23,  # left hip
    12: 24,  # right hip
}

# COCO upper-body skeleton connections (used for debug overlay)
_COCO_UPPER_BODY = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
]


@dataclass
class _KP:
    x: float          # normalised 0-1
    y: float          # normalised 0-1
    visibility: float # keypoint confidence


class _YOLOLandmarks:
    """YOLO COCO keypoints wrapped to match ArmTracker's _Landmarks protocol."""

    def __init__(self, kps: np.ndarray, frame_w: int, frame_h: int) -> None:
        # kps: shape [17, 3] — (x_px, y_px, conf)
        self._pts: dict[int, _KP] = {
            mp_idx: _KP(
                x=float(kps[coco_idx, 0]) / frame_w,
                y=float(kps[coco_idx, 1]) / frame_h,
                visibility=float(kps[coco_idx, 2]),
            )
            for coco_idx, mp_idx in _COCO_TO_MP.items()
        }

    def __getitem__(self, idx: int) -> _KP:
        return self._pts[idx]


def _person_centroid(kps_row: np.ndarray) -> Optional[tuple]:
    """Mean (x, y) pixel position of visible keypoints for one person."""
    vis = kps_row[kps_row[:, 2] > 0.3]
    if len(vis) == 0:
        return None
    return float(vis[:, 0].mean()), float(vis[:, 1].mean())


def _person_visible_kp_count(kps_row: np.ndarray) -> int:
    """Number of keypoints with confidence > 0.3 — proxy for detection quality."""
    return int((kps_row[:, 2] > 0.3).sum())


def _select_person(
    kps: np.ndarray,
    tracked_xy: Optional[tuple],
    frame_w: int,
    frame_h: int,
) -> int:
    """Pick which detected person to track.

    Strategy
    --------
    * If we have a previous centroid (*tracked_xy*), pick the nearest person to
      it — temporal continuity.  We only abandon the lock if every candidate is
      more than 30 % of the frame diagonal away, meaning the person genuinely
      left the frame.
    * With no history (first frame or after loss), pick the person whose
      centroid is closest to the frame centre.  Works well for a fixed camera
      pointed at an activity zone and is immune to fisheye area distortion.

    Both strategies ignore keypoint confidence entirely, so low-contrast
    clothing or poor lighting does not affect selection stability.
    """
    if kps.shape[0] == 1:
        return 0

    centroids = [_person_centroid(kps[i]) for i in range(kps.shape[0])]

    if tracked_xy is not None:
        tx, ty = tracked_xy
        max_dist = 0.30 * (frame_w ** 2 + frame_h ** 2) ** 0.5
        best, best_dist = 0, float("inf")
        for i, c in enumerate(centroids):
            if c is None:
                continue
            d = ((c[0] - tx) ** 2 + (c[1] - ty) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best = i
        if best_dist <= max_dist:
            return best
        # Tracking lost — fall through to centre heuristic

    # No history or re-acquisition: score by center-distance penalised by
    # visible-keypoint count.  A real person has many confident keypoints;
    # a painting or furniture ghost has very few.  Combined score:
    #   score = (dist / diag) - 0.4 * (visible_kps / 17)
    # Lower score wins.  The kp bonus can offset up to 0.4 of normalised
    # distance, so a person 40% of the diagonal away beats a painting 10%
    # away with only 2 visible keypoints.
    cx, cy = frame_w * 0.5, frame_h * 0.5
    diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
    best, best_score = 0, float("inf")
    for i, c in enumerate(centroids):
        if c is None:
            continue
        dist = ((c[0] - cx) ** 2 + (c[1] - cy) ** 2) ** 0.5
        kp_bonus = 0.4 * (_person_visible_kp_count(kps[i]) / 17)
        score = dist / diag - kp_bonus
        if score < best_score:
            best_score = score
            best = i
    return best


def _kps_to_landmarks(
    kps: Optional[np.ndarray],
    frame_w: int,
    frame_h: int,
    tracked_xy: Optional[tuple] = None,
) -> tuple:
    """Return *(landmarks, new_centroid)* for the selected person.

    *kps* is the [N, 17, 3] array returned by PoseModel (or None).
    *new_centroid* is the pixel (x, y) centre of the selected person —
    store it as the next frame's *tracked_xy*.  Both values are None when
    no person is detected.
    """
    if kps is None or kps.shape[0] == 0:
        return None, None
    best = _select_person(kps, tracked_xy, frame_w, frame_h)
    return _YOLOLandmarks(kps[best], frame_w, frame_h), _person_centroid(kps[best])



class CameraAnalyser(threading.Thread):
    def __init__(
        self,
        camera,
        settings,
        on_candidate: Callable[[Gesture, float, str], None],
        label: str = "cam",
        debug_frame_cb=None,
    ):
        super().__init__(daemon=True, name=f"Analyser-{label}")
        self.camera = camera
        self.s = settings
        self.on_candidate = on_candidate
        self.label = label
        self.debug_frame_cb = debug_frame_cb
        self._stop_evt = threading.Event()

        # CLAHE cache — object is expensive; recreate only when clip changes.
        self._clahe_obj = None
        self._clahe_clip: float = -1.0

        # Optional test recorder (set by main.py when ENDORA_RECORD_TESTS=1)
        self._recorder = None

        # Person tracking — centroid of the last selected person (pixels).
        # None until the first YOLO detection.
        self._track_xy: Optional[tuple] = None

        self._arm_tracker = ArmTracker(ArmTrackerConfig(
            arm_above_head_tolerance=float(getattr(settings, 'arm_above_head_tolerance', 0.15)),
            arm_above_head_tolerance_reclined=float(getattr(settings, 'arm_above_head_tolerance_reclined', 0.28)),
            body_upright_min=float(getattr(settings, 'body_upright_min', -0.15)),
            pose_visibility_min=float(getattr(settings, 'pose_visibility_min', 0.45)),
            state_confirm_s=float(getattr(settings, 'state_confirm_s', 0.20)),
            state_release_s=float(getattr(settings, 'state_release_s', 0.30)),
        ))
        self._state_machine = GestureStateMachine(StateMachineConfig(
            cooldown_s=float(getattr(settings, 'cooldown_s', 2.0)),
            snap_forearm_min=float(getattr(settings, 'snap_forearm_min', 0.10)),
            hold_duration_s=float(getattr(settings, 'hold_duration_s', 1.5)),
            double_snap_window_s=float(getattr(settings, 'double_snap_window_s', 3.0)),
            sustain_s=float(getattr(settings, 'sustain_s', 0.5)),
            snap_roll_threshold=float(getattr(settings, 'snap_roll_threshold', 0.0)),
        ))

    def stop(self):
        self._stop_evt.set()

    # ── Frame preprocessing ───────────────────────────────────────────────

    def _preprocess(self, frame):
        """Apply dewarp, flip, crop, CLAHE. Returns (proc_frame, w, h)."""
        h, w = frame.shape[:2]

        if getattr(self.s, 'dewarp_enable', False):
            from cameras.dewarp import build_dewarp_maps, apply_dewarp
            cx_raw = float(getattr(self.s, 'dewarp_cx', -1.0))
            cy_raw = float(getattr(self.s, 'dewarp_cy', -1.0))
            dw = int(getattr(self.s, 'dewarp_out_width', 640))
            dh = int(getattr(self.s, 'dewarp_out_height', 480))
            fov = float(getattr(self.s, 'dewarp_fov', 180.0))
            pan = float(getattr(self.s, 'dewarp_pan', 0.0))
            tilt = float(getattr(self.s, 'dewarp_tilt', 20.0))
            roll = float(getattr(self.s, 'dewarp_roll', 0.0))
            vfov = float(getattr(self.s, 'dewarp_vfov', 75.0))
            key = (w, h, dw, dh, fov, pan, tilt, roll, vfov, cx_raw, cy_raw)
            if getattr(self, '_dewarp_key', None) != key:
                self._dewarp_maps = build_dewarp_maps(
                    in_w=w, in_h=h, out_w=dw, out_h=dh,
                    fisheye_fov_deg=fov, pan_deg=pan, tilt_deg=tilt,
                    roll_deg=roll, vfov_deg=vfov,
                    cx=None if cx_raw < 0 else cx_raw,
                    cy=None if cy_raw < 0 else cy_raw,
                )
                self._dewarp_key = key
            frame = apply_dewarp(frame, *self._dewarp_maps)
            h, w = frame.shape[:2]

        if getattr(self.s, 'flip_image', False):
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            h, w = frame.shape[:2]

        ct = float(getattr(self.s, 'frame_crop_top', 0))
        cb = float(getattr(self.s, 'frame_crop_bottom', 0))
        cl = float(getattr(self.s, 'frame_crop_left', 0))
        cr = float(getattr(self.s, 'frame_crop_right', 0))
        y0, y1 = int(h * ct / 100), h - int(h * cb / 100)
        x0, x1 = int(w * cl / 100), w - int(w * cr / 100)
        if y0 > 0 or y1 < h or x0 > 0 or x1 < w:
            frame = frame[y0:y1, x0:x1]

        if getattr(self.s, 'low_light_enhance', False):
            clip = float(getattr(self.s, 'low_light_clip', 2.0))
            if clip != self._clahe_clip:
                self._clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
                self._clahe_clip = clip
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = self._clahe_obj.apply(l_ch)
            frame = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

        ph, pw = frame.shape[:2]
        return frame, pw, ph

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        try:
            self._run()
        except Exception:
            log.exception("[%s] Analyser crashed", self.label)
            raise

    def _run(self):
        import os
        from cameras.pose_model import PoseModel

        model_name = getattr(self.s, 'yolo_pose_model', 'yolo11n-pose.onnx')
        if not os.path.isabs(model_name):
            model_name = os.path.join('/app', model_name)

        yolo_imgsz = int(getattr(self.s, 'yolo_imgsz', 320))
        yolo_conf  = float(getattr(self.s, 'yolo_conf',  0.45))
        model = PoseModel(
            model_path=model_name,
            imgsz=yolo_imgsz,
            conf=yolo_conf,
            num_threads=0,          # 0 = all CPU cores
        )

        # grlib/MediaPipe Hands is initialized lazily on the first SINGLE_UP
        # frame to avoid loading two ML runtimes simultaneously at startup
        # (causes silent segfault/OOM on embedded hardware).
        _hand_pipeline = None
        _NoHandDetected = None
        _grlib_ok = True

        log.info("[%s] Analyser running (v%s — YOLO pose + grlib hands)",
                 self.label, __version__)

        _last_arm_state: ArmState = ArmState.DOWN
        _cached_kps: Optional[np.ndarray] = None   # [N, 17, 3] from PoseModel
        _cached_lm: object = None
        _cached_pw: int = 1
        _cached_ph: int = 1
        _prev_small: Optional[np.ndarray] = None   # for motion gate
        _frames_since_yolo: int = 999              # force run on first frame

        while not self._stop_evt.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            proc_frame, pw, ph = self._preprocess(frame)

            # ── Motion gate ───────────────────────────────────────────────
            # Resize to 80×60 (~0.1 ms) and diff against previous frame.
            # Skip YOLO when the scene is static — reuse cached landmarks.
            # Always run YOLO when:
            #   • significant motion detected  (something is moving)
            #   • arm already raised           (responsive snap detection)
            #   • heartbeat interval reached   (catch slow arm lifts)
            mot_thresh = float(getattr(self.s, 'motion_threshold', 0.015))
            max_skip   = int(getattr(self.s,   'yolo_max_skip',    12))
            yolo_conf  = float(getattr(self.s, 'yolo_conf',        0.45))

            gray  = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (80, 60), interpolation=cv2.INTER_AREA)
            if _prev_small is None:
                motion = True
            else:
                motion = (
                    float(cv2.absdiff(small, _prev_small).mean()) / 255.0
                ) > mot_thresh
            _prev_small = small
            _frames_since_yolo += 1

            arm_is_up = (_last_arm_state != ArmState.DOWN)
            run_yolo  = motion or arm_is_up or (_frames_since_yolo >= max_skip)

            now = time.monotonic()

            if run_yolo:
                # Confidence hysteresis:
                #   acquire  (no lock): base_conf * 1.3 — strict, rejects paintings/furniture
                #   maintain (locked):  base_conf * 0.65 — lenient, bridges arm-raise dropouts
                # Both multipliers are internal — not user-facing. Tune yolo_conf only.
                base_conf = float(getattr(self.s, 'yolo_conf', 0.45))
                model.conf = base_conf * 0.65 if self._track_xy is not None else base_conf * 1.3

                _cached_kps = model(proc_frame)    # Optional[ndarray [N,17,3]]
                _cached_lm, new_centroid = _kps_to_landmarks(
                    _cached_kps, pw, ph, self._track_xy
                )
                if new_centroid is not None:
                    self._track_xy = new_centroid
                # Keep _track_xy when nobody detected — quick re-acquisition
                _cached_pw, _cached_ph = pw, ph
                _frames_since_yolo = 0
                log.debug("[%s] YOLO ran (motion=%s arm_up=%s)",
                          self.label, motion, arm_is_up)
                # Feed recorder if active (keypoints for regression tests)
                if self._recorder is not None:
                    if _cached_kps is not None and _cached_kps.shape[0] > 0:
                        best = _select_person(_cached_kps, self._track_xy, pw, ph)
                        self._recorder.on_frame(_cached_kps[best], pw, ph, now)
                    else:
                        self._recorder.on_frame(
                            np.zeros((17, 3), dtype=np.float32), pw, ph, now
                        )

            landmarks = _cached_lm

            # ── Hand landmarks (grlib / MediaPipe Hands) ──────────────────
            # Only run when arm is already raised — avoids running both ML
            # models every frame on resource-constrained hardware.
            hand_lm: Optional[np.ndarray] = None
            if _last_arm_state == ArmState.SINGLE_UP and _grlib_ok:
                if _hand_pipeline is None:
                    try:
                        sys.modules.setdefault('cv2.cv2', cv2)
                        from grlib.feature_extraction.pipeline import Pipeline
                        from grlib.exceptions import NoHandDetectedException as _NHD
                        _NoHandDetected = _NHD
                        _hand_pipeline = Pipeline(num_hands=1, optimize_pipeline=True)
                        _hand_pipeline.add_stage()
                        log.info("[%s] grlib hand pipeline ready", self.label)
                    except Exception as e:
                        log.warning("[%s] grlib init failed, snap_roll disabled: %s",
                                    self.label, e)
                        _grlib_ok = False

                if _hand_pipeline is not None:
                    try:
                        flat_lm, _ = _hand_pipeline.get_landmarks_from_image(proc_frame)
                        hand_lm = flat_lm
                    except Exception as e:
                        if _NoHandDetected is None or not isinstance(e, _NoHandDetected):
                            log.debug("[%s] grlib hand error: %s", self.label, e)

            n_persons = 0 if _cached_kps is None else _cached_kps.shape[0]
            log.debug("[%s] YOLO: %d person(s) detected", self.label, n_persons)

            reading = self._arm_tracker.classify(landmarks, pw, ph, hand_lm)

            if reading is not None:
                _last_arm_state = reading.state
                if reading.state != getattr(self, '_last_logged_state', None):
                    log.info("[%s] state → %s", self.label, reading.state.name)
                    self._last_logged_state = reading.state
                if reading.state.name == 'SINGLE_UP':
                    log.debug("[%s] SINGLE_UP forearm_dy=%.3f snap_roll=%.3f",
                              self.label, reading.forearm_dy, reading.snap_roll)

            gesture = self._state_machine.tick(reading, now)

            if gesture is not None:
                log.debug("[%s] gesture candidate: %s", self.label, gesture)
                self.on_candidate(gesture, 1.0, self.label)
                # Feed the test recorder if active
                if self._recorder is not None:
                    self._recorder.on_gesture(gesture, self.label)

            if self.debug_frame_cb is not None:
                try:
                    # _cached_kps is already [N,17,3]; pick the tracked person
                    _dbg_kps = None
                    if _cached_kps is not None and _cached_kps.shape[0] > 0:
                        _idx = _select_person(_cached_kps, self._track_xy, pw, ph)
                        _dbg_kps = _cached_kps[_idx]
                    dbg = _draw_debug(proc_frame, _dbg_kps, hand_lm, reading, gesture)
                    self.debug_frame_cb(self.label, dbg)
                except Exception as e:
                    log.debug("[%s] debug render error: %s", self.label, e)

        log.info("[%s] Analyser stopped", self.label)




# ── Debug overlay ─────────────────────────────────────────────────────────────

def _draw_debug(frame, person_kps, hand_lm, reading, fired_gesture):
    """Draw YOLO skeleton + grlib hand indicator + gesture state overlay.

    *person_kps* is a [17, 3] numpy array for the already-selected person,
    or None if no pose was detected this frame.
    """
    img = frame.copy()
    h, w = img.shape[:2]

    detected = person_kps is not None
    if detected:
        person = person_kps   # [17, 3] — already the right person

        for a, b in _COCO_UPPER_BODY:
            x1, y1, c1 = person[a]
            x2, y2, c2 = person[b]
            if c1 > 0.5 and c2 > 0.5:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                         (0, 200, 0), 2)

        for i in range(5, 13):
            x, y, c = person[i]
            if c > 0.5:
                cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

    if not detected:
        msg = "NO POSE DETECTED"
        fs = max(0.6, w / 800)
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        tx, ty = (w - tw) // 2, 60
        cv2.rectangle(img, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6),
                      (0, 0, 180), -1)
        cv2.putText(img, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), 2, cv2.LINE_AA)

    # Wrist marker (only for SINGLE_UP)
    if reading and reading.state.name == 'SINGLE_UP':
        wx, wy = int(reading.wrist_x), int(reading.wrist_y)
        cv2.circle(img, (wx, wy), 12, (255, 255, 0), -1)
        cv2.circle(img, (wx, wy), 12, (0, 0, 0), 2)

    # Status panel
    if reading is not None:
        state_name = reading.state.name
        forearm = reading.forearm_dy
        snap_roll = reading.snap_roll if reading.state.name == 'SINGLE_UP' else 0.0
        hand_str = f"{snap_roll:+.2f}" if hand_lm is not None else "none"
        lines = [
            (f"state: {state_name}", (0, 255, 100)),
            (f"forearm_dy: {forearm:.3f}", (255, 255, 255)),
            (f"snap_roll:  {hand_str}", (255, 255, 255)),
            (f"upright: {reading.upright}", (255, 255, 255)),
        ]
    else:
        lines = [("state: none", (160, 160, 160))]

    fs = max(0.35, w / 1800)
    lh = int(fs * 42)
    pad = int(fs * 12)
    panel_h = len(lines) * lh + pad * 2
    panel_w = int(w * 0.32)
    y_start = h - panel_h - 6
    overlay = img.copy()
    cv2.rectangle(overlay, (4, y_start - 2),
                  (4 + panel_w, y_start + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, (line, color) in enumerate(lines):
        y = y_start + pad + i * lh + lh - 4
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, color, 1, cv2.LINE_AA)

    return img
