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


def _largest_person(kps: np.ndarray) -> int:
    """Return index of the person with the largest keypoint bounding box.

    Bounding box area (from visible keypoints) is far more stable frame-to-frame
    than average confidence, so this avoids flip-flopping when multiple people
    are in frame. The largest person is almost always the one closest to the
    camera — i.e. the one most likely to be doing the gesture.
    """
    if kps.shape[0] == 1:
        return 0
    best, best_area = 0, -1.0
    for i in range(kps.shape[0]):
        vis = kps[i][kps[i, :, 2] > 0.3]   # only confident keypoints
        if len(vis) < 2:
            continue
        area = float(
            (vis[:, 0].max() - vis[:, 0].min()) *
            (vis[:, 1].max() - vis[:, 1].min())
        )
        if area > best_area:
            best_area = area
            best = i
    return best


def _yolo_to_landmarks(
    results, frame_w: int, frame_h: int
) -> Optional[_YOLOLandmarks]:
    """Return landmarks for the largest (closest) detected person, or None."""
    if not results:
        return None
    kps_data = results[0].keypoints
    if kps_data is None or kps_data.data.shape[0] == 0:
        return None
    kps = kps_data.data.cpu().numpy()  # [num_persons, 17, 3]
    best = _largest_person(kps)
    return _YOLOLandmarks(kps[best], frame_w, frame_h)



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

        self._arm_tracker = ArmTracker(ArmTrackerConfig(
            arm_above_head_tolerance=float(getattr(settings, 'arm_above_head_tolerance', 0.15)),
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
        os.environ.setdefault('ULTRALYTICS_SYNC', 'False')

        from ultralytics import YOLO
        try:
            from ultralytics import settings as _yolo_settings
            _yolo_settings.update({'sync': False})
        except Exception:
            pass

        model_name = getattr(self.s, 'yolo_pose_model', 'yolo11n-pose.onnx')
        if not os.path.isabs(model_name):
            model_name = os.path.join('/app', model_name)
        yolo = YOLO(model_name)

        # grlib/MediaPipe Hands is initialized lazily on the first SINGLE_UP
        # frame to avoid loading two ML runtimes simultaneously at startup
        # (causes silent segfault/OOM on embedded hardware).
        _hand_pipeline = None
        _NoHandDetected = None
        _grlib_ok = True

        log.info("[%s] Analyser running (v%s — YOLO pose + grlib hands)",
                 self.label, __version__)

        _last_arm_state: ArmState = ArmState.DOWN
        _cached_yolo: object = None
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

            if run_yolo:
                _cached_yolo = yolo(proc_frame, verbose=False, conf=yolo_conf)
                _cached_lm   = _yolo_to_landmarks(_cached_yolo, pw, ph)
                _cached_pw, _cached_ph = pw, ph
                _frames_since_yolo = 0
                log.debug("[%s] YOLO ran (motion=%s arm_up=%s)",
                          self.label, motion, arm_is_up)
                # Feed recorder if active (keypoints for regression tests)
                if self._recorder is not None:
                    kps_data = (
                        _cached_yolo[0].keypoints.data.cpu().numpy()
                        if _cached_yolo and _cached_yolo[0].keypoints is not None
                        and _cached_yolo[0].keypoints.data.shape[0] > 0
                        else np.zeros((17, 3), dtype=np.float32)
                    )
                    best = _largest_person(kps_data) \
                        if kps_data.ndim == 3 and kps_data.shape[0] > 1 else 0
                    self._recorder.on_frame(
                        kps_data[best] if kps_data.ndim == 3 else kps_data,
                        pw, ph, now,
                    )

            yolo_results = _cached_yolo
            landmarks    = _cached_lm

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

            n_persons = (
                yolo_results[0].keypoints.data.shape[0]
                if yolo_results and yolo_results[0].keypoints is not None
                else 0
            )
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

            now = time.monotonic()
            gesture = self._state_machine.tick(reading, now)

            if gesture is not None:
                log.debug("[%s] gesture candidate: %s", self.label, gesture)
                self.on_candidate(gesture, 1.0, self.label)
                # Feed the test recorder if active
                if self._recorder is not None:
                    self._recorder.on_gesture(gesture, self.label)

            if self.debug_frame_cb is not None:
                try:
                    dbg = _draw_debug(proc_frame, yolo_results, hand_lm, reading, gesture)
                    self.debug_frame_cb(self.label, dbg)
                except Exception as e:
                    log.debug("[%s] debug render error: %s", self.label, e)

        log.info("[%s] Analyser stopped", self.label)




# ── Debug overlay ─────────────────────────────────────────────────────────────

def _draw_debug(frame, yolo_results, hand_lm, reading, fired_gesture):
    """Draw YOLO skeleton + grlib hand indicator + gesture state overlay."""
    img = frame.copy()
    h, w = img.shape[:2]

    detected = False
    if yolo_results and yolo_results[0].keypoints is not None:
        kps_data = yolo_results[0].keypoints.data
        if kps_data.shape[0] > 0:
            detected = True
            kps = kps_data.cpu().numpy()
            best = _largest_person(kps)
            person = kps[best]  # [17, 3]

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
        forearm = reading.forearm_dy if reading.state.name == 'SINGLE_UP' else 0.0
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
