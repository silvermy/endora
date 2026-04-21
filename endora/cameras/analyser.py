"""
cameras/analyser.py

Thin orchestration layer. Per frame:
  1. Preprocess (dewarp / crop / CLAHE)
  2. Run MediaPipe Pose
  3. ArmTracker.classify() → ArmReading
  4. GestureStateMachine.tick() → Gesture or None
  5. Debug overlay render
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import cv2

from version import __version__
from cameras.arm_tracker import ArmTracker, ArmTrackerConfig
from core.state_machine import (
    Gesture, GestureStateMachine, StateMachineConfig,
)

log = logging.getLogger(__name__)


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

        # Build subsystems from settings
        self._arm_tracker = ArmTracker(ArmTrackerConfig(
            arm_above_head_tolerance=float(getattr(settings, 'arm_above_head_tolerance', 0.15)),
            body_upright_min=float(getattr(settings, 'body_upright_min', -0.15)),
            pose_visibility_min=float(getattr(settings, 'pose_visibility_min', 0.45)),
        ))
        self._state_machine = GestureStateMachine(StateMachineConfig(
            cooldown_s=float(getattr(settings, 'cooldown_s', 2.0)),
            snap_forearm_min=float(getattr(settings, 'snap_forearm_min', 0.10)),
            hold_duration_s=float(getattr(settings, 'hold_duration_s', 1.5)),
            double_snap_window_s=float(getattr(settings, 'double_snap_window_s', 3.0)),
            sustain_s=float(getattr(settings, 'sustain_s', 0.5)),
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
        import mediapipe as mp
        mp_pose = mp.solutions.pose

        complexity = max(0, min(2, int(getattr(self.s, 'pose_model_complexity', 1))))
        pose = mp_pose.Pose(
            model_complexity=complexity,
            min_detection_confidence=float(getattr(
                self.s, 'pose_min_detection_confidence', 0.5)),
            min_tracking_confidence=float(getattr(
                self.s, 'pose_min_tracking_confidence', 0.5)),
            enable_segmentation=False,
            static_image_mode=False,
        )

        log.info("[%s] Analyser running (v%s — pose-only gesture set)",
                 self.label, __version__)

        while not self._stop_evt.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            proc_frame, pw, ph = self._preprocess(frame)
            rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_res = pose.process(rgb)
            rgb.flags.writeable = True

            landmarks = pose_res.pose_landmarks.landmark if (
                pose_res and pose_res.pose_landmarks) else None

            reading = self._arm_tracker.classify(landmarks, pw, ph)

            # Log state transitions at INFO so the user can see recognition
            # without having to enable debug logging.
            if reading is not None:
                state_now = reading.state
                if state_now != getattr(self, '_last_logged_state', None):
                    log.info("[%s] state → %s", self.label, state_now.name)
                    self._last_logged_state = state_now

            now = time.monotonic()
            gesture = self._state_machine.tick(reading, now)

            if gesture is not None:
                self.on_candidate(gesture, 1.0, self.label)

            if self.debug_frame_cb is not None:
                try:
                    dbg = _draw_debug(proc_frame, pose_res, reading, gesture)
                    self.debug_frame_cb(self.label, dbg)
                except Exception as e:
                    log.debug("[%s] debug render error: %s", self.label, e)

        pose.close()
        log.info("[%s] Analyser stopped", self.label)


# ── Debug overlay ─────────────────────────────────────────────────────────────

def _draw_debug(frame, pose_res, reading, fired_gesture):
    """Draw skeleton + gesture state overlay."""
    import mediapipe as mp
    img = frame.copy()
    h, w = img.shape[:2]

    if pose_res and pose_res.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            img, pose_res.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 200, 0), thickness=2),
        )
    else:
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
        lines = [
            (f"state: {state_name}", (0, 255, 100)),
            (f"forearm_dy: {forearm:.3f}", (255, 255, 255)),
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
