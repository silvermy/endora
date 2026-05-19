"""
cameras/recorder.py

Capture YOLO keypoints + gesture labels to disk for regression testing.

Activate by setting the environment variable ENDORA_RECORD_TESTS=1 before
starting the add-on (or by calling debug_server's /start_capture endpoint).
Each time a gesture fires (or a manual capture is requested), the last
`window_s` seconds of YOLO keypoints are saved as a .npz file.

Layout of each saved file:
  keypoints   float32  [N, 17, 3]  — COCO pose keypoints (x_px, y_px, conf)
  t_offsets   float64  [N]         — seconds relative to first frame in window
  frame_w     int                  — frame width used for normalisation
  frame_h     int                  — frame height
  label       str                  — human description (e.g. "snap_right_arm")
  gesture     str                  — gesture enum name (e.g. "SNAP")

Replay in tests:
  from cameras.analyser import _YOLOLandmarks
  lm = _YOLOLandmarks(kps[i], frame_w, frame_h)
  reading = tracker.classify(lm, frame_w, frame_h, now=t_offsets[i])
  gesture = sm.tick(reading, t_offsets[i])
"""
from __future__ import annotations

import collections
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_SAVE_DIR = Path(os.environ.get("ENDORA_CAPTURE_DIR", "/data/test_captures"))


@dataclass
class _Frame:
    keypoints: np.ndarray   # [17, 3] float32
    frame_w: int
    frame_h: int
    t: float                # monotonic timestamp


class TestRecorder:
    """
    Rolling buffer of recent YOLO keypoints.  Thread-safe.

    Usage in analyser._run():
        if self._recorder:
            self._recorder.on_frame(kps_array, pw, ph, now)
        ...
        if gesture and self._recorder:
            self._recorder.on_gesture(gesture, self.label)
    """

    def __init__(self, window_s: float = 5.0, save_dir: Path = _SAVE_DIR):
        self._window_s = window_s
        self._save_dir = save_dir
        self._lock = threading.Lock()
        # deque of _Frame; we keep at most window_s seconds
        self._buf: collections.deque[_Frame] = collections.deque()
        self._active = True
        log.info("TestRecorder active — captures → %s", save_dir)

    # ── Called from analyser ──────────────────────────────────────────────

    def on_frame(self, keypoints: np.ndarray, frame_w: int, frame_h: int,
                 t: float) -> None:
        """Append a frame of YOLO keypoints to the rolling buffer."""
        if not self._active:
            return
        frame = _Frame(
            keypoints=keypoints.astype(np.float32),
            frame_w=frame_w,
            frame_h=frame_h,
            t=t,
        )
        with self._lock:
            self._buf.append(frame)
            # Trim old frames outside the window
            cutoff = t - self._window_s
            while self._buf and self._buf[0].t < cutoff:
                self._buf.popleft()

    def on_gesture(self, gesture, camera_label: str = "") -> None:
        """Auto-save when a gesture fires (gesture is a Gesture enum value)."""
        label = f"{camera_label}_{gesture.name.lower()}".strip("_")
        self.save(gesture_name=gesture.name, label=label)

    # ── Save ─────────────────────────────────────────────────────────────

    def save(self, gesture_name: str = "UNKNOWN", label: str = "manual") -> Optional[Path]:
        """Flush current buffer to a .npz file and return the path."""
        with self._lock:
            frames = list(self._buf)

        if not frames:
            log.warning("TestRecorder.save: buffer empty, nothing to save")
            return None

        kps_list = [f.keypoints for f in frames]
        t0 = frames[0].t
        t_offsets = np.array([f.t - t0 for f in frames], dtype=np.float64)
        frame_w = frames[-1].frame_w
        frame_h = frames[-1].frame_h

        self._save_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        fname = self._save_dir / f"{ts}_{label}.npz"
        try:
            np.savez_compressed(
                fname,
                keypoints=np.stack(kps_list, axis=0),
                t_offsets=t_offsets,
                frame_w=np.int32(frame_w),
                frame_h=np.int32(frame_h),
                label=np.array(label),
                gesture=np.array(gesture_name),
            )
            log.info("TestRecorder: saved %d frames → %s", len(frames), fname)
            return fname
        except Exception as e:
            log.error("TestRecorder.save failed: %s", e)
            return None

    def manual_capture(self, label: str = "manual") -> Optional[Path]:
        """Triggered by the debug page 'Capture' button."""
        return self.save(gesture_name="MANUAL", label=label)

    def stop(self) -> None:
        self._active = False
