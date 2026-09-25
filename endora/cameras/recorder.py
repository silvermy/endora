"""
cameras/recorder.py

Capture YOLO keypoints + gesture labels to disk for regression testing.

Activate by setting the environment variable ENDORA_RECORD_TESTS=1 before
starting the add-on (or by calling debug_server's /start_capture endpoint).
Each time a gesture fires (or a manual capture is requested), the last
`window_s` seconds of YOLO keypoints are saved as a .npz file.

Buffers are kept PER TRACKED PERSON. They used to be a single buffer fed with
`valid[0]` — the first detection of each frame by array index, which is not
stable between frames and is not necessarily the person who fired. One
occupant routinely produces two concurrent detections here, so a saved trace
could interleave two different bodies frame by frame, and replaying it
reproduced neither. Captures recorded on 2026-09-25 alternate between
shoulders of 22 px and 78 px; the gesture they were supposed to explain does
not fire on replay at all. Keyed by pid, a trace is one body throughout.

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


# A pid whose buffer has gone this long without a frame is dropped, so that
# person-id churn cannot grow the buffer dict without limit. Generous next to
# the capture window, since the point is only to bound memory.
_PID_IDLE_DROP_S = 30.0


class TestRecorder:
    """
    Rolling buffer of recent YOLO keypoints.  Thread-safe.

    Usage in analyser._run(), once per tracked person:
        if self._recorder:
            self._recorder.on_frame(kps_row, pw, ph, now, pid=pid)
        ...
        if gesture and self._recorder:
            self._recorder.on_gesture(gesture, self.label, pid=pid)
    """

    def __init__(self, window_s: float = 5.0, save_dir: Path = _SAVE_DIR):
        self._window_s = window_s
        self._save_dir = save_dir
        self._lock = threading.Lock()
        # One rolling buffer per tracked person — see the module docstring
        # for why a single shared buffer could not be trusted. The key is
        # whatever the caller passes as pid; None is its own buffer, which is
        # what a caller with no person tracking gets.
        self._bufs: dict[object, collections.deque] = {}
        self._active = True
        log.info("TestRecorder active — captures → %s", save_dir)

    # ── Called from analyser ──────────────────────────────────────────────

    def on_frame(self, keypoints: np.ndarray, frame_w: int, frame_h: int,
                 t: float, pid: object = None) -> None:
        """Append one person's YOLO keypoints to that person's buffer."""
        if not self._active:
            return
        frame = _Frame(
            keypoints=keypoints.astype(np.float32),
            frame_w=frame_w,
            frame_h=frame_h,
            t=t,
        )
        cutoff = t - self._window_s
        with self._lock:
            buf = self._bufs.setdefault(pid, collections.deque())
            buf.append(frame)
            while buf and buf[0].t < cutoff:
                buf.popleft()
            # Person ids churn — ten new ones in 159 s was measured on the
            # deployed camera — so retire the buffers nobody is feeding.
            for key in [k for k, b in self._bufs.items()
                        if b and t - b[-1].t > _PID_IDLE_DROP_S]:
                del self._bufs[key]

    def on_gesture(self, gesture, camera_label: str = "",
                   pid: object = None) -> None:
        """Auto-save when a gesture fires (gesture is a Gesture enum value).

        Saves the buffer belonging to the person who fired it, so the trace
        and the gesture describe the same body.
        """
        label = f"{camera_label}_{gesture.name.lower()}".strip("_")
        if pid is not None:
            label = f"{label}_pid{pid}"
        self.save(gesture_name=gesture.name, label=label, pid=pid)

    # ── Save ─────────────────────────────────────────────────────────────

    def save(self, gesture_name: str = "UNKNOWN", label: str = "manual",
             pid: object = None) -> Optional[Path]:
        """Flush one person's buffer to a .npz file and return the path.

        With no pid, saves the longest buffer currently held — a manual
        capture has no person in mind, and the busiest track is the best
        guess at the one worth looking at.
        """
        with self._lock:
            if pid is not None:
                frames = list(self._bufs.get(pid, ()))
            elif self._bufs:
                frames = list(max(self._bufs.values(), key=len))
            else:
                frames = []

        if not frames:
            log.warning("TestRecorder.save: buffer empty, nothing to save")
            return None

        kps_list = [f.keypoints for f in frames]
        t0 = frames[0].t
        t_offsets = np.array([f.t - t0 for f in frames], dtype=np.float64)
        frame_w = frames[-1].frame_w
        frame_h = frames[-1].frame_h

        # Two people firing in the same second used to land on one filename
        # and silently overwrite each other; that is exactly the case these
        # per-person buffers exist to tell apart.
        self._save_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        fname = self._save_dir / f"{ts}_{label}.npz"
        n = 1
        while fname.exists():
            fname = self._save_dir / f"{ts}_{label}_{n}.npz"
            n += 1
        try:
            np.savez_compressed(
                fname,
                keypoints=np.stack(kps_list, axis=0),
                t_offsets=t_offsets,
                frame_w=np.int32(frame_w),
                frame_h=np.int32(frame_h),
                label=np.array(label),
                gesture=np.array(gesture_name),
                pid=np.array("" if pid is None else str(pid)),
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
