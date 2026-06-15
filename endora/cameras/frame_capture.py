"""
cameras/frame_capture.py — Save annotated debug frames on gesture events.

Frames (JPEG) and metadata (JSON) are written to /data/debug_frames/.
A rolling max of MAX_FRAMES pairs is kept; oldest are evicted automatically.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

CAPTURE_DIR = Path("/data/debug_frames")
MAX_FRAMES = 200


class FrameCapture:
    """Thread-safe frame saver for post-hoc gesture debugging."""

    def __init__(
        self,
        capture_dir: Path = CAPTURE_DIR,
        max_frames: int = MAX_FRAMES,
    ) -> None:
        self._dir = Path(capture_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max = max_frames
        self._lock = threading.Lock()

    def save(
        self,
        frame: np.ndarray,
        event_type: str,
        *,
        camera: str = "cam",
        arm_state: str = "UNKNOWN",
        gesture: Optional[str] = None,
        forearm_dy: float = 0.0,
        upright: Optional[bool] = None,
    ) -> Optional[Path]:
        """Save *frame* as JPEG plus a JSON sidecar. Returns saved path or None."""
        ts = time.time()
        stem = f"{ts:.3f}_{camera}_{event_type}".replace(" ", "_").replace("/", "_")
        jpg_path = self._dir / f"{stem}.jpg"
        meta_path = self._dir / f"{stem}.json"

        meta = {
            "timestamp": ts,
            "camera": camera,
            "event_type": event_type,
            "arm_state": arm_state,
            "gesture": gesture,
            "forearm_dy": round(forearm_dy, 4),
            "upright": upright,
        }

        try:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                return None
            with self._lock:
                jpg_path.write_bytes(buf.tobytes())
                meta_path.write_text(json.dumps(meta))
                self._evict()
            log.debug("Captured frame: %s", jpg_path.name)
            return jpg_path
        except Exception as exc:
            log.debug("FrameCapture.save error: %s", exc)
            return None

    def _evict(self) -> None:
        """Remove oldest files when over the limit. Must be called under self._lock."""
        jpgs = sorted(self._dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
        for p in jpgs[: max(0, len(jpgs) - self._max)]:
            p.unlink(missing_ok=True)
            meta = p.with_suffix(".json")
            if meta.exists():
                meta.unlink(missing_ok=True)

    def list_captures(self) -> list[dict]:
        """Return metadata for all saved captures, newest first."""
        result = []
        for meta_path in sorted(
            self._dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            try:
                data = json.loads(meta_path.read_text())
                data["filename"] = meta_path.stem + ".jpg"
                result.append(data)
            except Exception:
                pass
        return result

    def get_jpeg(self, filename: str) -> Optional[bytes]:
        """Return the raw JPEG bytes for *filename*, or None if not found."""
        p = self._dir / Path(filename).name
        if p.suffix != ".jpg" or not p.exists():
            return None
        try:
            return p.read_bytes()
        except Exception:
            return None
