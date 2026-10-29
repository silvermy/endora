"""
cameras/frame_capture.py — Save annotated debug frames on gesture events.

Frames (JPEG) and metadata (JSON) are written to /data/debug_frames/.
A rolling max of MAX_FRAMES pairs is kept; oldest are evicted automatically.

Gesture photos are the exception. Every arm-state change also lands in that
rolling set, so 200 frames covered about an hour and a half on the Jetson:
195 of them state changes, five gestures. A fire's photo was gone long before
anyone sat down to label it, and the trace alone could not say whether a
"not intended" snap was a hand behind the head or a hallucinated arm. So a
gesture frame, and the state frame just before it (for a held pose that is the
photo of the pose itself, see debug_server._photo_for), are moved to KEEP_DIR
beside the keypoint traces. About 100 KB each.

Kept photos of fires nobody labelled are deleted after KEEP_UNLABELLED_DAYS;
by then nobody is going to judge them. A labelled fire's photos stay as long
as its trace, which is forever: they are what a later disagreement about that
label gets settled with.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

CAPTURE_DIR = Path("/data/debug_frames")
MAX_FRAMES = 200
KEEP_DIR = Path(os.environ.get("ENDORA_CAPTURE_DIR", "/data/test_captures")) / "photos"
# How far back a gesture reaches for the state frame to keep with it. Matches
# the window debug_server._photo_for searches for a held pose's photo.
KEEP_STATE_BEFORE_S = 4.0
KEEP_UNLABELLED_DAYS = 30
_EXPIRE_EVERY_S = 24 * 3600
# A photo belongs to a labelled capture when the capture's timestamp (whole
# seconds, from the trace's filename) is this close: the gesture frame lands
# within a second or two of the trace, the state frame up to
# KEEP_STATE_BEFORE_S before it.
_LABEL_MATCH_S = KEEP_STATE_BEFORE_S + 2.0


class FrameCapture:
    """Thread-safe frame saver for post-hoc gesture debugging."""

    def __init__(
        self,
        capture_dir: Path = CAPTURE_DIR,
        max_frames: int = MAX_FRAMES,
        keep_dir: Path = KEEP_DIR,
    ) -> None:
        self._dir = Path(capture_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max = max_frames
        self._keep_dir = Path(keep_dir)
        self._lock = threading.Lock()
        self._last_expiry = 0.0

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
                if event_type.startswith("gesture_"):
                    jpg_path = self._keep(stem, ts)
                self._evict()
            log.debug("Captured frame: %s", jpg_path.name)
            if ts - self._last_expiry >= _EXPIRE_EVERY_S:
                # The first save after startup, then daily. Off the camera
                # loop: it reads the label store and lists every kept photo.
                self._last_expiry = ts
                threading.Thread(target=self.expire_kept, name="photo-expiry",
                                 daemon=True).start()
            return jpg_path
        except Exception as exc:
            log.debug("FrameCapture.save error: %s", exc)
            return None

    def _keep(self, stem: str, ts: float) -> Path:
        """Move this gesture frame, and the last state frame before it, out
        of the rolling set. Must be called under self._lock. Returns where the
        gesture frame now is; on any failure it stays where it was."""
        try:
            self._keep_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            log.warning("FrameCapture: cannot create %s: %s", self._keep_dir, exc)
            return self._dir / f"{stem}.jpg"
        stems = [stem]
        states = []
        for meta_path in self._dir.glob("*_state_*.json"):
            try:
                t = float(meta_path.name.split("_", 1)[0])
            except ValueError:
                continue
            if ts - KEEP_STATE_BEFORE_S <= t <= ts:
                states.append((t, meta_path.stem))
        if states:
            stems.append(max(states)[1])
        for s in stems:
            for suffix in (".jpg", ".json"):
                try:
                    (self._dir / f"{s}{suffix}").replace(self._keep_dir / f"{s}{suffix}")
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    log.warning("FrameCapture: could not keep %s%s: %s", s, suffix, exc)
        kept = self._keep_dir / f"{stem}.jpg"
        return kept if kept.exists() else self._dir / f"{stem}.jpg"

    def expire_kept(self, now: Optional[float] = None) -> int:
        """Delete kept photos older than KEEP_UNLABELLED_DAYS that belong to
        no labelled capture. Returns how many photos were deleted."""
        if not self._keep_dir.is_dir():
            return 0
        from cameras import labeler
        now = time.time() if now is None else now
        cutoff = now - KEEP_UNLABELLED_DAYS * 86400
        labelled = sorted(
            labeler._parse_name(name)["ts"] for name, v in labeler.get_labels().items()
            if v.get("label") and labeler._parse_name(name)["ts"] is not None)
        removed = 0
        with self._lock:
            for jpg in self._keep_dir.glob("*.jpg"):
                try:
                    t = float(jpg.name.split("_", 1)[0])
                except ValueError:
                    continue
                if t >= cutoff:
                    continue
                if any(abs(t - lt) <= _LABEL_MATCH_S for lt in labelled):
                    continue
                jpg.unlink(missing_ok=True)
                jpg.with_suffix(".json").unlink(missing_ok=True)
                removed += 1
        if removed:
            log.info("FrameCapture: deleted %d unlabelled photos older than %d days",
                     removed, KEEP_UNLABELLED_DAYS)
        return removed

    def _evict(self) -> None:
        """Remove oldest files when over the limit. Must be called under self._lock."""
        jpgs = sorted(self._dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
        for p in jpgs[: max(0, len(jpgs) - self._max)]:
            p.unlink(missing_ok=True)
            meta = p.with_suffix(".json")
            if meta.exists():
                meta.unlink(missing_ok=True)

    def list_captures(self) -> list[dict]:
        """Return metadata for all saved captures, kept ones included, newest
        first."""
        result = []
        metas = list(self._dir.glob("*.json"))
        if self._keep_dir.is_dir():
            metas += self._keep_dir.glob("*.json")
        for meta_path in metas:
            try:
                data = json.loads(meta_path.read_text())
                data["filename"] = meta_path.stem + ".jpg"
                result.append(data)
            except Exception:
                pass
        result.sort(key=lambda d: float(d.get("timestamp", 0)), reverse=True)
        return result

    def get_jpeg(self, filename: str) -> Optional[bytes]:
        """Return the raw JPEG bytes for *filename*, or None if not found."""
        name = Path(filename).name
        p = self._dir / name
        if not p.exists():
            p = self._keep_dir / name
        if p.suffix != ".jpg" or not p.exists():
            return None
        try:
            return p.read_bytes()
        except Exception:
            return None
