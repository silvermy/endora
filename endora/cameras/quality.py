"""
cameras/quality.py

How well people are being seen, hour by hour.

The question "is detection worse in the dark?" could only be answered from
the recordings, and those exist only when a gesture fired — so they are
biased towards the moments detection worked. A dark-room failure leaves no
recording at all. This measures every tracked detection instead, whether or
not anything fires, and summarises it once an hour:

* how confidently the pose model sees tracked people (box confidence,
  upper-body keypoint confidence, how often both wrists are clearly seen);
* how often they are lost (dropouts of 0.3 s or more, tracks dropped);
* the lighting: how much of the hour the camera spent in its black-and-white
  infrared night mode, and how bright the picture was.

Summaries go to the log and to a JSON-lines file that outlives restarts and
the 300-line log buffer, so nights can be compared with days — and an IR
illuminator, if one is added, with no illuminator.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

QUALITY_PATH = Path("/data/quality.jsonl")

# A frame whose MEDIAN pixel has channels differing by less than this is black
# and white — the camera's IR night mode. Measured on this camera's photos: IR
# frames 0.0; colour frames 2.5-3.0 by lamplight in the evening (its colour is
# muted) and 4-9.5 by day. The median, because a few coloured objects — or the
# debug overlay's boxes on a saved photo — barely move it, where they dragged
# the mean of an IR frame up to 5.
_GREY_COLOURFULNESS = 1.0

# Upper-body COCO keypoints: nose, shoulders, elbows, wrists.
_UPPER = [0, 5, 6, 7, 8, 9, 10]
_WRISTS = [9, 10]


class HourlyQuality:
    """Accumulates one hour at a time; returns a summary when the hour turns."""

    def __init__(self, label: str = "A",
                 clock: Callable[[], datetime.datetime] = datetime.datetime.now,
                 path: Optional[Path] = QUALITY_PATH) -> None:
        self.label = label
        self._clock = clock
        self._path = path
        self._hour = self._hour_key()
        self._reset()

    def _hour_key(self) -> str:
        return self._clock().strftime("%Y-%m-%d %H:00")

    def _reset(self) -> None:
        self._box: list = []
        self._kp: list = []
        self._wrists = 0
        self._samples = 0
        self._frames = 0
        self._grey_frames = 0
        self._brightness = 0.0
        self._dropouts = 0
        self._lost = 0

    # ── feeding ───────────────────────────────────────────────────────────

    def add_frame(self, small_bgr: np.ndarray) -> None:
        """Lighting, from any small BGR copy of the frame."""
        # Every fourth pixel each way: plenty to judge colour and brightness,
        # and keeps the median off the loop's budget (it cost ~3 ms a frame
        # over the whole 320-wide image).
        f = small_bgr[::4, ::4].astype(np.int16)
        b, g, r = f[..., 0], f[..., 1], f[..., 2]
        colourfulness = float(np.median(np.abs(b - g) + np.abs(g - r)) / 2.0)
        self._frames += 1
        self._grey_frames += colourfulness < _GREY_COLOURFULNESS
        self._brightness += float(f.mean())

    def add_detection(self, kps_row: np.ndarray, box_conf: Optional[float]) -> None:
        """One detection of a tracked person, on one pose-model run."""
        self._samples += 1
        if box_conf is not None:
            self._box.append(float(box_conf))
        self._kp.append(float(np.median(kps_row[_UPPER, 2])))
        self._wrists += bool((kps_row[_WRISTS, 2] > 0.5).all())

    def note_dropout(self) -> None:
        self._dropouts += 1

    def note_lost(self) -> None:
        self._lost += 1

    # ── reporting ─────────────────────────────────────────────────────────

    def tick(self) -> Optional[dict]:
        """Call often. When the hour has turned, write and return the
        summary of the hour just finished; otherwise None."""
        now = self._hour_key()
        if now == self._hour:
            return None
        summary = self.summary()
        self._hour = now
        self._reset()
        if summary["frames"]:
            self._write(summary)
        return summary

    def summary(self) -> dict:
        n = self._samples
        return {
            "hour": self._hour,
            "camera": self.label,
            "frames": self._frames,
            "ir_fraction": round(self._grey_frames / self._frames, 2) if self._frames else None,
            "brightness": round(self._brightness / self._frames, 1) if self._frames else None,
            "tracked_samples": n,
            "box_conf": round(float(np.median(self._box)), 3) if self._box else None,
            "upper_body_conf": round(float(np.median(self._kp)), 3) if self._kp else None,
            "wrists_seen": round(self._wrists / n, 2) if n else None,
            "dropouts": self._dropouts,
            "lost": self._lost,
        }

    def _write(self, s: dict) -> None:
        log.info("[%s] quality %s: %d tracked samples, box conf %s, upper body %s, "
                 "wrists seen %s, %d dropouts, %d lost, IR %s of the hour, "
                 "brightness %s", self.label, s["hour"], s["tracked_samples"],
                 s["box_conf"], s["upper_body_conf"], s["wrists_seen"],
                 s["dropouts"], s["lost"], s["ir_fraction"], s["brightness"])
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a") as fh:
                fh.write(json.dumps(s) + "\n")
        except Exception as e:
            log.warning("quality: could not write %s: %s", self._path, e)


def read_recent(limit: int = 24 * 14, path: Path = QUALITY_PATH) -> list:
    """The last *limit* hourly summaries, oldest first."""
    try:
        lines = path.read_text().splitlines()[-limit:]
    except Exception:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out
