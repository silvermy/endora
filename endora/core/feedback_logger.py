"""
core/feedback_logger.py

Gesture feedback collection for threshold tuning.

Records:
  - Every gesture that fires (label="fired") with the ArmReading that triggered it.
  - User-marked false positives: press 'n' within 5s of a gesture to mark it wrong.
  - User-marked false negatives: press 'f' at any time to capture the last few
    readings as a missed gesture (ring buffer of recent frames).
  - Near-misses: frames where snap condition almost triggered (logged from
    state machine when arm is up but threshold not met).

Output: JSONL at /data/feedback.jsonl (or ./feedback.jsonl outside HA).
Each line is one labeled event — load with pandas.read_json(lines=True) for analysis.

Keyboard controls (requires a TTY — no-op in Docker without one):
  n  — mark last fired gesture as FALSE POSITIVE
  f  — mark recent readings as FALSE NEGATIVE (missed gesture)
  ?  — print a summary of counts so far
"""
from __future__ import annotations

import collections
import dataclasses
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Deque, Optional

log = logging.getLogger(__name__)

_DATA_DIR = Path("/data") if Path("/data").exists() else Path(".")
_LOG_PATH = _DATA_DIR / "feedback.jsonl"

# How long after a gesture fires to accept 'n' (false-positive mark).
_FP_WINDOW_S = 5.0
# How many recent ArmReadings to snapshot for false-negative reports.
_FN_BUFFER_SIZE = 30


@dataclasses.dataclass
class _ReadingSnapshot:
    ts: float
    state: str
    forearm_dy: float
    snap_roll: float
    upright: bool
    raised_side: Optional[str]


def _snap(reading) -> dict:
    """Serialise an ArmReading (or None) to a plain dict."""
    if reading is None:
        return {}
    return {
        "state": reading.state.name,
        "forearm_dy": round(float(reading.forearm_dy), 4),
        "snap_roll": round(float(reading.snap_roll), 4),
        "upright": bool(reading.upright),
        "raised_side": reading.raised_side.name if reading.raised_side else None,
    }


class FeedbackLogger:
    """Thread-safe feedback logger with optional keyboard listener."""

    def __init__(self):
        self._lock = threading.Lock()
        self._fh = self._open_log()

        # Ring buffer of recent ArmReadings for false-negative snapshots.
        self._recent: Deque[tuple[float, dict]] = collections.deque(maxlen=_FN_BUFFER_SIZE)

        # Last fired event info (for false-positive marking).
        self._last_gesture: Optional[str] = None
        self._last_gesture_ts: float = 0.0
        self._last_gesture_reading: dict = {}

        # Counters
        self._counts: dict[str, int] = collections.defaultdict(int)

        self._kb_thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def push_reading(self, reading) -> None:
        """Call every frame with the current ArmReading (may be None)."""
        if reading is None:
            return
        with self._lock:
            self._recent.append((time.monotonic(), _snap(reading)))

    def on_gesture_fired(self, gesture_name: str, confidence: float, reading=None) -> None:
        """Call when a gesture fires. Uses most recent buffered reading if none given."""
        now = time.monotonic()
        with self._lock:
            if reading is not None:
                reading_dict = _snap(reading)
            elif self._recent:
                _, reading_dict = self._recent[-1]
            else:
                reading_dict = {}
            entry = {
                "ts": time.time(),
                "label": "fired",
                "gesture": gesture_name,
                "confidence": round(confidence, 3),
                "reading": reading_dict,
            }
            self._last_gesture = gesture_name
            self._last_gesture_ts = now
            self._last_gesture_reading = reading_dict
            self._write(entry)
            self._counts["fired"] += 1

    def on_near_miss(self, gesture_name: str, reason: str, reading) -> None:
        """Call when detection almost fired but a threshold blocked it."""
        entry = {
            "ts": time.time(),
            "label": "near_miss",
            "gesture": gesture_name,
            "reason": reason,
            "reading": _snap(reading),
        }
        with self._lock:
            self._write(entry)
            self._counts["near_miss"] += 1

    def mark_false_positive(self) -> bool:
        """Mark the most recent gesture as a false positive. Returns True if within window."""
        now = time.monotonic()
        with self._lock:
            if not self._last_gesture or (now - self._last_gesture_ts) > _FP_WINDOW_S:
                log.info("[feedback] No recent gesture to mark as false positive "
                         "(window is %.0fs)", _FP_WINDOW_S)
                return False
            entry = {
                "ts": time.time(),
                "label": "false_positive",
                "gesture": self._last_gesture,
                "reading": self._last_gesture_reading,
                "seconds_after_fire": round(now - self._last_gesture_ts, 2),
            }
            self._write(entry)
            self._counts["false_positive"] += 1
            age = round(now - self._last_gesture_ts, 1)
            log.info("[feedback] Marked %s (%.1fs ago) as FALSE POSITIVE",
                     self._last_gesture, age)
            self._last_gesture = None  # consumed
        return True

    def mark_wrong_gesture(self, intended: str = "unknown") -> bool:
        """Mark the most recent gesture as the wrong type. Returns True if within window."""
        now = time.monotonic()
        with self._lock:
            if not self._last_gesture or (now - self._last_gesture_ts) > _FP_WINDOW_S:
                log.info("[feedback] No recent gesture to mark as wrong gesture "
                         "(window is %.0fs)", _FP_WINDOW_S)
                return False
            entry = {
                "ts": time.time(),
                "label": "wrong_gesture",
                "gesture_fired": self._last_gesture,
                "gesture_intended": intended,
                "reading": self._last_gesture_reading,
                "seconds_after_fire": round(now - self._last_gesture_ts, 2),
            }
            self._write(entry)
            self._counts["wrong_gesture"] = self._counts.get("wrong_gesture", 0) + 1
            log.info("[feedback] Marked %s as WRONG GESTURE (intended: %s)",
                     self._last_gesture, intended)
            self._last_gesture = None
        return True

    def mark_no_pose(self) -> None:
        """I was visible on camera but YOLO didn't detect me at all."""
        entry = {
            "ts": time.time(),
            "label": "no_pose",
        }
        with self._lock:
            self._write(entry)
            self._counts["no_pose"] += 1
        log.info("[feedback] Recorded NO POSE DETECTED")

    def mark_false_negative(self, gesture_hint: str = "unknown") -> None:
        """Snapshot the recent reading buffer as a missed gesture."""
        now = time.monotonic()
        with self._lock:
            recent = list(self._recent)
        entry = {
            "ts": time.time(),
            "label": "false_negative",
            "gesture_hint": gesture_hint,
            "recent_readings": [r for _, r in recent[-10:]],  # last 10 frames
        }
        with self._lock:
            self._write(entry)
            self._counts["false_negative"] += 1
        log.info("[feedback] Recorded FALSE NEGATIVE (captured %d recent frames)",
                 len(recent))

    def reset_counts(self) -> None:
        """Reset event counters after the log has been downloaded and cleared."""
        with self._lock:
            self._counts.clear()

    def print_summary(self) -> None:
        with self._lock:
            counts = dict(self._counts)
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        log.info("[feedback] Summary — %s | log: %s", parts or "no events yet", _LOG_PATH)

    def start_keyboard_listener(self) -> None:
        """Start a background thread that reads keypresses from stdin (TTY only)."""
        if not sys.stdin.isatty():
            log.debug("[feedback] stdin is not a TTY — keyboard listener disabled")
            return
        self._kb_thread = threading.Thread(
            target=self._kb_loop, name="feedback-kb", daemon=True
        )
        self._kb_thread.start()
        log.info("[feedback] Keyboard feedback active — "
                 "press 'n'=false-positive  'f'=false-negative  '?'=summary")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _kb_loop(self) -> None:
        try:
            import tty
            import termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            tty.setraw(fd)
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if ch == "n":
                        self.mark_false_positive()
                    elif ch == "f":
                        self.mark_false_negative()
                    elif ch == "?":
                        self.print_summary()
                    elif ch in ("\x03", "\x04", "q"):
                        break  # Ctrl-C / Ctrl-D / q — let the main loop handle it
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception as exc:
            log.debug("[feedback] Keyboard listener stopped: %s", exc)

    def _open_log(self):
        try:
            fh = open(_LOG_PATH, "a", buffering=1)
            log.info("[feedback] Logging to %s", _LOG_PATH)
            return fh
        except OSError as exc:
            log.warning("[feedback] Cannot open %s: %s — feedback disabled", _LOG_PATH, exc)
            return None

    def _write(self, entry: dict) -> None:
        if self._fh:
            self._fh.write(json.dumps(entry) + "\n")
