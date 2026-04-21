"""
core/fusion.py

Two-camera gesture fusion with single-camera fallback.

Single-camera mode activates automatically when both RTSP URLs are identical,
or when single_camera_mode is set to true in config. In this mode a gesture
fires as soon as one camera sustains it, without waiting for a second source.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Tuple

from core.state_machine import Gesture

log = logging.getLogger(__name__)


class GestureFusion:
    def __init__(self, settings, on_gesture: Callable[[Gesture, float, list], None]):
        self.s = settings
        self.on_gesture = on_gesture
        self._lock = threading.Lock()

        # Detect single-camera mode automatically
        self._single_cam = (
            settings.single_camera_mode or
            settings.rtsp_url_a == settings.rtsp_url_b
        )
        if self._single_cam:
            log.info("Single-camera mode active — gesture fires from one source")
        else:
            log.info("Dual-camera mode active — both cameras used for fusion")

        self._pending: Dict[Gesture, List[Tuple[float, float, str]]] = {
            g: [] for g in Gesture
        }
        self._last_emitted: Dict[Gesture, float] = {g: 0.0 for g in Gesture}
        # Tracks the last time ANY gesture fired — blocks all gestures during cooldown.
        # This prevents a second gesture (e.g. wave) firing immediately after the
        # first (e.g. snap) because the motion that triggered the first gesture
        # produces residual velocity that crosses the wave threshold a frame later.
        self._last_emitted_any: float = 0.0
        self.total_emitted = 0

    def receive(self, gesture: Gesture, confidence: float, source: str):
        with self._lock:
            now = time.monotonic()

            # ── Global cooldown check FIRST ───────────────────────────────
            # A single cooldown window covers ALL gesture types.  Per-gesture
            # clocks are kept for logging/stats but the global gate fires first.
            global_elapsed = now - self._last_emitted_any
            if global_elapsed < self.s.cooldown_s:
                log.debug("Gesture %s suppressed by global cooldown (%.1fs remaining)",
                          gesture, self.s.cooldown_s - global_elapsed)
                return

            # Per-gesture cooldown (secondary, guards same-gesture repeats)
            if now - self._last_emitted[gesture] < self.s.cooldown_s:
                log.debug("Gesture %s suppressed by per-gesture cooldown (%.1fs remaining)",
                          gesture,
                          self.s.cooldown_s - (now - self._last_emitted[gesture]))
                return

            window = self.s.fusion_agreement_window_s
            self._pending[gesture] = [
                (ts, conf, src)
                for ts, conf, src in self._pending[gesture]
                if now - ts < window
            ]
            self._pending[gesture].append((now, confidence, source))

            candidates = self._pending[gesture]
            sources = list({src for _, _, src in candidates})
            avg_conf = sum(c for _, c, _ in candidates) / len(candidates)

            both_agree = len(sources) >= 2
            if self._single_cam:
                # Analyser sustain already filters noise — fire immediately
                should_emit = len(candidates) >= 1
            else:
                should_emit = both_agree or len(candidates) >= 3

            if not should_emit:
                return

            boost = 1.15 if both_agree else 1.0
            final_conf = min(1.0, avg_conf * boost)

            # Mark emitted BEFORE calling callback to block any re-entrant
            # candidates that arrive while the callback is executing
            self._last_emitted[gesture] = now
            self._last_emitted_any = now
            self._pending[gesture].clear()
            self.total_emitted += 1

            log.info("✓ GESTURE: %s  confidence=%.2f  cameras=%s",
                     gesture, final_conf, sorted(sources))

        # Call outside the lock
        self.on_gesture(gesture, final_conf, sources)
