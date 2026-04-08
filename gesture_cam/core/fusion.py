"""
core/fusion.py

Receives gesture candidates from both camera analysers and decides when
to emit a confirmed event to the Home Assistant backend.

Rules:
  1. Both cameras agree within fusion_agreement_window_s  → high-conf emit
  2. Single camera fires >= 2 candidates in the window    → accepted (e.g. if
     the second camera has a blocked view)
  3. Per-gesture cooldown prevents rapid repeat firing
  4. Source camera list is passed through to the HA event payload
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Tuple

from cameras.analyser import Gesture

log = logging.getLogger(__name__)


class GestureFusion:
    def __init__(self, settings, on_gesture: Callable[[Gesture, float, list], None]):
        self.s = settings
        self.on_gesture = on_gesture
        self._lock = threading.Lock()

        self._pending: Dict[Gesture, List[Tuple[float, float, str]]] = {
            g: [] for g in Gesture
        }
        self._last_emitted: Dict[Gesture, float] = {g: 0.0 for g in Gesture}
        self.total_emitted = 0

    def receive(self, gesture: Gesture, confidence: float, source: str):
        with self._lock:
            now = time.monotonic()
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
            single_ok  = len(candidates) >= 2

            if not (both_agree or single_ok):
                return
            if now - self._last_emitted[gesture] < self.s.cooldown_s:
                return

            final_conf = min(1.0, avg_conf * (1.15 if both_agree else 1.0))
            self._last_emitted[gesture] = now
            self._pending[gesture].clear()
            self.total_emitted += 1

            log.info("✓ GESTURE: %s  confidence=%.2f  cameras=%s",
                     gesture, final_conf, sorted(sources))

        self.on_gesture(gesture, final_conf, sources)
