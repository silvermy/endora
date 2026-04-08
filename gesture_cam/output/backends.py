"""
output/backends.py

Home Assistant is the primary output target when running as an add-on.

HABackend fires a HA event via the Supervisor REST API:
  POST http://supervisor/core/api/events/<event_name>
  Authorization: Bearer <SUPERVISOR_TOKEN>

The event fires with event_data so automations can branch on gesture type:
  {
    "gesture": "wave_left",
    "confidence": 0.91,
    "source_cameras": ["A", "B"],
    "timestamp": "2024-04-05T14:32:01.123456"
  }

In Home Assistant, listen with:
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: wave_left

A fallback PrintBackend is included for local testing outside Docker.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from cameras.analyser import Gesture

log = logging.getLogger(__name__)


class BaseBackend:
    def send(self, gesture: Gesture, confidence: float, sources: list[str] = None) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ── Home Assistant Supervisor API ─────────────────────────────────────────────

class HABackend(BaseBackend):
    """
    Fires a Home Assistant event via the internal Supervisor API.

    The SUPERVISOR_TOKEN env var is injected automatically by the HA
    Supervisor into every add-on container — no user configuration needed.

    The ha_url should be http://supervisor/core/api (the internal Docker
    network hostname, not the external IP).
    """

    def __init__(self, settings):
        self._token = os.environ.get("SUPERVISOR_TOKEN", "")
        if not self._token:
            log.warning(
                "SUPERVISOR_TOKEN not set — HA events will be rejected (401). "
                "This is expected when testing outside Home Assistant."
            )
        self._event_name = settings.ha_event_name
        self._base_url = settings.ha_url.rstrip("/")
        self._event_url = f"{self._base_url}/events/{self._event_name}"
        log.info("HA backend → %s", self._event_url)

    def send(self, gesture: Gesture, confidence: float, sources: list[str] = None) -> None:
        payload = json.dumps({
            "gesture": gesture.name.lower(),
            "confidence": round(confidence, 3),
            "source_cameras": sorted(sources or []),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }).encode("utf-8")

        req = urllib.request.Request(
            self._event_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                log.info(
                    "HA event fired: %s  confidence=%.2f  cameras=%s  status=%d",
                    gesture, confidence, sources, resp.status,
                )
        except urllib.error.HTTPError as e:
            log.error("HA API HTTP %d for gesture %s: %s", e.code, gesture, e.reason)
        except urllib.error.URLError as e:
            log.error("HA API unreachable: %s", e)


# ── Print (dev / test) ────────────────────────────────────────────────────────

class PrintBackend(BaseBackend):
    SYMBOLS = {
        Gesture.WAVE_LEFT:    "← WAVE LEFT ",
        Gesture.WAVE_RIGHT:   "→ WAVE RIGHT",
        Gesture.PALM_UP:      "↑ PALM UP   ",
        Gesture.PALM_DOWN:    "↓ PALM DOWN ",
        Gesture.FIST_GESTURE: "✊ FIST      ",
    }

    def send(self, gesture: Gesture, confidence: float, sources: list[str] = None) -> None:
        sym = self.SYMBOLS.get(gesture, str(gesture))
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {sym}  conf={confidence:.2f}  cams={sources}", flush=True)


# ── Factory ───────────────────────────────────────────────────────────────────

def make_backend(settings) -> BaseBackend:
    token = os.environ.get("SUPERVISOR_TOKEN", "")
    if token or settings.ha_url != "http://supervisor/core/api":
        # We're inside HA (token present) or user explicitly configured ha_url
        return HABackend(settings)
    log.info("No SUPERVISOR_TOKEN — using PrintBackend (dev mode)")
    return PrintBackend()
