"""
output/backends.py

Fires Home Assistant events via the REST API.

Two auth modes — automatically selected:

  1. HA Add-on (Home Assistant OS / Supervised):
     SUPERVISOR_TOKEN is injected by the Supervisor.
     API endpoint: http://supervisor/core/api  (internal Docker network)

  2. Standalone Docker (Home Assistant Container / Core):
     HA_TOKEN must be set to a Long-Lived Access Token from your HA profile.
     HA_URL must point to your HA instance, e.g. http://192.168.1.x:8123/api

Event payload:
  {
    "gesture":        "endora-snap",
    "confidence":     0.91,
    "source_cameras": ["A"],
    "timestamp":      "2024-04-05T14:32:01.123456+00:00"
  }

HA automation trigger:
  platform: event
  event_type: gesture_detected
  event_data:
    gesture: endora-snap
"""

from __future__ import annotations

import collections
import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from cameras.analyser import Gesture

log = logging.getLogger(__name__)

# Hard minimum between any two firings of the same gesture at the output layer.
# This is a safety net on top of the fusion cooldown — protects against
# duplicate events reaching HA even if the fusion layer lets one through.
OUTPUT_DEBOUNCE_S = 2.0


class BaseBackend:
    def __init__(self):
        self._last_sent: dict[Gesture, float] = collections.defaultdict(float)

    def _debounced(self, gesture: Gesture) -> bool:
        """Returns True if this gesture should be suppressed."""
        now = time.monotonic()
        if now - self._last_sent[gesture] < OUTPUT_DEBOUNCE_S:
            log.debug("Output debounce suppressed %s", gesture)
            return True
        self._last_sent[gesture] = now
        return False

    def send(self, gesture: Gesture, confidence: float, sources: list = None) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ── Home Assistant REST API ───────────────────────────────────────────────────

class HABackend(BaseBackend):

    def __init__(self, settings):
        super().__init__()
        # Supervisor token takes priority (add-on mode)
        self._token = (
            os.environ.get("SUPERVISOR_TOKEN") or
            os.environ.get("HA_TOKEN") or
            ""
        )

        ha_url = (
            os.environ.get("HA_URL") or
            settings.ha_url or
            "http://supervisor/core/api"
        ).rstrip("/")

        self._event_name = settings.ha_event_name
        self._event_url = f"{ha_url}/events/{self._event_name}"

        if not self._token:
            log.warning(
                "No auth token found. Set SUPERVISOR_TOKEN (add-on) or "
                "HA_TOKEN (standalone Docker). Events will be rejected with 401."
            )
        else:
            mode = "add-on (Supervisor)" if os.environ.get("SUPERVISOR_TOKEN") else "standalone (Long-Lived Token)"
            log.info("HA backend ready — mode=%s url=%s event=%s",
                     mode, self._event_url, self._event_name)

    def send(self, gesture: Gesture, confidence: float, sources: list = None) -> None:
        if self._debounced(gesture):
            return

        payload = json.dumps({
            "gesture":        gesture.event_name,
            "confidence":     round(confidence, 3),
            "source_cameras": sorted(sources or []),
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        }).encode("utf-8")

        req = urllib.request.Request(
            self._event_url,
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self._token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                log.info("HA event fired: %s  conf=%.2f  cams=%s  status=%d",
                         gesture, confidence, sources, resp.status)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                log.error(
                    "HA API returned 401 Unauthorized. "
                    "Check your SUPERVISOR_TOKEN or HA_TOKEN is valid."
                )
            else:
                log.error("HA API HTTP %d for %s: %s", e.code, gesture, e.reason)
        except urllib.error.URLError as e:
            log.error("HA API unreachable (%s): %s", self._event_url, e)


# ── Print fallback (dev / no token) ──────────────────────────────────────────

class PrintBackend(BaseBackend):
    SYMBOLS = {
        Gesture.SNAP:        "☝️  endora-snap       ",
        Gesture.HOLD:        "✋  endora-hold       ",
        Gesture.DOUBLE_SNAP: "✌️  endora-double-snap",
        Gesture.THUMBS_UP:   "👍  endora-thumbs-up  ",
    }

    def __init__(self):
        super().__init__()

    def send(self, gesture: Gesture, confidence: float, sources: list = None) -> None:
        if self._debounced(gesture):
            return
        sym = self.SYMBOLS.get(gesture, str(gesture))
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {sym}  conf={confidence:.2f}  cams={sources}", flush=True)


# ── Factory ───────────────────────────────────────────────────────────────────

def make_backend(settings) -> BaseBackend:
    """
    Returns HABackend if any auth token is present or ha_url is configured.
    Falls back to PrintBackend for zero-config local testing.
    """
    has_supervisor = bool(os.environ.get("SUPERVISOR_TOKEN"))
    has_token      = bool(os.environ.get("HA_TOKEN"))
    has_custom_url = settings.ha_url != "http://supervisor/core/api"

    if has_supervisor or has_token or has_custom_url:
        return HABackend(settings)

    log.info("No HA token or custom URL configured — using PrintBackend (dev mode)")
    return PrintBackend()
