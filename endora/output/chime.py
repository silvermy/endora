"""
output/chime.py

Plays a short audio clip on any HA-integrated speaker when Endora detects
an arm-up transition — immediate audible feedback before the gesture fires.

Uses Home Assistant's media_player.play_media service with announce=true,
which overlays the clip on whatever is currently playing (TV, music, etc.)
and resumes automatically. Works with any speaker HA knows about: Sonos,
Chromecast, Echo, HomePod, Spotify Connect, DLNA, etc.

Config:
    chime_enable: true
    chime_entity_id: "media_player.living_room"
    chime_volume: 40        # 0-100
    chime_debounce_s: 4.0   # min seconds between chimes
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

log = logging.getLogger(__name__)

_POST_TIMEOUT = 3.0


class ChimeNotifier:
    """Send an audio clip to any HA media_player on arm-up transitions."""

    def __init__(self, settings, chime_url: str):
        self._chime_url  = chime_url
        self._entity_id  = getattr(settings, "chime_entity_id", "")
        self._volume     = int(getattr(settings, "chime_volume", 40))
        self._debounce_s = float(getattr(settings, "chime_debounce_s", 4.0))
        self._last_played = 0.0
        self._lock       = threading.Lock()

        token = (
            os.environ.get("SUPERVISOR_TOKEN") or
            os.environ.get("HA_TOKEN") or ""
        )
        ha_url = (
            os.environ.get("HA_URL") or
            getattr(settings, "ha_url", "") or
            "http://supervisor/core/api"
        ).rstrip("/")

        self._service_url = f"{ha_url}/services/media_player/play_media"
        self._headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {token}",
        }

        if not token:
            log.warning("Chime: no HA token — calls will fail with 401")
        if not self._chime_url:
            log.warning("Chime: no chime URL resolved — chime disabled")
        elif not self._entity_id:
            log.warning("Chime: chime_entity_id not set — chime disabled. "
                        "Set it to e.g. 'media_player.living_room' in config.")
        else:
            log.info("Chime ready — entity=%s url=%s", self._entity_id, self._chime_url)

    def notify(self):
        """Play the chime if debounce allows. Non-blocking."""
        if not self._entity_id or not self._chime_url:
            return
        with self._lock:
            now = time.monotonic()
            if now - self._last_played < self._debounce_s:
                return
            self._last_played = now
        threading.Thread(target=self._post, daemon=True).start()

    def _post(self):
        if not self._entity_id or not self._chime_url:
            return
        payload = json.dumps({
            "entity_id":          self._entity_id,
            "media_content_id":   self._chime_url,
            "media_content_type": "music",
            "announce":           True,
            "extra":              {"volume": self._volume / 100.0},
        }).encode()
        req = urllib.request.Request(
            self._service_url, data=payload,
            headers=self._headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_POST_TIMEOUT) as r:
                log.info("Chime sent (status=%d entity=%s)", r.status, self._entity_id)
        except urllib.error.HTTPError as e:
            log.warning("Chime HTTP %d: %s", e.code, e.reason)
        except Exception as e:
            log.warning("Chime error: %s", e)


def make_chime_notifier(settings, chime_url: str) -> Optional[ChimeNotifier]:
    """Return a ChimeNotifier if chime is enabled, else None."""
    if not getattr(settings, "chime_enable", False):
        return None
    return ChimeNotifier(settings, chime_url)
