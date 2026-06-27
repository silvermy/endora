"""
output/chime.py

Plays a short audio clip on any speaker when Endora detects an arm-up
transition — giving immediate audible feedback before the gesture fires.

Two backends, auto-selected by config:

  1. HAChimeBackend  (default / recommended)
     Calls Home Assistant's media_player.play_media service.
     Works with any speaker HA knows about: Sonos, Chromecast, Echo,
     HomePod, Spotify Connect, DLNA, etc.
     Requires: chime_entity_id = "media_player.your_speaker"

  2. SonosDirectBackend  (standalone fallback)
     Calls the Sonos local REST API directly (no HA needed).
     Requires: sonos_ip = "192.168.x.y"  (or SSDP auto-discovery)

Selection logic:
  - chime_entity_id set  → HAChimeBackend
  - sonos_ip set         → SonosDirectBackend
  - neither set          → HAChimeBackend with SSDP discovery disabled;
                           logs a warning on first notify()

The clip plays on top of whatever the speaker is currently doing
(TV, music, etc.) and the speaker resumes automatically.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

log = logging.getLogger(__name__)

_CLIP_POST_TIMEOUT = 3.0
_SONOS_PORT        = 1443
_SSDP_ADDR         = "239.255.255.250"
_SSDP_PORT         = 1900


# ── Shared debounce mixin ─────────────────────────────────────────────────────

class _ChimeBase:
    def __init__(self, debounce_s: float):
        self._debounce_s   = debounce_s
        self._last_played  = 0.0
        self._debounce_lock = threading.Lock()

    def notify(self):
        """Fire the chime if debounce allows. Non-blocking."""
        with self._debounce_lock:
            now = time.monotonic()
            if now - self._last_played < self._debounce_s:
                return
            self._last_played = now
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self):
        raise NotImplementedError


# ── Backend 1: Home Assistant media_player ────────────────────────────────────

class HAChimeBackend(_ChimeBase):
    """
    Plays the chime via HA's media_player.play_media service call.
    announce=true ducks/overlays current playback on supported platforms.
    """

    def __init__(self, settings, chime_url: str):
        super().__init__(float(getattr(settings, "chime_debounce_s",
                                       getattr(settings, "sonos_debounce_s", 4.0))))
        self._chime_url   = chime_url
        self._entity_id   = getattr(settings, "chime_entity_id", "")
        self._volume      = int(getattr(settings, "chime_volume",
                                        getattr(settings, "sonos_volume", 30)))

        token = (
            os.environ.get("SUPERVISOR_TOKEN") or
            os.environ.get("HA_TOKEN") or
            ""
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
            log.warning("Chime: no HA token found — chime calls will fail with 401")
        if not self._entity_id:
            log.warning("Chime: chime_entity_id not set — chime disabled. "
                        "Set it to e.g. 'media_player.living_room' in config.")
        else:
            log.info("Chime: HA backend ready — entity=%s url=%s",
                     self._entity_id, self._chime_url)

    def _play(self):
        if not self._entity_id:
            return
        payload = json.dumps({
            "entity_id":          self._entity_id,
            "media_content_id":   self._chime_url,
            "media_content_type": "music",
            "announce":           True,
            "extra": {"volume": self._volume / 100.0},
        }).encode()
        req = urllib.request.Request(
            self._service_url, data=payload,
            headers=self._headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_CLIP_POST_TIMEOUT) as r:
                log.info("Chime sent via HA (status=%d entity=%s)", r.status, self._entity_id)
        except urllib.error.HTTPError as e:
            log.warning("Chime HA HTTP %d: %s", e.code, e.reason)
        except Exception as e:
            log.warning("Chime HA error: %s", e)


# ── Backend 2: Sonos local API (standalone fallback) ──────────────────────────

def _ssl_no_verify():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _ssdp_find_sonos(timeout: float = 2.0) -> Optional[str]:
    msg = (
        "M-SEARCH * HTTP/1.1\r\n"
        f"HOST: {_SSDP_ADDR}:{_SSDP_PORT}\r\n"
        "MAN: \"ssdp:discover\"\r\n"
        f"MX: {int(timeout)}\r\n"
        "ST: urn:schemas-upnp-org:device:ZonePlayer:1\r\n\r\n"
    ).encode()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.settimeout(timeout + 0.5)
        s.sendto(msg, (_SSDP_ADDR, _SSDP_PORT))
        _, addr = s.recvfrom(4096)
        s.close()
        log.info("Sonos SSDP: found %s", addr[0])
        return addr[0]
    except socket.timeout:
        log.warning("Sonos SSDP: no device found within %ss", timeout)
        return None
    except OSError as e:
        log.warning("Sonos SSDP error: %s", e)
        return None


def _sonos_player_id(ip: str) -> Optional[str]:
    url = f"https://{ip}:{_SONOS_PORT}/api/v1/players/local/info"
    try:
        with urllib.request.urlopen(url, timeout=4.0, context=_ssl_no_verify()) as r:
            data = json.loads(r.read())
            pid = data.get("playerId") or data.get("id")
            if pid:
                log.info("Sonos player ID: %s", pid)
            return pid
    except Exception as e:
        log.warning("Sonos player-ID fetch failed (%s): %s", ip, e)
        return None


class SonosDirectBackend(_ChimeBase):
    """Plays the chime via the Sonos local audioClip REST API."""

    def __init__(self, settings, chime_url: str):
        super().__init__(float(getattr(settings, "chime_debounce_s",
                                       getattr(settings, "sonos_debounce_s", 4.0))))
        self._chime_url  = chime_url
        self._volume     = int(getattr(settings, "chime_volume",
                                       getattr(settings, "sonos_volume", 30)))
        self._ip: Optional[str]        = getattr(settings, "sonos_ip", "") or None
        self._player_id: Optional[str] = getattr(settings, "sonos_player_id", "") or None
        self._resolved   = bool(self._ip and self._player_id)
        self._res_lock   = threading.Lock()

        if self._resolved:
            log.info("Sonos direct: ip=%s player=%s", self._ip, self._player_id)
        else:
            threading.Thread(target=self._resolve, daemon=True).start()

    def _resolve(self):
        with self._res_lock:
            if self._resolved:
                return
            if not self._ip:
                self._ip = _ssdp_find_sonos()
            if self._ip and not self._player_id:
                self._player_id = _sonos_player_id(self._ip)
            self._resolved = bool(self._ip and self._player_id)
            if not self._resolved:
                log.warning("Sonos direct: could not resolve device — chime disabled")

    def _play(self):
        if not self._resolved:
            self._res_lock.acquire(timeout=5.0)
            try:
                pass
            finally:
                try:
                    self._res_lock.release()
                except RuntimeError:
                    pass
        if not (self._resolved and self._ip and self._player_id):
            return
        url = (f"https://{self._ip}:{_SONOS_PORT}"
               f"/api/v1/players/{self._player_id}/audioClip")
        payload = json.dumps({
            "name": "endora", "appId": "com.endora.gesture",
            "streamUrl": self._chime_url,
            "volume": self._volume, "clipType": "CUSTOM",
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_CLIP_POST_TIMEOUT,
                                        context=_ssl_no_verify()) as r:
                log.info("Sonos clip posted (status=%d)", r.status)
        except urllib.error.HTTPError as e:
            log.warning("Sonos clip HTTP %d: %s", e.code, e.reason)
        except Exception as e:
            log.warning("Sonos clip error: %s", e)


# ── Factory ───────────────────────────────────────────────────────────────────

def make_chime_notifier(settings, chime_url: str) -> Optional[_ChimeBase]:
    """
    Return the appropriate chime backend, or None if chime is disabled.

    Priority:
      1. chime_entity_id set  → HAChimeBackend  (any HA-integrated speaker)
      2. sonos_ip set         → SonosDirectBackend  (Sonos without HA)
      3. neither              → HAChimeBackend with a warning (entity_id missing)
    """
    if not getattr(settings, "chime_enable",
                   getattr(settings, "sonos_enable", False)):
        return None

    entity_id = getattr(settings, "chime_entity_id", "")
    sonos_ip  = getattr(settings, "sonos_ip", "")

    if entity_id:
        log.info("Chime: using HA media_player backend (entity=%s)", entity_id)
        return HAChimeBackend(settings, chime_url)
    elif sonos_ip:
        log.info("Chime: using Sonos direct backend (ip=%s)", sonos_ip)
        return SonosDirectBackend(settings, chime_url)
    else:
        log.info("Chime: using HA media_player backend (entity_id not yet set)")
        return HAChimeBackend(settings, chime_url)
