"""
output/sonos.py

Plays a short audio clip on a Sonos speaker via the local Sonos REST API
(audioClip endpoint).  The clip plays on top of whatever is already playing
(TV, music, etc.) then the speaker resumes automatically.

Usage
-----
The notifier is triggered on arm-up state transitions (SINGLE_UP / BOTH_UP)
so users get immediate haptic-style feedback before the gesture fully fires.

Discovery
---------
If ``sonos_ip`` is not set, the module will attempt a one-shot SSDP discovery
on the LAN and cache the first Sonos device it finds.  Set ``sonos_ip`` in
config to skip discovery and reduce startup latency.

The ``sonos_player_id`` is fetched automatically from the device's local API
the first time a clip is requested, then cached for the life of the process.

Sonos local API reference
-------------------------
POST http://{ip}:1443/api/v1/players/{playerId}/audioClip
{
  "name":      "endora",
  "appId":     "com.endora.gesture",
  "streamUrl": "http://{endora_ip}:{port}/chime.wav",
  "volume":    30,
  "clipType":  "CUSTOM"
}
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

log = logging.getLogger(__name__)

_SONOS_CLIP_PORT = 1443
_SSDP_ADDR = "239.255.255.250"
_SSDP_PORT = 1900
_SSDP_MX   = 2          # seconds to wait for SSDP responses
_PLAYER_FETCH_TIMEOUT = 4.0
_CLIP_POST_TIMEOUT    = 3.0


# ── SSDP discovery ────────────────────────────────────────────────────────────

def _ssdp_discover_sonos(timeout: float = _SSDP_MX) -> Optional[str]:
    """Return the IP of the first Sonos device found on the LAN, or None."""
    msg = (
        "M-SEARCH * HTTP/1.1\r\n"
        f"HOST: {_SSDP_ADDR}:{_SSDP_PORT}\r\n"
        "MAN: \"ssdp:discover\"\r\n"
        f"MX: {int(timeout)}\r\n"
        "ST: urn:schemas-upnp-org:device:ZonePlayer:1\r\n"
        "\r\n"
    ).encode()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(timeout + 0.5)
        sock.sendto(msg, (_SSDP_ADDR, _SSDP_PORT))
        data, addr = sock.recvfrom(4096)
        sock.close()
        log.info("Sonos SSDP: found device at %s", addr[0])
        return addr[0]
    except socket.timeout:
        log.warning("Sonos SSDP: no device found within %ss", timeout)
        return None
    except OSError as e:
        log.warning("Sonos SSDP error: %s", e)
        return None


# ── Player-ID fetch ───────────────────────────────────────────────────────────

def _fetch_player_id(ip: str) -> Optional[str]:
    """Ask the Sonos device for its player ID via the local REST API."""
    url = f"https://{ip}:{_SONOS_CLIP_PORT}/api/v1/players/local/info"
    ctx = _ssl_no_verify_ctx()
    try:
        with urllib.request.urlopen(url, timeout=_PLAYER_FETCH_TIMEOUT, context=ctx) as r:
            data = json.loads(r.read())
            pid = data.get("playerId") or data.get("id")
            if pid:
                log.info("Sonos player ID: %s", pid)
                return pid
    except Exception as e:
        log.warning("Sonos player-ID fetch failed (%s): %s", ip, e)
    return None


def _ssl_no_verify_ctx():
    """Return an SSL context that skips certificate verification.

    Sonos local API uses a self-signed cert — verification would always fail.
    """
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ── Main notifier class ───────────────────────────────────────────────────────

class SonosNotifier:
    """
    Send an audio clip to a Sonos speaker on arm-up state transitions.

    Parameters
    ----------
    settings   : Settings object (reads sonos_* fields)
    chime_url  : Full URL Sonos can reach to fetch the WAV, e.g.
                 ``http://192.168.1.50:8765/chime.wav``
    """

    def __init__(self, settings, chime_url: str):
        self._chime_url   = chime_url
        self._volume      = int(getattr(settings, "sonos_volume", 30))
        self._debounce_s  = float(getattr(settings, "sonos_debounce_s", 4.0))
        self._last_played: float = 0.0
        self._lock        = threading.Lock()

        # IP / player-ID are resolved lazily so __init__ is instant
        self._ip: Optional[str] = getattr(settings, "sonos_ip", "") or None
        self._player_id: Optional[str] = getattr(settings, "sonos_player_id", "") or None
        self._resolved   = False  # set True once both are known
        self._resolve_lock = threading.Lock()

        if self._ip and self._player_id:
            self._resolved = True
            log.info("Sonos: using configured ip=%s player=%s", self._ip, self._player_id)
        elif self._ip:
            # Have IP, need player ID — fetch in background
            threading.Thread(target=self._resolve, daemon=True).start()
        else:
            # Need full discovery
            threading.Thread(target=self._resolve, daemon=True).start()

    # ── Resolution ────────────────────────────────────────────────────────

    def _resolve(self):
        with self._resolve_lock:
            if self._resolved:
                return
            if not self._ip:
                self._ip = _ssdp_discover_sonos()
            if not self._ip:
                log.warning("Sonos: could not find a device; chime disabled. "
                            "Set sonos_ip in config to enable.")
                return
            if not self._player_id:
                self._player_id = _fetch_player_id(self._ip)
            if self._player_id:
                self._resolved = True
                log.info("Sonos ready: ip=%s player=%s chime=%s",
                         self._ip, self._player_id, self._chime_url)
            else:
                log.warning("Sonos: device found at %s but player ID unavailable", self._ip)

    # ── Public API ────────────────────────────────────────────────────────

    def notify(self):
        """Play the chime if debounce allows.  Non-blocking — spawns a thread."""
        with self._lock:
            now = time.monotonic()
            if now - self._last_played < self._debounce_s:
                return
            self._last_played = now
        threading.Thread(target=self._post_clip, daemon=True).start()

    # ── Internal ─────────────────────────────────────────────────────────

    def _post_clip(self):
        if not self._resolved:
            # Give the background resolver up to 5 s on first call
            self._resolve_lock.acquire(timeout=5.0)
            try:
                pass
            finally:
                try:
                    self._resolve_lock.release()
                except RuntimeError:
                    pass
        if not self._resolved or not self._ip or not self._player_id:
            log.debug("Sonos: not resolved, skipping clip")
            return

        url = f"https://{self._ip}:{_SONOS_CLIP_PORT}/api/v1/players/{self._player_id}/audioClip"
        payload = json.dumps({
            "name":      "endora",
            "appId":     "com.endora.gesture",
            "streamUrl": self._chime_url,
            "volume":    self._volume,
            "clipType":  "CUSTOM",
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            ctx = _ssl_no_verify_ctx()
            with urllib.request.urlopen(req, timeout=_CLIP_POST_TIMEOUT, context=ctx) as resp:
                log.info("Sonos clip posted (status=%d)", resp.status)
        except urllib.error.HTTPError as e:
            log.warning("Sonos clip HTTP %d: %s", e.code, e.reason)
        except Exception as e:
            log.warning("Sonos clip error: %s", e)
