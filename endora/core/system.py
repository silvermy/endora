"""
core/system.py

Wires RTSP capture → analyser → fusion → HA event backend.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time

from cameras.capture import RtspCapture
from cameras.analyser import CameraAnalyser
from core.state_machine import Gesture
from cameras import debug_server
from cameras.recorder import TestRecorder
from core import deployment
from core.feedback_logger import FeedbackLogger
from core.fusion import GestureFusion
from output.backends import make_backend
from output.chime import make_chime_notifier

log = logging.getLogger(__name__)

# Gestures that get their own sound, distinct from the arm-raise chime.
# Keyed by Gesture so the mapping is checked by the type system rather than
# by a string that could drift from the enum.
_GESTURE_SOUNDS = {
    Gesture.FOLDED_ARMS: "folded_arms.mp3",
}


class GestureSystem:

    def __init__(self, settings):
        self.s = settings
        self.backend = make_backend(settings)
        self.feedback = FeedbackLogger()

        self.fusion = GestureFusion(settings, on_gesture=self._on_gesture,
                                    on_suppressed=self._on_suppressed)

        # Single-camera mode: explicit flag OR both URLs identical
        self._single = (
            getattr(settings, 'single_camera_mode', False)
            or settings.rtsp_url_a == settings.rtsp_url_b
        )
        if self._single:
            log.info("Single-camera mode: running one analyser only")

        self.cam_a = RtspCapture(
            rtsp_url=settings.rtsp_url_a,
            width=settings.frame_width,
            height=settings.frame_height,
            reconnect_delay_s=settings.rtsp_reconnect_delay_s,
            rtsp_transport=settings.rtsp_transport,
            name="CamA",
        )
        self.cam_b = None if self._single else RtspCapture(
            rtsp_url=settings.rtsp_url_b,
            width=settings.frame_width,
            height=settings.frame_height,
            reconnect_delay_s=settings.rtsp_reconnect_delay_s,
            rtsp_transport=settings.rtsp_transport,
            name="CamB",
        )

        # Optional debug stream
        self._debug_enabled = settings.debug_port > 0
        self._host_ip = _detect_host_ip()
        if self._debug_enabled:
            debug_server.configure(camera_count=1 if self._single else 2)
            debug_server.set_settings(settings)
            debug_server.set_host_info(self._host_ip, settings.debug_port)
            debug_server.set_feedback_logger(self.feedback)
            debug_server.start(settings.debug_port, ingress_port=8766)

        # Optional chime on arm-up transitions
        self._chime = None
        # Per-gesture sounds, played when the gesture itself fires rather than
        # on the arm-raise that precedes it. FOLDED_ARMS raises no arm, so the
        # confirmation chime never reaches it — without this the gesture is
        # silent. Each notifier keeps its own debounce clock, so one sound
        # cannot mute another.
        self._gesture_chimes: dict = {}
        chime_on = getattr(settings, "chime_enable", False)
        if chime_on:
            chime_url = _install_chime_wav(self._host_ip, settings.debug_port)
            self._chime = make_chime_notifier(settings, chime_url)
            for gesture, filename in _GESTURE_SOUNDS.items():
                url = _install_sound(filename, self._host_ip, settings.debug_port)
                notifier = make_chime_notifier(settings, url) if url else None
                if notifier is not None:
                    self._gesture_chimes[gesture] = notifier

        dbg_cb = debug_server.update_frame if self._debug_enabled else None
        # Only render the live overlay while someone is watching it — see
        # debug_server.is_being_viewed. Captures and logs stay unconditional.
        dbg_wanted = debug_server.is_being_viewed if self._debug_enabled else None

        # Regression-test recorder — activated by ENDORA_RECORD_TESTS=1
        self._recorder: TestRecorder | None = None
        if os.environ.get("ENDORA_RECORD_TESTS", "").strip() == "1":
            self._recorder = TestRecorder()
            debug_server.set_recorder(self._recorder)

        # Each analyser runs its own ONNX Runtime pose-model session. Left at
        # 0 (= os.cpu_count()) per session, two simultaneous analysers would
        # each try to claim every core, oversubscribing the CPU and pinning
        # it at 100%. Split the machine's cores evenly across analysers.
        num_analysers = 1 if self._single else 2
        model_threads = max(1, (os.cpu_count() or 4) // num_analysers)

        self.analyser_a = CameraAnalyser(
            camera=self.cam_a, settings=settings,
            on_candidate=self.fusion.receive, label="A",
            debug_frame_cb=dbg_cb,
            debug_wanted_cb=dbg_wanted,
            feedback_logger=self.feedback,
            chime_notifier=self._chime,
            gesture_sound_cb=self._play_gesture_sound,
            num_threads=model_threads,
        )
        self.analyser_a._recorder = self._recorder
        if self.analyser_a._frame_capture is not None:
            debug_server.set_frame_capture(self.analyser_a._frame_capture)

        self.analyser_b = None if self._single else CameraAnalyser(
            camera=self.cam_b, settings=settings,
            on_candidate=self.fusion.receive, label="B",
            debug_frame_cb=dbg_cb,
            debug_wanted_cb=dbg_wanted,
            feedback_logger=self.feedback,
            chime_notifier=self._chime,
            gesture_sound_cb=self._play_gesture_sound,
            num_threads=model_threads,
        )
        if self.analyser_b:
            self.analyser_b._recorder = self._recorder

    def run(self):
        self.feedback.start_keyboard_listener()
        self.cam_a.start()
        if self.cam_b:
            self.cam_b.start()

        log.info("Waiting for RTSP stream(s) (up to 30 s each)…")
        ok_a = self.cam_a.wait_ready(timeout=30)
        ok_b = self.cam_b.wait_ready(timeout=30) if self.cam_b else True

        if not ok_a:
            log.error("Camera A stream not available: %s", self.s.rtsp_url_a)
        if self.cam_b and not ok_b:
            log.error("Camera B stream not available: %s", self.s.rtsp_url_b)
        if not (ok_a or ok_b):
            log.critical("No RTSP stream is available — exiting")
            return

        self.analyser_a.start()
        if self.analyser_b:
            self.analyser_b.start()

        log.info("Gesture system running. Listening for gestures…")
        try:
            while self.analyser_a.is_alive() or (
                self.analyser_b and self.analyser_b.is_alive()
            ):
                time.sleep(10)
                self._log_stats()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self.analyser_a.stop()
        if self.analyser_b:
            self.analyser_b.stop()
        self.cam_a.stop()
        if self.cam_b:
            self.cam_b.stop()
        self.backend.close()

    def _play_gesture_sound(self, gesture: Gesture) -> None:
        """One sound per gesture: its own if it has one, else the chime."""
        notifier = self._gesture_chimes.get(gesture) or self._chime
        if notifier is not None:
            notifier.notify()

    def _on_suppressed(self, gesture_name: str, reason: str) -> None:
        """A gesture fired in the analyser but never reached Home Assistant."""
        if self.feedback:
            self.feedback.on_suppressed(gesture_name, reason)

    def _on_gesture(self, gesture: Gesture, confidence: float, sources: list):
        # Update UI immediately — don't wait for the HTTP round-trip to HA
        if self._debug_enabled:
            debug_server.notify_gesture(str(gesture))
        self.feedback.on_gesture_fired(gesture.name, confidence, reading=None)
        # Fire HA event in a background thread so it never stalls the pipeline
        threading.Thread(
            target=self.backend.send,
            args=(gesture, confidence, sources),
            daemon=True,
        ).start()

    _last_stats = 0.0

    def _log_stats(self):
        now = time.monotonic()
        if now - self._last_stats < 30.0:
            return
        self._last_stats = now
        if self.cam_b:
            log.info(
                "Stats | CamA: %d frames %.0ffps | CamB: %d frames %.0ffps | "
                "Events fired: %d",
                self.cam_a.frames_captured, self.cam_a._fps_actual,
                self.cam_b.frames_captured, self.cam_b._fps_actual,
                self.fusion.total_emitted,
            )
        else:
            log.info(
                "Stats | CamA: %d frames %.0ffps | Events fired: %d",
                self.cam_a.frames_captured, self.cam_a._fps_actual,
                self.fusion.total_emitted,
            )


def _install_chime_wav(host_ip: str = "", debug_port: int = 0,
                       media_dir: "Path | None" = None) -> str:
    """The arm-raise confirmation chime. See _install_sound."""
    return _install_sound("chime.wav", host_ip, debug_port, media_dir)


def _install_sound(filename: str, host_ip: str = "", debug_port: int = 0,
                   media_dir: "Path | None" = None) -> str:
    """Return a URL Home Assistant can hand to a speaker for *filename*.

    Two routes, because the two deployments have different access to HA:

    * **Add-on** — /media is mapped, so the clip is copied there and HA
      proxies it via media-source://media_source/local/. No firewall issues,
      and it works whether or not the debug server is running.
    * **Standalone (Jetson)** — /media belongs to the HA machine and is not
      reachable from here. The debug server already serves the bundled clip
      at /chime.wav, so the speaker fetches it from this host over the LAN.
      That makes the chime depend on debug_port being set, which is why the
      failure below says so explicitly rather than going quiet.

    Either way the URL carries a hash of the audio's own bytes: a fixed URL
    meant HA's media proxy and the speaker both kept playing a replaced clip
    from cache indefinitely (v1.9.140). A content-derived URL changes exactly
    when the audio does, which no cache can defeat.
    """
    import hashlib
    import shutil
    from pathlib import Path
    src = Path(__file__).parent.parent / "cameras" / "static" / filename
    if not src.exists():
        log.error("Sound: bundled %s not found at %s", filename, src)
        return ""
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:8]
    stem, suffix = src.stem, src.suffix

    # Injectable so the two routes can be tested without a real /media.
    media_dir = Path("/media") if media_dir is None else media_dir
    # Gate on the deployment, not on whether a /media directory happens to
    # exist. The Jetson's L4T base image has one, so the add-on route was
    # taken on a standalone install: the clip was copied into the container's
    # own throwaway /media and Home Assistant was handed a
    # media-source:// URL for a file that only existed on the HA machine —
    # left there by an add-on install that had since been removed. It played
    # until someone cleaned up, then would have failed with nothing in the
    # log to explain why.
    if deployment.is_addon() and media_dir.is_dir():
        dest = media_dir / f"endora_{stem}_{digest}{suffix}"
        try:
            shutil.copy2(src, dest)
            # Drop clips we installed for previous versions of the sound.
            # Scoped to THIS sound's stem: a shared glob would have each
            # sound delete the others every startup.
            for stale in media_dir.glob(f"endora_{stem}_*{suffix}"):
                if stale != dest:
                    try:
                        stale.unlink()
                        log.info("Sound: removed superseded %s", stale.name)
                    except Exception as e:
                        log.debug("Sound: could not remove %s: %s", stale.name, e)
            log.info("Sound: installed %s → %s", src.name, dest)
            return f"media-source://media_source/local/{dest.name}"
        except PermissionError:
            log.warning(
                "Sound: cannot write to /media (uid=%d permissions=%s) — "
                "try adding 'full_access: true' to the add-on config",
                os.getuid(), oct(media_dir.stat().st_mode),
            )
        except Exception as e:
            log.warning("Sound: copy to /media failed: %s", e)
        # Fall through — the HTTP route below may still work.

    # Standalone route. The query string is what carries the digest; the
    # debug server matches on path alone, so /chime.wav serves it unchanged
    # while the URL a cache keys on still moves with the audio.
    if debug_port > 0 and host_ip:
        url = f"http://{host_ip}:{debug_port}/sound/{src.name}?v={digest}"
        log.info("Sound: serving %s from the debug server at %s", src.name, url)
        return url

    log.warning(
        "Sound: no way to serve the audio — /media is not mounted (expected "
        "outside the HA add-on) and the debug server is disabled. Set "
        "debug_port to a real port so the speaker can fetch the clip from "
        "this host.")
    return ""


def _detect_host_ip() -> str:
    """Return a hostname reachable by LAN devices for the chime URL.

    In HA add-on mode (host_network=false) the container gets a private Docker
    IP that LAN devices cannot reach. Use homeassistant.local instead, which
    mDNS advertises on the LAN and maps to the exposed ports.
    """
    if os.environ.get("SUPERVISOR_TOKEN"):
        return "homeassistant.local"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "homeassistant.local"
