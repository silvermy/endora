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
from core.feedback_logger import FeedbackLogger
from core.fusion import GestureFusion
from output.backends import make_backend
from output.chime import make_chime_notifier

log = logging.getLogger(__name__)


class GestureSystem:

    def __init__(self, settings):
        self.s = settings
        self.backend = make_backend(settings)
        self.feedback = FeedbackLogger()

        self.fusion = GestureFusion(settings, on_gesture=self._on_gesture)

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
        self._sonos = None
        chime_on = getattr(settings, "chime_enable",
                           getattr(settings, "sonos_enable", False))
        if chime_on:
            if settings.debug_port > 0:
                chime_url = f"http://{self._host_ip}:{settings.debug_port}/chime.wav"
                self._sonos = make_chime_notifier(settings, chime_url)
            else:
                log.warning("Chime: debug_port must be set so the speaker can "
                            "reach the chime WAV; chime disabled.")

        dbg_cb = debug_server.update_frame if self._debug_enabled else None

        # Regression-test recorder — activated by ENDORA_RECORD_TESTS=1
        self._recorder: TestRecorder | None = None
        if os.environ.get("ENDORA_RECORD_TESTS", "").strip() == "1":
            self._recorder = TestRecorder()
            debug_server.set_recorder(self._recorder)

        self.analyser_a = CameraAnalyser(
            camera=self.cam_a, settings=settings,
            on_candidate=self.fusion.receive, label="A",
            debug_frame_cb=dbg_cb,
            feedback_logger=self.feedback,
            sonos_notifier=self._sonos,
        )
        self.analyser_a._recorder = self._recorder
        if self.analyser_a._frame_capture is not None:
            debug_server.set_frame_capture(self.analyser_a._frame_capture)

        self.analyser_b = None if self._single else CameraAnalyser(
            camera=self.cam_b, settings=settings,
            on_candidate=self.fusion.receive, label="B",
            debug_frame_cb=dbg_cb,
            feedback_logger=self.feedback,
            sonos_notifier=self._sonos,
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


def _detect_host_ip() -> str:
    """Return the host's LAN IP by probing an external address (no data sent)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "homeassistant.local"
