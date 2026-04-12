"""
core/system.py

Wires RTSP capture → analyser → fusion → HA event backend.
"""

from __future__ import annotations

import logging
import time

from cameras.capture import RtspCapture
from cameras.analyser import CameraAnalyser, Gesture
from cameras import debug_server
from core.fusion import GestureFusion
from output.backends import make_backend

log = logging.getLogger(__name__)


class GestureSystem:

    def __init__(self, settings):
        self.s = settings
        self.backend = make_backend(settings)

        self.fusion = GestureFusion(settings, on_gesture=self._on_gesture)

        self.cam_a = RtspCapture(
            rtsp_url=settings.rtsp_url_a,
            width=settings.frame_width,
            height=settings.frame_height,
            reconnect_delay_s=settings.rtsp_reconnect_delay_s,
            rtsp_transport=settings.rtsp_transport,
            name="CamA",
        )
        self.cam_b = RtspCapture(
            rtsp_url=settings.rtsp_url_b,
            width=settings.frame_width,
            height=settings.frame_height,
            reconnect_delay_s=settings.rtsp_reconnect_delay_s,
            rtsp_transport=settings.rtsp_transport,
            name="CamB",
        )

        # Optional debug stream
        self._debug_enabled = settings.debug_port > 0
        if self._debug_enabled:
            debug_server.start(settings.debug_port)

        dbg_cb = debug_server.update_frame if self._debug_enabled else None

        self.analyser_a = CameraAnalyser(
            camera=self.cam_a, settings=settings,
            on_candidate=self.fusion.receive, label="A",
            debug_frame_cb=dbg_cb,
        )
        self.analyser_b = CameraAnalyser(
            camera=self.cam_b, settings=settings,
            on_candidate=self.fusion.receive, label="B",
            debug_frame_cb=dbg_cb,
        )

    def run(self):
        self.cam_a.start()
        self.cam_b.start()

        log.info("Waiting for RTSP streams (up to 30 s each)…")
        ok_a = self.cam_a.wait_ready(timeout=30)
        ok_b = self.cam_b.wait_ready(timeout=30)

        if not ok_a:
            log.error("Camera A stream not available: %s", self.s.rtsp_url_a)
        if not ok_b:
            log.error("Camera B stream not available: %s", self.s.rtsp_url_b)
        if not (ok_a or ok_b):
            log.critical("Neither RTSP stream is available — exiting")
            return

        self.analyser_a.start()
        self.analyser_b.start()

        log.info("Gesture system running. Listening for gestures…")
        try:
            while self.analyser_a.is_alive() or self.analyser_b.is_alive():
                time.sleep(10)
                self._log_stats()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self.analyser_a.stop()
        self.analyser_b.stop()
        self.cam_a.stop()
        self.cam_b.stop()
        self.backend.close()

    def _on_gesture(self, gesture: Gesture, confidence: float, sources: list):
        self.backend.send(gesture, confidence, sources)
        if self._debug_enabled:
            debug_server.notify_gesture(str(gesture))

    _last_stats = 0.0

    def _log_stats(self):
        now = time.monotonic()
        if now - self._last_stats < 30.0:
            return
        self._last_stats = now
        log.info(
            "Stats | CamA: %d frames %.0ffps | CamB: %d frames %.0ffps | "
            "Events fired: %d",
            self.cam_a.frames_captured, self.cam_a._fps_actual,
            self.cam_b.frames_captured, self.cam_b._fps_actual,
            self.fusion.total_emitted,
        )
