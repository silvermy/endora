"""
cameras/capture.py

Connects to an RTSP stream and exposes the latest frame via get_frame().

Each RtspCapture runs in its own daemon thread.  It continuously reads
frames from the RTSP source and stores only the most recent one — the
analyser always sees the newest image, never a stale queue back-log.

Reconnect logic:  If the stream drops (camera reboots, network blip),
the thread waits `reconnect_delay_s` seconds and retries indefinitely.
This is essential for a always-on home deployment.

FFmpeg transport options:
  - rtsp_transport tcp   → reliable, slightly higher latency (~200 ms)
  - rtsp_transport udp   → lower latency, may drop frames on WiFi
  Default is TCP which works with virtually all IP cameras.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class RtspCapture(threading.Thread):
    """
    Reads frames from one RTSP stream and exposes them via .get_frame().
    Non-blocking: always returns the most recent frame.
    Automatically reconnects on stream loss.
    """

    def __init__(
        self,
        rtsp_url: str,
        width: int = 640,
        height: int = 480,
        reconnect_delay_s: float = 5.0,
        rtsp_transport: str = "tcp",
        name: str = "RtspCapture",
    ):
        super().__init__(daemon=True, name=name)
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.reconnect_delay_s = reconnect_delay_s
        self.rtsp_transport = rtsp_transport
        self.label = name

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._ready = threading.Event()

        self.frames_captured: int = 0
        self._fps_actual: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest BGR frame, or None if not yet ready."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """Block until the first frame arrives or timeout elapses."""
        return self._ready.wait(timeout)

    def stop(self):
        self._stop_evt.set()

    # ── Thread body ───────────────────────────────────────────────────────

    def run(self):
        import cv2

        # Tell FFmpeg (OpenCV's backend) to use the chosen transport and to
        # open the stream quickly without a long probe.
        options = (
            f"rtsp_transport;{self.rtsp_transport}|"
            "stimeout;5000000|"        # 5 s connection timeout (microseconds)
            "fflags;nobuffer|"         # reduce buffering → lower latency
            "flags;low_delay"
        )

        masked = _mask_url(self.rtsp_url)
        log.info("[%s] Connecting to RTSP stream: %s", self.label, masked)

        frame_times: list[float] = []

        while not self._stop_evt.is_set():
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8_000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000)

            if not cap.isOpened():
                log.warning(
                    "[%s] Could not open stream %s — retrying in %.0fs",
                    self.label, masked, self.reconnect_delay_s,
                )
                cap.release()
                self._sleep_interruptible(self.reconnect_delay_s)
                continue

            log.info("[%s] Stream opened: %s", self.label, masked)
            consecutive_failures = 0

            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        log.warning(
                            "[%s] Too many read failures — reconnecting", self.label
                        )
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0

                # Resize if camera native resolution differs from target
                fh, fw = frame.shape[:2]
                if fw != self.width or fh != self.height:
                    frame = cv2.resize(
                        frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR
                    )

                now = time.monotonic()
                frame_times.append(now)
                frame_times = [t for t in frame_times if now - t < 1.0]
                self._fps_actual = len(frame_times)

                with self._lock:
                    self._frame = frame
                self.frames_captured += 1
                if not self._ready.is_set():
                    self._ready.set()

            cap.release()
            if not self._stop_evt.is_set():
                log.info(
                    "[%s] Stream lost — reconnecting in %.0fs",
                    self.label, self.reconnect_delay_s,
                )
                self._sleep_interruptible(self.reconnect_delay_s)

        log.info("[%s] Capture stopped", self.label)

    def _sleep_interruptible(self, seconds: float):
        """Sleep in small increments so stop() is responsive."""
        deadline = time.monotonic() + seconds
        while not self._stop_evt.is_set() and time.monotonic() < deadline:
            time.sleep(0.2)


def _mask_url(url: str) -> str:
    """Replace password in rtsp://user:pass@host/... for safe logging."""
    import re
    return re.sub(r"(rtsp://[^:]+:)[^@]+(@)", r"\1****\2", url)
