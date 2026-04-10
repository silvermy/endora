"""
cameras/debug_server.py

Optional MJPEG debug stream — serves annotated camera frames over HTTP.
Access at http://<ha-ip>:8765/  to see a live split view of both cameras
with skeleton overlay, wrist marker, velocity vector, and gesture state.

Only starts when debug_port is set in config (default 0 = disabled).
"""
from __future__ import annotations

import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_latest: Dict[str, Optional[np.ndarray]] = {"A": None, "B": None}
_lock = threading.Lock()
_last_gesture: Dict[str, float] = {"label": "", "ts": 0.0}


def update_frame(label: str, frame: np.ndarray) -> None:
    with _lock:
        _latest[label] = frame


def notify_gesture(label: str) -> None:
    with _lock:
        _last_gesture["label"] = label
        _last_gesture["ts"] = time.monotonic()


def _compose() -> bytes:
    """Compose side-by-side frame and encode as JPEG."""
    with _lock:
        a = _latest.get("A")
        b = _latest.get("B")
        gesture_label = _last_gesture.get("label", "")
        gesture_age = time.monotonic() - _last_gesture.get("ts", 0.0)

    placeholder = np.zeros((360, 480, 3), dtype=np.uint8)
    cv2.putText(placeholder, "No frame", (140, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

    fa = cv2.resize(a if a is not None else placeholder, (480, 360))
    fb = cv2.resize(b if b is not None else placeholder, (480, 360))
    combined = np.hstack([fa, fb])

    # Gesture flash overlay
    if gesture_age < 1.5 and gesture_label:
        txt = f"  {gesture_label.upper()}  "
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
        tx = (combined.shape[1] - tw) // 2
        ty = combined.shape[0] - 30
        cv2.rectangle(combined, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8),
                      (0, 200, 0), -1)
        cv2.putText(combined, txt, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)

    # Labels
    cv2.putText(combined, "Camera A", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(combined, "Camera B", (488, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    _, jpg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpg.tobytes()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # suppress access logs

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""<html><head>
<title>Gesture Cam Debug</title>
<style>body{background:#111;margin:0;display:flex;flex-direction:column;
align-items:center;font-family:monospace;color:#eee}
img{max-width:100%;border:1px solid #444;margin-top:8px}</style>
</head><body>
<h3>Gesture Cam Live Debug</h3>
<img src="/stream"><br>
<small>Refresh auto via MJPEG &mdash; green=arm ready, orange=warming up</small>
</body></html>""")

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = _compose()
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                        + jpg + b"\r\n"
                    )
                    time.sleep(0.1)  # ~10 fps debug stream
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()


def start(port: int) -> None:
    """Start MJPEG server in background thread."""
    server = HTTPServer(("0.0.0.0", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True,
                         name="DebugServer")
    t.start()
    log.info("Debug stream: http://0.0.0.0:%d/  (open in browser)", port)
