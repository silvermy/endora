"""
cameras/debug_server.py — MJPEG debug stream
"""
from __future__ import annotations

import logging
import threading
import time
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_latest: Dict[str, Optional[np.ndarray]] = {"A": None, "B": None}
_lock = threading.Lock()
_last_gesture: Dict = {"label": "", "ts": 0.0}


def update_frame(label: str, frame: np.ndarray) -> None:
    with _lock:
        _latest[label] = frame


def notify_gesture(label: str) -> None:
    with _lock:
        _last_gesture["label"] = label
        _last_gesture["ts"] = time.monotonic()


def _compose() -> bytes:
    with _lock:
        a = _latest.get("A")
        b = _latest.get("B")
        gesture_label = _last_gesture.get("label", "")
        gesture_age = time.monotonic() - _last_gesture.get("ts", 0.0)

    # Fixed 480x360 panels — same as the working 1.5.4 version
    placeholder = np.zeros((360, 480, 3), dtype=np.uint8)
    cv2.putText(placeholder, "No frame", (140, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

    fa = cv2.resize(a if a is not None else placeholder, (480, 360))
    fb = cv2.resize(b if b is not None else placeholder, (480, 360))
    combined = np.hstack([fa, fb])

    # Cam labels — pushed down to y=38 to clear Reolink OSD at top
    cv2.putText(combined, "Cam A", (8, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, "Cam A", (8, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(combined, "Cam B", (488, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, "Cam B", (488, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Gesture flash
    if gesture_age < 2.0 and gesture_label:
        txt = f"  {gesture_label.upper()}  "
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        tx = (combined.shape[1] - tw) // 2
        ty = combined.shape[0] - 20
        cv2.rectangle(combined, (tx-8, ty-th-8), (tx+tw+8, ty+8), (0, 170, 0), -1)
        cv2.putText(combined, txt, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    _, jpg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return jpg.tobytes()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""<!DOCTYPE html><html><head>
<title>Endora Debug</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{background:#0d0d0d;margin:0;padding:6px 0;display:flex;
     flex-direction:column;align-items:center;font-family:monospace;color:#aaa}
h3{margin:4px 0 6px;font-size:13px;letter-spacing:2px;color:#888}
img{max-width:100%;display:block}
small{margin-top:5px;font-size:10px;color:#555}
</style></head><body>
<h3>ENDORA DEBUG</h3>
<img src="/stream">
<small>cyan=ready &nbsp;|&nbsp; orange=warming &nbsp;|&nbsp; arrow=peak velocity</small>
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
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()


def start(port: int) -> None:
    class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True
    server = ThreadedHTTPServer(("0.0.0.0", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True,
                         name="DebugServer")
    t.start()
    log.info("Debug stream: http://0.0.0.0:%d/", port)
