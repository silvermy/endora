"""
cameras/debug_server.py

Optional MJPEG debug stream — serves annotated camera frames over HTTP.
Access at http://<ha-ip>:8765/  to see a live split view of both cameras.

Frames are served at native resolution — no resizing, no smearing.
Cameras are shown side by side. Each frame carries its own overlay.
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
        _latest[label] = frame.copy()


def notify_gesture(label: str) -> None:
    with _lock:
        _last_gesture["label"] = label
        _last_gesture["ts"] = time.monotonic()


def _label_frame(img: np.ndarray, cam_label: str) -> np.ndarray:
    """Add a small camera label in top-left, above the camera's own OSD."""
    h, w = img.shape[:2]
    txt = f"Cam {cam_label}"
    fs = max(0.4, w / 1600)          # scale font to frame width
    th = int(22 * fs / 0.4)
    pad = 4
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    # Dark pill behind label
    cv2.rectangle(img, (pad, pad), (pad + tw + 8, pad + th + 4), (0, 0, 0), -1)
    cv2.putText(img, txt, (pad + 4, pad + th),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def _compose() -> bytes:
    """Compose side-by-side frame at native resolution and encode as JPEG."""
    with _lock:
        a = _latest.get("A")
        b = _latest.get("B")
        gesture_label = _last_gesture.get("label", "")
        gesture_age = time.monotonic() - _last_gesture.get("ts", 0.0)

    # Use native frame sizes — no resize
    if a is not None:
        fa = a.copy()
        fh, fw = fa.shape[:2]
    else:
        fh, fw = 480, 640
        fa = np.zeros((fh, fw, 3), dtype=np.uint8)
        cv2.putText(fa, "No frame", (fw//2 - 60, fh//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)

    if b is not None:
        fb = b.copy()
        bh, bw = fb.shape[:2]
    else:
        bh, bw = fh, fw
        fb = np.zeros((bh, bw, 3), dtype=np.uint8)
        cv2.putText(fb, "No frame", (bw//2 - 60, bh//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)

    # Match heights if different
    if fh != bh:
        fb = cv2.resize(fb, (int(bw * fh / bh), fh))
        bh, bw = fb.shape[:2]

    # Add cam labels
    _label_frame(fa, "A")
    _label_frame(fb, "B")

    # Thin divider
    divider = np.zeros((fh, 2, 3), dtype=np.uint8)
    divider[:] = (60, 60, 60)
    combined = np.hstack([fa, divider, fb])

    # Gesture flash banner — centered, scaled to frame height
    if gesture_age < 2.0 and gesture_label:
        txt = f"  {gesture_label.upper()}  "
        fs = fh / 480.0
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
        tx = (combined.shape[1] - tw) // 2
        ty = combined.shape[0] - int(20 * fs)
        cv2.rectangle(combined,
                      (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10),
                      (0, 180, 0), -1)
        cv2.putText(combined, txt, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, fs, (0, 0, 0), 2, cv2.LINE_AA)

    _, jpg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
<title>Gesture Cam Debug</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{background:#0a0a0a;margin:0;padding:8px;
       display:flex;flex-direction:column;align-items:center;
       font-family:monospace;color:#ccc}
  h3{margin:4px 0 6px;font-size:14px;letter-spacing:1px;color:#aaa}
  img{max-width:100%;display:block;border:1px solid #333}
  small{margin-top:6px;font-size:11px;color:#666}
</style>
</head><body>
<h3>GESTURE CAM &mdash; LIVE DEBUG</h3>
<img src="/stream">
<small>cyan dot = arm ready &nbsp;|&nbsp; orange = warming &nbsp;|&nbsp;
magenta arrow = peak velocity</small>
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
    server = HTTPServer(("0.0.0.0", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True,
                         name="DebugServer")
    t.start()
    log.info("Debug stream: http://0.0.0.0:%d/  (open in browser)", port)
