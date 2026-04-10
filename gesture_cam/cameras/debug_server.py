"""
cameras/debug_server.py — MJPEG debug stream
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

_latest: Dict[str, Optional[bytes]] = {"A": None, "B": None}
_sizes: Dict[str, tuple] = {"A": (640, 480), "B": (640, 480)}
_lock = threading.Lock()
_last_gesture: Dict = {"label": "", "ts": 0.0}

# Max width per camera panel in the side-by-side view
MAX_PANEL_W = 640


def update_frame(label: str, frame: np.ndarray) -> None:
    """Receive a frame, downscale if needed, encode to JPEG, store bytes."""
    try:
        h, w = frame.shape[:2]
        # Downscale if wider than MAX_PANEL_W to keep stream fast
        if w > MAX_PANEL_W:
            scale = MAX_PANEL_W / w
            frame = cv2.resize(frame, (MAX_PANEL_W, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with _lock:
            _latest[label] = jpg.tobytes()
            _sizes[label] = (w, h)
    except Exception:
        pass


def notify_gesture(label: str) -> None:
    with _lock:
        _last_gesture["label"] = label
        _last_gesture["ts"] = time.monotonic()


def _compose() -> bytes:
    """Decode stored JPEGs, stack side by side, re-encode."""
    with _lock:
        ja = _latest.get("A")
        jb = _latest.get("B")
        gesture_label = _last_gesture.get("label", "")
        gesture_age = time.monotonic() - _last_gesture.get("ts", 0.0)

    def placeholder(w=640, h=480):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, "No frame", (w//2 - 55, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 70, 70), 2)
        return img

    fa = cv2.imdecode(np.frombuffer(ja, np.uint8), cv2.IMREAD_COLOR) \
         if ja else placeholder()
    fb = cv2.imdecode(np.frombuffer(jb, np.uint8), cv2.IMREAD_COLOR) \
         if jb else placeholder()

    # Match heights
    ah, aw = fa.shape[:2]
    bh, bw = fb.shape[:2]
    if ah != bh:
        scale = ah / bh
        fb = cv2.resize(fb, (int(bw * scale), ah), interpolation=cv2.INTER_AREA)
        bh, bw = fb.shape[:2]

    # Cam labels — small, top-left of each panel
    def add_label(img, txt):
        fs = 0.45
        cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (200, 200, 200), 1, cv2.LINE_AA)

    add_label(fa, "Cam A")
    add_label(fb, "Cam B")

    divider = np.full((ah, 2, 3), 50, dtype=np.uint8)
    combined = np.hstack([fa, divider, fb])
    ch, cw = combined.shape[:2]

    # Gesture flash
    if gesture_age < 2.0 and gesture_label:
        txt = f" {gesture_label.upper()} "
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        tx = (cw - tw) // 2
        ty = ch - 16
        cv2.rectangle(combined, (tx-6, ty-th-6), (tx+tw+6, ty+6),
                      (0, 170, 0), -1)
        cv2.putText(combined, txt, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    _, out = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return out.tobytes()


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
body{background:#0d0d0d;margin:0;padding:6px 0;display:flex;
     flex-direction:column;align-items:center;font-family:monospace;color:#aaa}
h3{margin:4px 0 6px;font-size:13px;letter-spacing:2px;color:#888}
img{max-width:100%;display:block}
small{margin-top:5px;font-size:10px;color:#555}
</style></head><body>
<h3>GESTURE CAM DEBUG</h3>
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
                    time.sleep(0.12)   # ~8 fps — easy on the Pi
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
    log.info("Debug stream: http://0.0.0.0:%d/", port)
