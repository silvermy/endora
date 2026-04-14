"""
cameras/debug_server.py — MJPEG debug stream + live parameter tuning UI
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

log = logging.getLogger(__name__)

_latest: Dict[str, Optional[np.ndarray]] = {"A": None, "B": None}
_lock = threading.Lock()
_last_gesture: Dict = {"label": "", "ts": 0.0}
_single_camera: bool = False   # set by configure() before start()
_settings = None               # live Settings instance, set by set_settings()


# ── Tunable parameter definitions ────────────────────────────────────────────
# (key, label, min, max, step, group)
_PARAMS = [
    ("wave_velocity_threshold_px", "Wave velocity threshold (px)", 1,   60,   1,    "Gesture"),
    ("wave_sustain_frames",        "Sustain frames",               1,   10,   1,    "Gesture"),
    ("arm_above_head_tolerance",   "Arm height tolerance",         0.30, 1.0, 0.01, "Gesture"),
    ("cooldown_s",                 "Cooldown (s)",                 0,   10,   0.25, "Gesture"),
    ("fist_curl_threshold",        "Fist curl threshold",          0.25, 1.0, 0.05, "Gesture"),
    ("pose_visibility_min",         "Min body visibility (furn.)",  0.05, 0.8, 0.01, "Body"),
    ("frame_crop_bottom",          "Crop bottom (%)",              0,   60,   1,    "Body"),
    ("frame_crop_top",             "Crop top (%)",                 0,   30,   1,    "Body"),
    ("dewarp_tilt",                "Tilt (° down)",               -10,  80,   1,    "Dewarp"),
    ("dewarp_pan",                 "Pan (° right)",               -30,  30,   1,    "Dewarp"),
    ("dewarp_roll",                "Roll (° CW)",                 -45,  45,   1,    "Dewarp"),
    ("dewarp_vfov",                "Vertical FOV (°)",             20,  100,  1,    "Dewarp"),
]


# ── Public API ────────────────────────────────────────────────────────────────

def configure(camera_count: int) -> None:
    """Call before start() to set 1- or 2-camera layout."""
    global _single_camera
    _single_camera = (camera_count == 1)


def set_settings(s) -> None:
    """Register the live Settings instance so sliders can read/write it."""
    global _settings
    _settings = s


def update_frame(label: str, frame: np.ndarray) -> None:
    with _lock:
        _latest[label] = frame


def notify_gesture(label: str) -> None:
    with _lock:
        _last_gesture["label"] = label
        _last_gesture["ts"] = time.monotonic()


# ── Frame composition ─────────────────────────────────────────────────────────

def _letterbox(src: Optional[np.ndarray], pw: int, ph: int,
               placeholder: str = "No frame") -> np.ndarray:
    if src is None:
        blank = np.zeros((ph, pw, 3), dtype=np.uint8)
        cv2.putText(blank, placeholder, (pw // 2 - 60, ph // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        return blank
    sh, sw = src.shape[:2]
    scale = min(pw / sw, ph / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((ph, pw, 3), dtype=np.uint8)
    out[(ph - nh) // 2:(ph - nh) // 2 + nh,
        (pw - nw) // 2:(pw - nw) // 2 + nw] = resized
    return out


def _compose() -> bytes:
    with _lock:
        a = _latest.get("A")
        b = _latest.get("B")
        gesture_label = _last_gesture.get("label", "")
        gesture_age = time.monotonic() - _last_gesture.get("ts", 0.0)

    if _single_camera:
        PANEL_W, PANEL_H = 960, 540
        combined = _letterbox(a, PANEL_W, PANEL_H)
        for col, th in [((0, 0, 0), 3), ((200, 200, 200), 1)]:
            cv2.putText(combined, "Cam A", (8, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)
    else:
        PANEL_W, PANEL_H = 480, 360
        combined = np.hstack([_letterbox(a, PANEL_W, PANEL_H),
                               _letterbox(b, PANEL_W, PANEL_H)])
        for label, x in [("Cam A", 8), ("Cam B", 488)]:
            for col, th in [((0, 0, 0), 3), ((200, 200, 200), 1)]:
                cv2.putText(combined, label, (x, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, th, cv2.LINE_AA)

    if gesture_age < 2.0 and gesture_label:
        txt = f"  {gesture_label.upper()}  "
        (tw, th2), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        tx = (combined.shape[1] - tw) // 2
        ty = combined.shape[0] - 20
        cv2.rectangle(combined, (tx - 8, ty - th2 - 8), (tx + tw + 8, ty + 8),
                      (0, 170, 0), -1)
        cv2.putText(combined, txt, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    _, jpg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return jpg.tobytes()


# ── Settings helpers ──────────────────────────────────────────────────────────

def _current_values() -> dict:
    if _settings is None:
        return {}
    return {key: getattr(_settings, key, None)
            for key, *_ in _PARAMS
            if getattr(_settings, key, None) is not None}


def _apply_setting(key: str, raw: str) -> bool:
    if _settings is None:
        return False
    current = getattr(_settings, key, None)
    if current is None:
        return False
    try:
        if isinstance(current, bool):
            v = raw.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            v = int(float(raw))
        elif isinstance(current, float):
            v = float(raw)
        else:
            v = raw
        setattr(_settings, key, v)
        return True
    except (ValueError, TypeError):
        return False


def _save_to_yaml() -> tuple[bool, str]:
    """Write current live values into /data/settings.yaml."""
    yaml_path = Path("/data/settings.yaml")
    vals = _current_values()
    try:
        text = yaml_path.read_text() if yaml_path.exists() else ""
        for key, value in vals.items():
            # Format floats nicely
            if isinstance(value, float):
                formatted = f"{value:.4g}"
            else:
                formatted = str(value).lower()  # bool → true/false
            pattern = rf'^({re.escape(key)}\s*:\s*)([^\n#]*)'
            new_text, n = re.subn(pattern, rf'\g<1>{formatted}',
                                  text, flags=re.MULTILINE)
            if n:
                text = new_text
            else:
                text += f"\n{key}: {formatted}"
        yaml_path.write_text(text)
        return True, ""
    except Exception as e:
        return False, str(e)


# ── HTML page ─────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<title>Endora Debug</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0d0d;font-family:monospace;color:#ccc;
     display:flex;flex-direction:column;align-items:center;padding:8px 10px;gap:8px}
h3{font-size:11px;letter-spacing:3px;color:#555;padding:2px 0}
#wrap{display:flex;gap:12px;width:100%;max-width:1300px;align-items:flex-start}
#vbox{flex:1 1 auto;min-width:0}
#vbox img{width:100%;display:block;border:1px solid #222}
#legend{font-size:10px;color:#444;text-align:center;padding:3px 0}
/* ── slider panel ── */
#panel{flex:0 0 270px;background:#111;border:1px solid #252525;
       border-radius:6px;padding:10px 12px;display:flex;flex-direction:column;gap:10px;
       max-height:92vh;overflow-y:auto}
.grp{font-size:9px;letter-spacing:2px;color:#555;text-transform:uppercase;
     border-bottom:1px solid #1e1e1e;padding-bottom:3px;margin-bottom:1px}
.row{display:flex;flex-direction:column;gap:2px;margin-bottom:1px}
.lbl{display:flex;justify-content:space-between;font-size:10px;color:#888}
.val{color:#e8e060;min-width:38px;text-align:right;font-weight:bold}
input[type=range]{width:100%;accent-color:#e8e060;height:3px;cursor:pointer;margin:1px 0}
#savebtn{width:100%;padding:6px;background:#0e2e0e;border:1px solid #2e6e2e;
         color:#5c5;border-radius:4px;cursor:pointer;font:11px monospace;
         letter-spacing:1px;margin-top:4px}
#savebtn:hover{background:#163016}
#savebtn:disabled{opacity:0.5;cursor:default}
#savemsg{font-size:10px;min-height:13px;text-align:center;padding-top:2px}
@media(max-width:860px){
  #wrap{flex-direction:column}
  #panel{flex:none;width:100%;max-height:none}
}
</style>
</head>
<body>
<h3>ENDORA DEBUG</h3>
<div id="wrap">
  <div id="vbox">
    <img src="/stream" alt="stream">
    <div id="legend">cyan=ready &nbsp;|&nbsp; orange=warming &nbsp;|&nbsp; arrow=peak velocity</div>
  </div>
  <div id="panel">
    <div id="sliders"></div>
    <button id="savebtn" onclick="doSave()">&#128190;&nbsp;SAVE TO settings.yaml</button>
    <div id="savemsg"></div>
  </div>
</div>
<script>
const PARAMS = __PARAMS_JSON__;

function fmt(v, step) {
  return step < 1 ? (+v).toFixed(step < 0.05 ? 2 : 2) : String(+v % 1 === 0 ? +v : (+v).toFixed(1));
}

function build(vals) {
  const c = document.getElementById('sliders');
  c.innerHTML = '';
  let grp = null;
  PARAMS.forEach(([key, label, mn, mx, step, group]) => {
    if (group !== grp) {
      grp = group;
      const d = document.createElement('div');
      d.className = 'grp'; d.textContent = group;
      c.appendChild(d);
    }
    const v = vals[key] !== undefined ? vals[key] : ((+mn + +mx) / 2);
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML =
      `<div class="lbl"><span>${label}</span><span class="val" id="v_${key}">${fmt(v, step)}</span></div>` +
      `<input type="range" id="s_${key}" min="${mn}" max="${mx}" step="${step}" value="${v}"` +
      ` oninput="onInput('${key}',this.value,${step})"` +
      ` onchange="onCommit('${key}',this.value)">`;
    c.appendChild(row);
  });
}

function onInput(key, v, step) {
  document.getElementById('v_' + key).textContent = fmt(v, step);
}

const _t = {};
function onCommit(key, value) {
  clearTimeout(_t[key]);
  _t[key] = setTimeout(() => {
    fetch('/set?key=' + encodeURIComponent(key) + '&value=' + encodeURIComponent(value))
      .catch(e => console.warn('set failed', e));
  }, 60);
}

function doSave() {
  const btn = document.getElementById('savebtn');
  const msg = document.getElementById('savemsg');
  btn.disabled = true;
  msg.style.color = '#888'; msg.textContent = 'saving…';
  fetch('/save', {method: 'POST'})
    .then(r => r.json())
    .then(d => {
      msg.style.color = d.ok ? '#5c5' : '#c55';
      msg.textContent  = d.ok ? '✓ saved' : '✗ ' + (d.error || 'error');
    })
    .catch(() => { msg.style.color='#c55'; msg.textContent='✗ request failed'; })
    .finally(() => { btn.disabled = false; });
}

fetch('/settings').then(r => r.json()).then(build).catch(() => build({}));
</script>
</body>
</html>"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/":
            html = _HTML_TEMPLATE.replace(
                "__PARAMS_JSON__", json.dumps(_PARAMS)
            )
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/settings":
            body = json.dumps(_current_values()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/set":
            key   = qs.get("key",   [None])[0]
            value = qs.get("value", [None])[0]
            if key and value and _apply_setting(key, value):
                log.info("Live tune: %s = %s", key, value)
                body = b'{"ok":true}'
                self.send_response(200)
            else:
                body = b'{"ok":false,"error":"unknown key or bad value"}'
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/stream":
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

    def do_POST(self):
        if urlparse(self.path).path == "/save":
            ok, err = _save_to_yaml()
            if ok:
                log.info("Live settings written to /data/settings.yaml")
                body = b'{"ok":true}'
                self.send_response(200)
            else:
                log.warning("Save failed: %s", err)
                body = json.dumps({"ok": False, "error": err}).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


# ── Start ─────────────────────────────────────────────────────────────────────

def start(port: int) -> None:
    class _Server(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True
    server = _Server(("0.0.0.0", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True,
                     name="DebugServer").start()
    log.info("Debug stream: http://0.0.0.0:%d/", port)
