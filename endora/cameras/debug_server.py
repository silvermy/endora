"""
cameras/debug_server.py — MJPEG debug stream + live parameter tuning UI
"""
from __future__ import annotations

import collections
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


def _host_ip() -> str:
    """Return the host's primary outbound IP (no packet sent)."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "homeassistant.local"

_latest: Dict[str, Optional[np.ndarray]] = {"A": None, "B": None}
_lock = threading.Lock()
_last_gesture: Dict = {"label": "", "ts": 0.0}
_single_camera: bool = False
_settings = None
_recorder = None
_frame_capture = None
_feedback_logger = None
_host_ip: str = "homeassistant.local"
_debug_port: int = 8765

# ── In-browser live log ───────────────────────────────────────────────────────
_LOG_BUF: collections.deque = collections.deque(maxlen=300)
_log_lock = threading.Lock()


class _LogHandler(logging.Handler):
    """Captures log records into a ring buffer for the /log endpoint."""

    _LEVEL_CHAR = {
        logging.DEBUG:    "D",
        logging.INFO:     "I",
        logging.WARNING:  "W",
        logging.ERROR:    "E",
        logging.CRITICAL: "E",
    }

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with _log_lock:
                _LOG_BUF.append({
                    "t": record.created,
                    "l": self._LEVEL_CHAR.get(record.levelno, "I"),
                    "n": record.name.split(".")[-1],
                    "m": record.getMessage(),
                })
        except Exception:
            self.handleError(record)


# ── Tunable parameter definitions ────────────────────────────────────────────
# (key, label, min, max, step, group)
# Trimmed to the sliders you actually reach for during live tuning.
_PARAMS = [
    ("arm_above_head_tolerance", "Arm raise margin",   0.0,  0.30, 0.01, "Gesture"),
    ("palm_twist_threshold",     "Snap sensitivity",   0.10, 1.20, 0.05, "Gesture"),
    ("cooldown_s",               "Cooldown (s)",       0,    10,   0.25, "Gesture"),
    ("yolo_conf",                "YOLO confidence",    0.10, 0.80, 0.01, "Body"),
    # View group — rendered as joystick (pan/tilt) + compact sliders
    ("dewarp_vfov",              "Vertical FOV (°)",   20,   100,  1,    "View"),
    ("frame_crop_bottom",        "Crop bottom (%)",    0,    60,   1,    "View"),
    ("pose_visibility_min",      "Min visibility",     0.05, 0.8,  0.01, "View"),
]

# Pan/tilt are the joystick axes — not sliders.
# (key, label, min, max, step)
_JOY_PARAMS = [
    ("dewarp_pan",  "Pan",  -30, 30, 1),   # X axis: ← left / right →
    ("dewarp_tilt", "Tilt", -10, 80, 1),   # Y axis: ↑ up  / down ↓
]

# Boolean toggles shown as switches above the sliders
# (key, label, description)
_TOGGLES = [
    ("low_light_enhance", "CLAHE enhance", "Boost local contrast before pose inference — helps dark clothing on dark backgrounds"),
]


# ── Public API ────────────────────────────────────────────────────────────────

def configure(camera_count: int) -> None:
    global _single_camera
    _single_camera = (camera_count == 1)


def set_settings(s) -> None:
    global _settings
    _settings = s


def set_recorder(r) -> None:
    global _recorder
    _recorder = r


def set_frame_capture(fc) -> None:
    global _frame_capture
    _frame_capture = fc


def set_feedback_logger(fl) -> None:
    global _feedback_logger
    _feedback_logger = fl


def set_host_info(ip: str, port: int) -> None:
    global _host_ip, _debug_port
    _host_ip = ip
    _debug_port = port


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
    result = {}
    for key, *_ in _PARAMS + _JOY_PARAMS:
        v = getattr(_settings, key, None)
        if v is not None:
            result[key] = v
    if hasattr(_settings, 'log_level'):
        result['log_level'] = _settings.log_level
    for key, *_ in _TOGGLES:
        v = getattr(_settings, key, None)
        if v is not None:
            result[key] = v
    return result


def _apply_setting(key: str, raw: str) -> bool:
    if _settings is None:
        return False
    current = getattr(_settings, key, None)
    # log_level is a string field not in _PARAMS — handle separately
    if key == 'log_level' and hasattr(_settings, 'log_level'):
        valid = ('debug', 'info', 'warning', 'error')
        if raw.lower() not in valid:
            return False
        _settings.log_level = raw.lower()
        _apply_log_level(raw.lower())
        return True
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


def _apply_log_level(level_str: str) -> None:
    """Update Python logging level across all loggers immediately."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    for name in list(logging.Logger.manager.loggerDict):
        lgr = logging.getLogger(name)
        if isinstance(lgr, logging.Logger):
            lgr.setLevel(level)
    log.info("Log level changed to %s", level_str.upper())


def _save_to_yaml() -> tuple[bool, str]:
    """
    Persist current live settings so they survive an add-on restart.

    Writes to two places:
      /data/runtime_overrides.yaml  — loaded by settings.py AFTER options.json,
                                      so these values win on every restart.
                                      The HA Supervisor never touches this file.
      /data/settings.yaml           — patched in-place (Docker / standalone path)
    """
    vals = _current_values()
    errors: list[str] = []

    # ── 1. runtime_overrides.yaml (survives HA Supervisor regeneration) ──
    # The Supervisor regenerates options.json from its stored config on every
    # restart, so patching options.json directly is not persistent.  This
    # file is loaded last in settings.py and takes the highest priority.
    overrides_path = Path("/data/runtime_overrides.yaml")
    try:
        lines = [
            "# Endora runtime overrides — written by debug page Save button.\n",
            "# Loaded after options.json; takes priority over HA UI values.\n",
            "# Delete this file to revert to the HA Configuration tab values.\n",
            "\n",
        ]
        for key, value in vals.items():
            if isinstance(value, float):
                formatted = f"{value:.4g}"
            elif isinstance(value, bool):
                formatted = str(value).lower()
            else:
                formatted = str(value)
            lines.append(f"{key}: {formatted}\n")
        overrides_path.write_text("".join(lines))
    except Exception as e:
        errors.append(f"runtime_overrides.yaml: {e}")

    # ── 2. settings.yaml (regex patch so comments are preserved) ─────────
    yaml_path = Path("/data/settings.yaml")
    try:
        text = yaml_path.read_text() if yaml_path.exists() else ""
        for key, value in vals.items():
            if isinstance(value, float):
                formatted = f"{value:.4g}"
            elif isinstance(value, bool):
                formatted = str(value).lower()
            else:
                formatted = str(value)
            pattern = rf'^({re.escape(key)}\s*:\s*)([^\n#]*)'
            new_text, n = re.subn(pattern, rf'\g<1>{formatted}',
                                  text, flags=re.MULTILINE)
            if n:
                text = new_text
            else:
                text += f"\n{key}: {formatted}"
        yaml_path.write_text(text)
    except Exception as e:
        errors.append(f"settings.yaml: {e}")

    if errors:
        return False, "; ".join(errors)
    return True, ""


# ── HTML page ─────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<title>Endora Debug</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{
  background:#0d0d0d;
  font-family:system-ui,-apple-system,'Segoe UI',sans-serif;
  color:#d0d0d0;
  -webkit-font-smoothing:antialiased;
  text-rendering:optimizeLegibility;
  display:flex;flex-direction:column;align-items:center;
  padding:8px 10px;gap:8px
}
h3{font-size:11px;letter-spacing:3px;color:#555;font-weight:500;text-transform:uppercase}
#wrap{display:flex;gap:12px;width:100%;max-width:1300px;align-items:flex-start}
#vbox{flex:1 1 auto;min-width:0}
#vbox img{width:100%;display:block;border:1px solid #1e1e1e;min-height:240px;background:#111}
#legend{font-size:11px;color:#444;text-align:center;padding:4px 0}
#fbrow{display:flex;gap:8px;align-items:center;padding:8px 0 2px}
#fpbtn,#fnbtn{
  flex:1;padding:11px 8px;border-radius:7px;cursor:pointer;
  font-size:15px;font-weight:700;border:2px solid;letter-spacing:.3px
}
#fpbtn{background:#3a0a0a;border-color:#cc3333;color:#ff6666}
#fpbtn:hover{background:#551515}
#fnbtn{background:#0a1a3a;border-color:#3366cc;color:#6699ff}
#fnbtn:hover{background:#152550}
#feedbackmsg{font-size:12px;min-width:120px;text-align:center;color:#888}
#panel{
  flex:0 0 272px;background:#111;border:1px solid #222;border-radius:8px;
  padding:12px;display:flex;flex-direction:column;gap:12px;
  max-height:92vh;overflow-y:auto
}
/* ── toggles ── */
.togrow{
  display:flex;align-items:center;justify-content:space-between;
  padding:7px 10px;background:#161616;border-radius:6px;border:1px solid #222
}
.togrow .rowlabel{font-size:13px;color:#999;font-weight:500}
.tog-wrap{display:flex;align-items:center;gap:8px}
#loglabel{font-size:11px;font-weight:600;color:#555;min-width:38px;text-align:right}
.toggle{position:relative;display:inline-block;width:42px;height:22px}
.toggle input{opacity:0;width:0;height:0;position:absolute}
.tog-track{
  position:absolute;inset:0;background:#2a2a2a;border-radius:22px;
  cursor:pointer;transition:background .2s
}
.tog-track:before{
  content:'';position:absolute;width:16px;height:16px;
  left:3px;top:3px;background:#555;border-radius:50%;
  transition:transform .2s,background .2s
}
.toggle input:checked + .tog-track{background:#163216}
.toggle input:checked + .tog-track:before{transform:translateX(20px);background:#5c5}
/* ── slider groups ── */
.grp{
  font-size:10px;letter-spacing:2px;color:#555;text-transform:uppercase;
  font-weight:600;border-bottom:1px solid #1e1e1e;padding-bottom:4px
}
.row{display:flex;flex-direction:column;gap:3px;margin-bottom:1px}
.lbl{display:flex;justify-content:space-between;align-items:baseline;font-size:12px;color:#999}
.val{
  color:#e8c040;font-weight:600;font-size:12px;
  min-width:44px;text-align:right;
  font-variant-numeric:tabular-nums
}
input[type=range]{
  width:100%;cursor:pointer;
  -webkit-appearance:none;appearance:none;
  height:4px;background:#2a2a2a;border-radius:2px;outline:none;margin:3px 0
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;appearance:none;
  width:16px;height:16px;border-radius:50%;
  background:#e8c040;cursor:pointer;border:none;
  box-shadow:0 0 0 2px #0d0d0d
}
input[type=range]::-moz-range-thumb{
  width:16px;height:16px;border-radius:50%;
  background:#e8c040;cursor:pointer;border:none
}
input[type=range]:focus::-webkit-slider-thumb{box-shadow:0 0 0 2px #0d0d0d,0 0 0 4px #e8c04066}
/* ── save ── */
#savebtn{
  width:100%;padding:8px;background:#0e2e0e;border:1px solid #2e6e2e;
  color:#5c5;border-radius:6px;cursor:pointer;
  font-family:system-ui,-apple-system,sans-serif;
  font-size:13px;font-weight:600;margin-top:2px
}
#savebtn:hover{background:#163016}
#savebtn:disabled{opacity:.45;cursor:default}
#savemsg{font-size:12px;min-height:16px;text-align:center;padding-top:3px}
@media(max-width:860px){
  #wrap{flex-direction:column}
  #panel{flex:none;width:100%;max-height:none}
}
/* ── joystick ── */
.joy-wrap{display:flex;gap:12px;align-items:center;padding:2px 0 8px}
.joy-pad{
  position:relative;width:140px;height:140px;flex-shrink:0;
  background:#080808;border:1px solid #2a2a2a;border-radius:6px;
  cursor:crosshair;touch-action:none;user-select:none
}
.joy-pad::before,.joy-pad::after{content:'';position:absolute;background:#1c1c1c}
.joy-pad::before{left:50%;top:6px;bottom:6px;width:1px}
.joy-pad::after {top:50%;left:6px;right:6px;height:1px}
.joy-dot{
  position:absolute;width:16px;height:16px;border-radius:50%;
  background:#e8c040;transform:translate(-50%,-50%);pointer-events:none;
  box-shadow:0 0 0 2px #0d0d0d,0 0 0 4px #e8c04055
}
.joy-axlbl{
  position:absolute;font-size:9px;color:#333;pointer-events:none;line-height:1
}
.joy-info{display:flex;flex-direction:column;gap:8px;justify-content:center}
.joy-kv{font-size:12px;color:#666}
.joy-kv b{color:#e8c040;font-variant-numeric:tabular-nums;display:inline-block;min-width:26px;text-align:right}
/* ── live log panel ── */
#logbox{
  width:100%;max-width:1300px;
  background:#080808;border:1px solid #1a1a1a;border-radius:6px;
  display:flex;flex-direction:column;height:220px;
}
.log-hdr{
  display:flex;align-items:center;gap:10px;
  padding:5px 10px;border-bottom:1px solid #1a1a1a;flex-shrink:0;
}
.log-title{font-size:10px;letter-spacing:2px;color:#555;text-transform:uppercase;font-weight:600;flex:1}
.log-hdr label{font-size:11px;color:#555;display:flex;align-items:center;gap:4px;cursor:pointer}
.log-hdr button{font-size:11px;color:#444;background:none;border:none;cursor:pointer;padding:0 4px}
.log-hdr button:hover{color:#888}
#loglines{
  overflow-y:auto;flex:1;padding:3px 8px;
  font-family:'SF Mono','Consolas','Fira Mono',monospace;font-size:11px;line-height:1.65;
}
.logline{white-space:pre-wrap;word-break:break-all}
.ll-D{color:#3a7a3a}
.ll-I{color:#6a6a6a}
.ll-W{color:#a07820}
.ll-E{color:#a03030}
</style>
</head>
<body>
<h3>Endora Debug</h3>
<div id="wrap">
  <div id="vbox">
    <img id="streamimg" alt="stream">
    <div id="legend">YOLO pose &nbsp;·&nbsp; grlib hands &nbsp;·&nbsp; state machine &nbsp;·&nbsp; <a href="captures" target="_blank" style="color:#555;text-decoration:none">&#128249; captures</a></div>
    <div id="fbrow">
      <button id="fpbtn" onclick="doFeedback('fp')" title="Mark the last gesture that fired as a false positive (within 5s)">&#10007; False positive</button>
      <button id="fnbtn" onclick="doFeedback('fn')" title="I just did a gesture and nothing was detected">&#63; Missed gesture</button>
      <span id="feedbackmsg"></span>
    </div>
  </div>
  <div id="panel">
    <div class="togrow">
      <span class="rowlabel">Debug logging</span>
      <div class="tog-wrap">
        <span id="loglabel">INFO</span>
        <label class="toggle" title="Toggle debug / info logging">
          <input type="checkbox" id="logtoggle" onchange="setLog(this.checked)">
          <span class="tog-track"></span>
        </label>
      </div>
    </div>
    <div id="booltoggles"></div>
    <div id="sliders"></div>
    <button id="savebtn" onclick="doSave()">&#128190;&nbsp; Save to settings.yaml</button>
    <div id="savemsg"></div>
    <button id="capturebtn" onclick="doCapture()" style="display:none">&#128249;&nbsp; Capture test case</button>
    <div id="capturemsg" style="font-size:12px;min-height:16px;text-align:center;padding-top:3px"></div>
  </div>
</div>
<div id="logbox">
  <div class="log-hdr">
    <span class="log-title">Live Log</span>
    <label><input type="checkbox" id="autoscroll" checked>&nbsp;auto-scroll</label>
    <button id="clearlog">clear</button>
  </div>
  <div id="loglines"></div>
</div>
<script>
const PARAMS  = __PARAMS_JSON__;
const TOGGLES = __TOGGLES_JSON__;
const JOY     = __JOY_PARAMS_JSON__;

function fmt(v, step) {
  const n = +v;
  if (step >= 1) return String(Number.isInteger(n) ? n : n.toFixed(1));
  return n.toFixed(2);
}

// ── Joystick ────────────────────────────────────────────────────────────────
var _joyActive = false;
function buildJoystick(vals) {
  const [panKey,, panMin, panMax, panStep]   = JOY[0];
  const [tiltKey,,tiltMin,tiltMax,tiltStep]  = JOY[1];
  const pv = vals[panKey]  !== undefined ? +vals[panKey]  : 0;
  const tv = vals[tiltKey] !== undefined ? +vals[tiltKey] : 20;

  // Percentage position so it works before layout
  const px = ((pv  - panMin)  / (panMax  - panMin)  * 100).toFixed(1) + '%';
  const py = ((1 - (tv - tiltMin) / (tiltMax - tiltMin)) * 100).toFixed(1) + '%';

  const wrap = document.createElement('div');
  wrap.className = 'joy-wrap';
  wrap.innerHTML =
    '<div class="joy-pad" id="joypad">' +
      '<div class="joy-dot" id="joydot" style="left:' + px + ';top:' + py + '"></div>' +
      '<span class="joy-axlbl" style="top:3px;left:50%;transform:translateX(-50%)">up</span>' +
      '<span class="joy-axlbl" style="bottom:3px;left:50%;transform:translateX(-50%)">down</span>' +
      '<span class="joy-axlbl" style="left:3px;top:50%;transform:translateY(-50%)">◀</span>' +
      '<span class="joy-axlbl" style="right:3px;top:50%;transform:translateY(-50%)">▶</span>' +
    '</div>' +
    '<div class="joy-info">' +
      '<div class="joy-kv">Pan&nbsp; <b id="jv_' + panKey  + '">' + pv + '</b>°</div>' +
      '<div class="joy-kv">Tilt <b id="jv_' + tiltKey + '">' + tv + '</b>°</div>' +
    '</div>';

  const pad = wrap.querySelector('#joypad');
  const dot = wrap.querySelector('#joydot');

  function applyXY(x, y) {
    var r = pad.getBoundingClientRect();
    var w = r.width || 140, h = r.height || 140;
    x = Math.max(0, Math.min(w, x));
    y = Math.max(0, Math.min(h, y));
    var pan  = panMin  + (x / w) * (panMax  - panMin);
    var tilt = tiltMin + (1 - y / h) * (tiltMax - tiltMin);
    var pr = Math.round(pan  / panStep)  * panStep;
    var tr = Math.round(tilt / tiltStep) * tiltStep;
    dot.style.left = (x / w * 100).toFixed(1) + '%';
    dot.style.top  = (y / h * 100).toFixed(1) + '%';
    var pe = document.getElementById('jv_' + panKey);
    var te = document.getElementById('jv_' + tiltKey);
    if (pe) pe.textContent = pr;
    if (te) te.textContent = tr;
    onCom(panKey, pr); onCom(tiltKey, tr);
  }

  function fromEvent(e) {
    var r = pad.getBoundingClientRect();
    var src = e.touches ? e.touches[0] : e;
    applyXY(src.clientX - r.left, src.clientY - r.top);
  }

  pad.addEventListener('mousedown',  function(e){ _joyActive=true; fromEvent(e); });
  pad.addEventListener('touchstart', function(e){ _joyActive=true; fromEvent(e); e.preventDefault(); }, {passive:false});
  if (!pad._joyDoc) {
    pad._joyDoc = true;
    document.addEventListener('mousemove',  function(e){ if(_joyActive) fromEvent(e); });
    document.addEventListener('touchmove',  function(e){ if(_joyActive){ fromEvent(e); e.preventDefault(); }}, {passive:false});
    document.addEventListener('mouseup',  function(){ _joyActive=false; });
    document.addEventListener('touchend', function(){ _joyActive=false; });
  }
  return wrap;
}

function build(vals) {
  // ── Bool toggles ──
  const tc = document.getElementById('booltoggles');
  tc.innerHTML = '';
  TOGGLES.forEach(([key, label, desc]) => {
    const on = vals[key] === true || vals[key] === 'true';
    const row = document.createElement('div');
    row.className = 'togrow'; row.title = desc;
    row.innerHTML =
      '<span class="rowlabel">' + label + '</span>' +
      '<div class="tog-wrap">' +
      '<span id="lbl_' + key + '" style="font-size:11px;font-weight:600;min-width:28px;text-align:right;color:' + (on?'#e8c040':'#555') + '">' + (on?'ON':'OFF') + '</span>' +
      '<label class="toggle"><input type="checkbox" id="tog_' + key + '" ' + (on?'checked':'') + ' onchange="setBool(\'' + key + '\',this.checked)">' +
      '<span class="tog-track"></span></label></div>';
    tc.appendChild(row);
  });

  // ── Sliders (with joystick injected before View group) ──
  const c = document.getElementById('sliders');
  c.innerHTML = '';
  let grp = null;
  PARAMS.forEach(([key, label, mn, mx, step, group]) => {
    if (group !== grp) {
      grp = group;
      const d = document.createElement('div');
      d.className = 'grp'; d.textContent = group;
      c.appendChild(d);
      if (group === 'View') c.appendChild(buildJoystick(vals));
    }
    const v = vals[key] !== undefined ? vals[key] : ((+mn + +mx) / 2);
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML =
      '<div class="lbl"><span>' + label + '</span>' +
      '<span class="val" id="v_' + key + '">' + fmt(v, step) + '</span></div>' +
      '<input type="range" id="s_' + key + '" min="' + mn + '" max="' + mx + '" step="' + step + '" value="' + v + '"' +
      ' oninput="onIn(\'' + key + '\',this.value,' + step + ')"' +
      ' onchange="onCom(\'' + key + '\',this.value)">';
    c.appendChild(row);
  });

  // ── Log toggle ──
  const ll = vals.log_level || 'info';
  const isDbg = ll === 'debug';
  document.getElementById('logtoggle').checked = isDbg;
  document.getElementById('loglabel').textContent = isDbg ? 'DEBUG' : 'INFO';
  document.getElementById('loglabel').style.color = isDbg ? '#e8c040' : '#555';
}

function onIn(key, v, step) {
  document.getElementById('v_' + key).textContent = fmt(v, step);
}

const _t = {};
function onCom(key, value) {
  clearTimeout(_t[key]);
  _t[key] = setTimeout(() =>
    fetch('set?key='+encodeURIComponent(key)+'&value='+encodeURIComponent(value))
      .catch(e => console.warn('set', e)), 60);
}

function setBool(key, on) {
  const lbl = document.getElementById('lbl_' + key);
  if (lbl) { lbl.textContent = on ? 'ON' : 'OFF'; lbl.style.color = on ? '#e8c040' : '#555'; }
  fetch('set?key='+encodeURIComponent(key)+'&value='+(on?'true':'false'))
    .catch(e => console.warn(e));
}

function setLog(isDebug) {
  const level = isDebug ? 'debug' : 'info';
  document.getElementById('loglabel').textContent = isDebug ? 'DEBUG' : 'INFO';
  document.getElementById('loglabel').style.color = isDebug ? '#e8c040' : '#555';
  fetch('set?key=log_level&value=' + level).catch(e => console.warn(e));
}

function doSave() {
  const btn = document.getElementById('savebtn');
  const msg = document.getElementById('savemsg');
  btn.disabled = true;
  msg.style.color = '#888'; msg.textContent = 'saving\u2026';
  fetch('save', {method:'POST'})
    .then(r => r.json())
    .then(d => {
      msg.style.color = d.ok ? '#5c5' : '#c55';
      msg.textContent = d.ok ? '\u2713 saved' : '\u2717 ' + (d.error || 'error');
    })
    .catch(() => { msg.style.color='#c55'; msg.textContent='\u2717 request failed'; })
    .finally(() => { btn.disabled = false; });
}

fetch('settings').then(r=>r.json()).then(build).catch(()=>build({}));

// Show capture button only when recorder is active
fetch('recorder_status').then(r=>r.json()).then(function(d){
  if(d.active) document.getElementById('capturebtn').style.display='';
}).catch(function(){});

function doCapture() {
  var btn = document.getElementById('capturebtn');
  var msg = document.getElementById('capturemsg');
  var label = prompt('Test case label (e.g. snap_right_arm):', 'manual');
  if (label === null) return;
  btn.disabled = true;
  msg.style.color='#888'; msg.textContent='saving…';
  fetch('capture', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({label: label})})
    .then(function(r){return r.json();})
    .then(function(d){
      msg.style.color = d.ok ? '#5c5' : '#c55';
      msg.textContent = d.ok ? '✓ saved: '+d.file.split('/').pop() : '✗ '+(d.error||'error');
    })
    .catch(function(){ msg.style.color='#c55'; msg.textContent='✗ request failed'; })
    .finally(function(){ btn.disabled=false; });
}

// ── Feedback ────────────────────────────────────────────────────────────────
function doFeedback(label) {
  var msg = document.getElementById('feedbackmsg');
  var hint = label === 'fn' ? (prompt('What gesture did you try? (e.g. SNAP)', 'SNAP') || 'unknown') : undefined;
  if (label === 'fn' && hint === null) return; // cancelled
  var body = label === 'fn' ? JSON.stringify({label:'fn', hint:hint}) : JSON.stringify({label:'fp'});
  msg.style.color = '#888'; msg.textContent = 'logging…';
  fetch('feedback', {method:'POST', headers:{'Content-Type':'application/json'}, body:body})
    .then(function(r){return r.json();})
    .then(function(d){
      msg.style.color = d.ok ? (label==='fp'?'#c55':'#59c') : '#888';
      msg.textContent = d.ok ? '✓ ' + d.msg : '✗ ' + (d.error || 'error');
      setTimeout(function(){ msg.textContent=''; }, 4000);
    })
    .catch(function(){ msg.style.color='#888'; msg.textContent='✗ request failed'; });
}

// ── Live log ────────────────────────────────────────────────────────────────
(function() {
  var _since = 0;
  function pollLog() {
    fetch('log?since=' + _since)
      .then(function(r) { return r.json(); })
      .then(function(data) {
        if (data.entries && data.entries.length) {
          var box = document.getElementById('loglines');
          var auto = document.getElementById('autoscroll').checked;
          data.entries.forEach(function(e) {
            var d = document.createElement('div');
            d.className = 'logline ll-' + e.l;
            var t = new Date(e.t * 1000).toTimeString().slice(0, 8);
            d.textContent = t + ' [' + e.n + '] ' + e.m;
            box.appendChild(d);
          });
          while (box.children.length > 300) box.removeChild(box.firstChild);
          if (auto) box.scrollTop = box.scrollHeight;
          _since = data.next_since;
        }
      })
      .catch(function() {})
      .then(function() { setTimeout(pollLog, 1000); });
  }
  document.getElementById('clearlog').addEventListener('click', function() {
    document.getElementById('loglines').innerHTML = '';
  });
  pollLog();
})();

// ── Frame-by-frame polling ──────────────────────────────────────────────────
// Fetches individual JPEGs from the relative 'frame' endpoint at ~10 fps.
// Relative URL works through both HA ingress (HTTPS) and direct HTTP access.
// Avoids multipart/x-mixed-replace (Safari issues, ingress buffering) and
// mixed-content blocking (no hardcoded http:// stream URL needed).
(function() {
  var img = document.getElementById('streamimg');
  var _prevUrl = null;
  var _delay = 100;

  function fetchFrame() {
    fetch('frame?' + Date.now())
      .then(function(r) {
        if (!r.ok) throw new Error(r.status);
        return r.blob();
      })
      .then(function(blob) {
        var url = URL.createObjectURL(blob);
        img.src = url;
        if (_prevUrl) URL.revokeObjectURL(_prevUrl);
        _prevUrl = url;
        _delay = 100;
        setTimeout(fetchFrame, _delay);
      })
      .catch(function() {
        _delay = Math.min(_delay * 2, 5000);
        setTimeout(fetchFrame, _delay);
      });
  }
  fetchFrame();
})();
</script>
</body>
</html>"""


# ── Captures gallery ─────────────────────────────────────────────────────────

def _captures_html() -> str:
    return r"""<!DOCTYPE html>
<html>
<head>
<title>Endora Captures</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0d0d;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;
  color:#d0d0d0;padding:12px}
h2{font-size:13px;letter-spacing:3px;color:#555;font-weight:500;text-transform:uppercase;
  margin-bottom:12px}
#toolbar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;align-items:center}
#toolbar select,#toolbar button{
  background:#161616;border:1px solid #2a2a2a;color:#aaa;
  padding:5px 10px;border-radius:5px;font-size:12px;cursor:pointer}
#toolbar button:hover{background:#222}
#count{font-size:12px;color:#444}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.card{background:#111;border:1px solid #1e1e1e;border-radius:7px;overflow:hidden}
.card img{width:100%;display:block;min-height:120px;background:#0a0a0a;cursor:pointer}
.card img:hover{opacity:.85}
.meta{padding:8px 10px;font-size:11px;line-height:1.7}
.ev{font-weight:700;font-size:12px;margin-bottom:2px}
.ev-gesture{color:#5c5}
.ev-state{color:#e8c040}
.ev-pose_lost{color:#c55}
.kv{color:#555}
.kv b{color:#888}
#lightbox{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);
  z-index:999;align-items:center;justify-content:center;cursor:zoom-out
}
#lightbox img{max-width:96vw;max-height:92vh;object-fit:contain}
#lightbox.open{display:flex}
</style>
</head>
<body>
<h2>Debug Captures</h2>
<div id="toolbar">
  <select id="filter">
    <option value="">All events</option>
    <option value="gesture">Gestures</option>
    <option value="state">State changes</option>
    <option value="pose_lost">Pose lost</option>
  </select>
  <button onclick="load()">&#8635; Refresh</button>
  <span id="count"></span>
</div>
<div id="grid"></div>
<div id="lightbox" onclick="this.classList.remove('open')">
  <img id="lbimg" src="" alt="">
</div>
<script>
var _all = [];
function evClass(ev) {
  if (ev.startsWith('gesture')) return 'ev-gesture';
  if (ev.startsWith('state'))   return 'ev-state';
  if (ev.startsWith('pose'))    return 'ev-pose_lost';
  return '';
}
function fmt(ts) {
  var d = new Date(ts * 1000);
  return d.toLocaleTimeString() + ' ' + d.toLocaleDateString();
}
function render(items) {
  var filt = document.getElementById('filter').value;
  var shown = filt ? items.filter(function(i){ return i.event_type.startsWith(filt); }) : items;
  document.getElementById('count').textContent = shown.length + ' / ' + items.length + ' frames';
  var g = document.getElementById('grid');
  g.innerHTML = '';
  shown.forEach(function(m) {
    var div = document.createElement('div');
    div.className = 'card';
    var ev = m.event_type || '';
    var meta = [
      ['time', fmt(m.timestamp || 0)],
      ['cam',  m.camera || '?'],
      ['arm',  m.arm_state || '?'],
      m.gesture ? ['gesture', m.gesture] : null,
      ['forearm_dy', (m.forearm_dy || 0).toFixed(3)],
      m.upright !== undefined && m.upright !== null ? ['upright', String(m.upright)] : null,
    ].filter(Boolean);
    div.innerHTML =
      '<img src="captures/' + encodeURIComponent(m.filename) + '" loading="lazy" ' +
        'onclick="zoom(this.src)" alt="">' +
      '<div class="meta"><div class="ev ' + evClass(ev) + '">' + ev + '</div>' +
      meta.map(function(kv){ return '<div class="kv"><b>' + kv[0] + ':</b> ' + kv[1] + '</div>'; }).join('') +
      '</div>';
    g.appendChild(div);
  });
}
function zoom(src) {
  document.getElementById('lbimg').src = src;
  document.getElementById('lightbox').classList.add('open');
}
function load() {
  fetch('captures/list')
    .then(function(r){ return r.json(); })
    .then(function(data){ _all = data; render(_all); })
    .catch(function(){ document.getElementById('count').textContent = 'error loading'; });
}
document.getElementById('filter').addEventListener('change', function(){ render(_all); });
load();
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
            html = (_HTML_TEMPLATE
                    .replace("__PARAMS_JSON__",     json.dumps(_PARAMS))
                    .replace("__TOGGLES_JSON__",    json.dumps(_TOGGLES))
                    .replace("__JOY_PARAMS_JSON__", json.dumps(_JOY_PARAMS))
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
                body = b'{"ok":true}'
                self.send_response(200)
            else:
                body = b'{"ok":false}'
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/frame":
            # Single JPEG — used by the JS frame-poller (works in all browsers
            # including Safari, and through HA ingress without mixed content).
            jpg = _compose()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpg)

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
                    # Flush after every frame — without this the OS send-buffer
                    # holds the data until it fills (~8 KB), which can delay the
                    # first visible frame by several seconds and makes the browser
                    # show a blank image until the buffer finally drains.
                    self.wfile.flush()
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        elif parsed.path == "/recorder_status":
            body = json.dumps({"active": _recorder is not None}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/log":
            since = float(qs.get("since", ["0"])[0])
            with _log_lock:
                entries = [e for e in _LOG_BUF if e["t"] > since]
            next_since = entries[-1]["t"] if entries else since
            body = json.dumps({"entries": entries, "next_since": next_since}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/captures":
            body = _captures_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/captures/list":
            items = _frame_capture.list_captures() if _frame_capture else []
            body = json.dumps(items).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path.startswith("/captures/") and parsed.path.endswith(".jpg"):
            filename = parsed.path[len("/captures/"):]
            jpg = _frame_capture.get_jpeg(filename) if _frame_capture else None
            if jpg:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(jpg)
            else:
                self.send_response(404)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/capture":
            if _recorder is None:
                body = json.dumps({
                    "ok": False,
                    "error": "recorder not active — set ENDORA_RECORD_TESTS=1 and restart",
                }).encode()
                self.send_response(503)
            else:
                import urllib.parse as _up
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b""
                try:
                    payload = json.loads(raw) if raw else {}
                except Exception:
                    payload = {}
                label = str(payload.get("label", "manual")).strip() or "manual"
                saved = _recorder.manual_capture(label=label)
                if saved:
                    body = json.dumps({"ok": True, "file": str(saved)}).encode()
                    self.send_response(200)
                else:
                    body = json.dumps({"ok": False, "error": "buffer empty"}).encode()
                    self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/feedback":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw) if raw else {}
            except Exception:
                payload = {}
            label = payload.get("label", "")
            if _feedback_logger is None:
                body = json.dumps({"ok": False, "error": "feedback logger not active"}).encode()
                self.send_response(503)
            elif label == "fp":
                ok = _feedback_logger.mark_false_positive()
                body = json.dumps({"ok": ok, "msg": "marked as false positive" if ok else "no recent gesture to mark (5s window expired)"}).encode()
                self.send_response(200)
            elif label == "fn":
                hint = str(payload.get("hint", "unknown"))
                _feedback_logger.mark_false_negative(gesture_hint=hint)
                body = json.dumps({"ok": True, "msg": "recorded as missed gesture"}).encode()
                self.send_response(200)
            else:
                body = json.dumps({"ok": False, "error": "label must be 'fp' or 'fn'"}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/save":
            ok, err = _save_to_yaml()
            if ok:
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

def start(port: int, ingress_port: int = 8766) -> None:
    class _Server(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True

    # Attach the in-browser log handler to the root logger so all log records
    # flow into the ring buffer regardless of which module emitted them.
    _handler = _LogHandler()
    _handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(_handler)

    # Debug stream server (direct access)
    debug_server = _Server(("0.0.0.0", port), _Handler)
    threading.Thread(target=debug_server.serve_forever, daemon=True,
                     name="DebugServer").start()
    log.info("Debug stream: http://%s:%d/", _host_ip, port)

    # Ingress server (HA sidebar) — serves the full debug UI directly so it
    # works through HA's HTTPS ingress proxy without popup/mixed-content issues.
    # All URLs in the HTML are relative (/stream, /settings, /set, /save) so
    # they resolve correctly whether accessed via ingress or the direct port.
    ingress_server = _Server(("0.0.0.0", ingress_port), _Handler)
    threading.Thread(target=ingress_server.serve_forever, daemon=True,
                     name="IngressServer").start()
    log.info("Ingress (sidebar): port %d serving full debug UI", ingress_port)
