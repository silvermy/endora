"""
cameras/labeler.py

Marking recorded captures with what was actually INTENDED.

Every .npz in the capture directory is named for what the state machine
decided. That is precisely the thing under review, so the filenames are not
labels — they are the hypothesis. Three separate fixes this week were left
half-finished because there was no way to tell a fire that was meant from
one that was not:

* the arm-span ceiling for FOLDED_ARMS was fitted against two known-false
  captures because the other seventy-five were unlabelled;
* flourish_require_progressive is measured, wired, and shipped switched OFF,
  because enabling it would suppress five of fourteen snaps and only two of
  those are confirmed false;
* the recline work could only be evaluated against captures whose status was
  inferred from when they happened.

So this is not a step towards anything in particular. It is the missing
denominator.

Labels live in one JSON file beside the captures, keyed by filename, and the
captures themselves are never modified — a trace is evidence and stays as it
was recorded. Deleting the label file loses opinions, never data.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

CAPTURE_DIR = Path(os.environ.get("ENDORA_CAPTURE_DIR", "/data/test_captures"))
LABELS_PATH = Path(os.environ.get("ENDORA_LABELS_PATH",
                                  str(CAPTURE_DIR.parent / "capture_labels.json")))

# The vocabulary is deliberately three words, not a scale. "Did you mean to
# do this?" has a yes, a no, and an honest shrug; anything finer invites
# guessing, and a guessed label is worse than an absent one because it looks
# like data.
LABELS = ("intended", "not_intended", "unsure")

_lock = threading.Lock()

# COCO skeleton edges, for drawing. Upper body plus legs — the legs matter
# for judging posture, which is what several of these disputes turned on.
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]


# ── label storage ────────────────────────────────────────────────────────────

def _read_all() -> dict:
    try:
        with LABELS_PATH.open() as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:                       # corrupt file: keep going
        log.warning("labeler: could not read %s: %s", LABELS_PATH, e)
        return {}


def _write_all(data: dict) -> None:
    """Atomic replace, so an interrupted write cannot lose the whole file."""
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LABELS_PATH.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    tmp.replace(LABELS_PATH)


def set_label(name: str, label: Optional[str], note: str = "") -> bool:
    """Record (or clear, with label=None) the intent behind one capture."""
    if not _is_safe_name(name):
        return False
    if label is not None and label not in LABELS:
        return False
    with _lock:
        data = _read_all()
        if label is None:
            data.pop(name, None)
        else:
            data[name] = {"label": label, "note": note[:500],
                          "at": time.time()}
        _write_all(data)
    return True


def get_labels() -> dict:
    with _lock:
        return _read_all()


# ── captures ─────────────────────────────────────────────────────────────────

def _is_safe_name(name: str) -> bool:
    """A capture name arrives from a URL; keep it to one plain filename in
    the capture directory."""
    return bool(name) and name.endswith(".npz") and Path(name).name == name


def _parse_name(name: str) -> dict:
    """`<epoch>_<camera>_<gesture>_pid<N>.npz`, tolerating older forms that
    predate the camera label or the pid suffix."""
    stem = name[:-4]
    parts = stem.split("_")
    out = {"ts": None, "camera": "", "gesture": "", "pid": ""}
    if parts and parts[0].isdigit():
        out["ts"] = int(parts[0])
        parts = parts[1:]
    if parts and len(parts[0]) == 1 and parts[0].isalpha():
        out["camera"] = parts[0]
        parts = parts[1:]
    if parts and parts[-1].startswith("pid"):
        out["pid"] = parts[-1][3:]
        parts = parts[:-1]
    out["gesture"] = "_".join(parts)
    return out


def list_captures() -> list[dict]:
    """Every capture, newest first, with its label if it has one."""
    labels = get_labels()
    rows = []
    try:
        paths = sorted(CAPTURE_DIR.glob("*.npz"))
    except Exception:
        paths = []
    for p in paths:
        meta = _parse_name(p.name)
        entry = labels.get(p.name) or {}
        rows.append({
            "name": p.name,
            "ts": meta["ts"],
            "gesture": meta["gesture"],
            "pid": meta["pid"],
            "bytes": p.stat().st_size if p.exists() else 0,
            "label": entry.get("label"),
            "note": entry.get("note", ""),
        })
    rows.sort(key=lambda r: (r["ts"] or 0), reverse=True)
    return rows


def load_trace(name: str) -> Optional[dict]:
    """Keypoints for one capture, rounded for transport.

    Confidences are kept: whether a joint was actually seen is often the
    whole story, and a skeleton drawn without them looks equally certain
    everywhere — which is exactly the illusion that made several of these
    captures hard to judge.
    """
    if not _is_safe_name(name):
        return None
    path = CAPTURE_DIR / name
    if not path.is_file():
        return None
    try:
        z = np.load(path, allow_pickle=True)
        kps = z["keypoints"]
        frames = [[[round(float(v), 1) if j < 2 else round(float(v), 2)
                    for j, v in enumerate(pt)] for pt in f] for f in kps]
        meta = _parse_name(name)
        return {
            "name": name,
            "frames": frames,
            "t": [round(float(x), 3) for x in z["t_offsets"]],
            "frame_w": int(z["frame_w"]),
            "frame_h": int(z["frame_h"]),
            "gesture": str(z["gesture"]) if "gesture" in z else meta["gesture"],
            "pid": str(z["pid"]) if "pid" in z else meta["pid"],
            "ts": meta["ts"],
        }
    except Exception as e:
        log.warning("labeler: could not load %s: %s", name, e)
        return None


def summary() -> dict:
    """Counts per gesture and label — the thing worth checking before
    drawing any conclusion from the set."""
    rows = list_captures()
    out: dict = {"total": len(rows), "labelled": 0, "by_gesture": {}}
    for r in rows:
        g = r["gesture"] or "?"
        slot = out["by_gesture"].setdefault(
            g, {"total": 0, "intended": 0, "not_intended": 0,
                "unsure": 0, "unlabelled": 0})
        slot["total"] += 1
        if r["label"]:
            out["labelled"] += 1
            slot[r["label"]] += 1
        else:
            slot["unlabelled"] += 1
    return out


# ── the page ─────────────────────────────────────────────────────────────────
#
# Deliberately one capture at a time with three big targets and keyboard
# shortcuts. There are a few hundred of these and the job only gets done if
# a decision costs one keystroke; a grid of thumbnails would be faster to
# build and never finished.

LABELER_HTML = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Endora — label captures</title>
<style>
 :root{--bg:#14161a;--fg:#e8eaed;--dim:#9aa0a6;--line:#2a2e35;
       --yes:#2e7d32;--no:#c62828;--maybe:#7a5c00;--acc:#4da3ff}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:15px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif}
 header{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;
        padding:10px 16px;border-bottom:1px solid var(--line)}
 h1{font-size:15px;margin:0;font-weight:600}
 .dim{color:var(--dim)}
 main{max-width:860px;margin:0 auto;padding:16px}
 .meta{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px}
 .meta b{font-weight:600}
 canvas{width:100%;height:auto;background:#0c0e11;border:1px solid var(--line);
        border-radius:8px;display:block;touch-action:none}
 .row{display:flex;gap:10px;align-items:center;margin:10px 0;flex-wrap:wrap}
 input[type=range]{flex:1;min-width:180px;accent-color:var(--acc)}
 button{font:inherit;color:var(--fg);background:#20242b;border:1px solid var(--line);
        border-radius:8px;padding:10px 14px;cursor:pointer}
 button:hover{border-color:var(--acc)}
 .choices{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}
 .choices button{padding:16px 10px;font-weight:600;font-size:16px}
 .yes{background:var(--yes)} .no{background:var(--no)} .maybe{background:var(--maybe)}
 .cur{outline:2px solid var(--acc)}
 .bar{height:6px;background:#20242b;border-radius:3px;overflow:hidden;margin-top:6px}
 .bar>i{display:block;height:100%;background:var(--acc)}
 table{border-collapse:collapse;width:100%;margin-top:18px;font-size:13px}
 td,th{border-top:1px solid var(--line);padding:4px 6px;text-align:left}
 kbd{background:#20242b;border:1px solid var(--line);border-radius:4px;
     padding:1px 5px;font-size:12px}
 .note{width:100%;background:#0c0e11;color:var(--fg);border:1px solid var(--line);
       border-radius:8px;padding:8px;margin-top:10px;font:inherit}
</style>
<header>
  <h1>Label captures</h1>
  <span class="dim" id="progress">loading…</span>
  <span class="dim" style="margin-left:auto">
    <kbd>1</kbd> meant it · <kbd>2</kbd> didn't · <kbd>3</kbd> unsure ·
    <kbd>←</kbd><kbd>→</kbd> move · <kbd>space</kbd> replay
  </span>
</header>
<main>
  <div class="bar"><i id="pbar" style="width:0"></i></div>
  <div class="meta" id="meta"></div>
  <canvas id="cv" width="640" height="320"></canvas>
  <div class="row">
    <button id="play">▶︎ replay</button>
    <input type="range" id="scrub" min="0" max="0" value="0">
    <span class="dim" id="tlabel">—</span>
  </div>
  <div class="choices">
    <button class="yes"   data-l="intended">I meant this</button>
    <button class="no"    data-l="not_intended">I didn't</button>
    <button class="maybe" data-l="unsure">Unsure</button>
  </div>
  <input class="note" id="note" placeholder="optional note (what were you doing?)">
  <div class="row">
    <button id="prev">← previous</button>
    <button id="next">next →</button>
    <button id="skipun">next unlabelled</button>
    <span class="dim" id="saved"></span>
  </div>
  <table id="tally"></table>
</main>
<script>
const SK = __SKELETON__;
let rows = [], i = 0, trace = null, fi = 0, timer = null, view = null;
const $ = s => document.querySelector(s);
const cv = $("#cv"), cx = cv.getContext("2d");

async function boot(){
  rows = await (await fetch("label/list")).json();
  if(!rows.length){ $("#progress").textContent = "no captures found"; return; }
  i = rows.findIndex(r => !r.label); if(i < 0) i = 0;
  await show();
}

function tally(){
  const by = {};
  for(const r of rows){
    const g = r.gesture || "?";
    by[g] = by[g] || {t:0, intended:0, not_intended:0, unsure:0};
    by[g].t++; if(r.label) by[g][r.label]++;
  }
  const done = rows.filter(r => r.label).length;
  $("#progress").textContent = `${done} of ${rows.length} labelled`;
  $("#pbar").style.width = (100*done/rows.length).toFixed(1) + "%";
  let h = "<tr><th>gesture</th><th>meant</th><th>didn't</th><th>unsure</th><th>left</th></tr>";
  for(const g of Object.keys(by).sort()){
    const b = by[g], left = b.t - b.intended - b.not_intended - b.unsure;
    h += `<tr><td>${g}</td><td>${b.intended}</td><td>${b.not_intended}</td>`
       + `<td>${b.unsure}</td><td class="dim">${left}</td></tr>`;
  }
  $("#tally").innerHTML = h;
}

async function show(){
  const r = rows[i];
  trace = await (await fetch("label/trace?name=" + encodeURIComponent(r.name))).json();
  const when = r.ts ? new Date(r.ts*1000).toLocaleString() : "unknown time";
  $("#meta").innerHTML =
      `<span><b>${r.gesture}</b></span><span>${when}</span>`
    + `<span class="dim">pid ${r.pid || "?"}</span>`
    + `<span class="dim">${trace.frames.length} frames</span>`
    + (r.label ? `<span style="color:var(--acc)">= ${r.label}</span>` : "");
  $("#note").value = r.note || "";
  document.querySelectorAll(".choices button").forEach(b =>
      b.classList.toggle("cur", b.dataset.l === r.label));
  $("#scrub").max = trace.frames.length - 1;
  // Zoom to the subject. At full frame a couch-distance person is a
  // hundred pixels in a 1280-wide view — far too small to judge a pose,
  // which is the entire job here. The box is taken over the WHOLE trace so
  // the view never moves while playing; a jittering crop would make
  // everything look like motion.
  view = fitView();
  cv.width = 860;
  cv.height = Math.round(cv.width * view.h / view.w);
  fi = 0; play();
  tally();
}

function fitView(){
  let x0=1e9, y0=1e9, x1=-1e9, y1=-1e9, any=false;
  for(const f of trace.frames) for(const p of f){
    if(p[2] < 0.3) continue;
    any = true;
    x0=Math.min(x0,p[0]); x1=Math.max(x1,p[0]);
    y0=Math.min(y0,p[1]); y1=Math.max(y1,p[1]);
  }
  if(!any) return {x:0,y:0,w:trace.frame_w,h:trace.frame_h};
  const pad = Math.max(40, 0.25*Math.max(x1-x0, y1-y0));
  x0-=pad; y0-=pad; x1+=pad; y1+=pad;
  // Keep a sane aspect so the body is not stretched.
  const w = Math.max(x1-x0, (y1-y0)*1.6);
  const h = Math.max(y1-y0, w/1.6);
  return {x:(x0+x1)/2 - w/2, y:(y0+y1)/2 - h/2, w, h};
}

function draw(){
  const f = trace.frames[fi], s = cv.width / view.w;
  const X = v => (v - view.x) * s, Y = v => (v - view.y) * s;
  cx.fillStyle = "#0c0e11"; cx.fillRect(0,0,cv.width,cv.height);
  // Where this is in the room, since the view is cropped to the subject.
  const ms = 90/trace.frame_w, mx = cv.width-98, my = 8;
  cx.strokeStyle = "#2a2e35"; cx.lineWidth = 1;
  cx.strokeRect(mx, my, 90, trace.frame_h*ms);
  cx.fillStyle = "rgba(77,163,255,.55)";
  cx.fillRect(mx + view.x*ms, my + view.y*ms, view.w*ms, view.h*ms);
  // Faint trail of where the wrists have been — a sweep is a shape over
  // time, and a single frame cannot show whether the arm travelled.
  for(const w of [9,10]){
    cx.beginPath();
    for(let k=0;k<=fi;k++){
      const p = trace.frames[k][w];
      if(p[2] < 0.3) continue;
      cx.lineTo(X(p[0]), Y(p[1]));
    }
    cx.strokeStyle = "rgba(77,163,255,.35)"; cx.lineWidth = 1.5; cx.stroke();
  }
  cx.lineWidth = 2.5;
  for(const [a,b] of SK){
    if(f[a][2] < 0.3 || f[b][2] < 0.3) continue;
    cx.beginPath(); cx.moveTo(X(f[a][0]), Y(f[a][1])); cx.lineTo(X(f[b][0]), Y(f[b][1]));
    cx.strokeStyle = "#39d353"; cx.stroke();
  }
  for(let k=0;k<f.length;k++){
    const p = f[k]; if(p[2] < 0.3) continue;
    cx.beginPath(); cx.arc(X(p[0]), Y(p[1]), (k===9||k===10)?6:4, 0, 7);
    // Confidence as opacity: a guessed joint must not look as sure as a
    // seen one.
    cx.fillStyle = (k===9||k===10)
        ? `rgba(77,163,255,${Math.min(1,p[2]+.25)})`
        : `rgba(232,234,237,${Math.min(1,p[2]+.15)})`;
    cx.fill();
  }
  $("#scrub").value = fi;
  $("#tlabel").textContent = (trace.t[fi] ?? 0).toFixed(2) + "s";
}

function play(){
  clearInterval(timer); fi = 0;
  timer = setInterval(() => {
    if(!trace) return;
    draw();
    if(++fi >= trace.frames.length){ clearInterval(timer); fi = trace.frames.length-1; }
  }, 90);
}

async function mark(label){
  const r = rows[i];
  await fetch("label/set", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({name:r.name, label, note:$("#note").value})});
  r.label = label; r.note = $("#note").value;
  $("#saved").textContent = "saved"; setTimeout(()=>$("#saved").textContent="", 900);
  if(i < rows.length-1){ i++; await show(); } else tally();
}

document.querySelectorAll(".choices button").forEach(b =>
    b.onclick = () => mark(b.dataset.l));
$("#prev").onclick = async () => { if(i>0){ i--; await show(); } };
$("#next").onclick = async () => { if(i<rows.length-1){ i++; await show(); } };
$("#skipun").onclick = async () => {
  const n = rows.findIndex((r,k) => k>i && !r.label);
  if(n >= 0){ i = n; await show(); }
};
$("#play").onclick = play;
$("#scrub").oninput = e => { clearInterval(timer); fi = +e.target.value; draw(); };
addEventListener("keydown", e => {
  if(e.target.tagName === "INPUT" && e.target.type !== "range") return;
  if(e.key === "1") mark("intended");
  else if(e.key === "2") mark("not_intended");
  else if(e.key === "3") mark("unsure");
  else if(e.key === "ArrowLeft")  $("#prev").click();
  else if(e.key === "ArrowRight") $("#next").click();
  else if(e.key === " "){ e.preventDefault(); play(); }
});
boot();
</script>
"""


def page() -> bytes:
    return LABELER_HTML.replace("__SKELETON__", json.dumps(SKELETON)).encode()
