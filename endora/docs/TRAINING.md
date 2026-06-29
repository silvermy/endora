# Endora — Detection Tuning & Feedback Workflow

> **There is no neural-network training step.** Endora uses stock YOLO11-pose
> weights and a deterministic pose→gesture pipeline. "Training" here means
> *tuning that pipeline to your room, body, and camera* using a feedback loop:
>
> **gather labeled feedback → analyse it → adjust thresholds → lock in a
> regression capture.**
>
> This document describes that loop end to end. For a quick symptom→fix table
> see the [Tuning reference](../README.md#tuning-reference) in the README; this
> doc is the deeper "how and why."

---

## 1. The pipeline you're tuning

Each frame flows through three stages. Knowing which stage a problem lives in
tells you which knob to turn.

```
 RTSP frame
   │  preprocess (dewarp / crop / CLAHE)
   ▼
 [1] YOLO pose  ──►  keypoints for every person in frame   (cameras/pose_model.py)
   │                  • confidence + visibility per joint
   ▼
 [2] ArmTracker ──►  ArmReading: state (DOWN / SINGLE_UP / BOTH_UP / T_POSE /
   │                  CROSS_ARMS), forearm_dy, upright, raised_side
   │                  • per-person, per-side, visibility-aware   (cameras/arm_tracker.py)
   ▼
 [3] StateMachine ►  fires a Gesture (SNAP / HOLD / DOUBLE_SNAP / RAISE_BOTH /
                      T_POSE / CROSS_ARMS) with timing/sustain rules   (core/state_machine.py)
```

Every person in frame gets their own ArmTracker + StateMachine, so anyone can
gesture independently.

**Mental model of the two key gates:**
- **Stage 2 (ArmTracker)** decides *"is an arm up?"* — geometry + visibility.
- **Stage 3 (StateMachine)** decides *"is this arm-up a deliberate gesture?"* —
  verticality (`forearm_dy`) and time held (`snap_sustain_s`, `hold_duration_s`…).

---

## 2. Gather labeled feedback

All feedback lands in **`/data/feedback.jsonl`** (or `./feedback.jsonl` when run
outside Home Assistant). One JSON object per line.

### Automatic (always on)
- **`fired`** — every gesture the system emits, with the `ArmReading` that triggered it.
- **`near_miss`** — a frame where a gesture *almost* fired but a threshold blocked
  it, with a human-readable `reason`. These are the most valuable tuning signal.

### Manual — debug UI buttons
Open the debug page (HA sidebar → **Open Web UI**, or the direct port) and use:

| Button | Label written | Meaning |
|---|---|---|
| ✗ False positive | `false_positive` | The last gesture that fired was wrong — nothing should have fired. |
| ⇄ Wrong gesture | `wrong_gesture` | A gesture fired, but the *wrong* one (records intended vs fired). |
| ? Missed gesture | `false_negative` | You performed a gesture and nothing fired (snapshots the last ~10 readings). |
| 🚫 No pose | `no_pose` | You were visible but YOLO didn't detect you at all. |

`False positive` / `Wrong gesture` mark **the most recent fired gesture** — there
is no time limit; the entry records `seconds_after_fire` so a late mark is still
visible in analysis. (On a TTY you can also press `n` / `f` / `?`.)

### Getting the log out
The **⬇ feedback.jsonl** button downloads the log *and* copies it to the
clipboard (the clipboard path is what works inside the HA mobile app, whose
webview can't save files). Downloading **rotates** the log: the batch is backed
up to `feedback.jsonl.prev` and a fresh log starts — so each download is a clean
batch and you never lose the previous one.

---

## 3. Read the feedback

Load it for analysis:

```python
import pandas as pd
df = pd.read_json("feedback.jsonl", lines=True)
df["label"].value_counts()
```

### Entry shapes
```jsonc
{"label":"fired","gesture":"SNAP","reading":{"state":"SINGLE_UP","forearm_dy":0.13,"upright":true,"raised_side":"RIGHT"}}
{"label":"near_miss","gesture":"SNAP","reason":"sustain=0.106s < 0.15s required","reading":{...}}
{"label":"false_negative","gesture_hint":"SNAP","recent_readings":[ ... last ~10 frames ... ]}
{"label":"no_pose"}
```

### How to interpret it
- **`forearm_dy` is only meaningful when `state == SINGLE_UP`.** A `DOWN` reading
  always shows `forearm_dy: 0.0` — that means "arm not detected up," *not* "arm flat."
- **`upright`**: `true` = sitting/standing confirmed; `false` = reclined **or**
  hips not visible (blanket/crop). The reclined path uses stricter thresholds.
- **`near_miss` reasons tell you exactly which gate blocked the gesture:**
  - `sustain=0.09s < 0.15s required` → the arm *was* a good raise but wasn't held
    long enough → lower **`snap_sustain_s`** (Stage 3).
  - `forearm_dy=0.04 < 0.07 (min)` → the raise was too shallow / looked like a wave
    → lower **`snap_forearm_min`**, or it's genuinely not a snap (Stage 3).
- **A `false_negative` whose `recent_readings` are all `DOWN`** → the arm never
  reached `SINGLE_UP` → it's a Stage-2 problem (visibility, camera angle, or
  `arm_above_head_tolerance`), not a Stage-3 timing problem.
- **Lots of `no_pose`** → Stage-1: YOLO isn't finding you (lighting, framing,
  occlusion, or model size — see §6).

---

## 4. The tuning loop — symptom → knob

Change one thing at a time, then re-test and re-gather feedback.

| Symptom (from feedback) | Stage | Knob | Direction |
|---|---|---|---|
| `false_negative`, readings all `DOWN`, arm clearly up | 2 | `arm_above_head_tolerance` | ↓ (e.g. 0.10 → 0.05) |
| Arm up but only from a high/angled camera (wrist near shoulder) | 2 | `forearm_vertical_min` | ↓ to allow the forearm-vertical shortcut |
| One shoulder hidden (couch/blanket) rejects you | 2 | `pose_visibility_min` / `keypoint_visibility_min` | ↓ slightly |
| Reclined raises missed (lying down) | 2 | `arm_above_head_tolerance_reclined` | ↓ (raises reclined false-positive risk) |
| `near_miss: sustain … < … required`, good `forearm_dy` | 3 | `snap_sustain_s` | ↓ (e.g. 0.15 → 0.05) for fast snaps |
| `near_miss: forearm_dy … < … (min)` | 3 | `snap_forearm_min` | ↓ toward 0.05 (raises wave→snap risk) |
| HOLD fires too soon/late | 3 | `hold_duration_s` | adjust |
| T-pose fires while raising both arms | 3 | `sustain_s` | ↑ toward 1.0 |
| Gesture flickers / drops mid-hold | 3 | `state_release_s` | ↑ to bridge YOLO dropouts |
| Single-frame phantom gestures | 3 | `state_confirm_s` | ↑ |
| Tracking furniture / paintings | 1–2 | `pose_visibility_min`, `yolo_conf` | ↑ |
| `no_pose`, slow arm lifts missed | 1 | `motion_threshold` ↓ / `yolo_max_skip` ↓ | more frequent YOLO |
| High CPU | 1 | `yolo_pose_model` → nano, or `yolo_imgsz` ↓ | — |
| Pose drops on unusual poses | 1 | `yolo_pose_model` → `yolo11s-pose.onnx`, `yolo_imgsz` ↑ | more accuracy |

Current defaults live in `config/settings.py` (each field is commented with its
rationale and tuning direction).

---

## 5. Apply changes live (no restart)

The debug page **Save** button writes **`/data/runtime_overrides.yaml`**, which is
loaded **last** — it overrides both `options.json` (HA add-on config) and
`settings.yaml`:

```
settings.py defaults  →  settings.yaml  →  options.json  →  runtime_overrides.yaml (wins)
```

So changing a default in code has **no effect** on an install that has an override
for that key — tune via the page, or clear the override. Changes take effect
immediately; no restart needed.

While testing, watch the live overlay — it shows `state`, `forearm_dy`,
`snap_roll`, and `upright` in real time. Tune until `forearm_dy` reads a healthy
0.10+ during your raise and the state reaches `SINGLE_UP`.

---

## 6. Lock in fixes with regression captures

Once a gesture detects reliably, capture it so future tuning can't silently break it.

1. **Enable recording:** set `ENDORA_RECORD_TESTS=1` before starting the add-on.
   The `TestRecorder` keeps a rolling ~5 s buffer of YOLO keypoints and auto-saves
   a `.npz` every time a gesture fires. The debug page also gains a **Capture test
   case** button for manual snapshots.
2. **Captures land in `/data/test_captures/`** as `<ts>_<label>.npz` with:
   `keypoints [N,17,3]`, `t_offsets [N]`, `frame_w`, `frame_h`, `label`, `gesture`.
3. **Commit a fixture:** copy a good `.npz` into `endora/tests/captures/` and set
   its `gesture` to the expected enum name (e.g. `SNAP`).
4. **It now runs in CI:** `tests/test_captured.py` replays the keypoints through the
   real `ArmTracker → GestureStateMachine` and asserts the expected gesture fires
   (and within 1 s). These are *positive* assertions — they guard against a tuning
   change that stops a known-good gesture from firing.

```
cd endora && .venv/bin/python -m pytest tests/ -q
```

---

## 7. Changing the YOLO model (the closest thing to "model training")

Endora ships two stock pose models (`cameras/pose_model.py`):
- `yolo11n-pose.onnx` — nano, ~25 ms/frame on a Pi 5, least accurate.
- `yolo11s-pose.onnx` — small, ~50 ms/frame, noticeably better on unusual poses
  (lounging, blanket, arm overhead). **Default.**

Inference resolution is set by `yolo_imgsz` (default 320). ONNX models are usually
exported at a fixed size; `resolve_model_path()` will use a bundled or cached
`<model>-<imgsz>.onnx`, or export one from a `.pt` you drop in `/data/` (export
runs on x86/macOS only — torch is unsafe on the Pi's CPU).

If you ever want a *custom-trained* pose model, train a YOLO11-pose network
externally, export it to ONNX, and point `yolo_pose_model` at it — the rest of
the pipeline is model-agnostic as long as it emits 17 COCO keypoints.

---

## 8. Worked example

From a real session: many `near_miss` entries reading
`sustain=0.085–0.115s < 0.15s required`, all with **good** `forearm_dy` (0.08–0.15).

- **Diagnosis:** the snaps were genuine (strong verticality) but *fast* — held
  only ~0.1 s, under the 0.15 s sustain. A Stage-3 timing problem, not detection.
- **Fix:** lower `snap_sustain_s` to ~0.05 via the debug page. Re-test → the
  near-misses convert to `fired`.
- **Lock it in:** with `ENDORA_RECORD_TESTS=1`, perform the fast snap, copy the
  saved `.npz` into `tests/captures/`, commit. Now a future change that
  re-introduces an over-long sustain fails the test.
