# Changelog

## 1.9.134

### Fixed — chiming for a sweep that never became a raise

Read live from a running install: **three chimes in 151 seconds, with no `SINGLE_UP` transition and no gesture at all.**

Sweep is computed for the highest arm regardless of its state, so an arm swung up with a bent elbow — one that never clears `arm_extension_min` and therefore can never become a gesture — still produced a climb and rate that satisfied the chime. The chime now additionally requires the reading to be a confirmed raise, so a sweep that goes nowhere stays silent.

## 1.9.133

### Fixed / Added — the live diagnostics were nearly unusable

Pulled from a running install, the debug endpoints turned out to hide the things that matter most:

- **The log ring buffer held under a minute.** `"N person(s) tracked"` was logged every single frame, so 300 entries covered 57 seconds — by the time you look, the gesture has scrolled away. It now logs only when the count changes.
- **The "What's actually running?" panel omitted the flourish settings** — `snap_require_flourish`, `flourish_min_climb`, `flourish_min_rate` — which are the settings that now decide whether a gesture fires at all, plus the per-gesture enables and `cross_gesture_cooldown_s`. All are listed now, in the panel and in the startup log.
- **Added `/version`**, because "which build is actually deployed" has been guesswork more than once, and a feedback log records no version at all.

Measured on the live install for reference: the analyser loop runs at **4.8 fps** at `yolo_imgsz: 320` with `yolo11s-pose.onnx` on this hardware — lower than most of the frame rates the flourish logic had been validated against.

## 1.9.132

### Fixed — a switched-off gesture still shadowed the raise

The per-gesture flags added in 1.9.131 acted only at the firing stage, which silenced the unwanted Home Assistant event but did not help recognition. The two-handed poses are tested **before** the single-arm raise and return early, so a frame misread as CROSS_ARMS is consumed there and SINGLE_UP is never evaluated at all — the raise was still lost, just silently.

Disabling a pose now skips its detection outright, so it cannot shadow a frame. That is what makes the switches genuinely improve raise recognition rather than only reducing noise.

## 1.9.131

### Added — per-gesture on/off switches

`gesture_snap_enable`, `gesture_hold_enable`, `gesture_double_snap_enable`, `gesture_cross_arms_enable`, `gesture_t_pose_enable`, `gesture_raise_both_enable` — all default on, all in the Configuration tab.

A gesture you never perform is not free. It fires Home Assistant events you did not ask for, and until v1.9.130 an unwanted one could suppress a real gesture through the shared cooldown: a field log showed 16 spurious CROSS_ARMS fires interleaved with SNAP attempts that never reached HA, from a user who was only ever testing SNAP.

Disabling DOUBLE_SNAP is handled properly — a second snap then registers as a plain SNAP rather than disappearing into the switched-off gesture.

## 1.9.130

### Fixed — a gesture could fire perfectly and never reach Home Assistant

The chime is emitted in the analyser, but the HA event only happens if **fusion** emits, and fusion applied its full `cooldown_s` (2 s) across *all* gesture types. So a single spurious CROSS_ARMS swallowed a real SNAP for two seconds — after the sound had already played. The symptom is "the chime fires at the right moment but nothing happens", with **no trace at all in `feedback.jsonl`**, because the `fired` row is only written after fusion emits. A correctly-detected gesture that was dropped downstream looked identical to one that was never detected.

- **Suppressed gestures are now recorded** as a `suppressed` row in `feedback.jsonl` with the reason, and logged at info rather than debug. This was the actual diagnostic gap — it cost a full round-trip to find.
- **The cross-gesture cooldown is now separate and short** (`cross_gesture_cooldown_s`, 0.5 s). Its only job is stopping one gesture's residual motion from immediately triggering a different one; the full `cooldown_s` still applies per gesture, so repeats are unchanged. Sustained poses already have their own re-arm latch, so they never needed a two-second veto over everything else.

## 1.9.129

### Fixed — SNAP arriving seconds late, or not at all

Firing required three things **in the same frame**: a high enough arm, a sweep that qualified *at that instant*, and the sustain window already elapsed. But a sweep only qualifies transiently — for the moments around the top of the arc — so whether those coincided was a matter of timing. Miss the overlap and the gesture arrived seconds later, when the numbers happened to line up again, or never. Meanwhile the chime had already fired during the ascent, which is the remaining "sound with no gesture" case.

- **The qualifying sweep is now latched for the duration of the raise.** The question it answers is "did this raise arrive on a flourish", which does not stop being true a moment later. The latch clears when the arm comes down, so one flourish cannot arm later fires.
- **The sweep replaces `snap_sustain_s` rather than being charged on top of it.** Filtering accidental raises is exactly the job that setting existed for, and the flourish test does it better; requiring both meant the sustain window had to elapse while the sweep still qualified.

Measured from the arm reaching the top of the arc, SNAP now fires between 0.07 s and 0.40 s **before** it — as soon as the sweep is unmistakable — at 5, 8, 10 and 15 fps, for both a brisk and a slow human lift. An arm at rest for 20 seconds, hanging or horizontal, still fires nothing and chimes not at all.

## 1.9.128

### Fixed — "sounds but no gestures": the sweep evidence expired before it was used

One bug produced both symptoms. Live feedback showed seven SNAP near-misses reporting `rise_delta` of **1.15–1.94** — a full arm sweep, unmistakable — alongside `sweep_climb: 0.000`, rejected as *"arm is up but did not sweep"*.

The sweep window was 0.8 s, and the climb was measured from the lowest point **in that window to the present moment**. Once the arm had been up for longer than the window, every sample in it was the top: the minimum equalled the current value and the climb collapsed to zero. The evidence of the lift evaporated in the time it took to confirm the raise. The chime, meanwhile, had already fired mid-ascent while the climb was briefly non-zero — hence sound with no gesture behind it.

- **The sweep is now measured over the ascent**: from the last time the arm was at its lowest, to the highest point reached after it. Holding the arm up afterwards no longer dilutes either the climb or the rate — the speed of the lift is what identifies a flourish, and that does not change because you kept your arm there.
- **`flourish_window_s` 0.80 → 2.50**, so the whole lift plus the confirm/sustain delay fits comfortably inside it.

Verified: a raise held for 1, 2, 3, 5 or 10 seconds now fires (all previously silent), a classic up-and-down flourish still fires, and each produces exactly one chime. An arm at rest for 20 seconds — hanging or horizontal — and a slow 9-second lift all remain silent.

## 1.9.127

### Fixed — joystick knob rendered off the pad after Reset

The dewarp pan slider ran `-30…30`, but a perfectly ordinary saved value like `-35` sits outside that. Reset reloads the page and re-renders the joystick from the restored settings, so the knob was placed at **−8.3%** — off the pad's left edge rather than at the aimed position.

- Pan is now `-90…90` and tilt `-30…90`, which covers real aiming into a 180° fisheye; both stay symmetric about their defaults, so the knob starts on the crosshair.
- The rendered knob position is clamped to the pad regardless, so no saved value can put it outside again.

Your own pan/tilt values are untouched — only the pad they are drawn on changed.

## 1.9.126

### Fixed — CROSS_ARMS stealing an obvious snap

Both two-handed gestures measured sideways position in **raw image x**, which silently assumes which way the person faces. A person facing the camera is mirrored: their right shoulder appears on the image's *left*, so their right wrist sits left of the midline **just by hanging at their side** — which the crossing test read as "crossed past the midline onto the other side". With both arms down, all four CROSS_ARMS conditions were satisfied, and because two-handed gestures are checked before SINGLE_UP, it stole the raise. Loosening the wrist-proximity ceiling in 1.9.124 made it easier to reach.

Sideways position is now projected onto the person's **own left-right axis** (right shoulder → left shoulder), so "toward my other shoulder" means the same thing whichever way they face. This fixes T_POSE identically — it had the same assumption, and would only have fired for someone facing one particular way.

Verified in both orientations: arms at rest no longer classify as CROSS_ARMS, genuinely crossed arms still do, and a raise is unaffected. A `_mirror()` regression test covers it.

### Changed

- The dewarp joystick's tilt range is now symmetric about its default (`-20…80`, midpoint 30), so the knob starts **on** the pad's crosshair. Previously the range was `-10…80` with midpoint 35, leaving the home position visibly off centre while pan sat exactly on it.

## 1.9.125

### Fixed — chimes with no gesture behind them

The sweep-onset chime added in 1.9.123 tested `sweep_rate` **alone**. Rate is climb divided by elapsed time, so a hand's worth of keypoint jitter between two adjacent frames divides a tiny climb by a tiny interval and produces a large rate. An arm resting near horizontal — an armrest, a lap — is where elevation is most sensitive to wrist noise, and it chimed repeatedly with no gesture behind it. Reproduced at 1280×640 with realistic per-keypoint noise: **11 phantom chimes in a single 15-second window** at 4 px, zero gestures. (At `yolo_imgsz: 320` upscaled to a 1280-wide frame, every inference pixel becomes four, so 2–4 px is conservative.)

- The chime now requires **both** climb and rate, exactly the bar the gesture gate uses. Phantom chimes across every resting posture and noise level: **0**.
- **The chime also fires when a gesture actually fires**, so a sound never happens without one. The sweep-onset chime is now only a head start on speaker latency; `chime_debounce_s` dedupes the pair. This also gives sustained poses (CROSS_ARMS, T_POSE, RAISE_BOTH) a chime, which they never had — they have no sweep.
- The condition is extracted as `_sweep_meets_flourish()` and covered by regression tests.

Detection itself was verified unaffected: a deliberate flourish fires 12/12 at three body sizes and four noise levels, and the gesture path (tracker → state machine → fusion → HA) tested clean end-to-end.

## 1.9.124

### Fixed

- **Crossing your arms more tightly stopped registering as CROSS_ARMS.** The two requirements pull against each other — crossing further past the body midline necessarily pushes the wrists apart — so with the proximity ceiling at 1.20 shoulder-widths only a narrow 0.80–1.20 band satisfied both. Raised to 2.00, which leaves the crossing minimum (the guard against hands-in-front-while-typing) doing the real work. Pre-existing, not introduced by the geometry rewrite.

All six gestures are verified working end-to-end after the flourish change: SNAP (one flourish), DOUBLE_SNAP (two inside the window), HOLD (flourish then keep the arm up), RAISE_BOTH, T_POSE and CROSS_ARMS. Note that the sustained poses — RAISE_BOTH, T_POSE, CROSS_ARMS — still require holding for `sustain_s`, which is right for a pose; only the single-arm gesture is flourish-triggered. Sweeping both arms up and straight back down therefore fires nothing.

## 1.9.123

### Changed — detect the flourish, not a held pose

Endora's gesture is a theatrical arm sweep with a chime, not a hand raised and held like a question in class. The detector had been built around the latter, and testing made the gap unambiguous: **a real flourish — arm sweeping up over ~0.6 s and straight back down — fired nothing at all**, at any frame rate. The `wrist_still` gate, added to reject reaching for a phone, rejects a flourish for exactly the same reason: a sweep never holds still.

SNAP now fires on the sweep itself, measured on two new values carried by every reading:

- **`sweep_climb`** — elevation gained since the arm's lowest point in the last 0.8 s. A full down-to-up sweep is ~2.0; one starting from an armrest, ~0.7.
- **`sweep_rate`** — that climb per second. A resting arm sits at ~0; a deliberate flourish runs 2–3.

This is a stronger false-positive filter than everything it replaces. Every static false positive we have fought — an arm on an armrest, draped over a backrest, holding a phone, a hand at the face — is geometrically similar to a raise but sweeps at a rate of essentially zero, so requiring the sweep excludes them **by construction** rather than by accumulated gates. `snap_require_still` is now off by default and `snap_sustain_s` drops to 0: the sweep is the evidence, so the gesture fires at the top of the arc.

- **Fires roughly 0.6 s sooner**, because it no longer waits for stillness plus a sustain window after the arm arrives. A raise arriving on a large, *progressive* sweep also skips the `state_confirm_s` wait — that delay exists to reject single-frame phantoms, and a coherent multi-frame climb is already proof of real movement. "Progressive" matters: a keypoint flickering between a good position and a garbage one produces a large climb and an enormous rate but is bimodal, never passing through intermediate elevations, and is still rejected.
- **The chime now fires at sweep onset**, while the arm is still travelling up, so an Echo's 1–2 s latency lands the sound as the flourish completes instead of well after it.
- `snap_require_flourish` can be turned off to restore the hold-style gesture.

Verified end-to-end at 1280×640: the flourish fires at 8, 10 and 15 fps, fires before the arm comes back down, and fires when the sweep starts from an armrest; a statically raised arm, an armrest rest, a backrest drape, a hand at the face and a ten-second slow lift all stay silent.

## 1.9.122

### Added — you can finally see what is actually running

Three separate debugging dead-ends this month traced to the same root cause: a stale value in one config file silently outranked a shipped default, and nothing anywhere reported it. Two fixes:

- **Startup log lists every gesture-critical setting with the file its value came from** — `default`, `options.json`, `settings.yaml`, `runtime_overrides.yaml` or an env var — with overridden ones marked `*`. If behaviour does not match the code, this is now the first line of the log to check.
- **"What's actually running?" panel on the debug page** (under the Save/Reset buttons). Same information live, with file-sourced values highlighted, plus a count of how many settings are overridden.

### Fixed

- **Reset now clears `settings.yaml` too.** Save writes to both files, so clearing only `runtime_overrides.yaml` left a lower-priority copy behind that still won for any key the HA Configuration tab does not supply. Reset strips only the keys the debug page manages — hand-authored keys and comments in that file are preserved, since we never wrote them.

## 1.9.121

### Changed — gesture geometry rewritten around two dimensionless numbers

The raise test used to ask "is the wrist higher *in the frame* than the shoulder, by a tuned margin?" — with separate margins for upright, reclined and unknown posture, a secondary forearm-vertical route, a wrist-near-nose veto, and a per-person scale factor to keep the frame-fraction margins meaningful at different distances. That accumulated complexity was the source of most of the false positives and misses in the last several releases. It is replaced by a direct statement of the gesture:

- **elevation** = `(shoulder_y − wrist_y) / |shoulder − wrist|` — the sine of the arm's angle above horizontal. 1.0 is straight up, 0.0 level, −1.0 hanging down.
- **extension** = `|shoulder − wrist| / (upper arm + forearm)` — 1.0 fully straight, 0.71 elbow at 90°.

"Arm extended straight up" is now exactly `elevation ≥ raise_elevation_min AND extension ≥ arm_extension_min`. Both are dimensionless, so distance, frame size and posture all cancel: **one threshold covers standing, sitting and lying on the couch.**

**Unit-mixing bug fixed.** All geometry is now computed in pixels. Landmarks are normalised per-axis, so on the 1280×640 frame the fisheye dewarp emits, every previous distance calculation mixed two different units — shoulder width was understated 2× against torso length. That is why the old body-scale estimate collapsed to its 0.5 floor whenever the hips were hidden, silently halving every margin.

**What this removes:** `arm_above_head_tolerance`, `arm_above_head_tolerance_reclined`, `body_upright_min`, `forearm_vertical_min`, `forearm_route_min_margin`, `wrist_head_exclude_dist`, `body_scale_reference`, `snap_forearm_min`, `raise_travel_min` and `wrist_still_max_travel` — ten thresholds, replaced by four. They remain in the settings schema as deprecated no-ops so existing configs keep loading; **saved values for them stop having any effect**, which is deliberate — a stale saved number can no longer silently override a shipped default.

T-pose, raise-both and cross-arms are expressed in the same terms (angles, and lengths measured against shoulder width) rather than frame fractions. A hand held at your face is now rejected because a bent arm scores low on extension, not by a special-case nose rule. An arm pointing nearly at the camera is *refused* rather than guessed at, via `min_arm_len_frac`.

The temporal gates are unchanged in spirit but now speak elevation: `rose_recently` fires when the arm was seen low **or** its elevation climbed by `rise_elevation_delta`, and stillness is measured against the arm's own length.

Verified end-to-end at 1280×640 across ten scenarios drawn from real feedback: deliberate raises fire identically at three body sizes and while reclined; raises starting from an armrest fire; and armrest-resting, backrest-draped, hand-at-face and arm-down postures all stay silent — reading DOWN rather than SINGLE_UP, so they no longer trigger the chime either.

## 1.9.120

### Fixed — "just a sound, no gesture" and phantom chimes

Feedback from 3–5 Aug isolated both halves of the same problem: **the chime was decoupled from the gesture logic**, so what you heard had almost nothing to do with what fired.

- **Rise evidence now accepts travel, not just a below-shoulder start.** Requiring the wrist to be seen at/below shoulder level (v1.9.119) made a raise that *begins* with the arm on an armrest or sofa backrest impossible to fire — the wrist never goes below the shoulder. Seven deliberate-sized raises (reference margin 0.13–0.20) were blocked back-to-back with `no_rise` and never fired. A raise now qualifies if the wrist has **risen** by `raise_travel_min` (0.08, body-scaled) within the window, wherever it started; the below-shoulder test is kept as a second route. Movement is the signal that actually separates a raise from a resting arm.
- **The chime no longer sounds for raises that cannot fire.** It used to sound on *any* entry into SINGLE_UP, so a resting arm reading as raised chimed repeatedly with no gesture behind it — 28 such blocked raises came from one seat in this batch alone. Since the chime is all you hear, that is audibly identical to a false positive. It now requires the same rise evidence the SNAP gate uses.
- `rise_travel` is logged on every reading and in `feedback.jsonl`, and the `no_rise` near-miss message now reports the measured travel instead of a generic string.

## 1.9.119

### Fixed — sofa-backrest/armrest arm postures re-arming SNAP

Post-camera-move feedback showed the storms gone (CROSS_ARMS latch: 2 isolated fires vs ~100 the day before) but two remaining leaks, both from arms resting *above* shoulder level:

- **Rise evidence is now posture-aware.** For an upright body, the wrist must have been seen AT or BELOW shoulder level to count as "the arm rose" — an arm draped over the sofa backrest hovers a few percent above the shoulder and bobbed in and out of the old near-shoulder tolerance band, re-arming the gate every few minutes. A genuine upright raise starts from the lap and is unaffected. Reclined bodies keep the lenient band (a lying person's resting wrist sits at shoulder height by geometry).
- **`forearm_route_min_margin` 0.06 → 0.10.** After the camera move the resting-arm fires drifted to 0.065–0.088 (reference units), straddling the old bar. Deliberate raises on record all clear 0.16+.

## 1.9.118

### Added

- **"Reset overrides" button on the debug page.** Deletes `/data/runtime_overrides.yaml` and reloads settings live (no restart needed), reverting every debug-page override to the HA Configuration tab values / shipped defaults. Overrides accumulate every slider ever touched and silently outrank new defaults shipped in later releases; clearing them previously required shell access into the add-on container.

## 1.9.117

### Fixed

- **Configuration-tab translations refreshed.** `snap_forearm_min` shows there as "Snap forearm threshold", and its description still recommended a value from two defaults ago. Stale entries updated, deprecated ones removed, and friendly names/descriptions added for all the new tuning settings.

## 1.9.116

### Fixed — resting-arm SNAP storm and CROSS_ARMS re-fire storm

The first feedback batch with `raise_margin` logging made two failure modes precisely visible:

- **Resting-arm/phone posture fired SNAP+HOLD all day through the forearm-vertical route.** Every flagged false fire had the wrist sitting AT shoulder level (`raise_margin` 0.000–0.049) while every confirmed deliberate raise cleared 0.17+. Two compounding causes, both fixed:
  - Reclining feet-toward-the-camera foreshortens the torso in image space, which collapsed the body-scale factor to 0.5–0.65 and shrank every margin. The size estimate now takes the **larger** of the torso-length and shoulder-width estimates — each collapses under a different projection (foreshortening kills torso, side-on kills shoulder width), so the max is robust to both.
  - The forearm-vertical route accepted a wrist merely *at* shoulder height. It now requires `forearm_route_min_margin` (0.06, body-scaled) of actual clearance — right between the false fires (≤0.049) and the real ones (≥0.17).
- **CROSS_ARMS fired ~100 times in 20 minutes of sitting with arms crossed.** Sustained-pose gestures (CROSS_ARMS / T_POSE / RAISE_BOTH) now fire **once per pose entry** and latch until the pose has been released for `sustained_rearm_s` (2 s). A one-frame keypoint dropout does not re-arm them.
- `snap_forearm_min` 0.06 → 0.05: with the margin and trajectory gates carrying false-positive rejection, a slightly bent elbow on a clearly raised arm shouldn't block SNAP — feedback showed a genuine attempt retried four times at dy 0.070 against a scale-adjusted 0.071 bar, never firing.

## 1.9.115

### Fixed — false-positive burst after 1.9.114

Analysis of the first post-1.9.114 feedback batch (7 false SNAPs in 12 minutes, all `rose=true still=true`, `scale_factor` 0.61–0.80) found three contributors:

- **`body_scale_reference` default 0.25 → 0.18.** The initial reference was calibrated to the test fixtures, not the real room: the resident's typical seated position read `scale_factor` 0.6–0.8, which silently tightened every tuned margin by 20–40% and made the whole system more trigger-happy than before the update. At 0.18 the typical position reads ≈1.0, restoring the tuned margins' intended meaning. **If your add-on configuration already shows `body_scale_reference: 0.25` saved, change it to 0.18 by hand** — saved options override the new default.
- **`snap_roll` formula fixed; `snap_roll_threshold` default 0.65 → 0.0 (route disabled).** The old formula divided (index.x − pinky.x) by its own absolute value, so every detected hand read exactly ±1.0. That made the "roll ≥ threshold counts as snap even with a non-vertical forearm" OR-route degenerate into *"any visible hand while the arm is up counts as a snap"* — harmless while full-frame hand detection almost never fired (~1 in 15), but armed on nearly every raise once the 1.9.114 wrist-crop made hand detection reliable. Roll is now a real orientation signal (|roll| ≈ 1 palm-to-camera, ≈ 0 edge-on, clamped ±1.5); re-enable the threshold route only after feedback data shows the new values separate real snaps from false fires.
- **`raise_margin` now logged.** Readings (and feedback.jsonl rows) include the achieved wrist-above-shoulder margin, closing the long-standing gap where reclined-threshold tuning had to be done blind against `forearm_dy`, a different quantity.

## 1.9.114

### Changed — gesture recognition overhaul (fewer false positives AND fewer missed gestures)

- **SNAP now requires trajectory evidence, not just a raised-arm pose.** Two new gates, both on by default:
  - `snap_require_rise` — the wrist must have been seen below shoulder level within the last few seconds. Blocks fires from poses that have simply existed for a while (hand propped against the head, a ghost detection with a permanently "raised" arm, sleeping postures) — a deliberate gesture always starts with an actual upward motion.
  - `snap_require_still` — the raised wrist must hold still briefly (`wrist_still_max_travel` over ~0.3 s). Blocks pass-through reaches (phone, blanket, glass), which keep moving through the raised zone; a deliberate raise stops and holds.
  Both can be disabled live from the add-on configuration if they ever block genuine gestures, and blocked fires are logged to `feedback.jsonl` as `near_miss` rows with reasons `no_rise` / `wrist_moving`.
- **All geometric thresholds now scale with each person's detected body size** (torso length; shoulder-width fallback when a blanket hides the hips). Previously every margin was a fixed fraction of frame height, so a person lying far from the camera was asked to clear margins sized for someone standing right in front of it — the main reason reclined gestures were missed while standing gestures misfired. Tune with `body_scale_reference` (the torso size the thresholds are calibrated at); the live per-person factor is shown as `scale:` on the debug overlay.
- **Default pose model is now `yolo11s-pose.onnx` at 480×480** (was nano at 320×320). The nano model's keypoints are too noisy for reliable gesture geometry in exactly the hard cases (dark clothes on a dark sofa, blanket, reclined, folded legs). NOTE: existing installs with `yolo_pose_model`/`yolo_imgsz` saved in their options or runtime overrides keep their saved values — change them in the add-on configuration or debug page to pick up the new defaults.
- **Hand detection now runs on a crop around the raised wrist** (`hand_crop_enable`, on by default) instead of the full frame, upscaled so MediaPipe can actually see the hand at couch distance. This makes `snap_roll` — the one signal that has never been wrong in reviewed feedback — available on far more fires than the historical ~1 in 15.
- Debug overlay shows the new signals: `scale:` and, while an arm is up, `rose:`/`still:`.

## 1.9.92

### Fixed

- **`yolo_pose_model` and `single_camera_mode` no longer disappear when saving the add-on configuration.** These settings existed in the code but were missing from `config.json`'s options/schema, so the Home Assistant Supervisor silently stripped them from `options.json` on every save. Both are now valid add-on options and persist through the UI.
- **CPU pinned at 100% in two-camera setups.** Each camera runs its own `CameraAnalyser` thread, and each one loaded an independent ONNX Runtime pose-model session with `num_threads=0` (= all CPU cores). With two cameras configured, that meant two sessions simultaneously competing for every core, oversubscribing the CPU and pinning it permanently — even on capable hardware. Analysers now split the machine's cores evenly between them (`cpu_count() // number_of_analysers`) instead of each claiming all of them.

If you only use one camera, you can now also set `single_camera_mode: true` (or point `rtsp_url_b` at the same URL as `rtsp_url_a`) to run a single analyser using the full core count.
