# Changelog

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
