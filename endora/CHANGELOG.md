# Changelog

## 1.9.92

### Fixed

- **`yolo_pose_model` and `single_camera_mode` no longer disappear when saving the add-on configuration.** These settings existed in the code but were missing from `config.json`'s options/schema, so the Home Assistant Supervisor silently stripped them from `options.json` on every save. Both are now valid add-on options and persist through the UI.
- **CPU pinned at 100% in two-camera setups.** Each camera runs its own `CameraAnalyser` thread, and each one loaded an independent ONNX Runtime pose-model session with `num_threads=0` (= all CPU cores). With two cameras configured, that meant two sessions simultaneously competing for every core, oversubscribing the CPU and pinning it permanently — even on capable hardware. Analysers now split the machine's cores evenly between them (`cpu_count() // number_of_analysers`) instead of each claiming all of them.

If you only use one camera, you can now also set `single_camera_mode: true` (or point `rtsp_url_b` at the same URL as `rtsp_url_a`) to run a single analyser using the full core count.
