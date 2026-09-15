# Endora

![Status: Beta](https://img.shields.io/badge/status-beta-yellow)

Control Home Assistant with body gestures — lights, TV, scenes, anything.

Watches an RTSP camera stream for pose-based gestures and fires HA events you can use in any automation. Runs as a **Home Assistant Add-on** (HA OS / Supervised) or as a **standalone Docker container**.

Designed for use from a couch with a fisheye camera.

**Debug stream:** open via **Settings → Add-ons → Endora → Open Web UI** (works through HA ingress, no IP needed). Direct access: the exact URL is printed in the add-on log on startup.

---

## Gestures

All gestures are detected from body pose alone — no hand detection required. This means they work reliably even when you're across the room from the camera.

| HA event data | How to perform |
|---|---|
| `endora-snap` | Raise one arm straight up — fires quickly |
| `endora-hold` | Raise one arm and keep it up for `hold_duration_s` (default 1.5s) |
| `endora-double-snap` | Raise one arm, lower it, raise again within `double_snap_window_s` (default 3s) |
| `endora-raise-both` | Raise both arms straight up and hold for `sustain_s` (default 0.5s) |
| `endora-t-pose` | Extend both arms horizontally to the sides, hold for `sustain_s` |
| `endora-cross-arms` | Cross arms in front of chest (each wrist near opposite shoulder), hold for `sustain_s` |

**Priority:** SNAP fires with a small delay to let competing gestures (RAISE_BOTH, T_POSE, CROSS_ARMS) supersede it — if you raise one arm straight up and hold, SNAP fires first, then HOLD fires. If you raise both, RAISE_BOTH fires instead.

---

## Installation — HA Add-on

### 1. Add the repository

**Settings → Add-ons → Add-on Store → ⋮ → Repositories**

Add: `https://github.com/silvermy/endora`

### 2. Install & Configure

Find **Endora** in the add-on store, install, then set options in the **Configuration** tab:

```yaml
rtsp_url_a: "rtsp://admin:password@192.168.1.100:554/stream1"
rtsp_url_b: "rtsp://admin:password@192.168.1.100:554/stream1"
ha_event_name: gesture_detected
debug_port: 8765
log_level: info
```

### 3. Start

Click **Start** → check the **Log** tab for stream connection. Open the debug UI via **Open Web UI** (top of the add-on page) or find the exact direct URL in the log (`Debug stream: http://...`).

### 4. Sidebar (optional)

**Settings → Add-ons → Endora → Info tab → "Show in sidebar" toggle** — adds
an entry that opens the debug UI *inside* Home Assistant, via ingress. Home
Assistant renders sidebar entries in-page and offers no way to make one open
an external URL.

To get a separate browser tab instead — and the only option for a standalone
host, which has no ingress at all — see
**[docs/homeassistant/](docs/homeassistant/README.md)**: a small custom panel
that opens the console in its own tab and returns HA to your dashboard.

---

## Installation — Standalone container (Jetson / any Docker host)

The add-on above runs on the Home Assistant machine itself. The same
application also runs as a plain container on a separate box — the reason to
do that being a Jetson, where pose inference moves to the GPU instead of the
CPU. It reports gestures to Home Assistant over the network with a
Long-Lived Access Token rather than through the Supervisor.

Everything else — settings, the debug page, gesture logic — is identical.
Which deployment you are looking at is the first line of the log:

```
Endora v1.9.145 starting — standalone container, accelerators: TensorrtExecutionProvider
```

### 1. Prepare the host

For a Jetson Orin: JetPack 6.x, with the QSPI bootloader firmware updated to
match (a board still on JetPack 5 firmware will not boot a JP6 image, and the
67-TOPS `MAXN_SUPER` profile needs firmware 36.4.3+). Install
`nvidia-container-toolkit`.

For any other Docker host, use `docker-compose.yml` instead of
`docker-compose.jetson.yml` below — same application, CPU inference.

### 2. Create a Long-Lived Access Token

**HA → Profile (bottom-left) → Long-Lived Access Tokens → Create Token**

### 3. Configure

```bash
cp .env.example .env
```

Set `RTSP_URL_A` / `RTSP_URL_B`, `HA_TOKEN`, and — unlike the add-on —
`HA_URL` pointing at the **Home Assistant machine's address**, not
`localhost`. This container is on a different host.

Settings beyond those live in `/data/settings.yaml` on the container's
volume (there is no Configuration tab here); the full list is the same
[configuration reference](#full-configuration-reference) below.

### 4. Build and start

```bash
make up
```

The first build downloads a multi-gigabyte JetPack base image and is slow.

A `Makefile` wraps the Compose commands for this deployment (it is Jetson-only;
the add-on is managed by the Supervisor and ignores it). `make` on its own
lists everything. The ones you will use:

| | |
|---|---|
| `make rebuild` | `git pull`, rebuild, restart — the normal update path |
| `make logs` | follow the log |
| `make restart` | restart without rebuilding, to pick up `settings.yaml` edits |
| `make gpu` | confirm inference landed on the GPU |
| `make debug` | print the debug UI URL |
| `make disk` | where the space went |
| `make prune` | reclaim space from old images and build cache |

The disk targets matter more than they sound: a JetPack image is several GB
and every rebuild leaves the previous layers behind, which fills a microSD
card fast. None of them touch Docker volumes — `endora_data` holds your
settings, feedback log and TensorRT cache, so `make prune` is deliberately
never `docker system prune --volumes`. The destructive targets refuse to run
on anything that is not a Tegra board.

### 5. Confirm it is on the GPU

```bash
make gpu
```

Expect `provider=TensorrtExecutionProvider`. If it says
`provider=CPUExecutionProvider`, the GPU wheel or the nvidia runtime is not
in play and you are getting no benefit from the board — see
`yolo_execution_provider` in the configuration reference.

The first inference after a build stalls for several minutes while TensorRT
compiles an engine for the model. It is cached in the `/data` volume, so
later restarts are immediate — do not delete that volume casually.

### 6. Open the debug UI

There is no sidebar entry and no **Open Web UI** button here — those belong
to the add-on. Go to the port directly:

```
http://<jetson-ip>:8765/
```

The exact URL is printed at startup (`Debug stream: http://...`). It is the
same page as the add-on's, with the same live view, sliders and Save button;
Save writes `/data/runtime_overrides.yaml` on the container's volume exactly
as it does under Home Assistant.

`DEBUG_PORT` in `.env` controls the port and defaults to 8765 for this
deployment. (The add-on ships it off by default, since add-on users turn it
on in the Configuration tab — which does not exist here.) Set it to `0` to
disable the page.

**Leave it on.** This server also delivers the chime audio here, so `0`
silences the sound too. It is cheap to leave running: the overlay is drawn
only while a browser is actually requesting frames, so an unwatched debug
page costs nothing beyond the log ring buffer and the gesture captures —
both of which you want running unwatched, because their value is having the
record from *before* you noticed something was wrong.

For a sidebar entry that opens this page in its own tab, see
**[docs/homeassistant/](docs/homeassistant/README.md)**.

---

## HA Automation examples

```yaml
- alias: "Endora — snap → lights toggle"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: endora-snap
  action:
    service: light.toggle
    target:
      area_id: living_room

- alias: "Endora — hold → lights off"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: endora-hold
  action:
    service: light.turn_off
    target:
      area_id: living_room

- alias: "Endora — double-snap → movie scene"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: endora-double-snap
  action:
    service: scene.turn_on
    target:
      entity_id: scene.movie_mode

- alias: "Endora — raise both arms → max brightness"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: endora-raise-both
  action:
    service: light.turn_on
    target:
      area_id: living_room
    data:
      brightness: 255

- alias: "Endora — T-pose → pause all media"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: endora-t-pose
  action:
    service: media_player.media_pause
    target:
      area_id: living_room

- alias: "Endora — cross arms → stop everything"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: endora-cross-arms
  action:
    service: script.all_off
```

---

## Debug stream

| Overlay element | Meaning |
|---|---|
| Green skeleton | Body detected |
| `NO POSE DETECTED` (red) | Body not found — adjust camera or lighting |
| Yellow dot on wrist | Arm is classified as raised |
| `state:` | Current arm classification (DOWN / SINGLE_UP / BOTH_UP / T_POSE / CROSS_ARMS) |
| `forearm_dy` | Forearm verticality — should read 0.10+ for a clean SNAP |
| `upright` | Whether body is detected as upright |

---

## Chime — audio feedback on arm raise

Endora can play a short sound on any HA-integrated speaker the moment it detects an arm moving up — before the gesture fires. This gives you instant confirmation that Endora saw you, even if the gesture takes another second to complete.

Works with **any speaker HA knows about**: Sonos, Chromecast, Echo, HomePod, Spotify Connect, DLNA, etc. Uses HA's `media_player.play_media` with `announce: true`, so it overlays on whatever is currently playing (TV, music) and resumes automatically.

How the speaker gets the audio depends on the deployment. The **add-on**
copies the clip into HA's `/media` folder and hands the speaker a
`media-source://` URL. A **standalone container** has no access to that
folder — it serves the clip from its own debug server instead, so the chime
there requires `DEBUG_PORT` to be set. With the debug page disabled the log
says so rather than going quiet.

### Setup

1. Find your speaker's entity ID in HA → **Settings → Devices & Services → Entities**, filter by `media_player`.

2. Add to the add-on config (or `settings.yaml` on a standalone host):

```yaml
chime_enable: true
chime_entity_id: "media_player.living_room"
chime_volume: 40        # 0–100, absolute — not a fraction of current volume
chime_debounce_s: 6.0   # must exceed the clip length; the bundled clip is 4.0 s
```

3. Restart. The startup line tells you which route it took — **add-on**:

   ```
   Chime: installed chime.wav → /media/endora_chime_5192bb79.wav
   Chime ready — entity=media_player.living_room
   ```

   **Standalone**:

   ```
   Chime: serving from the debug server at http://10.0.0.141:8765/chime.wav?v=5192bb79
   ```

   If a standalone host reports the `/media` line instead, it has written the
   clip into its own container and handed HA a URL only the HA machine can
   serve — that was a real bug, fixed in v1.9.161.

The filename carries a hash of the audio's own bytes. A fixed URL let HA's
media proxy and the speaker both keep playing a replaced clip from cache
indefinitely; a content-derived one changes exactly when the audio does.

> On a standalone host the chime is delivered by the debug server, so
> **`debug_port: 0` silences the chime as well as the debug page.**

---

## Fisheye dewarping

If using a fisheye camera (e.g. Reolink in Fisheye mode):

```yaml
dewarp_enable: true
dewarp_fov: 180        # total lens FOV
dewarp_tilt: 30        # + = down toward floor
dewarp_pan: -25        # + = right, - = left
dewarp_roll: 0
dewarp_vfov: 50
dewarp_out_width: 1280
dewarp_out_height: 640
```

Tune `dewarp_pan` until you are roughly centred in the debug stream.

---

## Tuning reference

> For the full feedback-driven tuning workflow (how to gather labeled feedback,
> read `feedback.jsonl`, decide which threshold to change, and lock in fixes with
> regression captures), see **[docs/TRAINING.md](docs/TRAINING.md)**.

| Problem | Fix |
|---|---|
| No skeleton / tracking furniture | Raise `pose_visibility_min` toward `0.5`; centre yourself with `dewarp_pan` |
| Arm raise not detected | Lower `raise_elevation_min` toward `0.60` |
| Arm raise triggers too easily | Raise `raise_elevation_min` toward `0.80` |
| SNAP not firing at all | Check the pose sample rate first — see the row below. Then lower `flourish_min_climb` / `flourish_min_rate`, or set `snap_require_flourish: false` to fall back to a held raise |
| SNAP fires only occasionally, and the debug overlay shows an implausible `sweep` rate (10+/s) | The model is sampling too slowly to see the sweep: the log's `pose N sample/s` is the number that matters, not the camera fps. Raise it (faster model, lower `yolo_imgsz`, better hardware) rather than loosening thresholds |
| CPU pegged whenever someone is in the room | `yolo_max_skip_active` is `1` — every frame while a person is tracked. Correct on a GPU; on a Pi try `3` |
| Two events per gesture (a second chime ~1.5 s later) | That is HOLD. Set `gesture_hold_enable: false`, or adjust `hold_duration_s` |
| HOLD fires too soon / too late | Adjust `hold_duration_s` |
| T-pose fires when raising both arms | Raise `sustain_s` toward `1.0` |
| Cross-arms not detecting | Wrists need to be quite close to opposite shoulders; pose must be clean |
| High CPU | Switch to `yolo_pose_model: yolo11n-pose.onnx` (nano, ~25 ms/frame); lower `yolo_imgsz` toward `320` |
| Pose drops on unusual poses / distant people | Raise `yolo_imgsz` toward `640` (or try `480` as a middle ground), or switch `yolo_pose_model` to `yolo11s-pose.onnx` (small, more accurate) |
| `yolo_imgsz` change has no effect | Only `320`/`480`/`640` are bundled — any other value silently falls back to `640` on a Pi (no runtime ONNX export on aarch64) |
| SNAP fires with nobody in frame (framed pictures, mirrors, TV) | Raise `bg_subtract_min_foreground` toward `0.20`; check `/captures` on the debug page to confirm the ghost source |
| Real gesture rejected as a "ghost" | Lower `bg_subtract_min_foreground` toward `0.05`, or disable `bg_subtract_enable` |
| SNAP fires from resting a hand near your own face (glasses, phone, scratching) | Raise `arm_extension_min` toward `0.85` — a hand at the face bends the elbow, so it fails on straightness |
| Genuine raise near your head gets rejected | Lower `arm_extension_min` toward `0.75` |

> Settings marked **Deprecated** in the configuration reference are read by
> nothing: the v1.9.121 geometry rewrite replaced the forearm-angle and
> head-distance routes with `elevation` and `extension`. Changing
> `arm_above_head_tolerance`, `snap_forearm_min`, `forearm_vertical_min` or
> `wrist_head_exclude_dist` has no effect at all, and this table used to
> recommend three of them.

---

## Full configuration reference

| Option | Default | Description |
|---|---|---|
| `yolo_pose_model` | `yolo11s-pose.onnx` | Pose model: `yolo11n-pose.onnx` (fast/nano) or `yolo11s-pose.onnx` (accurate/small) |
| `yolo_imgsz` | `480` | Inference resolution — `320`/`480`/`640` are bundled; other values fall back to `640` on a Pi |
| `rtsp_url_a` | — | RTSP stream URL (required) |
| `rtsp_url_b` | same as A | Second camera; set equal to A for single-camera mode |
| `debug_port` | `8765` | Debug stream port |
| `ha_event_name` | `gesture_detected` | HA event type fired on gesture |
| `log_level` | `info` | `debug` / `info` / `warning` / `error` |
| `raise_elevation_min` | `0.70` | How far above horizontal an arm must point to count as raised (1.0 = straight up). Same value standing, sitting or lying down |
| `arm_extension_min` | `0.80` | How straight the arm must be (1.0 = fully extended, 0.71 = elbow at 90°) |
| `body_upright_min` | `-0.15` | Hip-shoulder gap to confirm upright (negative OK for fisheye) |
| `pose_visibility_min` | `0.45` | Min landmark visibility to accept a pose (filters furniture) |
| `wrist_head_exclude_dist` | `0.09` | Reject a raised wrist within this distance of the nose keypoint (filters resting a hand against your own face) |
| `snap_elevation_min` | `0.70` | Minimum arm elevation for the gesture to fire |
| `snap_require_flourish` | `true` | Fire on the arm sweep (Endora's flourish), not on a held raise |
| `flourish_min_climb` | `0.60` | How much elevation the sweep must gain |
| `flourish_min_rate` | `0.80` | How fast the sweep must be (elevation per second) |
| `snap_sustain_s` | `0.20` | Seconds the arm must stay up before SNAP fires |
| `hold_duration_s` | `1.5` | Seconds after SNAP that arm must stay up to fire HOLD |
| `double_snap_window_s` | `3.0` | Seconds within which two snaps count as DOUBLE_SNAP |
| `sustain_s` | `0.5` | Seconds held for CROSS_ARMS / T_POSE / RAISE_BOTH |
| `cooldown_s` | `2.0` | Minimum seconds between any two gestures |
| `bg_subtract_enable` | `true` | Reject detections whose wrist never moves against the learned background (filters framed pictures/mirrors/TV mis-read as a raised arm) |
| `bg_subtract_min_foreground` | `0.12` | Min fraction of a wrist's check-patch that must be "moving" pixels to count as live |
| `frame_crop_bottom` | `0` | % of frame to crop from bottom |
| `flip_image` | `false` | Rotate frame 180° |
| `mirror_camera` | `false` | Reserved for future use |
| `low_light_enhance` | `false` | CLAHE contrast boost |
| `chime_enable` | `false` | Play a sound on arm-up detection |
| `chime_entity_id` | `""` | HA `media_player` entity to play chime on (e.g. `media_player.living_room`) |
| `chime_volume` | `40` | Chime volume 0–100 (100 = speaker's current max) |
| `chime_debounce_s` | `4.0` | Min seconds between chimes |
| `dewarp_*` | — | Fisheye dewarping parameters |
