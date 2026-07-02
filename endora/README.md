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

**Settings → Add-ons → Endora → Info tab → "Show in sidebar" toggle** — adds a shortcut that opens the debug stream in a new tab.

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

### Setup

1. Find your speaker's entity ID in HA → **Settings → Devices & Services → Entities**, filter by `media_player`.

2. Add to add-on config:

```yaml
chime_enable: true
chime_entity_id: "media_player.living_room"
chime_volume: 40        # 0–100
chime_debounce_s: 4.0   # minimum seconds between chimes
```

3. Restart the add-on. On startup you should see:
   ```
   Chime: installed chime.wav → /media/endora_chime.wav
   Chime ready — entity=media_player.living_room
   ```

The chime sound is bundled with the add-on and automatically installed to HA's `/media` folder on startup.

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
| Arm raise not detected | Lower `arm_above_head_tolerance` toward `0.10` |
| Arm raise triggers too easily | Raise `arm_above_head_tolerance` toward `0.20` |
| SNAP not firing | Lower `snap_forearm_min` toward `0.05`; watch `forearm_dy` in debug |
| HOLD fires too soon / too late | Adjust `hold_duration_s` |
| T-pose fires when raising both arms | Raise `sustain_s` toward `1.0` |
| Cross-arms not detecting | Wrists need to be quite close to opposite shoulders; pose must be clean |
| High CPU | Switch to `yolo_pose_model: yolo11n-pose.onnx` (nano, ~25 ms/frame) |
| Pose drops on unusual poses | Switch to `yolo_pose_model: yolo11s-pose.onnx` (small, ~50 ms/frame, more accurate) |

---

## Full configuration reference

| Option | Default | Description |
|---|---|---|
| `yolo_pose_model` | `yolo11s-pose.onnx` | Pose model: `yolo11n-pose.onnx` (fast/nano) or `yolo11s-pose.onnx` (accurate/small) |
| `rtsp_url_a` | — | RTSP stream URL (required) |
| `rtsp_url_b` | same as A | Second camera; set equal to A for single-camera mode |
| `debug_port` | `8765` | Debug stream port |
| `ha_event_name` | `gesture_detected` | HA event type fired on gesture |
| `log_level` | `info` | `debug` / `info` / `warning` / `error` |
| `arm_above_head_tolerance` | `0.15` | Wrist must be this far above shoulder (frame fraction) |
| `body_upright_min` | `-0.15` | Hip-shoulder gap to confirm upright (negative OK for fisheye) |
| `pose_visibility_min` | `0.45` | Min landmark visibility to accept a pose (filters furniture) |
| `snap_forearm_min` | `0.06` | Minimum forearm verticality for SNAP/HOLD |
| `hold_duration_s` | `1.5` | Seconds after SNAP that arm must stay up to fire HOLD |
| `double_snap_window_s` | `3.0` | Seconds within which two snaps count as DOUBLE_SNAP |
| `sustain_s` | `0.5` | Seconds held for CROSS_ARMS / T_POSE / RAISE_BOTH |
| `cooldown_s` | `2.0` | Minimum seconds between any two gestures |
| `frame_crop_bottom` | `0` | % of frame to crop from bottom |
| `flip_image` | `false` | Rotate frame 180° |
| `mirror_camera` | `false` | Reserved for future use |
| `low_light_enhance` | `false` | CLAHE contrast boost |
| `chime_enable` | `false` | Play a sound on arm-up detection |
| `chime_entity_id` | `""` | HA `media_player` entity to play chime on (e.g. `media_player.living_room`) |
| `chime_volume` | `40` | Chime volume 0–100 (100 = speaker's current max) |
| `chime_debounce_s` | `4.0` | Min seconds between chimes |
| `dewarp_*` | — | Fisheye dewarping parameters |
