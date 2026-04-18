# Endora

Control Home Assistant with arm gestures — lights, TV, volume, anything.

Watches an RTSP camera stream for hand/arm gestures and fires HA events you can use in any automation. Runs as a **Home Assistant Add-on** (HA OS / Supervised) or as a **standalone Docker container**.

Designed for use from a couch with a fisheye camera mounted overhead or at eye level.

**Debug stream:** [http://homeassistant.local:8765/](http://homeassistant.local:8765/) — live camera overlay with skeleton, arm state, and gesture candidate. Enable with `debug_port: 8765`.

---

## Gestures

All gestures require the arm to be **raised above the head** — wrist above shoulder by at least `arm_above_head_tolerance`.

| HA event data | How to perform |
|---|---|
| `endora-snap` | Raise arm straight up — fires immediately when arm goes vertical |
| `endora-hold` | Raise arm and keep it up — fires `hold_duration_s` seconds after snap |
| `endora-double-snap` | Raise arm twice within `double_snap_window_s` seconds |
| `endora-thumbs-up` | Raise arm with thumb extended upward, other fingers curled |

**SNAP → HOLD sequence:** raising your arm always fires `endora-snap` first. If you keep your arm up, `endora-hold` fires after `hold_duration_s` (default 1.5s). Lower and raise again within `double_snap_window_s` (default 3s) for `endora-double-snap`.

---

## Installation — HA Add-on

### 1. Add the repository

**Settings → Add-ons → Add-on Store → ⋮ → Repositories**

Add: `https://github.com/silvermy/endora`

### 2. Install & Configure

Find **Endora** in the add-on store, install, then set options in the **Configuration** tab:

```yaml
rtsp_url_a: "rtsp://admin:password@192.168.1.100:554/stream1"
rtsp_url_b: "rtsp://admin:password@192.168.1.100:554/stream1"  # same as A for single-camera
ha_event_name: gesture_detected
debug_port: 8765
log_level: info
```

### 3. Start

Click **Start** → check the **Log** tab for stream connection. Open the debug stream at [http://homeassistant.local:8765/](http://homeassistant.local:8765/) to verify camera view and tune.

---

## Installation — Standalone Docker

### 1. Get a Long-Lived Access Token

**HA → Profile → Long-Lived Access Tokens → Create Token**

### 2. Configure

```bash
git clone https://github.com/silvermy/endora
cd endora
cp .env.example .env
```

Edit `.env`:

```env
RTSP_URL_A=rtsp://admin:password@192.168.1.100:554/stream1
RTSP_URL_B=rtsp://admin:password@192.168.1.100:554/stream1
HA_TOKEN=your_long_lived_token_here
HA_URL=http://homeassistant.local:8123/api
```

### 3. Run

```bash
docker compose up -d --build
docker compose logs -f endora
```

---

## HA Automation examples

Every gesture fires event type `gesture_detected`:

```json
{
  "gesture":        "endora-snap",
  "confidence":     0.91,
  "source_cameras": ["A"],
  "timestamp":      "2024-04-05T14:32:01.123456+00:00"
}
```

### Lights

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

- alias: "Endora — double-snap → scene"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: endora-double-snap
  action:
    service: scene.turn_on
    target:
      entity_id: scene.movie_mode

- alias: "Endora — thumbs-up → volume up"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: endora-thumbs-up
  action:
    service: media_player.volume_up
    target:
      entity_id: media_player.living_room_tv
```

---

## Debug stream

Navigate to [http://homeassistant.local:8765/](http://homeassistant.local:8765/) after enabling `debug_port: 8765`.

| Overlay element | Meaning |
|---|---|
| Green skeleton | Body detected and passing visibility filter |
| `NO POSE DETECTED` (red box) | Body not found — check lighting, camera angle |
| Cyan dot on wrist | ARM READY — gestures can fire |
| Orange dot on wrist | Warming up (not enough consecutive frames yet) |
| `forearm_dy` | Forearm verticality — should read 0.10+ for a clean snap |
| `hold_elapsed` | Seconds since snap fired — approaches `hold_duration_s` for HOLD |
| `cand:` | Current gesture candidate this frame |

---

## Fisheye dewarping

If using a fisheye camera (e.g. Reolink with fisheye mode), enable dewarping:

```yaml
dewarp_enable: true
dewarp_fov: 180        # total lens FOV
dewarp_tilt: 30        # + = camera looks down toward floor
dewarp_pan: -25        # + = right, - = left (centre yourself in frame)
dewarp_roll: 0         # level the horizon
dewarp_vfov: 50        # vertical FOV of output (narrower = more zoomed in)
dewarp_out_width: 1280
dewarp_out_height: 640
```

Set the Reolink camera to **Fisheye** mode (not Defisheye) in the app so the raw fisheye circle is streamed.

Tune `dewarp_pan` until you are roughly centred in the debug stream — this prevents MediaPipe from latching onto furniture instead of you.

---

## Tuning reference

| Problem | Fix |
|---|---|
| No skeleton / tracking furniture | Raise `pose_visibility_min` toward `0.5`; adjust `dewarp_pan` to centre yourself |
| ARM READY never triggers | Lower `arm_above_head_tolerance` (try `0.15`) |
| ARM READY triggers too easily | Raise `arm_above_head_tolerance` (try `0.25`) |
| SNAP not firing | Lower `snap_forearm_min` toward `0.07`; watch `forearm_dy` in debug |
| SNAP fires when arm is sideways | Raise `snap_forearm_min` toward `0.13` |
| HOLD fires too soon / too late | Adjust `hold_duration_s` |
| DOUBLE_SNAP window too tight | Raise `double_snap_window_s` to `4` or `5` |
| THUMBS_UP not detecting | Set `log_level: debug` and watch `thumbs_up=` in logs; lower `fist_curl_threshold` |
| Feet-up on couch fires false snaps | Raise `leg_raise_margin` toward `0.25` |
| Sitting cross-legged suppresses gestures | Lower `leg_raise_margin` toward `0.15` |
| High CPU on Pi | Set `pose_model_complexity: 0` |
| Stream dropping | Switch `rtsp_transport: udp` on wired LAN |
| 401 from HA | Check token; for add-on ensure `homeassistant_api: true` in config.json |

---

## Full configuration reference

| Option | Default | Description |
|---|---|---|
| `rtsp_url_a` | — | RTSP stream URL (required) |
| `rtsp_url_b` | same as A | Second camera; set equal to A for single-camera mode |
| `debug_port` | `8765` | Debug stream port; `0` = disabled |
| `ha_event_name` | `gesture_detected` | HA event type fired on gesture |
| `log_level` | `info` | `debug` / `info` / `warning` / `error` |
| `arm_above_head_tolerance` | `0.15` | Wrist must be this far above shoulder (frame fraction) |
| `body_upright_min` | `-0.15` | Hip-shoulder gap to confirm upright posture (negative OK for fisheye) |
| `leg_raise_margin` | `0.20` | Both ankles must exceed this above hip to suppress gestures (couch guard) |
| `snap_forearm_min` | `0.10` | Minimum forearm verticality for SNAP/HOLD (`forearm_dy` in debug) |
| `hold_duration_s` | `1.5` | Seconds after SNAP that arm must stay up to fire HOLD |
| `double_snap_window_s` | `3.0` | Seconds within which two snaps count as DOUBLE_SNAP |
| `fist_curl_threshold` | `0.75` | Fraction of fingers curled for THUMBS_UP detection |
| `cooldown_s` | `2.0` | Minimum seconds between any two gestures |
| `pose_visibility_min` | `0.45` | Min shoulder/hip visibility to accept a pose (filters furniture) |
| `frame_crop_bottom` | `0` | % of frame to crop from bottom (removes coffee table) |
| `mirror_camera` | `false` | Flip left/right gesture direction |
| `flip_image` | `false` | Rotate frame 180° for upside-down cameras |
| `low_light_enhance` | `false` | CLAHE contrast boost for low-light / IR scenes |
| `dewarp_enable` | `false` | Enable fisheye-to-perspective dewarping |
| `dewarp_fov` | `180.0` | Total fisheye lens FOV in degrees |
| `dewarp_tilt` | `30.0` | Tilt angle: + = down toward floor |
| `dewarp_pan` | `0.0` | Pan angle: + = right, - = left |
| `dewarp_roll` | `0.0` | Roll to level horizon |
| `dewarp_vfov` | `75.0` | Vertical FOV of dewarped output |
| `dewarp_out_width` | `1280` | Output frame width (wider = more horizontal FOV) |
| `dewarp_out_height` | `480` | Output frame height |
