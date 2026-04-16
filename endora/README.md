# Endora

Wave your hand to control Home Assistant — lights, TV, volume, anything.

Watches an RTSP camera stream for hand gestures and fires HA events you can use in any automation. Runs as a **Home Assistant Add-on** (HA OS / Supervised) or as a **standalone Docker container**.

**Debug page:** [http://homeassistant.local:8765/](http://homeassistant.local:8765/) — live camera overlay, gesture state, and tuning sliders. Enable by setting `debug_port: 8765` in Configuration.

---

## Gestures

All gestures require the arm to be **fully extended above head** — elbow above shoulder, wrist above elbow.

| HA event data | Motion | Hand shape |
|---|---|---|
| `endora-snap` | Wrist snap — flip palm from facing forward to backward (or vice versa) | Open hand |
| `endora-fist` | Raise arm, close fist | Closed fist |
| `endora-wave-left` | Raise arm, sweep wrist to the left | Open hand |
| `endora-wave-right` | Raise arm, sweep wrist to the right | Open hand |

---

## Installation — HA Add-on

### 1. Add the repository

**Settings → Add-ons → Add-on Store → ⋮ → Repositories**

Add: `https://github.com/silvermy/endora`

Or as a local add-on:

```bash
cd /addons
git clone https://github.com/silvermy/endora endora
```

Then: **Settings → Add-ons → Add-on Store → ⋮ → Check for updates**

### 2. Configure

In the add-on **Configuration** tab:

```yaml
rtsp_url_a: "rtsp://admin:password@192.168.1.100:554/stream1"
ha_event_name: gesture_detected
debug_port: 8765
log_level: info
```

Set `rtsp_url_b` to the same value as `rtsp_url_a` for single-camera mode.

### 3. Start

Click **Start** → check the **Log** tab for the stream connecting.

Open the debug page at [http://homeassistant.local:8765/](http://homeassistant.local:8765/) to verify the camera view and tune settings.

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

### Lights on/off

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

- alias: "Endora — fist → lights off"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: endora-fist
  action:
    service: light.turn_off
    target:
      area_id: living_room
```

### Volume

```yaml
- alias: "Endora — wave left → volume up"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: endora-wave-left
  action:
    service: media_player.volume_up
    target:
      entity_id: media_player.living_room_tv

- alias: "Endora — wave right → volume down"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: endora-wave-right
  action:
    service: media_player.volume_down
    target:
      entity_id: media_player.living_room_tv
```

---

## Debug page

Navigate to [http://homeassistant.local:8765/](http://homeassistant.local:8765/) after enabling `debug_port: 8765`.

The page shows a live camera view with skeleton overlay and a right-side panel with live tuning sliders. Changes take effect immediately — no restart needed. Use **Save to settings.yaml** to persist them.

| Overlay | Meaning |
|---|---|
| Green skeleton | Body detected |
| `NO POSE DETECTED` (red) | Body not found — check lighting, camera angle |
| Cyan dot on wrist | ARM READY — gestures can fire |
| Orange dot on wrist | Warming up (not enough consecutive frames yet) |
| Bottom-left panel | Arm state, twist swing value, palm orientation, current candidate |
| Green banner | Gesture fired |

### Tuning

1. Open debug page and raise your arm — look for the cyan wrist dot (ARM READY).
2. If arm isn't triggering: lower **Arm raise margin** slider.
3. If arm triggers when it shouldn't: raise **Arm raise margin** slider.
4. For snap gesture: watch `twist=` value while snapping. If it doesn't exceed **Snap sensitivity**, lower that slider.
5. Use **CLAHE enhance** toggle if detection is poor in low light or dark clothing against dark background.

---

## Camera placement

An **eye-level mount at 4–4.5 ft** (e.g., on a mantel shelf or TV console) gives the best results. MediaPipe Pose was trained on frontal views — overhead mounting degrades accuracy significantly.

Point the camera at your typical sitting/standing position. With a fisheye lens the field of view is wide enough to cover a couch + a few feet either side without issue.

---

## Tuning reference

| Problem | Fix |
|---|---|
| No skeleton | Check lighting; try `pose_model_complexity: 2`; lower `pose_min_detection_confidence` to `0.3` |
| ARM READY triggers too easily | Raise **Arm raise margin** (`arm_above_head_tolerance`) |
| ARM READY never triggers | Lower **Arm raise margin** |
| Snap not detecting | Lower **Snap sensitivity** (`palm_twist_threshold`); watch `twist=` on debug page |
| False fist triggers | Raise `fist_curl_threshold` toward `0.9` |
| High CPU on Pi | Set `pose_model_complexity: 0`; `frame_width: 320` |
| Stream dropping | Switch `rtsp_transport: udp` on wired LAN |
| 401 from HA | Check token; for add-on ensure `homeassistant_api: true` |

---

## Configuration reference

| Option | Default | Description |
|---|---|---|
| `rtsp_url_a` | — | RTSP URL (required) |
| `rtsp_url_b` | same as A | Second camera; set equal to A for single-camera mode |
| `debug_port` | `8765` | Debug web page port; `0` = disabled |
| `ha_event_name` | `gesture_detected` | HA event type |
| `log_level` | `info` | `debug` / `info` / `warning` / `error` |
| `arm_above_head_tolerance` | `0.05` | How far above shoulder the wrist must be (frame fraction) |
| `wave_velocity_threshold_px` | `20` | Minimum wrist pixel speed for wave-left / wave-right |
| `palm_twist_threshold` | `0.40` | 2D knuckle-line swing required for snap gesture |
| `fist_curl_threshold` | `0.75` | Curl fraction to count as fist (0–1) |
| `cooldown_s` | `2.0` | Min seconds between gestures |
| `pose_visibility_min` | `0.35` | Shoulder visibility threshold (filters furniture) |
| `frame_crop_bottom` | `0` | % of frame to crop from bottom |
| `pose_model_complexity` | `1` | 0=fastest, 1=balanced, 2=accurate |
| `low_light_enhance` | `false` | CLAHE contrast boost before inference |
| `mirror_camera` | `true` | Flip left/right for forward-facing camera |
| `dewarp_enable` | `false` | Enable fisheye dewarping |
| `dewarp_tilt` | `30.0` | Camera tilt angle (° down) |
| `dewarp_pan` | `0.0` | Camera pan angle (° right) |
| `dewarp_vfov` | `75.0` | Vertical FOV of output (°) |
