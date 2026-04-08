# Gesture Cam — Home Assistant Add-on

Watches two RTSP camera streams for five hand gestures and fires
Home Assistant events you can use in any automation.

---

## Gestures

| Gesture | Movement | Hand |
|---|---|---|
| `wave_left` | Wrist sweeps left | Open palm |
| `wave_right` | Wrist sweeps right | Open palm |
| `palm_up` | Wrist moves upward | Flat palm |
| `palm_down` | Wrist moves downward | Flat palm |
| `fist_gesture` | Any direction | Closed fist |

All gestures require the arm to be raised above shoulder height first,
eliminating accidental triggers from normal activity.

---

## Installation

### 1. Add the repository to Home Assistant

Settings → Add-ons → Add-on Store → ⋮ → Repositories  
Add: `https://github.com/your-repo/gesture-cam-addon`

Or install as a **local add-on**:

```bash
# On your HA host (SSH add-on or terminal)
cd /addons
git clone https://github.com/your-repo/gesture-cam-addon gesture_cam
```

Then: Settings → Add-ons → Add-on Store → ⋮ → Check for updates

### 2. Configure

In the add-on Configuration tab, set your camera URLs:

```yaml
rtsp_url_a: "rtsp://admin:password@192.168.1.100:554/stream1"
rtsp_url_b: "rtsp://admin:password@192.168.1.101:554/stream1"
rtsp_transport: tcp
ha_event_name: gesture_detected
log_level: info
```

### 3. Start

Click **Start** in the add-on Info tab.  Check the Log tab — you should
see both streams connect and the gesture engine start.

---

## Camera placement

```
Corner A ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━● Corner B
          ↘ 120°+ FOV            FOV ↙
           ┌────────────────────────────┐
           │  kitchen / dining / lounge │
           └────────────────────────────┘
```

- Mount each camera **in the top corner**, height 2.0–2.5 m
- Use a wide-angle lens (90–170° FOV) so one camera covers the full length
- Overlapping coverage in the middle is fine — the fusion layer
  gives higher confidence when both cameras agree
- Good even lighting matters more than camera resolution

### Compatible cameras

Any camera with an RTSP stream works:

| Type | Notes |
|---|---|
| Reolink, Dahua, Hikvision | Standard `rtsp://ip:554/...` URL |
| Frigate NVR | `rtsp://frigate-ip:8554/camera_name` |
| go2rtc | Acts as a relay — normalises any source to RTSP |
| UniFi Protect | Requires Unifi Protect integration RTSP URL |

---

## Home Assistant automation examples

### The event payload

Every gesture fires `gesture_detected` with this data:

```json
{
  "gesture": "wave_left",
  "confidence": 0.91,
  "source_cameras": ["A", "B"],
  "timestamp": "2024-04-05T14:32:01.123456+00:00"
}
```

### Lights — wave left = off, wave right = on

```yaml
# automations.yaml
- alias: "Gesture — wave right → living room on"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: wave_right
  action:
    service: light.turn_on
    target:
      area_id: living_room
    data:
      brightness_pct: 80

- alias: "Gesture — wave left → living room off"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: wave_left
  action:
    service: light.turn_off
    target:
      area_id: living_room
```

### Volume — palm up / down

```yaml
- alias: "Gesture — palm up → volume up"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: palm_up
  action:
    service: media_player.volume_up
    target:
      entity_id: media_player.living_room_tv

- alias: "Gesture — palm down → volume down"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: palm_down
  action:
    service: media_player.volume_down
    target:
      entity_id: media_player.living_room_tv
```

### Fist — toggle TV

```yaml
- alias: "Gesture — fist → TV toggle"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: fist_gesture
  action:
    service: media_player.toggle
    target:
      entity_id: media_player.living_room_tv
```

### Template sensor (last gesture)

```yaml
# configuration.yaml
template:
  - sensor:
      - name: "Last Gesture"
        unique_id: last_gesture
        state: "none"
        attributes:
          confidence: 0
          cameras: []
        trigger:
          - platform: event
            event_type: gesture_detected
        action:
          - variables:
              d: "{{ trigger.event.data }}"
          - service: template.reload   # no-op; just to satisfy action block
        state: "{{ trigger.event.data.gesture }}"
```

---

## Local development / testing (without HA)

```bash
# Build the image locally (amd64)
docker build \
  --build-arg BUILD_FROM=ghcr.io/home-assistant/amd64-base-python:3.11-alpine3.19 \
  -t gesture_cam_dev .

# Run with env-var camera URLs (PrintBackend auto-selected — no SUPERVISOR_TOKEN)
docker run --rm \
  -e RTSP_URL_A="rtsp://admin:pass@192.168.1.100:554/stream1" \
  -e RTSP_URL_B="rtsp://admin:pass@192.168.1.101:554/stream1" \
  -e LOG_LEVEL=debug \
  gesture_cam_dev

# Or mount a local settings file
docker run --rm \
  -v $(pwd)/config/settings.yaml:/data/settings.yaml:ro \
  gesture_cam_dev
```

---

## Configuration reference

| Option | Default | Description |
|---|---|---|
| `rtsp_url_a` | — | RTSP URL for camera A (required) |
| `rtsp_url_b` | — | RTSP URL for camera B (required) |
| `rtsp_transport` | `tcp` | `tcp` or `udp` |
| `rtsp_reconnect_delay_s` | `5.0` | Seconds before reconnect on stream loss |
| `frame_width` | `640` | Resize incoming frames to this width |
| `frame_height` | `480` | Resize incoming frames to this height |
| `pose_model_complexity` | `0` | MediaPipe model: 0=lite, 1=full, 2=heavy |
| `arm_raised_wrist_above_shoulder_frac` | `0.10` | Arm-raise sensitivity (increase to reduce false triggers) |
| `wave_velocity_threshold_px` | `18.0` | px/frame needed for a wave |
| `wave_sustain_frames` | `3` | Consistent frames required before emitting |
| `vertical_velocity_threshold_px` | `15.0` | px/frame for up/down moves |
| `vertical_sustain_frames` | `3` | Consistent frames for vertical moves |
| `fist_curl_threshold` | `0.65` | Fraction of fingers curled = fist |
| `fusion_agreement_window_s` | `0.5` | Agreement window between two cameras |
| `cooldown_s` | `1.2` | Minimum seconds between same gesture |
| `ha_event_name` | `gesture_detected` | HA event type to fire |
| `log_level` | `info` | `debug` \| `info` \| `warning` \| `error` |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Stream not available" | Check RTSP URL, firewall, camera credentials |
| 401 from HA API | Ensure `homeassistant_api: true` in config.json, restart add-on |
| False triggers | Raise `arm_raised_wrist_above_shoulder_frac` to 0.20 |
| Gesture not detected | Lower `wave_velocity_threshold_px` to 12, check lighting |
| High CPU | Reduce `frame_width`/`frame_height` to 320×240; use `pose_model_complexity: 0` |
| Stream lag | Switch `rtsp_transport` to `udp` on a wired LAN |

---

## Project structure

```
gesture_cam/
├── Dockerfile
├── build.json          # Multi-arch build targets (aarch64, amd64)
├── config.json         # HA add-on manifest
├── run.sh              # Supervisor startup script
├── requirements.txt
├── main.py
├── config/
│   ├── settings.py     # Settings dataclass + options.json loader
│   └── settings.yaml   # Dev-only config
├── cameras/
│   ├── capture.py      # RTSP capture thread (auto-reconnect)
│   └── analyser.py     # MediaPipe pose + hand → gesture candidates
├── core/
│   ├── system.py       # Orchestrator
│   └── fusion.py       # Two-camera agreement + cooldown
└── output/
    └── backends.py     # HABackend (Supervisor API) + PrintBackend (dev)
```
