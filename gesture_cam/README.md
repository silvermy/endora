# Gesture Cam

Watches two RTSP camera streams for five hand gestures and fires
Home Assistant events you can use in any automation.

Runs as a Docker container — either as a **Home Assistant Add-on** (HA OS /
Supervised) or as a **standalone Docker container** alongside HA Container/Core.

---

## Gestures

| Event data value | Movement | Hand |
|---|---|---|
| `wave_left` | Wrist sweeps left | Open palm |
| `wave_right` | Wrist sweeps right | Open palm |
| `palm_up` | Wrist moves upward | Flat palm |
| `palm_down` | Wrist moves downward | Flat palm |
| `fist_gesture` | Any direction | Closed fist |

All gestures require the arm to be raised above shoulder height first.

---

## Installation — HA Add-on (HA OS or Supervised)

### 1. Add the repository

**Settings → Add-ons → Add-on Store → ⋮ → Repositories**

Add: `https://github.com/silvermy/gesture_cam`

Or install as a local add-on:

```bash
# SSH into your HA host
cd /addons
git clone https://github.com/silvermy/gesture_cam gesture_cam
```

Then: **Settings → Add-ons → Add-on Store → ⋮ → Check for updates**

### 2. Configure

In the add-on **Configuration** tab:

```yaml
rtsp_url_a: "rtsp://admin:password@192.168.1.100:554/stream1"
rtsp_url_b: "rtsp://admin:password@192.168.1.101:554/stream1"
ha_event_name: gesture_detected
log_level: info
```

### 3. Start

Click **Start** → check the **Log** tab for both streams connecting.

---

## Installation — Standalone Docker (HA Container or Core)

Use this path if you have no Add-on Store.

### 1. Get a Long-Lived Access Token

**HA → Profile (your name, bottom-left) → Long-Lived Access Tokens → Create Token**

Copy the token — you only see it once.

### 2. Configure

```bash
git clone https://github.com/silvermy/gesture_cam
cd gesture_cam
cp .env.example .env
```

Edit `.env`:

```env
RTSP_URL_A=rtsp://admin:password@192.168.1.100:554/stream1
RTSP_URL_B=rtsp://admin:password@192.168.1.101:554/stream1
HA_TOKEN=your_long_lived_token_here
HA_URL=http://localhost:8123/api
```

### 3. Build and run

```bash
docker compose up -d --build
docker compose logs -f gesture_cam
```

---

## HA Automation examples

Every gesture fires event type `gesture_detected`:

```json
{
  "gesture": "wave_right",
  "confidence": 0.91,
  "source_cameras": ["A", "B"],
  "timestamp": "2024-04-05T14:32:01.123456+00:00"
}
```

### Lights on/off

```yaml
- alias: "Gesture — wave right → lights on"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: wave_right
  action:
    service: light.turn_on
    target:
      area_id: living_room

- alias: "Gesture — wave left → lights off"
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

### Volume

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

---

## Tuning

| Problem | Fix |
|---|---|
| False triggers | Raise `arm_raised_wrist_above_shoulder_frac` to `0.20` |
| Gesture not detecting | Lower `wave_velocity_threshold_px` to `12`; check lighting |
| High CPU | Set `frame_width: 320`, `frame_height: 240`; use `pose_model_complexity: 0` |
| Stream dropping | Switch `rtsp_transport` to `udp` on wired LAN |
| 401 from HA | Check token; for add-on ensure `homeassistant_api: true` in config.json |

---

## Configuration reference

| Option | Default | Description |
|---|---|---|
| `rtsp_url_a` | — | RTSP URL for camera A (required) |
| `rtsp_url_b` | — | RTSP URL for camera B (required) |
| `rtsp_transport` | `tcp` | `tcp` (reliable) or `udp` (lower latency) |
| `rtsp_reconnect_delay_s` | `5.0` | Seconds before reconnect on stream loss |
| `frame_width` | `640` | Processing frame width |
| `frame_height` | `480` | Processing frame height |
| `pose_model_complexity` | `0` | 0=fastest, 1=balanced, 2=accurate |
| `arm_raised_wrist_above_shoulder_frac` | `0.10` | Arm-raise sensitivity |
| `wave_velocity_threshold_px` | `18.0` | px/frame needed for a wave |
| `wave_sustain_frames` | `3` | Consistent frames before firing |
| `vertical_velocity_threshold_px` | `15.0` | px/frame for palm up/down |
| `vertical_sustain_frames` | `3` | Consistent frames for vertical |
| `fist_curl_threshold` | `0.65` | Curl fraction = fist (0–1) |
| `fusion_agreement_window_s` | `0.5` | Agreement window between cameras |
| `cooldown_s` | `1.2` | Min seconds between same gesture |
| `ha_event_name` | `gesture_detected` | HA event type name |
| `log_level` | `info` | `debug` \| `info` \| `warning` \| `error` |

---

## Project structure

```
gesture_cam/
├── Dockerfile
├── docker-compose.yml     # For standalone (non-add-on) deployment
├── .env.example           # Copy to .env for standalone mode
├── build.json             # Multi-arch build config (add-on mode)
├── config.json            # HA add-on manifest
├── repository.yaml        # HA add-on repository descriptor
├── run.sh                 # S6 Stage 2 startup script
├── requirements.txt
├── main.py
├── config/
│   ├── settings.py        # Settings dataclass + options.json loader
│   └── settings.yaml      # Dev-only config override
├── cameras/
│   ├── capture.py         # RTSP capture thread (auto-reconnect)
│   └── analyser.py        # MediaPipe pose + hands → gesture candidates
├── core/
│   ├── system.py          # Orchestrator
│   └── fusion.py          # Two-camera agreement + cooldown
├── output/
│   └── backends.py        # HABackend (Supervisor + Long-Lived Token) / PrintBackend
└── translations/
    └── en.json            # HA UI option labels
```
