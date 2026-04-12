# Endora

Watches two RTSP camera streams for hand gestures and fires
Home Assistant events you can use in any automation.

Runs as a Docker container — either as a **Home Assistant Add-on** (HA OS /
Supervised) or as a **standalone Docker container** alongside HA Container/Core.

---

## Gestures

| Event data value | Movement | Hand |
|---|---|---|
| `wave_left` | Wrist flicks left | Open palm, arm raised |
| `wave_right` | Wrist flicks right | Open palm, arm raised |
| `palm_up` | Palm facing ceiling | Flat palm, arm raised |
| `palm_down` | Palm facing floor | Flat palm, arm raised |
| `fist_pump` | Upward punch | Closed fist, arm raised |

All gestures require the arm to be raised above head level first.

---

## Installation — HA Add-on (HA OS or Supervised)

### 1. Add the repository

**Settings → Add-ons → Add-on Store → ⋮ → Repositories**

Add: `https://github.com/silvermy/endora`

Or install as a local add-on:

```bash
# SSH into your HA host
cd /addons
git clone https://github.com/silvermy/endora endora
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
git clone https://github.com/silvermy/endora
cd endora
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
docker compose logs -f endora
```

---

## Debug web page

Endora includes a live MJPEG debug stream that lets you see exactly what the
gesture engine sees — skeleton overlay, wrist tracking, velocity readout, and
gesture state — all in your browser, in real time.

### Enable it

Set `debug_port` in your configuration:

```yaml
debug_port: 8765
```

For the HA add-on, also expose the port in `config.json` (already present) and
open it in your firewall if needed.

### Open it

Navigate to:

```
http://<your-ha-ip>:8765/
```

### What you'll see

The page shows a side-by-side MJPEG stream from both cameras (Cam A left,
Cam B right). Each camera panel overlays:

| Overlay element | Meaning |
|---|---|
| **Green skeleton** | MediaPipe Pose detected your body — landmarks and connections drawn |
| **`NO POSE DETECTED`** (red banner) | MediaPipe cannot find a body in the frame — check lighting, camera angle, or try a higher `pose_model_complexity` |
| **Cyan dot on wrist** | Arm is raised and ready — gestures can fire |
| **Orange dot on wrist** | Arm is raised but warming up (not enough consecutive frames yet) |
| **Magenta arrow** | Peak wrist velocity vector — shows direction and magnitude of last movement |
| **Bottom-left panel** | Live readout: arm state, vx/pvx, vy/pvy, fist flag, palm orientation, current candidate gesture |
| **Green banner (center-bottom)** | Gesture fired — shows the gesture name for 2 seconds |

### Tuning workflow

1. Open the debug page and stand in front of the camera.
2. If you see **"NO POSE DETECTED"**: MediaPipe can't find you. Try `pose_model_complexity: 2`, lower `pose_min_detection_confidence`, improve lighting, or move closer.
3. If the skeleton appears but gestures don't fire: watch the bottom-left panel. Raise your arm and check that the wrist dot appears (cyan). Then wave and observe the `pvx` value — if it stays below `wave_velocity_threshold_px`, lower that threshold.
4. If gestures fire too easily: raise `wave_velocity_threshold_px` or `arm_above_head_tolerance`.

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
- alias: "Endora — wave right → lights on"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: wave_right
  action:
    service: light.turn_on
    target:
      area_id: living_room

- alias: "Endora — wave left → lights off"
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
- alias: "Endora — palm up → volume up"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: palm_up
  action:
    service: media_player.volume_up
    target:
      entity_id: media_player.living_room_tv

- alias: "Endora — palm down → volume down"
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

### Fist pump — toggle TV

```yaml
- alias: "Endora — fist pump → TV toggle"
  trigger:
    platform: event
    event_type: gesture_detected
    event_data:
      gesture: fist_pump
  action:
    service: media_player.toggle
    target:
      entity_id: media_player.living_room_tv
```

---

## Tuning

| Problem | Fix |
|---|---|
| No skeleton on debug page | Check lighting; try `pose_model_complexity: 2`; lower `pose_min_detection_confidence` to `0.3` |
| False triggers | Raise `wave_velocity_threshold_px`; raise `arm_above_head_tolerance` |
| Gesture not detecting | Lower `wave_velocity_threshold_px`; check the debug page pvx readout |
| High CPU | Set `frame_width: 320`, `frame_height: 240`; use `pose_model_complexity: 0` |
| Stream dropping | Switch `rtsp_transport` to `udp` on wired LAN |
| 401 from HA | Check token; for add-on ensure `homeassistant_api: true` in config.json |
| Left/right reversed | Set `mirror_camera: true` |

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
| `pose_min_detection_confidence` | `0.4` | Lower = detects more; raises false positives |
| `arm_above_head_tolerance` | `0.86` | Absolute Y threshold (0–1); wrist must be above this Y in frame |
| `wave_velocity_threshold_px` | `15.0` | px/frame needed for a wave |
| `wave_sustain_frames` | `1` | Consistent frames before firing |
| `fist_curl_threshold` | `0.65` | Curl fraction = fist (0–1) |
| `palm_orientation_threshold` | `0.05` | z-depth sensitivity for palm up/down |
| `mirror_camera` | `true` | Flip left/right if camera faces you |
| `fusion_agreement_window_s` | `1.0` | Agreement window between cameras |
| `cooldown_s` | `2.0` | Min seconds between same gesture |
| `ha_event_name` | `gesture_detected` | HA event type name |
| `debug_port` | `0` | Set to e.g. `8765` to enable debug web page; `0` = disabled |
| `log_level` | `info` | `debug` \| `info` \| `warning` \| `error` |

---

## Project structure

```
endora/
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
│   ├── analyser.py        # MediaPipe pose + hands → gesture candidates
│   └── debug_server.py    # MJPEG debug stream (http://<ip>:<debug_port>/)
├── core/
│   ├── system.py          # Orchestrator
│   └── fusion.py          # Two-camera agreement + cooldown
├── output/
│   └── backends.py        # HABackend (Supervisor + Long-Lived Token) / PrintBackend
└── translations/
    └── en.json            # HA UI option labels
```
