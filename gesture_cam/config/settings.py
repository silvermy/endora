"""
config/settings.py

Settings for hands-only gesture detection.
Pose detection has been removed — only MediaPipe Hands is used.
This allows detection from any body position: standing, sitting, laying down.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import typing
from pathlib import Path

log = logging.getLogger(__name__)

HA_OPTIONS_PATH = Path("/data/options.json")


@dataclasses.dataclass
class Settings:
    # ── RTSP streams ──────────────────────────────────────────────────────
    rtsp_url_a: str = "rtsp://user:pass@192.168.1.100:554/stream1"
    rtsp_url_b: str = "rtsp://user:pass@192.168.1.101:554/stream1"
    rtsp_transport: str = "tcp"
    rtsp_reconnect_delay_s: float = 5.0

    # ── Frame size ────────────────────────────────────────────────────────
    frame_width: int = 640
    frame_height: int = 480

    # ── MediaPipe Hands ───────────────────────────────────────────────────
    # max_num_hands: keep at 1 for speed and to avoid detecting bystanders
    hand_model_max_hands: int = 1
    # Lower detection confidence = more sensitive, more CPU, more false positives
    # 0.5 is a good balance for a room camera
    hand_min_detection_confidence: float = 0.5
    hand_min_tracking_confidence: float = 0.5

    # ── Gesture thresholds ────────────────────────────────────────────────
    # Minimum wrist velocity (px/frame) to register a gesture.
    # A deliberate Bewitched-style flick hits 20-40 px/frame.
    # Idle hand movement while sitting is typically under 5 px/frame.
    # 15 is a good starting point — lower if not triggering, raise if
    # getting false triggers from normal hand movement.
    wave_velocity_threshold_px: float = 15.0
    wave_sustain_frames: int = 1
    vertical_velocity_threshold_px: float = 15.0
    vertical_sustain_frames: int = 1
    fist_curl_threshold: float = 0.65

    # ── Fusion / debounce ─────────────────────────────────────────────────
    fusion_agreement_window_s: float = 1.0
    # Minimum seconds before same gesture can fire again
    cooldown_s: float = 2.0
    # Set to true if both RTSP URLs point to the same camera
    single_camera_mode: bool = False

    # ── Home Assistant ────────────────────────────────────────────────────
    ha_event_name: str = "gesture_detected"
    ha_url: str = "http://supervisor/core/api"

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = "info"

    # ── Display ───────────────────────────────────────────────────────────
    show_display: bool = False

    @classmethod
    def load(cls) -> "Settings":
        data: dict = {}

        yaml_path = Path("/data/settings.yaml")
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path) as f:
                    data = yaml.safe_load(f) or {}
                log.info("Loaded settings from %s", yaml_path)
            except Exception as e:
                log.warning("Could not parse %s: %s", yaml_path, e)

        if HA_OPTIONS_PATH.exists():
            try:
                with open(HA_OPTIONS_PATH) as f:
                    options = json.load(f)
                data.update(options)
                log.info("Loaded add-on options from %s", HA_OPTIONS_PATH)
            except Exception as e:
                log.warning("Could not parse %s: %s", HA_OPTIONS_PATH, e)

        try:
            hints = typing.get_type_hints(cls)
        except Exception:
            hints = {f.name: type(getattr(cls, f.name, None))
                     for f in dataclasses.fields(cls)}

        coerced: dict = {}
        for k, v in data.items():
            if k not in hints:
                continue
            t = hints[k]
            try:
                if t is int:
                    v = int(v)
                elif t is float:
                    v = float(v)
                elif t is bool:
                    if isinstance(v, str):
                        v = v.lower() in ("true", "1", "yes")
                    else:
                        v = bool(v)
                elif t is str:
                    v = str(v)
            except (ValueError, TypeError) as e:
                log.warning("Could not coerce %s=%r to %s: %s — using default",
                            k, v, t, e)
                continue
            coerced[k] = v
        instance = cls(**coerced)

        for var, field in [
            ("RTSP_URL_A", "rtsp_url_a"),
            ("RTSP_URL_B", "rtsp_url_b"),
            ("HA_URL",     "ha_url"),
            ("LOG_LEVEL",  "log_level"),
        ]:
            val = os.environ.get(var)
            if val:
                setattr(instance, field, val)

        return instance
