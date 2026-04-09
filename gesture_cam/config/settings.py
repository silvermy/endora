"""
config/settings.py

Hybrid pose+hands gesture detection settings.
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
    # ── RTSP ──────────────────────────────────────────────────────────────
    rtsp_url_a: str = "rtsp://user:pass@192.168.1.100:554/stream1"
    rtsp_url_b: str = "rtsp://user:pass@192.168.1.101:554/stream1"
    rtsp_transport: str = "tcp"
    rtsp_reconnect_delay_s: float = 5.0

    # ── Frame ─────────────────────────────────────────────────────────────
    frame_width: int = 640
    frame_height: int = 640

    # ── Pose (arm-raise detection) ────────────────────────────────────────
    # Lite model (0) is fast enough now that hands runs separately
    pose_model_complexity: int = 1
    pose_min_detection_confidence: float = 0.4
    pose_min_tracking_confidence: float = 0.4

    # How far below nose level the wrist can be and still count as
    # "arm above head". 0.0 = wrist must be at or above nose.
    # 0.05 = wrist may be 5% of frame height below nose (a little slack).
    # Increase if arm-raise isn't triggering; decrease to require higher raise.
    arm_above_head_tolerance: float = 0.05

    # ── Hands (gesture classification) ───────────────────────────────────
    hand_model_max_hands: int = 1
    hand_min_detection_confidence: float = 0.3
    hand_min_tracking_confidence: float = 0.1

    # ── Palm orientation ──────────────────────────────────────────────────
    # Z-depth difference threshold between wrist and MCP knuckles.
    # Larger = requires more extreme wrist bend to register palm_up/down.
    # Start at 0.05 — lower to 0.03 if palm gestures aren't triggering.
    palm_orientation_threshold: float = 0.05

    # ── Gesture thresholds ────────────────────────────────────────────────
    wave_velocity_threshold_px: float = 10.0
    wave_sustain_frames: int = 1
    vertical_velocity_threshold_px: float = 10.0
    vertical_sustain_frames: int = 1
    fist_curl_threshold: float = 0.65

    # ── Fusion ────────────────────────────────────────────────────────────
    fusion_agreement_window_s: float = 1.0
    cooldown_s: float = 2.0
    single_camera_mode: bool = False

    # ── HA ────────────────────────────────────────────────────────────────
    ha_event_name: str = "gesture_detected"
    ha_url: str = "http://supervisor/core/api"

    # ── Misc ──────────────────────────────────────────────────────────────
    log_level: str = "info"
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
                if t is int:   v = int(v)
                elif t is float: v = float(v)
                elif t is bool:
                    v = v.lower() in ("true","1","yes") if isinstance(v,str) else bool(v)
                elif t is str: v = str(v)
            except (ValueError, TypeError) as e:
                log.warning("Could not coerce %s=%r to %s: %s", k, v, t, e)
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
