"""
config/settings.py

All tunable parameters.  When running as a Home Assistant add-on, values
are read from the add-on Options (written by the Supervisor to
/data/options.json).  A plain YAML file is also supported for dev/test.

Priority:  options.json  >  settings.yaml  >  dataclass defaults
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

    # ── MediaPipe pose ────────────────────────────────────────────────────
    pose_model_complexity: int = 0
    pose_min_detection_confidence: float = 0.6
    pose_min_tracking_confidence: float = 0.5

    # ── MediaPipe hands ───────────────────────────────────────────────────
    hand_model_max_hands: int = 1
    hand_min_detection_confidence: float = 0.6
    hand_min_tracking_confidence: float = 0.5

    # ── Arm-raised trigger ────────────────────────────────────────────────
    # diff = shoulder_y - wrist_y  (MediaPipe: 0=top, 1=bottom of frame)
    # With a HIGH/CEILING camera the wrist appears BELOW the shoulder in the
    # frame even when fully raised, so diff is negative.
    # From the log: raised arm gives diff ~ -0.07 to -0.10
    # Set threshold to -0.15 so any arm raise well above waist triggers.
    # If your camera is at eye level or below, use +0.05 instead.
    arm_raised_wrist_above_shoulder_frac: float = -0.15
    arm_raised_elbow_above_shoulder_frac: float = -0.20

    # ── Gesture thresholds ────────────────────────────────────────────────
    wave_velocity_threshold_px: float = 12.0    # lowered from 18 — easier to trigger
    wave_sustain_frames: int = 2                # lowered from 3 — fewer frames needed
    vertical_velocity_threshold_px: float = 10.0
    vertical_sustain_frames: int = 2
    fist_curl_threshold: float = 0.65
    hand_confidence_threshold: float = 0.55

    # ── Fusion ────────────────────────────────────────────────────────────
    fusion_agreement_window_s: float = 1.0      # widened from 0.5
    cooldown_s: float = 1.5

    # ── Single camera mode ────────────────────────────────────────────────
    # If both RTSP URLs are identical, treat as single-camera deployment.
    # The fusion layer will fire on a single sustained candidate instead of
    # requiring agreement between two independent sources.
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

        # get_type_hints resolves forward references / __future__ annotations
        # to actual Python types (int, float, str, bool) reliably.
        try:
            hints = typing.get_type_hints(cls)
        except Exception:
            hints = {f.name: type(getattr(cls, f.name, None)) for f in dataclasses.fields(cls)}

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
                log.warning("Could not coerce %s=%r to %s: %s — using default", k, v, t, e)
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
