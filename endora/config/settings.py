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
    # Advanced: override in settings.yaml if needed
    rtsp_transport: str = "tcp"
    rtsp_reconnect_delay_s: float = 5.0

    # ── Frame ─────────────────────────────────────────────────────────────
    # Advanced: override in settings.yaml if needed
    frame_width: int = 640
    frame_height: int = 640
    frame_crop_pct: float = 100.0
    frame_crop_top: float = 0.0
    frame_crop_bottom: float = 0.0
    frame_crop_left: float = 0.0
    frame_crop_right: float = 0.0

    # ── Pose (arm-raise detection) ────────────────────────────────────────
    # Advanced: override in settings.yaml if needed
    # Complexity 2 gives best detection; 0 is fastest. Max valid value is 2.
    pose_model_complexity: int = 2
    pose_min_detection_confidence: float = 0.3
    pose_min_tracking_confidence: float = 0.3
    arm_above_head_tolerance: float = 0.02
    # Furniture filter: minimum average visibility of shoulders+hips.
    # MediaPipe assigns high visibility to real body landmarks and near-zero
    # to furniture false-detections. 0.35 rejects furniture without touching
    # real people. Lower = more permissive; raise to 0.5 if still seeing table.
    pose_visibility_min: float = 0.35

    # ── Hands (gesture classification) ───────────────────────────────────
    # Advanced: override in settings.yaml if needed
    hand_model_max_hands: int = 1
    hand_min_detection_confidence: float = 0.1   # low = faster first-frame detect
    hand_min_tracking_confidence: float = 0.1
    palm_orientation_threshold: float = 0.05

    # ── Gesture thresholds ────────────────────────────────────────────────
    mirror_camera: bool = True
    wave_velocity_threshold_px: float = 20.0
    wave_sustain_frames: int = 3
    # Advanced: override in settings.yaml if needed
    vertical_velocity_threshold_px: float = 20.0
    vertical_sustain_frames: int = 1
    fist_curl_threshold: float = 0.75
    # Minimum peak single-frame swing in 2D hand_roll to register a snap.
    # hand_roll = (index_mcp.x − pinky_mcp.x) / hand_width; ranges ±1.
    # A full palm flip ≈ 0.8–1.2 swing.  0.40 catches deliberate snaps
    # while ignoring small lateral sways.
    palm_twist_threshold: float = 0.40

    # ── Fusion ────────────────────────────────────────────────────────────
    # Advanced: override in settings.yaml if needed
    fusion_agreement_window_s: float = 1.0
    cooldown_s: float = 2.0
    single_camera_mode: bool = False

    # ── HA ────────────────────────────────────────────────────────────────
    ha_event_name: str = "gesture_detected"
    ha_url: str = "http://supervisor/core/api"

    # ── Fisheye dewarping ─────────────────────────────────────────────────
    # Converts raw equidistant fisheye → flat perspective before MediaPipe.
    # Requires the RAW fisheye RTSP stream (disable in-camera dewarping).
    # Maps are built once on the first frame — restart the add-on to apply
    # changes to pan/tilt/fov settings.
    dewarp_enable: bool = False
    # Total FOV of the fisheye lens in degrees (180 = hemisphere).
    dewarp_fov: float = 180.0
    # Virtual camera pan (+= right, -= left) and tilt (+= down toward floor).
    # Tune these to point the virtual viewport toward where you stand/sit.
    dewarp_pan: float = 0.0
    dewarp_tilt: float = 20.0
    # Roll to level a tilted horizon. + = clockwise, - = counter-clockwise.
    # If the scene leans to the right use a negative value (e.g. -20).
    dewarp_roll: float = 0.0
    # Virtual camera vertical FOV — wider sees more room, more distortion.
    dewarp_vfov: float = 75.0
    # Output frame size of the dewarped image.
    dewarp_out_width: int = 640
    dewarp_out_height: int = 480
    # Fisheye circle centre in the input image (-1 = use frame geometric centre).
    dewarp_cx: float = -1.0
    dewarp_cy: float = -1.0

    # ── Low-light / night-vision enhancement ─────────────────────────────
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) boosts local
    # contrast in dark/IR frames before MediaPipe inference.  Helps pose
    # detection in dim rooms without amplifying noise like a brightness boost.
    low_light_enhance: bool = False
    # CLAHE clip limit — higher = stronger contrast boost, more noise risk.
    # 2.0 is a safe default; try 3.0–4.0 for very dark scenes.
    low_light_clip: float = 2.0

    # ── Misc ──────────────────────────────────────────────────────────────
    log_level: str = "info"
    show_display: bool = False
    # Set to e.g. 8765 to enable MJPEG debug stream at http://<ha-ip>:8765/
    debug_port: int = 0

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

        # Runtime overrides — written by the debug page Save button.
        # Loaded last so they take priority over both settings.yaml and
        # options.json (which the HA Supervisor regenerates on every restart,
        # making direct patches to options.json non-persistent).
        runtime_path = Path("/data/runtime_overrides.yaml")
        if runtime_path.exists():
            try:
                import yaml
                with open(runtime_path) as f:
                    overrides = yaml.safe_load(f) or {}
                data.update(overrides)
                log.info(
                    "Loaded runtime overrides from %s (%d keys)",
                    runtime_path, len(overrides),
                )
            except Exception as e:
                log.warning("Could not parse %s: %s", runtime_path, e)

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

        for var, field, cast in [
            ("RTSP_URL_A",  "rtsp_url_a",  str),
            ("RTSP_URL_B",  "rtsp_url_b",  str),
            ("HA_URL",      "ha_url",      str),
            ("LOG_LEVEL",   "log_level",   str),
            ("DEBUG_PORT",  "debug_port",  int),
        ]:
            val = os.environ.get(var)
            if val:
                try:
                    setattr(instance, field, cast(val))
                except (ValueError, TypeError):
                    pass

        return instance
