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
    # YOLO pose model weights file.  "yolo11n-pose.pt" is fastest (nano);
    # other sizes: yolo11s/m/l/x-pose.pt for more accuracy at the cost of speed.
    # The model is pre-downloaded into the Docker image at build time.
    yolo_pose_model: str = "yolo11n-pose.onnx"
    # Minimum YOLO detection confidence (0–1). Raise to reduce false detections
    # from furniture/shadows, especially in low light. Default 0.25 is too
    # permissive; 0.45 filters most ghost detections without missing real people.
    yolo_conf: float = 0.30
    # Inference resolution (square, must be multiple of 32).
    # 320 uses one-quarter the FLOPs of 640 — default and recommended on Pi.
    # Increase to 640 only if you need to detect very distant or small people.
    yolo_imgsz: int = 320
    # Motion gate: only run YOLO when the frame changes by more than this
    # fraction (0–1 mean absolute pixel difference over an 80×60 thumbnail).
    # 0.015 ≈ any visible arm movement; 0.0 = always run YOLO (no gate).
    motion_threshold: float = 0.015
    # Heartbeat: even with no motion, run YOLO at least every N frames so
    # slow arm lifts are eventually detected. 12 ≈ re-confirm every ~5s at 2.5fps.
    yolo_max_skip: int = 12
    # Minimum keypoint confidence for YOLO to count a landmark as visible.
    pose_min_detection_confidence: float = 0.3
    # Deprecated — no longer used (was MediaPipe tracking threshold).
    pose_min_tracking_confidence: float = 0.3
    # Deprecated — was MediaPipe model complexity (0/1/2).  Kept so old
    # settings.yaml files don't cause load errors.
    pose_model_complexity: int = 2
    # 0.15 = wrist must be 15 % of frame height above shoulder.
    # Prevents scratching the top of the head from triggering gestures.
    # Lower (0.05) is more permissive but can fire on incidental head touches.
    arm_above_head_tolerance: float = 0.15
    # Minimum gap (frame fraction) between average hip_y and average shoulder_y.
    # Guards against arm-raise false positives when lying down: when horizontal,
    # hips and shoulders converge; when upright, hips are 0.2–0.4 below shoulders.
    # 0.10 = overhead cameras (hips below shoulders in image).
    # -0.15 = frontal fisheye, fully upright seated.
    # -0.50 = allows reclined/lounging posture (hips appear much higher in frame)
    #         while still rejecting fully horizontal (lying flat).
    body_upright_min: float = -0.50
    # Leg-raise guard: if any ankle or knee is this far above hip level
    # (normalised frame fraction), all gesture detection is suppressed.
    # Prevents feet-up-on-couch from triggering false snaps.
    # 0.05 = 5% of frame height clearance above hip.
    leg_raise_margin: float = 0.05
    # Furniture filter: minimum average visibility of shoulders+hips.
    # MediaPipe assigns high visibility to real body landmarks and near-zero
    # to furniture false-detections. 0.35 rejects furniture without touching
    # real people. Lower = more permissive; raise to 0.5 if still seeing table.
    pose_visibility_min: float = 0.45

    # ── Hands (gesture classification) ───────────────────────────────────
    # Advanced: override in settings.yaml if needed
    hand_model_max_hands: int = 1
    hand_min_detection_confidence: float = 0.1   # low = faster first-frame detect
    hand_min_tracking_confidence: float = 0.1
    palm_orientation_threshold: float = 0.05

    # ── Gesture thresholds ────────────────────────────────────────────────
    # Flip the image 180° (useful for cameras mounted upside-down).
    flip_image: bool = False
    # Flip gesture left/right (set True if the camera faces you and you have
    # NOT already mirrored it in the camera's own app).
    mirror_camera: bool = False
    # Minimum forearm vertical extent (normalised frame height) to classify
    # an arm raise as SNAP.  snap_forearm_min = elbow_y_norm − wrist_y_norm.
    # Arm straight up → forearm_dy ≈ 0.08–0.18  (wrist clearly above elbow)
    # Arm swept sideways → forearm_dy ≈ −0.05–0.04 (wrist at elbow height)
    # 0.06 is the crossover.  Watch forearm_dy in the debug overlay:
    #   snap should read 0.10+, wave should read 0.00 or negative.
    # Lower toward 0.03 if snaps misfire as wave.
    # Raise toward 0.10 if waves misfire as snap.
    snap_forearm_min: float = 0.10
    # Deprecated name — kept so old settings.yaml files don't cause errors.
    snap_elbow_min: float = 0.08
    # wave_lateral_fraction: wrist offset from body midline as a fraction of
    # frame width required to classify as wave (vs snap).
    # Deprecated — no longer used for classification; snap_elbow_min is used.
    wave_lateral_fraction: float = 0.10
    # Deprecated — no longer used for classification; kept for backward compat
    # so existing options.json/settings.yaml files don't cause load errors.
    wave_velocity_threshold_px: float = 150.0
    wave_sustain_frames: int = 3
    # Advanced: override in settings.yaml if needed
    vertical_velocity_threshold_px: float = 20.0
    vertical_sustain_frames: int = 1
    fist_curl_threshold: float = 0.85
    # Minimum peak single-frame swing in 2D hand_roll to register a snap.
    # hand_roll = (index_mcp.x − pinky_mcp.x) / hand_width; ranges ±1.
    # A full palm flip ≈ 0.8–1.2 swing.  0.40 catches deliberate snaps
    # while ignoring small lateral sways.
    palm_twist_threshold: float = 0.40
    # Absolute hand_roll magnitude threshold for snap detection.
    # hand_roll ≈ ±1 when palm is 90° rotated sideways; ≈ 0 when facing camera.
    # If |hand_roll| exceeds this on the first raised frame, snap fires even
    # without a cross-raise swing baseline.  0.65 = palm tilted >~40° from neutral.
    snap_roll_threshold: float = 0.65

    # ── Hysteresis timing ─────────────────────────────────────────────────
    # Seconds a new arm state must be seen before being accepted.
    # Lower = more responsive but may get single-frame false positives.
    state_confirm_s: float = 0.20
    # Seconds of contradictory frames before dropping a confirmed arm state.
    # Higher = more stable mid-gesture but slower to release after arm down.
    state_release_s: float = 0.30

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
    # Set to e.g. 8765 to enable MJPEG debug stream at http://homeassistant.local:8765/
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
