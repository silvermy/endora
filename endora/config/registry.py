"""
config/registry.py

Single source of truth for every Endora setting: name, type, default,
description, and (for the subset that are user- or debug-page-tunable)
UI metadata. `Settings` (settings.py), config.json's options/schema blocks,
and debug_server.py's slider/joystick/toggle lists are all derived from
or checked against this list — see tests/test_registry_sync.py.

Adding a new setting: add one SettingField entry here. Set user_facing=True
to expose it in the HA Configuration tab (regenerate config.json via
scripts/gen_config_json.py), and set ui=UIMeta(...) to also expose it as a
debug-page slider/joystick/toggle. Never rename an existing `key` — real
installs have it saved in /data/options.json and/or
/data/runtime_overrides.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class UIMeta:
    label: str
    kind: Literal["slider", "joystick_x", "joystick_y", "toggle"]
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    ui_group: Optional[str] = None
    description: Optional[str] = None
    # Explicit render position among other fields of the same `kind` — the
    # debug page groups sliders by ui_group and assumes same-group entries
    # are contiguous, so order can't be left to registry declaration order
    # (which is grouped by setting *category*, not UI group).
    order: int = 0


@dataclass(frozen=True)
class SettingField:
    key: str
    type: type
    default: Any
    doc: str
    group: Optional[str] = None
    user_facing: bool = False
    enum: Optional[tuple] = None
    ui: Optional[UIMeta] = None
    deprecated: bool = False


REGISTRY: list[SettingField] = [
    # ── RTSP ─────────────────────────────────────────────────────────────
    SettingField("rtsp_url_a", str, "rtsp://user:pass@192.168.1.100:554/stream1",
                  "RTSP stream URL (required)", group="RTSP", user_facing=True),
    SettingField("rtsp_url_b", str, "rtsp://user:pass@192.168.1.100:554/stream1",
                  "Second camera; set equal to A for single-camera mode", group="RTSP", user_facing=True),
    SettingField("rtsp_transport", str, "tcp",
                  "RTSP transport protocol", group="RTSP"),
    SettingField("rtsp_reconnect_delay_s", float, 5.0,
                  "Seconds to wait before reconnecting a dropped stream", group="RTSP"),

    # ── Frame ────────────────────────────────────────────────────────────
    SettingField("frame_width", int, 640, "Fallback frame width", group="Frame"),
    SettingField("frame_height", int, 640, "Fallback frame height", group="Frame"),
    SettingField("frame_crop_pct", float, 100.0, "Legacy uniform crop percentage", group="Frame"),
    SettingField("frame_crop_top", float, 0.0, "% of frame to crop from top", group="Frame"),
    SettingField("frame_crop_bottom", float, 0.0, "% of frame to crop from bottom", group="Frame",
                  user_facing=True,
                  ui=UIMeta("Crop bottom (%)", "slider", 0, 60, 1, "View", order=7)),
    SettingField("frame_crop_left", float, 0.0, "% of frame to crop from left", group="Frame"),
    SettingField("frame_crop_right", float, 0.0, "% of frame to crop from right", group="Frame"),

    # ── Pose (arm-raise detection) ──────────────────────────────────────
    SettingField("yolo_pose_model", str, "yolo11n-pose.onnx",
                  "Pose model: yolo11n-pose.onnx (fast/nano) or yolo11s-pose.onnx (accurate/small)",
                  group="Pose", user_facing=True,
                  enum=("yolo11n-pose.onnx", "yolo11s-pose.onnx")),
    SettingField("yolo_conf", float, 0.30,
                  "Minimum YOLO detection confidence (0-1)", group="Pose", user_facing=True,
                  ui=UIMeta("YOLO confidence", "slider", 0.10, 0.80, 0.01, "Body", order=5)),
    SettingField("yolo_imgsz", int, 320,
                  "Inference resolution — 320, 480, or 640 are bundled; other "
                  "sizes silently fall back to 640 on aarch64 (no runtime export)",
                  group="Pose", user_facing=True),
    SettingField("motion_threshold", float, 0.015,
                  "Motion gate: only run YOLO when the frame changes by more than this fraction",
                  group="Pose", user_facing=True),
    SettingField("yolo_max_skip", int, 4,
                  "Run YOLO at least every N frames even with no motion", group="Pose", user_facing=True),
    SettingField("bg_subtract_enable", bool, True,
                  "Reject detections whose wrist never moves against the learned background "
                  "(filters framed pictures/mirrors/TV mis-read as a raised arm)",
                  group="Pose", user_facing=True,
                  ui=UIMeta("Ghost rejection", "toggle", order=1,
                            description="Reject detections whose wrist never moves against the learned "
                                        "background — filters framed pictures, mirrors, TV content "
                                        "mis-read as a raised arm")),
    SettingField("bg_subtract_min_foreground", float, 0.12,
                  "Min fraction of a wrist's check-patch that must be moving pixels to count as live",
                  group="Pose", user_facing=True,
                  ui=UIMeta("Ghost rejection", "slider", 0.0, 0.50, 0.01, "Gesture", order=4)),
    SettingField("pose_min_detection_confidence", float, 0.3,
                  "Minimum keypoint confidence for YOLO to count a landmark as visible", group="Pose"),
    SettingField("pose_min_tracking_confidence", float, 0.3,
                  "Deprecated — was MediaPipe tracking threshold", group="Pose", deprecated=True),
    SettingField("pose_model_complexity", int, 2,
                  "Deprecated — was MediaPipe model complexity (0/1/2)", group="Pose", deprecated=True),
    SettingField("arm_above_head_tolerance", float, 0.15,
                  "Wrist must be this far above shoulder (frame fraction) to count as raised",
                  group="Pose", user_facing=True,
                  ui=UIMeta("Arm raise margin", "slider", 0.0, 0.30, 0.01, "Gesture", order=0)),
    SettingField("arm_above_head_tolerance_reclined", float, 0.30,
                  "Stricter arm-raise threshold used when reclined or upright status is unknown",
                  group="Pose"),
    SettingField("body_upright_min", float, -0.50,
                  "Hip-shoulder gap to confirm upright (negative allows reclined/fisheye)",
                  group="Pose", user_facing=True),
    SettingField("leg_raise_margin", float, 0.05,
                  "Leg-raise guard: suppresses gestures if ankle/knee is this far above hip", group="Pose"),
    SettingField("pose_visibility_min", float, 0.45,
                  "Min landmark visibility to accept a pose (filters furniture)",
                  group="Pose", user_facing=True,
                  ui=UIMeta("Min visibility", "slider", 0.05, 0.8, 0.01, "View", order=8)),
    SettingField("keypoint_visibility_min", float, 0.30,
                  "Per-keypoint confidence below which a landmark is treated as not-visible", group="Pose"),
    SettingField("forearm_vertical_min", float, 0.10,
                  "Secondary raised-arm route: forearm verticality threshold when wrist is at/above shoulder",
                  group="Pose"),

    # ── Hands (gesture classification) ──────────────────────────────────
    SettingField("hand_model_max_hands", int, 1, "Max hands for grlib/MediaPipe hand pipeline", group="Hands"),
    SettingField("hand_min_detection_confidence", float, 0.1, "Hand model detection confidence", group="Hands"),
    SettingField("hand_min_tracking_confidence", float, 0.1, "Hand model tracking confidence", group="Hands"),
    SettingField("palm_orientation_threshold", float, 0.05, "Palm orientation threshold", group="Hands"),

    # ── Gesture thresholds ───────────────────────────────────────────────
    SettingField("flip_image", bool, False, "Rotate frame 180 degrees", group="Gesture", user_facing=True),
    SettingField("mirror_camera", bool, False, "Reserved for future use", group="Gesture", user_facing=True),
    SettingField("snap_forearm_min", float, 0.06,
                  "Minimum forearm verticality for SNAP/HOLD", group="Gesture", user_facing=True,
                  ui=UIMeta("Snap sensitivity", "slider", 0.03, 0.20, 0.01, "Gesture", order=1)),
    SettingField("snap_elbow_min", float, 0.06,
                  "Deprecated name for snap_forearm_min", group="Gesture", deprecated=True),
    SettingField("wave_lateral_fraction", float, 0.10,
                  "Deprecated — no longer used for classification", group="Gesture", deprecated=True),
    SettingField("wave_velocity_threshold_px", float, 150.0,
                  "Deprecated — no longer used for classification", group="Gesture", deprecated=True),
    SettingField("wave_sustain_frames", int, 3, "Legacy wave-gesture sustain frame count", group="Gesture"),
    SettingField("vertical_velocity_threshold_px", float, 20.0,
                  "Vertical raise velocity threshold", group="Gesture"),
    SettingField("vertical_sustain_frames", int, 1, "Vertical raise sustain frame count", group="Gesture"),
    SettingField("fist_curl_threshold", float, 0.85, "Fist curl threshold", group="Gesture"),
    SettingField("palm_twist_threshold", float, 0.40,
                  "Minimum peak single-frame swing in 2D hand_roll to register a snap", group="Gesture"),
    SettingField("snap_roll_threshold", float, 0.65,
                  "Absolute hand_roll magnitude threshold for snap detection", group="Gesture"),

    # ── Hysteresis timing ────────────────────────────────────────────────
    SettingField("snap_sustain_s", float, 0.10,
                  "Seconds the arm must stay up before SNAP fires", group="Hysteresis", user_facing=True,
                  ui=UIMeta("Snap hold time (s)", "slider", 0.0, 1.0, 0.05, "Gesture", order=2)),
    SettingField("state_confirm_s", float, 0.20,
                  "Seconds a new arm state must be seen before being accepted",
                  group="Hysteresis", user_facing=True),
    SettingField("state_release_s", float, 0.30,
                  "Seconds of contradictory frames before dropping a confirmed arm state",
                  group="Hysteresis", user_facing=True),

    # ── Fusion ───────────────────────────────────────────────────────────
    SettingField("fusion_agreement_window_s", float, 1.0,
                  "Window for cross-camera gesture agreement", group="Fusion"),
    SettingField("cooldown_s", float, 2.0,
                  "Minimum seconds between any two gestures", group="Fusion", user_facing=True,
                  ui=UIMeta("Cooldown (s)", "slider", 0, 10, 0.25, "Gesture", order=3)),
    SettingField("single_camera_mode", bool, False,
                  "Run a single analyser using the full core count", group="Fusion", user_facing=True),

    # ── HA ───────────────────────────────────────────────────────────────
    SettingField("ha_event_name", str, "gesture_detected",
                  "HA event type fired on gesture", group="HA", user_facing=True),
    SettingField("ha_url", str, "http://supervisor/core/api", "HA core API base URL", group="HA"),

    # ── Fisheye dewarping ────────────────────────────────────────────────
    SettingField("dewarp_enable", bool, False,
                  "Enable fisheye dewarping", group="Dewarp", user_facing=True),
    SettingField("dewarp_fov", float, 180.0,
                  "Total lens FOV in degrees", group="Dewarp", user_facing=True),
    SettingField("dewarp_pan", float, 0.0,
                  "Virtual camera pan (+ = right, - = left)", group="Dewarp", user_facing=True,
                  ui=UIMeta("Pan", "joystick_x", -30, 30, 1, order=0)),
    SettingField("dewarp_tilt", float, 30.0,
                  "Virtual camera tilt (+ = down toward floor)", group="Dewarp", user_facing=True,
                  ui=UIMeta("Tilt", "joystick_y", -10, 80, 1, order=1)),
    SettingField("dewarp_roll", float, 0.0,
                  "Roll to level a tilted horizon", group="Dewarp", user_facing=True),
    SettingField("dewarp_vfov", float, 75.0,
                  "Virtual camera vertical FOV", group="Dewarp", user_facing=True,
                  ui=UIMeta("Vertical FOV (°)", "slider", 20, 100, 1, "View", order=6)),
    SettingField("dewarp_out_width", int, 1280,
                  "Output frame width of the dewarped image", group="Dewarp", user_facing=True),
    SettingField("dewarp_out_height", int, 480,
                  "Output frame height of the dewarped image", group="Dewarp", user_facing=True),
    SettingField("dewarp_cx", float, -1.0,
                  "Fisheye circle centre X (-1 = frame geometric centre)", group="Dewarp"),
    SettingField("dewarp_cy", float, -1.0,
                  "Fisheye circle centre Y (-1 = frame geometric centre)", group="Dewarp"),

    # ── Low-light / night-vision enhancement ────────────────────────────
    SettingField("low_light_enhance", bool, False,
                  "CLAHE contrast boost before pose inference", group="Low-light", user_facing=True,
                  ui=UIMeta("CLAHE enhance", "toggle", order=0,
                            description="Boost local contrast before pose inference — helps dark "
                                        "clothing on dark backgrounds")),
    SettingField("low_light_clip", float, 2.0, "CLAHE clip limit", group="Low-light"),

    # ── Chime (arm-up audio feedback) ───────────────────────────────────
    SettingField("chime_enable", bool, False,
                  "Play a sound on arm-up detection", group="Chime", user_facing=True),
    SettingField("chime_entity_id", str, "",
                  "HA media_player entity to play chime on", group="Chime", user_facing=True),
    SettingField("chime_volume", int, 40,
                  "Chime volume 0-100", group="Chime", user_facing=True),
    SettingField("chime_debounce_s", float, 4.0,
                  "Min seconds between chimes", group="Chime", user_facing=True),

    # ── Misc ─────────────────────────────────────────────────────────────
    SettingField("log_level", str, "info", "Log verbosity", group="Misc", user_facing=True,
                  enum=("debug", "info", "warning", "error")),
    SettingField("show_display", bool, False, "Show a local OpenCV debug window", group="Misc"),
    SettingField("debug_port", int, 0,
                  "MJPEG debug stream port (0 = disabled)", group="Misc", user_facing=True),

    # ── Timing fields historically read via getattr() only — see
    # tests/test_registry_sync.py for why these must be real Settings fields ──
    SettingField("hold_duration_s", float, 1.5,
                  "Seconds after SNAP that arm must stay up to fire HOLD", group="Hysteresis", user_facing=True),
    SettingField("double_snap_window_s", float, 3.0,
                  "Seconds within which two snaps count as DOUBLE_SNAP", group="Hysteresis", user_facing=True),
    SettingField("sustain_s", float, 0.5,
                  "Seconds held for CROSS_ARMS / T_POSE / RAISE_BOTH", group="Hysteresis", user_facing=True),
]

REGISTRY_BY_KEY: dict[str, SettingField] = {f.key: f for f in REGISTRY}
