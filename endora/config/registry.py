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
                  ui=UIMeta("Crop bottom (%)", "slider", 0, 60, 1, "View", order=9)),
    SettingField("frame_crop_left", float, 0.0, "% of frame to crop from left", group="Frame"),
    SettingField("frame_crop_right", float, 0.0, "% of frame to crop from right", group="Frame"),

    # ── Pose (arm-raise detection) ──────────────────────────────────────
    SettingField("yolo_pose_model", str, "yolo11s-pose.onnx",
                  "Pose model: yolo11n-pose.onnx (fast/nano) or yolo11s-pose.onnx (accurate/small)",
                  group="Pose", user_facing=True,
                  enum=("yolo11n-pose.onnx", "yolo11s-pose.onnx")),
    SettingField("yolo_conf", float, 0.30,
                  "Minimum YOLO detection confidence (0-1)", group="Pose", user_facing=True,
                  ui=UIMeta("YOLO confidence", "slider", 0.10, 0.80, 0.01, "Body", order=7)),
    SettingField("yolo_imgsz", str, "480",
                  "Inference resolution — only bundled sizes actually take "
                  "effect (others silently fall back to 640 on aarch64, "
                  "since there is no runtime export there)",
                  group="Pose", user_facing=True,
                  enum=("320", "480", "640")),
    SettingField("yolo_execution_provider", str, "auto",
                  "ONNX Runtime execution provider. 'auto' uses the GPU when this "
                  "onnxruntime build has one and the CPU otherwise, so it is correct "
                  "on both a Pi and a Jetson; the explicit values exist to pin or "
                  "rule out a provider when debugging",
                  group="Pose", enum=("auto", "tensorrt", "cuda", "cpu")),
    SettingField("motion_threshold", float, 0.015,
                  "Motion gate: only run YOLO when the frame changes by more than this fraction",
                  group="Pose", user_facing=True),
    SettingField("motion_area_min", float, 0.002,
                  "Fraction of the frame that must change appreciably to count as "
                  "motion — catches a moving limb, which the frame-average test "
                  "misses entirely at room distance",
                  group="Pose", user_facing=True),
    SettingField("yolo_max_skip", int, 4,
                  "Run YOLO at least every N frames even with no motion", group="Pose", user_facing=True),
    SettingField("yolo_max_skip_active", int, 1,
                  "Run YOLO at least every N frames while a person is tracked. A "
                  "sweep is only measurable if several samples land inside it, and "
                  "during the ascent the arm is still DOWN, so nothing else forces "
                  "a run — 1 means every frame once somebody is in view, while an "
                  "empty room stays on the cheaper yolo_max_skip heartbeat",
                  group="Pose", user_facing=True),
    SettingField("crop_refine_enable", bool, False,
                  "Re-run pose on a crop around each under-resolved person. A "
                  "full-frame pass scales the whole image into one square input, "
                  "so someone across the room lands on a few dozen rows — the "
                  "regime where a pose model stops locating an elbow and starts "
                  "placing it on the shoulder-wrist line",
                  group="Pose", user_facing=True),
    SettingField("crop_refine_margin", float, 0.15,
                  "Grow the detection box by this fraction per side before cropping. "
                  "The box hugs the body the first pass found, so a wrist it missed "
                  "sits just outside it and too tight a crop inherits the blind spot",
                  group="Pose", user_facing=True),
    SettingField("crop_refine_max_persons", int, 2,
                  "At most this many persons re-inferred per frame, worst-resolved "
                  "first — a latency budget, since each one costs a second inference",
                  group="Pose", user_facing=True),
    SettingField("crop_refine_min_box_frac", float, 0.55,
                  "Only refine a person spanning less than this fraction of the "
                  "frame's long edge; above it they already fill the model input and "
                  "a second pass buys nothing but latency",
                  group="Pose", user_facing=True),
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
    # ── Raise geometry (v1.9.121 rewrite) ────────────────────────────────
    # elevation = (shoulder_y - wrist_y) / |shoulder - wrist|, measured in
    # pixels: the sine of the arm's angle above horizontal. Dimensionless,
    # so it needs no body-size or frame-size scaling.
    SettingField("raise_elevation_min", float, 0.70,
                  "How far above horizontal an arm must point to count as raised "
                  "(1.0 = straight up, 0.70 ~ 45 degrees). Replaces the old "
                  "frame-fraction margins",
                  group="Pose", user_facing=True,
                  ui=UIMeta("Arm raise angle", "slider", 0.30, 1.0, 0.01, "Gesture", order=0)),
    SettingField("arm_extension_min", float, 0.80,
                  "How straight the arm must be (1.0 = fully extended, 0.71 = elbow "
                  "at 90 degrees) — rejects a hand held at the face",
                  group="Pose", user_facing=True,
                  ui=UIMeta("Arm straightness", "slider", 0.50, 1.0, 0.01, "Gesture", order=6)),
    SettingField("min_arm_len_frac", float, 0.55,
                  "Refuse to judge an arm whose projection is shorter than this multiple "
                  "of shoulder width (too foreshortened to read reliably)",
                  group="Pose"),
    # Superseded by the two above — kept so existing settings files load,
    # but no longer read by any code path.
    SettingField("arm_above_head_tolerance", float, 0.15,
                  "Deprecated — replaced by raise_elevation_min", group="Pose",
                  deprecated=True),
    SettingField("arm_above_head_tolerance_reclined", float, 0.38,
                  "Deprecated — posture-specific thresholds are no longer needed",
                  group="Pose", deprecated=True),
    SettingField("body_upright_min", float, -0.50,
                  "Deprecated — upright is reported for diagnostics only", group="Pose",
                  deprecated=True),
    SettingField("leg_raise_margin", float, 0.05,
                  "Leg-raise guard: suppresses gestures if ankle/knee is this far above hip", group="Pose"),
    SettingField("pose_visibility_min", float, 0.45,
                  "Min landmark visibility to accept a pose (filters furniture)",
                  group="Pose", user_facing=True,
                  ui=UIMeta("Min visibility", "slider", 0.05, 0.8, 0.01, "View", order=10)),
    SettingField("keypoint_visibility_min", float, 0.30,
                  "Per-keypoint confidence below which a landmark is treated as not-visible", group="Pose"),
    SettingField("forearm_vertical_min", float, 0.10,
                  "Deprecated — the forearm-vertical route is gone; elevation measures "
                  "the whole arm", group="Pose", deprecated=True),
    SettingField("forearm_route_min_margin", float, 0.10,
                  "Deprecated — the forearm-vertical route is gone", group="Pose",
                  deprecated=True),
    SettingField("wrist_head_exclude_dist", float, 0.09,
                  "Deprecated — a hand at the face now fails arm_extension_min",
                  group="Pose", deprecated=True),
    SettingField("body_scale_reference", float, 0.18,
                  "Deprecated — thresholds are dimensionless or measured against the "
                  "body itself, so there is nothing left to calibrate",
                  group="Pose", deprecated=True),

    # ── Hands (gesture classification) ──────────────────────────────────
    SettingField("hand_model_max_hands", int, 1, "Max hands for grlib/MediaPipe hand pipeline", group="Hands"),
    SettingField("hand_crop_enable", bool, True,
                  "Run hand detection on a crop around the raised wrist instead of the full "
                  "frame — makes snap_roll usable at couch distance",
                  group="Hands", user_facing=True),
    SettingField("hand_min_detection_confidence", float, 0.1, "Hand model detection confidence", group="Hands"),
    SettingField("hand_min_tracking_confidence", float, 0.1, "Hand model tracking confidence", group="Hands"),
    SettingField("palm_orientation_threshold", float, 0.05, "Palm orientation threshold", group="Hands"),

    # ── Gesture thresholds ───────────────────────────────────────────────
    SettingField("flip_image", bool, False, "Rotate frame 180 degrees", group="Gesture", user_facing=True),
    SettingField("mirror_camera", bool, False, "Reserved for future use", group="Gesture", user_facing=True),
    SettingField("snap_elevation_min", float, 0.70,
                  "Minimum arm elevation for SNAP/HOLD to fire — raise above "
                  "raise_elevation_min to demand a straighter-up arm for firing than "
                  "for showing 'arm up'", group="Gesture", user_facing=True,
                  ui=UIMeta("Snap angle", "slider", 0.30, 1.0, 0.01, "Gesture", order=1)),
    SettingField("snap_forearm_min", float, 0.05,
                  "Deprecated — replaced by snap_elevation_min", group="Gesture",
                  deprecated=True),
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
    SettingField("snap_roll_threshold", float, 0.0,
                  "Absolute snap_roll magnitude that counts a raise as snap even below "
                  "snap_forearm_min (0 = disabled; see settings.py for why)", group="Gesture"),

    # ── Hysteresis timing ────────────────────────────────────────────────
    SettingField("snap_sustain_s", float, 0.0,
                  "Seconds the arm must stay up before SNAP fires (0 with the flourish "
                  "test, which fires at the top of the sweep)", group="Hysteresis", user_facing=True,
                  ui=UIMeta("Snap hold time (s)", "slider", 0.0, 1.0, 0.05, "Gesture", order=2)),
    SettingField("snap_require_flourish", bool, True,
                  "Fire on the arm SWEEP (Endora's flourish) rather than on a raised arm "
                  "held still — also makes resting-arm false positives impossible",
                  group="Hysteresis", user_facing=True),
    SettingField("flourish_min_climb", float, 0.60,
                  "How much elevation the sweep must gain (full down-to-up sweep is ~2.0; "
                  "starting from an armrest, ~0.7)",
                  group="Hysteresis", user_facing=True),
    SettingField("flourish_min_rate", float, 0.80,
                  "How fast the sweep must be, in elevation units per second "
                  "(a resting arm is ~0, a deliberate flourish 2-3)",
                  group="Hysteresis", user_facing=True),
    SettingField("snap_require_rise", bool, True,
                  "Legacy hold-style gate (only when snap_require_flourish is off): "
                  "require the arm to have risen recently",
                  group="Hysteresis", user_facing=True),
    SettingField("snap_require_still", bool, False,
                  "Legacy hold-style gate (only when snap_require_flourish is off): "
                  "require the raised wrist to hold still. Off by default — a flourish "
                  "never holds still",
                  group="Hysteresis", user_facing=True),
    SettingField("wrist_still_max_travel_arm", float, 0.15,
                  "Max wrist travel during the stillness window, as a multiple of that "
                  "arm's own length", group="Hysteresis", user_facing=True),
    SettingField("rise_elevation_delta", float, 0.35,
                  "How much an arm's elevation must climb to count as a deliberate lift "
                  "when it starts already raised (armrest/backrest)",
                  group="Hysteresis", user_facing=True),
    SettingField("rise_start_elevation_max", float, 0.35,
                  "An arm seen at or below this elevation within the rise window counts "
                  "as having started from a lowered position",
                  group="Hysteresis", user_facing=True),
    SettingField("wrist_still_max_travel", float, 0.05,
                  "Deprecated — replaced by wrist_still_max_travel_arm (arm-relative)",
                  group="Hysteresis", deprecated=True),
    SettingField("raise_travel_min", float, 0.08,
                  "Deprecated — replaced by rise_elevation_delta", group="Hysteresis",
                  deprecated=True),
    SettingField("state_confirm_s", float, 0.20,
                  "Seconds a new arm state must be seen before being accepted",
                  group="Hysteresis", user_facing=True),
    SettingField("state_release_s", float, 0.45,
                  "Seconds of contradictory frames before dropping a confirmed arm state",
                  group="Hysteresis", user_facing=True),

    # ── Fusion ───────────────────────────────────────────────────────────
    SettingField("fusion_agreement_window_s", float, 1.0,
                  "Window for cross-camera gesture agreement", group="Fusion"),
    SettingField("cooldown_s", float, 2.0,
                  "Minimum seconds between any two gestures", group="Fusion", user_facing=True,
                  ui=UIMeta("Cooldown (s)", "slider", 0, 10, 0.25, "Gesture", order=3)),
    SettingField("flourish_max_rate", float, 4.00,
                  "Upper bound on sweep rate. The minimum alone rewards nonsense: "
                  "rate is climb over the interval it happened in, so a keypoint "
                  "jumping between two adjacent samples reports an enormous one. "
                  "Recorded false snaps measured 8.82/s and 15.25/s; every genuine "
                  "one sat between 0.89 and 1.85",
                  group="Gesture", user_facing=True),
    SettingField("folded_resolution_min", float, 1.60,
                  "FOLDED_ARMS: minimum shoulder width over head height. Separates a "
                  "properly resolved person (2.52-4.56 measured) from a detection too "
                  "small to measure (1.05 on the one that fired at a laptop), using "
                  "only upper-body points the pose does not conceal",
                  group="Gesture"),
    SettingField("folded_visibility_min", float, 0.50,
                  "FOLDED_ARMS: minimum confidence across the shoulders and wrists "
                  "this gesture reads. Genuine captures 0.60-1.00; the false positive "
                  "0.39",
                  group="Gesture"),
    SettingField("gesture_snap_enable", bool, True,
                  "Fire the SNAP gesture. Turn off gestures you do not use — they still send HA events and can suppress ones you do use",
                  group="Gesture", user_facing=True),
    SettingField("gesture_hold_enable", bool, True,
                  "Fire the HOLD gesture. Turn off gestures you do not use — they still send HA events and can suppress ones you do use",
                  group="Gesture", user_facing=True),
    SettingField("gesture_double_snap_enable", bool, True,
                  "Fire the DOUBLE_SNAP gesture. Turn off gestures you do not use — they still send HA events and can suppress ones you do use",
                  group="Gesture", user_facing=True),
    SettingField("gesture_cross_arms_enable", bool, True,
                  "Fire the CROSS_ARMS gesture. Turn off gestures you do not use — they still send HA events and can suppress ones you do use",
                  group="Gesture", user_facing=True),
    SettingField("gesture_t_pose_enable", bool, True,
                  "Fire the T_POSE gesture. Turn off gestures you do not use — they still send HA events and can suppress ones you do use",
                  group="Gesture", user_facing=True),
    SettingField("gesture_raise_both_enable", bool, True,
                  "Fire the RAISE_BOTH gesture. Turn off gestures you do not use — they still send HA events and can suppress ones you do use",
                  group="Gesture", user_facing=True),
    SettingField("gesture_folded_arms_enable", bool, True,
                  "Fire the FOLDED_ARMS gesture — hands folded together at the chest, "
                  "facing the camera, sitting or standing. Reclining is excluded on "
                  "purpose: lying down, forearms on the chest look identical",
                  group="Gesture", user_facing=True),
    SettingField("folded_wrist_proximity", float, 0.50,
                  "FOLDED_ARMS: how close the wrists must be, in shoulder widths",
                  group="Gesture"),
    SettingField("folded_midline_max", float, 0.35,
                  "FOLDED_ARMS: how far each wrist may sit from the body midline, in "
                  "shoulder widths. Must stay below cross_arms_min_crossing or the two "
                  "poses overlap",
                  group="Gesture"),
    SettingField("folded_chest_depth", float, 0.55,
                  "FOLDED_ARMS: how far below the shoulder line the hands may sit, "
                  "as a fraction of torso length. The whole torso was accepted at "
                  "first, which includes hands resting on a laptop at roughly 0.6-0.9 "
                  "down — the default posture on a couch, and the gesture's first "
                  "false positive",
                  group="Gesture"),
    SettingField("folded_extension_max", float, 0.80,
                  "FOLDED_ARMS: maximum arm straightness — folded arms are bent arms",
                  group="Gesture"),
    SettingField("facing_shoulder_min", float, 0.45,
                  "FOLDED_ARMS: minimum shoulder width as a fraction of torso length, "
                  "below which the body is not square to the camera. In profile the "
                  "wrists overlap in the image whatever the hands are doing",
                  group="Gesture"),
    SettingField("folded_arm_span_max", float, 1.60,
                  "FOLDED_ARMS: ceiling on arm span (upper arm + forearm) in shoulder "
                  "widths. An occluded body reads LONG — hands hidden behind furniture "
                  "make the model invent wrists in the visible chest and splay the "
                  "elbows against narrow visible shoulders. Usable window measured "
                  "end-to-end on recorded captures: 1.52-1.70",
                  group="Gesture"),
    SettingField("folded_lean_max_deg", float, 22.0,
                  "FOLDED_ARMS: maximum torso lean off vertical, in degrees. Reclining "
                  "on a couch puts your forearms on your chest by itself, and the "
                  "generic upright test tolerates a 45 degree lean. Recorded traces: "
                  "the real gesture never exceeded 17 degrees, reclining ran 25-56",
                  group="Gesture"),
    SettingField("folded_knee_above_hip_max", float, 0.15,
                  "FOLDED_ARMS: how far the knees may rise above the hips, in shoulder "
                  "widths, before the pose is read as feet-up rather than seated. "
                  "Ignored when the knees are not visible — legs out of frame is normal "
                  "for someone sitting close to the camera",
                  group="Gesture"),
    SettingField("folded_recline_memory_s", float, 1.5,
                  "FOLDED_ARMS: for how long a confident sighting of a reclining body "
                  "keeps suppressing the gesture. The pose model alternates between a "
                  "correct reclined skeleton and a hallucinated upright one on the same "
                  "body, so a per-frame posture test alone is outvoted by the bad frames",
                  group="Gesture"),
    SettingField("folded_posture_visibility_min", float, 0.50,
                  "FOLDED_ARMS: confidence floor for the hips and knees before they may "
                  "veto the gesture on posture. Higher than folded_visibility_min "
                  "because a guessed hip that blocks a real gesture is worse than one "
                  "that fails to block a false one",
                  group="Gesture"),
    SettingField("cross_gesture_cooldown_s", float, 0.5,
                  "Minimum seconds before a DIFFERENT gesture type may fire. Deliberately "
                  "short — its only job is to stop one gesture's residual motion triggering "
                  "another; the full cooldown_s still applies per gesture",
                  group="Fusion", user_facing=True),
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
                  # +/-90 covers real aiming into a 180-degree fisheye. At
                  # +/-30 a perfectly ordinary saved value like -35 fell
                  # outside the pad, so after a reload the knob rendered off
                  # its left edge instead of at the aimed position.
                  ui=UIMeta("Pan", "joystick_x", -90, 90, 1, order=0)),
    SettingField("dewarp_tilt", float, 30.0,
                  "Virtual camera tilt (+ = down toward floor)", group="Dewarp", user_facing=True,
                  # Symmetric about the default (30) so the knob starts on
                  # the pad's crosshair, and wide enough that a saved value
                  # cannot fall outside the pad (see Pan).
                  ui=UIMeta("Tilt", "joystick_y", -30, 90, 1, order=1)),
    SettingField("dewarp_roll", float, 0.0,
                  "Roll to level a tilted horizon", group="Dewarp", user_facing=True),
    SettingField("dewarp_vfov", float, 75.0,
                  "Virtual camera vertical FOV", group="Dewarp", user_facing=True,
                  ui=UIMeta("Vertical FOV (°)", "slider", 20, 100, 1, "View", order=8)),
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
    SettingField("chime_debounce_s", float, 6.0,
                  "Min seconds between chimes — must exceed the clip length or one "
                  "sound runs into the next (the bundled chime is 4.0 s)",
                  group="Chime", user_facing=True),

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
    SettingField("sustain_credit_ticks", float, 2.5,
                  "How much of one gap between matched frames may count towards "
                  "sustain_s, in TICKS rather than seconds so it does not become a "
                  "frame-rate-dependent threshold. Forgiving a gap and crediting it "
                  "are different questions: crediting it let two lone frames 0.6 s "
                  "apart satisfy a 0.5 s sustain",
                  group="Gesture"),
    SettingField("sustain_gap_s", float, 0.85,
                  "How long a sustained pose may go unmatched and still count as held. "
                  "Folded hands mutually occlude, so the measured wrist gap swings "
                  "between 0.06 and 0.85 shoulder widths in adjacent frames while the "
                  "person has not moved — demanding an unbroken run meant firing only "
                  "when a good frame happened to land under the timer",
                  group="Hysteresis", user_facing=True),
    SettingField("sustain_s", float, 0.5,
                  "Seconds held for CROSS_ARMS / T_POSE / RAISE_BOTH", group="Hysteresis", user_facing=True),
    SettingField("sustained_rearm_s", float, 2.0,
                  "Sustained-pose gestures fire once per pose entry; the pose must be released "
                  "this long before it can fire again", group="Hysteresis", user_facing=True),
]

REGISTRY_BY_KEY: dict[str, SettingField] = {f.key: f for f in REGISTRY}


# Settings that decide whether a gesture fires. Logged at startup with the
# file each value came from: a stale entry in settings.yaml or
# runtime_overrides.yaml silently outranks a shipped default, and with
# nothing reporting that, three separate debugging sessions were spent
# chasing behaviour that no longer matched the code.
GESTURE_CRITICAL: list[str] = [
    "yolo_pose_model", "yolo_imgsz", "yolo_conf",
    "raise_elevation_min", "arm_extension_min", "min_arm_len_frac",
    "snap_elevation_min", "snap_sustain_s",
    "snap_require_flourish", "flourish_min_climb", "flourish_min_rate",
    "flourish_max_rate", "folded_visibility_min",
    "gesture_snap_enable", "gesture_cross_arms_enable",
    "gesture_t_pose_enable", "gesture_raise_both_enable",
    "gesture_folded_arms_enable", "facing_shoulder_min",
    "folded_wrist_proximity", "folded_midline_max",
    "folded_chest_depth", "folded_extension_max",
    "folded_resolution_min", "sustain_gap_s",
    "sustain_credit_ticks",
    "folded_lean_max_deg", "folded_knee_above_hip_max",
    "folded_arm_span_max",
    "folded_recline_memory_s", "folded_posture_visibility_min",
    # HOLD was missing here while every other gesture flag was listed, and
    # it is the one that most visibly surprises: it fires hold_duration_s
    # after a successful SNAP, from the same raised arm, with its own HA
    # event and its own chime. Someone who left their arm up reads that as
    # the detector firing twice for one gesture, and nothing in the startup
    # diagnostic said HOLD was enabled.
    "gesture_hold_enable", "hold_duration_s",
    "gesture_double_snap_enable", "double_snap_window_s",
    "cross_gesture_cooldown_s",
    "snap_require_rise", "snap_require_still",
    "rise_elevation_delta", "rise_start_elevation_max",
    "wrist_still_max_travel_arm",
    "state_confirm_s", "state_release_s", "cooldown_s", "sustained_rearm_s",
    "pose_visibility_min", "keypoint_visibility_min",
    # The pose sample rate ceiling (see _StageTimer): a sweep is only
    # measurable if several samples land inside flourish_window_s, and
    # these two decide how often the model looks.
    "yolo_max_skip", "yolo_max_skip_active",
]

# Two copies of this list existed — one for the startup log, one for the
# debug page's /effective endpoint — and they drifted: HOLD was added to the
# first in v1.9.149 "so the log can explain HOLD" and never reached the
# second, so the page a user actually reads went on omitting it. The test
# that was supposed to guard the list checked only one copy.
