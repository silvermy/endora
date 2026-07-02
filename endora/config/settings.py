"""
config/settings.py

Hybrid pose+hands gesture detection settings.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path

from config.registry import REGISTRY_BY_KEY

log = logging.getLogger(__name__)

HA_OPTIONS_PATH = Path("/data/options.json")


@dataclasses.dataclass
class Settings:
    # ── RTSP ──────────────────────────────────────────────────────────────
    rtsp_url_a: str = "rtsp://user:pass@192.168.1.100:554/stream1"
    rtsp_url_b: str = "rtsp://user:pass@192.168.1.100:554/stream1"
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
    # YOLO pose model weights file.
    #   yolo11n-pose.onnx  — nano,  fastest (~25 ms/frame on Pi 5),  least accurate
    #   yolo11s-pose.onnx  — small, ~2× slower (~50 ms), noticeably better at
    #                        unusual poses (lounging, arm raised, blanket-covered).
    #                        Recommended if Pi 5 CPU headroom allows it.
    # Both models are bundled in the Docker image.
    yolo_pose_model: str = "yolo11n-pose.onnx"
    # Minimum YOLO detection confidence (0–1). Raise to reduce false detections
    # from furniture/shadows, especially in low light. Default 0.25 is too
    # permissive; 0.45 filters most ghost detections without missing real people.
    yolo_conf: float = 0.30
    # Inference resolution (square, must be multiple of 32). Only sizes with
    # a matching bundled/cached .onnx actually take effect — today that's
    # 320, 480, and 640 (see Dockerfile). Any other value silently falls
    # back to 640 on aarch64 (Pi 4/5), since ONNX export at runtime is
    # unconditionally disabled there (PyTorch causes SIGILL). A missing
    # size can still be added by hand: generate it on an x86/macOS machine
    # and drop <model>-<imgsz>.onnx into the add-on's /data/ folder.
    # 320 uses one-quarter the FLOPs of 640 — fastest, but may miss distant
    # or small people. 480 is a middle ground. 640 gives the most detail.
    yolo_imgsz: int = 320
    # Motion gate: only run YOLO when the frame changes by more than this
    # fraction (0–1 mean absolute pixel difference over an 80×60 thumbnail).
    # 0.008 catches slow arm raises (low per-frame velocity); 0.0 = always run.
    motion_threshold: float = 0.015
    # Heartbeat: even with no motion, run YOLO at least every N frames so
    # slow arm lifts are eventually detected. 6 ≈ re-confirm every ~0.6s at 10fps.
    yolo_max_skip: int = 4
    # Background-subtraction liveness filter: rejects a YOLO detection whose
    # wrist(s) sit entirely over pixels the adaptive background model considers
    # static. Catches things like a framed picture on the wall that YOLO
    # mis-reads as a person with a permanently raised arm — a real arm-raise
    # always shows up as freshly-changed (foreground) pixels at the wrist, no
    # matter how long the rest of the room has looked the same. Continuously
    # re-learns the scene, so gradual lighting drift (day/night, lamps) doesn't
    # trip it. Disable if it ever suppresses a real gesture.
    bg_subtract_enable: bool = True
    # Minimum fraction of a wrist's small check-patch that must read as
    # foreground for that wrist to count as "moving". Lower = more permissive
    # (catches subtler motion, e.g. under a blanket) but slower to flag ghosts.
    bg_subtract_min_foreground: float = 0.12
    # Minimum keypoint confidence for YOLO to count a landmark as visible.
    pose_min_detection_confidence: float = 0.3
    # Deprecated — no longer used (was MediaPipe tracking threshold).
    pose_min_tracking_confidence: float = 0.3
    # Deprecated — was MediaPipe model complexity (0/1/2).  Kept so old
    # settings.yaml files don't cause load errors.
    pose_model_complexity: int = 2
    # Wrist must be this fraction of frame height above the shoulder to count
    # as a raised arm.  0.10 is more permissive than 0.15 — better for seated
    # or lounging postures where the wrist doesn't travel as high in the frame.
    # Lower toward 0.05 if still missing raises; raise toward 0.20 if you get
    # false triggers from resting your hand on top of your head.
    arm_above_head_tolerance: float = 0.15
    # Stricter threshold used when the body is reclined OR when upright status
    # cannot be confirmed (hips hidden by blanket).  Requires a deliberate
    # straight-up arm.  0.40 = wrist must clear shoulder by ~40% of frame
    # height — roughly "arm pointing straight at the ceiling".
    arm_above_head_tolerance_reclined: float = 0.30
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
    # Furniture filter: at least ONE shoulder must exceed this confidence
    # (uses max, not average, so a person with one shoulder hidden by a blanket
    # or turned side-on is not rejected). YOLO assigns high confidence to real
    # body landmarks and near-zero to furniture false-detections. 0.35 rejects
    # furniture without touching real people. Raise to 0.5 if still seeing table.
    pose_visibility_min: float = 0.45
    # Per-keypoint confidence below which one landmark (shoulder/wrist/elbow) is
    # treated as not-visible. Drives per-side arm-raise detection so an occluded
    # or mis-placed keypoint can't block a real raise on the other, visible arm.
    keypoint_visibility_min: float = 0.30
    # Forearm-vertical secondary route to a raised-arm: if the forearm is at
    # least this vertical (elbow_y − wrist_y, frame fraction) and the wrist is
    # at/above shoulder height, the arm counts as raised even if the wrist
    # doesn't clear the full arm_above_head_tolerance. Helps when the camera is
    # mounted high/at an angle so a raised arm's wrist stays near shoulder level
    # in the image. Raise toward 0.15 if resting a hand near your head misfires.
    forearm_vertical_min: float = 0.10

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
    snap_forearm_min: float = 0.06
    # Deprecated name — kept so old settings.yaml files don't cause errors.
    snap_elbow_min: float = 0.06
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
    # Minimum time the arm must stay up before SNAP fires (seconds), measured
    # from the first confirmed SINGLE_UP frame.  Filters out brief accidental
    # raises; ArmTracker's state_confirm_s adds another 0.20s on top.
    snap_sustain_s: float = 0.10

    # Seconds a new arm state must be seen before being accepted.
    # Lower = more responsive but may get single-frame false positives.
    state_confirm_s: float = 0.20
    # Seconds of contradictory frames before dropping a confirmed arm state.
    # Higher = more stable mid-gesture but slower to release after arm down.
    # 0.60 bridges YOLO pose-detection dropouts that occur when the arm is
    # raised and temporarily changes the body silhouette.
    state_release_s: float = 0.30
    # Seconds after SNAP that the arm must stay up to also fire HOLD.
    hold_duration_s: float = 1.5
    # Seconds within which two SNAPs count as DOUBLE_SNAP instead of two SNAPs.
    double_snap_window_s: float = 3.0
    # Seconds held for CROSS_ARMS / T_POSE / RAISE_BOTH before firing.
    sustain_s: float = 0.5

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
    dewarp_tilt: float = 30.0
    # Roll to level a tilted horizon. + = clockwise, - = counter-clockwise.
    # If the scene leans to the right use a negative value (e.g. -20).
    dewarp_roll: float = 0.0
    # Virtual camera vertical FOV — wider sees more room, more distortion.
    dewarp_vfov: float = 75.0
    # Output frame size of the dewarped image.
    dewarp_out_width: int = 1280
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

    # ── Chime (arm-up audio feedback) ────────────────────────────────────
    # Set True to play a short sound when an arm-up is detected.
    chime_enable: bool = False
    # HA entity ID of the speaker to play the chime on.
    # Find it in HA → Settings → Devices & Services → Entities, filter by
    # "media_player".  Works with any HA-integrated speaker (Sonos,
    # Chromecast, Echo, HomePod, DLNA, Spotify Connect, etc.).
    # Example: "media_player.living_room_sonos"
    chime_entity_id: str = ""
    # Volume for the chime clip (0–100).  40 is audible but not jarring
    # when the TV is playing at normal levels.
    chime_volume: int = 40
    # Minimum seconds between chimes — prevents rapid-fire if the arm
    # bobs up and down or two cameras both fire the transition.
    chime_debounce_s: float = 4.0
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

        # Types come from the registry (config/registry.py), not
        # typing.get_type_hints(cls) — this is the single source of truth
        # every field's type is checked against (see test_registry_sync.py),
        # so a field declared in the registry but missing from this
        # dataclass can no longer be silently dropped during coercion.
        hints = {k: f.type for k, f in REGISTRY_BY_KEY.items()}

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
