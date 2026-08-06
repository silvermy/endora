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
    # Both models are bundled in the Docker image. Default is the small model:
    # the deployment scenario (reclined on a dark couch, blanket, low light)
    # is exactly where nano's keypoints get too noisy to classify reliably,
    # and the motion gate keeps the average CPU cost well below worst case.
    yolo_pose_model: str = "yolo11s-pose.onnx"
    # Minimum YOLO detection confidence (0–1). Raise to reduce false detections
    # from furniture/shadows, especially in low light. Default 0.25 is too
    # permissive; 0.45 filters most ghost detections without missing real people.
    yolo_conf: float = 0.30
    # Inference resolution (square). A string, not an int — HA presents this
    # as a fixed-choice dropdown (see config/registry.py's enum) rather than
    # a free-entry field, since only a size with a matching bundled/cached
    # .onnx actually takes effect (see Dockerfile); any other value silently
    # fell back to 640 on aarch64 before this was a dropdown, and that dead
    # end is exactly what this type change exists to prevent. Always cast
    # with int() at the point of use (see analyser.py).
    # 320 uses one-quarter the FLOPs of 640 — fastest, but may miss distant
    # or small people. 480 is a middle ground. 640 gives the most detail.
    # Default 480: at 320 a couch-distance person's wrist/elbow keypoints are
    # too coarse for reliable gesture geometry, especially in low contrast.
    yolo_imgsz: str = "480"
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
    # ── Raise geometry ────────────────────────────────────────────────────
    # elevation = (shoulder_y - wrist_y) / |shoulder - wrist|, computed in
    # PIXELS: the sine of the arm's angle above horizontal. 1.0 = straight
    # up, 0.0 = level, -1.0 = hanging down. Dimensionless, so it means the
    # same thing at any camera distance, any frame shape, and in any posture
    # — standing, sitting or lying on a couch.
    raise_elevation_min: float = 0.70
    # How straight the arm must be: |shoulder-wrist| / (upper arm + forearm).
    # 1.0 = fully extended, 0.71 = elbow at 90 degrees. A hand held against
    # your own face scores low, which is why the old wrist-near-nose veto is
    # gone. Lower toward 0.70 if bent-elbow raises should count.
    arm_extension_min: float = 0.80
    # An arm pointing at the camera projects to almost nothing, and angles
    # computed from a few pixels are noise. If the projected arm is shorter
    # than this multiple of shoulder width, refuse to judge it rather than
    # guess.
    min_arm_len_frac: float = 0.55
    # Deprecated — superseded by the settings above. Kept so existing
    # settings files keep loading; no code reads them.
    arm_above_head_tolerance: float = 0.15
    arm_above_head_tolerance_reclined: float = 0.38
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
    forearm_vertical_min: float = 0.10
    forearm_route_min_margin: float = 0.10
    wrist_head_exclude_dist: float = 0.09
    body_scale_reference: float = 0.18

    # ── Hands (gesture classification) ───────────────────────────────────
    # Advanced: override in settings.yaml if needed
    hand_model_max_hands: int = 1
    hand_min_detection_confidence: float = 0.1   # low = faster first-frame detect
    hand_min_tracking_confidence: float = 0.1
    palm_orientation_threshold: float = 0.05
    # Run hand detection on a crop around the raised wrist instead of the
    # full frame. At couch distance a hand is a few dozen pixels in the full
    # frame and MediaPipe rarely finds it (snap_roll came back ~1 in 15
    # fires); the crop makes the hand large enough to detect reliably.
    hand_crop_enable: bool = True

    # ── Gesture thresholds ────────────────────────────────────────────────
    # Flip the image 180° (useful for cameras mounted upside-down).
    flip_image: bool = False
    # Flip gesture left/right (set True if the camera faces you and you have
    # NOT already mirrored it in the camera's own app).
    mirror_camera: bool = False
    # Minimum arm elevation for SNAP/HOLD to fire. The tracker already
    # applied raise_elevation_min to reach SINGLE_UP, so this only bites when
    # set higher — i.e. to demand a straighter-up arm for firing than for
    # showing "arm up" on the overlay.
    snap_elevation_min: float = 0.70
    # Deprecated — the forearm-only test was replaced by whole-arm elevation.
    snap_forearm_min: float = 0.05
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
    # Absolute snap_roll magnitude that lets a raise count as a snap even
    # when forearm_dy is below snap_forearm_min (an OR-route in the state
    # machine). 0.0 = disabled.
    # Disabled by default since v1.9.115: the old roll formula returned
    # exactly ±1.0 for ANY detected hand, so once the wrist-crop (v1.9.114)
    # made hand detection reliable, this route degenerated into "hand
    # visible while arm up ⇒ snap" and contributed to a false-positive
    # burst. The formula is fixed now (|roll|≈1 palm-to-camera, ≈0 edge-on),
    # but re-enable only after feedback.jsonl shows the new values actually
    # separate real snaps from false fires.
    snap_roll_threshold: float = 0.0

    # ── Hysteresis timing ─────────────────────────────────────────────────
    # Minimum time the arm must stay up before SNAP fires (seconds), measured
    # from the first confirmed SINGLE_UP frame.  Filters out brief accidental
    # raises; ArmTracker's state_confirm_s adds another 0.20s on top.
    # Was a hold timer; with the flourish test the sweep itself is the
    # evidence, so this defaults to 0 — SNAP fires at the top of the arc
    # instead of waiting. Historical note: raised 0.10 -> 0.20 in v1.9.113 after feedback.jsonl showed a cluster of
    # false SNAPs during a busy morning-routine window that forearm_dy/snap_roll
    # couldn't separate from real fires — a moderate middle ground between this
    # and the 0.50 default that v1.9.93 lowered specifically because real
    # snaps were being missed at higher sustain values; watch near_miss entries
    # for "sustain … < 0.20s required" if genuine snaps start getting dropped.
    snap_sustain_s: float = 0.0
    # ── The flourish ──────────────────────────────────────────────────────
    # The gesture is Endora's: a theatrical arm sweep, up and usually
    # straight back down — not a raised hand held still like a question in
    # class. Requiring the sweep is also what makes the static false
    # positives impossible by construction: an arm resting on an armrest,
    # draped over a backrest or holding a phone sweeps at a rate of
    # essentially zero, however much it resembles a raise geometrically.
    # Turn this off to fall back to the older hold-style detection.
    snap_require_flourish: bool = True
    # How much elevation the sweep must gain. A full sweep from hanging-down
    # to straight-up is ~2.0; one starting from an armrest has ~0.7
    # available. 0.60 admits both while rejecting a draped arm shifting
    # position (~0.4). Lower if flourishes are missed, raise if a lazy arm
    # movement triggers.
    flourish_min_climb: float = 0.60
    # …and how fast, in elevation units per second. A resting arm sits near
    # 0, a deliberate flourish runs 2–3. 0.80 leaves room for an unhurried
    # sweep without admitting slow drift.
    flourish_min_rate: float = 0.80
    # Legacy hold-style gates — used only when snap_require_flourish is off.
    snap_require_rise: bool = True
    snap_require_still: bool = False
    # Max wrist travel during the stillness window, as a multiple of that
    # arm's OWN length — so it means the same thing near or far from the
    # camera. Raise toward 0.25 if deliberate raises with a wobbly hand get
    # blocked; lower toward 0.10 if slow reaches still fire.
    wrist_still_max_travel_arm: float = 0.15
    # An arm counts as having risen if its elevation climbed by at least this
    # much within the rise window, wherever it started. This is what lets a
    # raise that begins with the arm already up (armrest, sofa backrest) fire
    # at all — such an arm never reads low.
    rise_elevation_delta: float = 0.35
    # …or if it was seen at/below this elevation within the window, i.e. it
    # started from a normally lowered position.
    rise_start_elevation_max: float = 0.35
    # Deprecated — replaced by the two settings above (which are measured
    # against the arm rather than the frame).
    wrist_still_max_travel: float = 0.05
    raise_travel_min: float = 0.08

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
    # A sustained-pose gesture fires ONCE per pose entry, then can't re-fire
    # until the pose has been released for this many seconds. Stops sitting
    # with crossed arms from re-firing CROSS_ARMS every cooldown (~100 fires
    # in 20 minutes of TV-watching seen in live feedback before this).
    sustained_rearm_s: float = 2.0

    # ── Per-gesture enable ────────────────────────────────────────────────
    # A gesture you never perform is not free: it still fires HA events, and
    # an unwanted one used to suppress a real gesture through the shared
    # cooldown. Turn off whatever you don't use.
    gesture_snap_enable: bool = True
    gesture_hold_enable: bool = True
    gesture_double_snap_enable: bool = True
    gesture_cross_arms_enable: bool = True
    gesture_t_pose_enable: bool = True
    gesture_raise_both_enable: bool = True

    # ── Fusion ────────────────────────────────────────────────────────────
    # Advanced: override in settings.yaml if needed
    fusion_agreement_window_s: float = 1.0
    cooldown_s: float = 2.0
    # Minimum seconds before a DIFFERENT gesture type may fire. Deliberately
    # much shorter than cooldown_s: its only job is to stop the residual
    # motion of one gesture immediately triggering another. Sharing the full
    # cooldown_s meant a single spurious CROSS_ARMS swallowed a real SNAP for
    # two seconds — after the chime had already played, which reads as "the
    # sound fired but nothing happened".
    cross_gesture_cooldown_s: float = 0.5
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
        # Where each key's winning value came from. Attached to the instance
        # as `_sources` (deliberately NOT a dataclass field — the registry
        # sync test compares dataclass fields against the registry). Three
        # separate debugging dead-ends have been caused by a stale value in
        # one of these files silently outranking a shipped default with
        # nothing anywhere reporting it, so record it as we merge.
        sources: dict = {}

        yaml_path = Path("/data/settings.yaml")
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path) as f:
                    data = yaml.safe_load(f) or {}
                sources.update({k: "settings.yaml" for k in data})
                log.info("Loaded settings from %s", yaml_path)
            except Exception as e:
                log.warning("Could not parse %s: %s", yaml_path, e)

        if HA_OPTIONS_PATH.exists():
            try:
                with open(HA_OPTIONS_PATH) as f:
                    options = json.load(f)
                data.update(options)
                sources.update({k: "options.json" for k in options})
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
                sources.update({k: "runtime_overrides.yaml" for k in overrides})
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
        # Keys that were present in a file but never made it into the
        # dataclass (unknown or uncoercible) must not claim a source.
        instance._sources = {k: v for k, v in sources.items() if k in coerced}

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
                    instance._sources[field] = f"env:{var}"
                except (ValueError, TypeError):
                    pass

        return instance

    # ── Effective-value reporting ─────────────────────────────────────────

    def source_of(self, key: str) -> str:
        """Where this setting's live value came from: a filename, an env var,
        or "default" when nothing overrode the shipped value.
        """
        return getattr(self, "_sources", {}).get(key, "default")

    def effective(self, keys=None) -> dict:
        """{key: (value, source)} for *keys* (all registry keys by default).

        This is the answer to "what is this thing actually running?" — a
        question that previously required reading three files by hand.
        """
        if keys is None:
            keys = [f.key for f in REGISTRY_BY_KEY.values()]
        return {k: (getattr(self, k, None), self.source_of(k)) for k in keys}
