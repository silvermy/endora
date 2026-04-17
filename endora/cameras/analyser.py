"""
cameras/analyser.py

Hybrid gesture detection: MediaPipe Pose + Hands.

Gesture set:
  WAVE_LEFT    — arm raised above head, open palm, wrist flicks left
  WAVE_RIGHT   — arm raised above head, open palm, wrist flicks right
  PALM_UP      — arm raised above head, palm rotated to face ceiling
  PALM_DOWN    — arm raised above head, palm rotated to face floor
  FIST_PUMP    — arm raised above head, closed fist, upward pump motion

Detection pipeline per frame:
  1. Pose  → is arm raised above head? (wrist above nose level)
  2. Hands → wrist flick velocity (wave left/right)
             palm orientation (up/down via wrist roll angle)
             hand shape (open/fist)
  3. Velocity tracker → directional velocity of wrist
  4. State machine → N consistent frames before firing
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from enum import Enum, auto
from typing import Callable, Deque, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── Gesture enum ──────────────────────────────────────────────────────────────

class Gesture(Enum):
    SNAP       = auto()  # arm raised, rapid wrist snap/rotation
    FIST       = auto()  # arm raised, closed fist (static)
    WAVE_LEFT  = auto()  # arm raised, sweep wrist to the left
    WAVE_RIGHT = auto()  # arm raised, sweep wrist to the right

    @property
    def event_name(self) -> str:
        """HA event data value, e.g. 'endora-wave-left'."""
        return f"endora-{self.name.lower().replace('_', '-')}"

    def __str__(self) -> str:
        return self.event_name


# ── Velocity tracker ──────────────────────────────────────────────────────────

WristSample = collections.namedtuple("WristSample", ["x", "y", "t"])


class VelocityTracker:
    HISTORY = 6

    def __init__(self):
        self._samples: Deque[WristSample] = collections.deque(maxlen=self.HISTORY)

    def update(self, x: float, y: float):
        self._samples.append(WristSample(x, y, time.monotonic()))

    def velocity(self) -> tuple[float, float]:
        if len(self._samples) < 2:
            return 0.0, 0.0
        oldest = self._samples[0]
        newest = self._samples[-1]
        n = len(self._samples) - 1
        return (newest.x - oldest.x) / n, (newest.y - oldest.y) / n

    def peak_velocity(self) -> tuple[float, float]:
        samples = list(self._samples)
        recent = samples[-3:] if len(samples) >= 3 else samples
        if len(recent) < 2:
            return 0.0, 0.0
        max_vx = max_vy = 0.0
        for i in range(1, len(recent)):
            dvx = recent[i].x - recent[i-1].x
            dvy = recent[i].y - recent[i-1].y
            if abs(dvx) > abs(max_vx):
                max_vx = dvx
            if abs(dvy) > abs(max_vy):
                max_vy = dvy
        return max_vx, max_vy

    def reset(self):
        self._samples.clear()


# ── Palm twist tracker ────────────────────────────────────────────────────────

class PalmTwistTracker:
    """
    Detects a palm snap by tracking 2D hand_roll over a rolling window.

    hand_roll = (index_mcp.x - pinky_mcp.x) / distance(index_mcp, pinky_mcp)
    Ranges roughly −1 to +1 based on how the knuckle line is oriented in the
    2D image.  Works at any camera angle — no z-depth estimation required.

    Swing = max(window) − min(window): captures both sharp snaps (large
    single-frame jump) and smooth twists spread over several frames.
    A full palm flip produces a swing of ~1.2–1.6; natural arm sway ~0.1–0.2.
    """
    HISTORY = 8   # ~800 ms at 10 fps — wide enough for a deliberate twist

    def __init__(self):
        self._samples: Deque[float] = collections.deque(maxlen=self.HISTORY)

    def update(self, hand_roll: float):
        self._samples.append(hand_roll)

    def peak_swing(self) -> tuple[float, str]:
        """
        Returns (range, direction) over the history window.
        range = max(samples) − min(samples).
        direction: 'up' if the window ended higher than it started, else 'down'.
        """
        if len(self._samples) < 2:
            return 0.0, "none"
        s = list(self._samples)
        swing = max(s) - min(s)
        direction = "up" if s[-1] > s[0] else "down"
        return swing, direction

    def reset(self):
        self._samples.clear()


# ── Analyser ──────────────────────────────────────────────────────────────────

class CameraAnalyser(threading.Thread):

    def __init__(
        self,
        camera,
        settings,
        on_candidate: Callable[[Gesture, float, str], None],
        label: str = "cam",
        debug_frame_cb=None,
    ):
        super().__init__(daemon=True, name=f"Analyser-{label}")
        self.camera = camera
        self.s = settings
        self.on_candidate = on_candidate
        self.label = label
        self.debug_frame_cb = debug_frame_cb
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        import mediapipe as mp

        mp_pose  = mp.solutions.pose
        mp_hands = mp.solutions.hands

        _complexity = max(0, min(2, int(self.s.pose_model_complexity)))
        if _complexity != int(self.s.pose_model_complexity):
            log.warning(
                "[%s] pose_model_complexity=%s is invalid — "
                "MediaPipe Pose only accepts 0, 1, or 2. Clamping to %d.",
                self.label, self.s.pose_model_complexity, _complexity,
            )
        pose = mp_pose.Pose(
            model_complexity=_complexity,
            min_detection_confidence=float(self.s.pose_min_detection_confidence),
            min_tracking_confidence=float(self.s.pose_min_tracking_confidence),
            enable_segmentation=False,
            static_image_mode=False,
        )
        hands = mp_hands.Hands(
            max_num_hands=int(self.s.hand_model_max_hands),
            min_detection_confidence=float(self.s.hand_min_detection_confidence),
            min_tracking_confidence=float(self.s.hand_min_tracking_confidence),
            # static_image_mode=True: run full detection on every crop frame.
            # We feed a freshly-cropped patch each frame (centred on the Pose
            # wrist) so tracking mode breaks; static mode is more reliable here.
            static_image_mode=True,
        )

        palm_twists  = PalmTwistTracker()
        wrist_tracker = VelocityTracker()   # tracks wrist pixel position for wave
        last_peak_vx  = 0.0                 # most recent peak horizontal velocity
        # Approach cache: collects hand_roll samples while the wrist is above
        # the shoulder but BEFORE the arm officially passes the raise check.
        # On the first arm-ready frame this data seeds the main tracker so a
        # fluid raise→snap is detected without needing a pause at the top.
        _approach_cache: collections.deque = collections.deque(maxlen=50)  # ~5 s
        sustain_counts: dict[Gesture, int] = {g: 0 for g in Gesture}
        # Frames each gesture must appear consecutively before firing.
        # All set to 1: single-frame sustain for fastest possible response.
        SUSTAIN_NEEDED: dict[Gesture, int] = {
            Gesture.SNAP:       1,
            Gesture.FIST:       1,
            Gesture.WAVE_LEFT:  1,
            Gesture.WAVE_RIGHT: 1,
        }
        last_arm_raised = False
        last_is_fist = False          # track fist→open transitions
        last_hand_roll = 0.0          # carry-forward when palm=unknown
        last_hand_roll_age = 0        # frames since last valid hand_roll
        MAX_ROLL_CARRY = 2            # max frames to carry forward
        # Per-side wrist pixel history: populated every frame (arm up or down)
        # so that on the very first raised frame we can immediately seed the
        # velocity tracker and detect a wave from a quick flick gesture.
        _right_wrist_history: collections.deque = collections.deque(maxlen=15)
        _left_wrist_history:  collections.deque = collections.deque(maxlen=15)
        # Cross-raise snap baseline: the hand_roll seen at the END of the
        # previous arm session.  Used as the "before" snapshot on the next
        # raise so a snap (roll changes between raises) is detectable even
        # when the arm only crosses the threshold for one frame.
        _prev_raise_roll: float = 0.0
        _prev_raise_roll_time: float = 0.0
        SNAP_ROLL_TTL_S: float = 15.0   # baseline expires after 15 s of inactivity
        consecutive_no_pose = 0
        NO_POSE_TOLERANCE = 4   # frames of arm-down before resetting state
        arm_raised_since: float = 0.0
        # Reset tracking if arm held still for this many seconds
        ARM_HELD_TIMEOUT_S = 10.0
        # Arm must be raised for this many consecutive frames before
        # gestures can fire.  1 = respond on the very first detected frame,
        # giving a fluid raise → gesture → lower flow.  Increase to 2 only
        # if phantom detections are a problem.
        consecutive_arm_raised = 0
        ARM_RAISE_MIN_FRAMES = 1
        # Furniture-lock breaker: if pose keeps landing on furniture
        # (shoulders too low) for this many frames, recreate the model
        _furniture_rejection_streak = 0
        FURNITURE_RESET_FRAMES = 12
        # Multi-person / idle-lock breaker: if no arm has been raised for
        # this many seconds, recreate the pose model so MediaPipe rescans
        # the whole scene.  Set high (60 s) so brief pauses between gestures
        # don't cause a model re-init stall mid-session.
        IDLE_RESET_S = 60.0
        _last_arm_up_time: float = 0.0
        _idle_reset_done: bool = False

        log.info("[%s] Analyser running (hybrid pose+hands mode)", self.label)

        while not self._stop_evt.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # ── Optional fisheye dewarping ────────────────────────────────
            # Converts raw equidistant fisheye to flat perspective before
            # any cropping or MediaPipe processing.  Maps are built once on
            # the first frame (lazy-init so we know the actual input size).
            # Requires raw fisheye RTSP — disable in-camera dewarping first.
            if getattr(self.s, 'dewarp_enable', False):
                from cameras.dewarp import build_dewarp_maps, apply_dewarp
                cx_raw = float(getattr(self.s, 'dewarp_cx',   -1.0))
                cy_raw = float(getattr(self.s, 'dewarp_cy',   -1.0))
                dw     = int(getattr(self.s,   'dewarp_out_width',  640))
                dh     = int(getattr(self.s,   'dewarp_out_height', 480))
                fov    = float(getattr(self.s, 'dewarp_fov',   180.0))
                pan    = float(getattr(self.s, 'dewarp_pan',     0.0))
                tilt   = float(getattr(self.s, 'dewarp_tilt',   20.0))
                roll   = float(getattr(self.s, 'dewarp_roll',    0.0))
                vfov   = float(getattr(self.s, 'dewarp_vfov',   75.0))
                # Cache key — rebuild maps if any param changes
                _key = (w, h, dw, dh, fov, pan, tilt, roll, vfov, cx_raw, cy_raw)
                if getattr(self, '_dewarp_key', None) != _key:
                    self._dewarp_maps = build_dewarp_maps(
                        in_w=w, in_h=h,
                        out_w=dw, out_h=dh,
                        fisheye_fov_deg=fov,
                        pan_deg=pan,
                        tilt_deg=tilt,
                        roll_deg=roll,
                        vfov_deg=vfov,
                        cx=None if cx_raw < 0 else cx_raw,
                        cy=None if cy_raw < 0 else cy_raw,
                    )
                    self._dewarp_key = _key
                frame = apply_dewarp(frame, *self._dewarp_maps)
                h, w = frame.shape[:2]

            # ── Optional asymmetric crop (removes fisheye distortion) ────
            # frame_crop_top/bottom/left/right = % to remove from each edge
            ct = float(getattr(self.s, 'frame_crop_top',    0))
            cb = float(getattr(self.s, 'frame_crop_bottom', 0))
            cl = float(getattr(self.s, 'frame_crop_left',   0))
            cr = float(getattr(self.s, 'frame_crop_right',  0))
            # Legacy symmetric crop_pct support
            crop_pct = float(getattr(self.s, 'frame_crop_pct', 100))
            if crop_pct < 100.0 and ct == 0 and cb == 0 and cl == 0 and cr == 0:
                margin = (100.0 - crop_pct) / 2.0
                ct = cb = cl = cr = margin
            y0 = int(h * ct / 100)
            y1 = h - int(h * cb / 100)
            x0 = int(w * cl / 100)
            x1 = w - int(w * cr / 100)
            if y0 > 0 or y1 < h or x0 > 0 or x1 < w:
                proc_frame = frame[y0:y1, x0:x1]
                ph, pw = y1 - y0, x1 - x0
                crop_offset = (x0, y0)
            else:
                proc_frame = frame
                ph, pw = h, w
                crop_offset = (0, 0)

            # ── Optional CLAHE low-light enhancement ──────────────────────
            # Boosts local contrast in dark/IR images before MediaPipe sees
            # the frame. Much better than simple brightness — preserves
            # structure while making body landmarks pop against background.
            if getattr(self.s, 'low_light_enhance', False):
                clip = float(getattr(self.s, 'low_light_clip', 2.0))
                lab = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
                l_ch = clahe.apply(l_ch)
                proc_frame = cv2.cvtColor(
                    cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR
                )

            rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_res = pose.process(rgb)

            # ── Lazy Hands: loose trigger (wrist above shoulder) ──────────
            # Fire Hands as soon as either wrist crosses its shoulder line —
            # earlier than the strict arm-raise check (which also requires the
            # elbow to clear the shoulder).  The earlier start lets the twist
            # approach-cache fill during the raise motion so a fluid
            # raise→snap is captured on the very first arm-ready frame.
            _run_hands = False
            if pose_res and pose_res.pose_landmarks:
                _lm_q  = pose_res.pose_landmarks.landmark
                _PL_q  = mp.solutions.pose.PoseLandmark
                _marg  = float(self.s.arm_above_head_tolerance)
                _rsh_y = _lm_q[_PL_q.RIGHT_SHOULDER].y
                _lsh_y = _lm_q[_PL_q.LEFT_SHOULDER].y
                _rw_y  = _lm_q[_PL_q.RIGHT_WRIST].y
                _lw_y  = _lm_q[_PL_q.LEFT_WRIST].y
                # Loose trigger: just wrist above shoulder on either side.
                _run_hands = (_rw_y < _rsh_y) or (_lw_y < _lsh_y)
                # Always track raw wrist pixel positions so we can seed the
                # velocity tracker on the first raised frame even when the arm
                # jumps from below-shoulder to raised in a single frame.
                _right_wrist_history.append(
                    (int(_lm_q[_PL_q.RIGHT_WRIST].x * pw), int(_rw_y * ph))
                )
                _left_wrist_history.append(
                    (int(_lm_q[_PL_q.LEFT_WRIST].x * pw), int(_lw_y * ph))
                )
            if _run_hands:
                # Crop a 300×300 patch centred on the raised wrist before
                # running Hands.  On a 1280-wide dewarped frame the hand is
                # only ~50 px across; MediaPipe's palm detector needs the
                # hand to fill a meaningful fraction of the image.
                # Cropping around the wrist (already known from Pose) gives
                # a hand-centric view and makes detection rock-solid.
                # Landmarks are returned as normalised [0,1] coords within
                # the crop — that's fine because we only use relative
                # positions between landmarks, never absolute frame coords.
                # Select which wrist to crop around.
                # Use the wrist that passes the strict raise check (wrist above
                # shoulder by margin); when both or neither pass, prefer the one
                # that's higher in the frame (smaller y).
                _rw_fully = _rw_y < (_rsh_y - _marg)
                _lw_fully = _lw_y < (_lsh_y - _marg)
                if _rw_fully or (_rw_y < _rsh_y and not _lw_fully):
                    _wx_n = _lm_q[_PL_q.RIGHT_WRIST].x
                    _wy_n = _lm_q[_PL_q.RIGHT_WRIST].y
                else:
                    _wx_n = _lm_q[_PL_q.LEFT_WRIST].x
                    _wy_n = _lm_q[_PL_q.LEFT_WRIST].y
                _wx_px = int(_wx_n * pw)
                _wy_px = int(_wy_n * ph)
                _ch = 180   # half-side of crop in pixels
                _cx1 = max(0, _wx_px - _ch)
                _cx2 = min(pw, _wx_px + _ch)
                _cy1 = max(0, _wy_px - _ch)
                _cy2 = min(ph, _wy_px + _ch)
                _hands_crop = rgb[_cy1:_cy2, _cx1:_cx2]
                if _hands_crop.size > 0:
                    _hands_rgb = cv2.resize(
                        _hands_crop, (256, 256),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    hand_res = hands.process(_hands_rgb)
                else:
                    hand_res = None
            else:
                hand_res = None

            # ── Approach-cache pre-classify ───────────────────────────────
            # Classify the hand here — before the arm-raise gate — so we can
            # feed hand_roll samples into the approach cache during the wrist's
            # rise toward the fully-extended position.  Only accumulates while
            # the arm is NOT yet officially raised (last_arm_raised=False) to
            # avoid mixing gesture-phase data back into the approach window.
            _pre_is_fist, _pre_palm, _pre_roll, _pre_conf = False, "unknown", 0.0, 0.0
            if _run_hands and hand_res is not None:
                _pre_is_fist, _pre_palm, _pre_roll, _pre_conf = _classify_hand_full(
                    hand_res, self.s
                )
                if _pre_palm != "unknown" and not _pre_is_fist and not last_arm_raised:
                    _approach_cache.append(_pre_roll)

            rgb.flags.writeable = True

            # ── Furniture / false-pose filter ─────────────────────────────
            # Uses MediaPipe landmark VISIBILITY scores rather than Y-position.
            # A real person has decent visibility on shoulders+hips even from
            # overhead. Furniture false-detections score near 0.0 on all of
            # these because MediaPipe has no confidence in any body landmark.
            #
            # After FURNITURE_RESET_FRAMES consecutive rejections the Pose
            # model is recreated to break the tracking lock — otherwise
            # MediaPipe keeps re-latching onto the same object each frame.
            if pose_res and pose_res.pose_landmarks:
                _lm_f = pose_res.pose_landmarks.landmark
                _PL_f = mp.solutions.pose.PoseLandmark
                # Use shoulder visibility only — hips are hidden from overhead
                # cameras and would drag the average below threshold for real people.
                # Furniture false-detections still score near 0 on shoulders.
                _vis_scores = [
                    _lm_f[_PL_f.LEFT_SHOULDER].visibility,
                    _lm_f[_PL_f.RIGHT_SHOULDER].visibility,
                ]
                _avg_vis = sum(_vis_scores) / len(_vis_scores)
                _min_vis = float(getattr(self.s, 'pose_visibility_min', 0.35))
                if _avg_vis < _min_vis:
                    _furniture_rejection_streak += 1
                    log.debug(
                        "[%s] pose rejected: avg_torso_visibility=%.2f < min=%.2f "
                        "(furniture? streak=%d)",
                        self.label, _avg_vis, _min_vis, _furniture_rejection_streak,
                    )
                    if _furniture_rejection_streak >= FURNITURE_RESET_FRAMES:
                        log.info(
                            "[%s] Breaking furniture tracking lock — recreating pose model "
                            "(avg_vis=%.2f)",
                            self.label, _avg_vis,
                        )
                        pose.close()
                        pose = mp_pose.Pose(
                            model_complexity=_complexity,
                            min_detection_confidence=float(self.s.pose_min_detection_confidence),
                            min_tracking_confidence=float(self.s.pose_min_tracking_confidence),
                            enable_segmentation=False,
                            static_image_mode=False,
                        )
                        _furniture_rejection_streak = 0
                    pose_res = None   # treat this frame as no-pose
                else:
                    _furniture_rejection_streak = 0

            # ── 1. Arm raised above head? ─────────────────────────────────
            arm_raised, wrist_xy, raised_side = _arm_above_head(
                pose_res, self.s, pw, ph
            )

            # wrist_xy stays in proc_frame pixel space — used for velocity
            # tracking (relative movement only) and debug overlay which draws
            # on proc_frame.  No remapping to full-frame needed.

            if not arm_raised:
                pose_detected = bool(pose_res and pose_res.pose_landmarks)
                consecutive_no_pose += 1
                if consecutive_no_pose >= NO_POSE_TOLERANCE:
                    # Save the last known roll as a cross-raise snap baseline
                    # BEFORE clearing state so the next raise can compare
                    # against it and detect a snap (roll change between raises).
                    if last_arm_raised and last_hand_roll != 0.0:
                        _prev_raise_roll = last_hand_roll
                        _prev_raise_roll_time = time.monotonic()
                    consecutive_arm_raised = 0
                    palm_twists.reset()
                    wrist_tracker.reset()
                    _approach_cache.clear()
                if consecutive_no_pose >= NO_POSE_TOLERANCE:
                    if last_arm_raised:
                        log.debug("[%s] arm lowered — resetting", self.label)
                        for g in Gesture:
                            sustain_counts[g] = 0
                    last_arm_raised = False
                if log.isEnabledFor(logging.DEBUG) and consecutive_no_pose % 10 == 1:
                    if not pose_detected:
                        log.debug("[%s] NO POSE DETECTED — body not found in frame", self.label)
                    else:
                        log.debug("[%s] arm not raised (pose OK, arm down)", self.label)

                # ── Idle-lock breaker (multi-person support) ──────────────
                # After IDLE_RESET_S seconds with no arm raised, recreate the
                # pose model so MediaPipe rescans the whole scene.  This lets
                # a different person "claim" detection by raising their arm.
                now_idle = time.monotonic()
                if _last_arm_up_time == 0.0:
                    _last_arm_up_time = now_idle   # initialise on first frame
                if now_idle - _last_arm_up_time > IDLE_RESET_S and not _idle_reset_done:
                    log.info(
                        "[%s] Idle %.0fs — recreating pose model for fresh person detection",
                        self.label, now_idle - _last_arm_up_time,
                    )
                    pose.close()
                    pose = mp_pose.Pose(
                        model_complexity=_complexity,
                        min_detection_confidence=float(self.s.pose_min_detection_confidence),
                        min_tracking_confidence=float(self.s.pose_min_tracking_confidence),
                        enable_segmentation=False,
                        static_image_mode=False,
                    )
                    _idle_reset_done = True
                # Debug: still render frame even when arm not raised
                if self.debug_frame_cb is not None:
                    try:
                        _debug_frame_counter = getattr(self, '_dfc', 0) + 1
                        self._dfc = _debug_frame_counter
                        if True:  # send every processed frame
                            dbg = _draw_debug(proc_frame, pose_res, None,
                                              0.0, 0.0, False, None,
                                              consecutive_arm_raised, ARM_RAISE_MIN_FRAMES)
                            self.debug_frame_cb(self.label, dbg)
                    except Exception as e:
                        log.debug("[%s] debug render error: %s", self.label, e)
                continue

            consecutive_no_pose = 0
            consecutive_arm_raised += 1
            wx, wy = wrist_xy
            # Arm is up — reset idle timer so we don't recreate mid-gesture
            _last_arm_up_time = time.monotonic()
            _idle_reset_done = False

            # Don't process gestures until arm has been raised for
            # enough consecutive frames to rule out phantom detections
            if consecutive_arm_raised < ARM_RAISE_MIN_FRAMES:
                last_arm_raised = True
                # Debug: render warming-up state
                if self.debug_frame_cb is not None:
                    try:
                        dbg = _draw_debug(proc_frame, pose_res, wrist_xy,
                                          0.0, 0.0, False, None,
                                          consecutive_arm_raised, ARM_RAISE_MIN_FRAMES)
                        self.debug_frame_cb(self.label, dbg)
                    except Exception as e:
                        log.debug("[%s] debug render error: %s", self.label, e)
                continue

            # ── 2. Hand shape and orientation ─────────────────────────────
            # Re-use the pre-classification computed in the approach-cache
            # section above (avoids running the model a second time).
            is_fist   = _pre_is_fist
            palm_facing = _pre_palm
            hand_roll = _pre_roll
            hand_conf = _pre_conf

            if not last_arm_raised:
                # ── First officially-raised frame ─────────────────────────
                # Seed the twist tracker from the approach cache so a fluid
                # raise→snap is detected immediately.  The cache holds up to
                # ~5 s of hand_roll samples collected while the wrist was
                # above the shoulder but before the arm officially raised.
                _n_seed = len(_approach_cache)
                palm_twists.reset()
                for _sv in list(_approach_cache)[-palm_twists.HISTORY:]:
                    palm_twists._samples.append(_sv)
                _approach_cache.clear()
                twist_swing, twist_dir = palm_twists.peak_swing()
                # Seed the velocity tracker from the pre-raise wrist history
                # for this side.  This lets a quick flick gesture (arm goes
                # from below to above shoulder in one frame) be detected
                # immediately — no need to hold the arm up for 2+ seconds.
                wrist_tracker.reset()
                _v_hist = (_right_wrist_history
                           if raised_side == "RIGHT"
                           else _left_wrist_history)
                for _hwx, _hwy in list(_v_hist)[-wrist_tracker.HISTORY:]:
                    wrist_tracker.update(_hwx, _hwy)
                wrist_tracker.update(wx, wy)   # include this raised frame
                last_peak_vx, _ = wrist_tracker.peak_velocity()
                last_is_fist = is_fist   # initialise for future fist transitions
                last_hand_roll = 0.0     # clear carry-forward on new arm raise
                last_hand_roll_age = 0
                # If this frame's roll wasn't captured in the approach cache
                # (hand not yet detected on the pre-raise frames), add it now
                # so we have at least one current-raise sample.
                if (len(palm_twists._samples) == 0
                        and palm_facing != "unknown"
                        and not is_fist
                        and hand_roll != 0.0):
                    palm_twists._samples.append(hand_roll)
                # Cross-raise snap baseline: prepend the roll from the previous
                # arm session so we can measure the delta even when the arm
                # only crosses the threshold for one frame.  If the user snapped
                # (wrist rotated between raises), the delta will exceed the
                # snap threshold.  Baseline expires after SNAP_ROLL_TTL_S.
                _now_raise = time.monotonic()
                if (len(palm_twists._samples) < 2
                        and _prev_raise_roll != 0.0
                        and _now_raise - _prev_raise_roll_time < SNAP_ROLL_TTL_S):
                    palm_twists._samples.appendleft(_prev_raise_roll)
                twist_swing, twist_dir = palm_twists.peak_swing()
                if palm_facing != "unknown" and not is_fist and hand_roll != 0.0:
                    last_hand_roll = hand_roll
                log.debug(
                    "[%s] arm raised (%s side) wrist=(%.0f,%.0f) "
                    "[seeded %d twist frames → swing=%.3f  "
                    "%d vel frames → pvx=%.1f]",
                    self.label, raised_side, wx, wy,
                    _n_seed, twist_swing,
                    len(_v_hist), last_peak_vx,
                )
                arm_raised_since = time.monotonic()
            else:
                # ── Subsequent arm-raised frames ──────────────────────────
                # Update trackers normally.  Fist→open transitions produce a
                # large hand_roll swing that looks like a snap; reset on
                # every fist state change to prevent those false fires.
                if is_fist != last_is_fist:
                    palm_twists.reset()
                    _approach_cache.clear()
                    last_hand_roll = 0.0
                    last_hand_roll_age = 0
                last_is_fist = is_fist
                if palm_facing != "unknown" and not is_fist:
                    # Fresh valid sample — update normally and record it.
                    last_hand_roll = hand_roll
                    last_hand_roll_age = 0
                    palm_twists.update(hand_roll)
                elif not is_fist and last_hand_roll != 0.0 and last_hand_roll_age < MAX_ROLL_CARRY:
                    # palm=unknown for a brief gap (Hands missed a frame).
                    # Carry the last known roll so the swing window stays
                    # continuous; the next valid frame will correct it.
                    last_hand_roll_age += 1
                    palm_twists.update(last_hand_roll)
                else:
                    last_hand_roll_age += 1
                twist_swing, twist_dir = palm_twists.peak_swing()
                # Track wrist pixel position for wave detection
                wrist_tracker.update(wx, wy)
                last_peak_vx, _ = wrist_tracker.peak_velocity()
            last_arm_raised = True

            # Reset stale gesture state if arm held still for too long
            now = time.monotonic()
            if now - arm_raised_since > ARM_HELD_TIMEOUT_S:
                palm_twists.reset()
                wrist_tracker.reset()
                last_peak_vx = 0.0
                _approach_cache.clear()
                sustain_counts = {g: 0 for g in Gesture}
                arm_raised_since = now
                log.debug("[%s] arm held still — resetting gesture state", self.label)

            # ── 3. Pick candidate ─────────────────────────────────────────
            candidate = _pick_candidate(
                is_fist, last_peak_vx, self.s
            )

            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "[%s] arm up | wrist=(%.0f,%.0f) fist=%s "
                    "roll=%.3f twist=%.3f/%s pvx=%.1f candidate=%s sustain=%s",
                    self.label, wx, wy,
                    is_fist, hand_roll,
                    twist_swing, twist_dir, last_peak_vx,
                    str(candidate) if candidate else "none",
                    {g.name: sustain_counts[g] for g in Gesture
                     if sustain_counts[g] > 0},
                )

            # ── 4. Sustain ────────────────────────────────────────────────
            for g in Gesture:
                if g == candidate:
                    sustain_counts[g] += 1
                else:
                    sustain_counts[g] = max(0, sustain_counts[g] - 1)

            needed = SUSTAIN_NEEDED.get(candidate, 1) if candidate else 1

            if candidate and sustain_counts.get(candidate, 0) >= needed:
                confidence = min(1.0, sustain_counts[candidate] / (needed * 2))
                log.debug("[%s] FIRING %s conf=%.2f", self.label, candidate, confidence)
                self.on_candidate(candidate, confidence, self.label)
                sustain_counts = {g: 0 for g in Gesture}
                palm_twists.reset()
                wrist_tracker.reset()
                last_peak_vx = 0.0
                consecutive_arm_raised = 0

            # ── Debug overlay ─────────────────────────────────────────────
            if self.debug_frame_cb is not None:
                try:
                    dbg = _draw_debug(
                        proc_frame, pose_res,
                        wrist_xy if arm_raised else None,
                        twist_swing, last_peak_vx, is_fist, candidate,
                        consecutive_arm_raised, ARM_RAISE_MIN_FRAMES,
                    )
                    self.debug_frame_cb(self.label, dbg)
                except Exception as e:
                    log.debug("[%s] debug render error: %s", self.label, e)

        pose.close()
        hands.close()
        log.info("[%s] Analyser stopped", self.label)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _arm_above_head(
    pose_res, settings, frame_w: int, frame_h: int
) -> tuple[bool, tuple[float, float], str]:
    """
    Returns (raised, (wrist_x, wrist_y), side).

    Arm-raise check for a FRONTAL camera: wrist clearly above shoulder.

    The previous two-part check (elbow above shoulder AND wrist above elbow)
    was designed for overhead cameras where the arm foreshortens badly.
    For a frontal eye-level mount the elbow rarely clears the shoulder in
    MediaPipe's normalised Y coordinates even during a full arm raise —
    the elbow swings out sideways while rising.  Checking only the wrist
    vs. the shoulder is simpler, more stable, and correct for frontal use.

    arm_above_head_tolerance:
      0.02 = wrist must be 2 % of frame height above shoulder (permissive)
      0.05 = 5 % above shoulder (stricter)
      0.00 = any wrist-above-shoulder position counts
    """
    if not pose_res or not pose_res.pose_landmarks:
        return False, (0.0, 0.0), ""

    import mediapipe as mp
    lm  = pose_res.pose_landmarks.landmark
    PL  = mp.solutions.pose.PoseLandmark

    margin = float(settings.arm_above_head_tolerance)

    # ── Body-upright check ────────────────────────────────────────────────
    # Hips must be sufficiently below shoulders, confirming the person is
    # sitting or standing rather than lying down.  When horizontal (couch),
    # hip_y ≈ shoulder_y so the threshold is not met and we bail early.
    # When upright, hips are typically 0.2–0.4 below shoulders.
    upright_min = float(getattr(settings, 'body_upright_min', 0.10))
    avg_sh_y = (lm[PL.LEFT_SHOULDER].y + lm[PL.RIGHT_SHOULDER].y) / 2.0
    avg_hp_y = (lm[PL.LEFT_HIP].y     + lm[PL.RIGHT_HIP].y)     / 2.0
    if avg_hp_y < avg_sh_y + upright_min:
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "  [arm-check] body not upright — hips=%.3f shoulders=%.3f "
                "(need gap≥%.2f, got %.3f)",
                avg_hp_y, avg_sh_y, upright_min, avg_hp_y - avg_sh_y,
            )
        return False, (0.0, 0.0), ""

    pairs = [
        ("RIGHT", PL.RIGHT_SHOULDER, PL.RIGHT_WRIST),
        ("LEFT",  PL.LEFT_SHOULDER,  PL.LEFT_WRIST),
    ]

    for side, sh_id, wr_id in pairs:
        sh = lm[sh_id]
        wr = lm[wr_id]

        # Wrist must be above the shoulder by at least `margin` (frame fraction).
        # In MediaPipe normalised coords y=0 is the top of the frame, so
        # "above" means a smaller y value.
        wrist_above_shoulder = wr.y < (sh.y - margin)

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "  [arm-check] %s sh_y=%.3f wr_y=%.3f margin=%.3f "
                "→ wrist_up=%s (sh-wr gap=%.3f)",
                side, sh.y, wr.y, margin,
                wrist_above_shoulder, sh.y - wr.y,
            )

        if wrist_above_shoulder:
            return True, (wr.x * frame_w, wr.y * frame_h), side

    return False, (0.0, 0.0), ""



def _classify_hand_full(
    hand_res, settings
) -> tuple[bool, str, float, float]:
    """
    Returns (is_fist, palm_facing, hand_roll, confidence).

    palm_facing:
      'camera'  — hand detected (orientation not used for gestures)
      'unknown' — hand not detected

    hand_roll: 2D orientation of the knuckle line.
      = (index_mcp.x − pinky_mcp.x) / distance(index_mcp, pinky_mcp)
      Ranges −1 to +1; sign flips when palm snaps front-to-back.
      Used by PalmTwistTracker to detect rapid rotations without needing
      z-depth (which is unreliable at frontal camera angles).
    """
    if not hand_res or not hand_res.multi_hand_landmarks:
        return False, "unknown", 0.0, 0.0

    lm = hand_res.multi_hand_landmarks[0].landmark

    # ── Fist detection (3-D distance, rotation-invariant) ────────────────
    # Measures wrist→tip vs wrist→MCP distance in 3D landmark space.
    # When a finger is extended the tip is 2–3× further from the wrist
    # than the MCP.  When curled into a fist the tip folds back and ends
    # up ≤ MCP distance from the wrist.
    #
    # Rotation-invariant: correct whether the palm faces the camera, faces
    # up, or faces sideways.  The old Y-position check (`tip.y > mcp.y`)
    # misfired when the palm faced upward because extended fingertips are
    # displaced vertically in image space.
    TIPS = [8, 12, 16, 20]
    MCPS = [5,  9, 13, 17]

    wrist = lm[0]

    def _d3(a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5

    curled = 0
    for tip_i, mcp_i in zip(TIPS, MCPS):
        tip_d = _d3(lm[tip_i], wrist)
        mcp_d = _d3(lm[mcp_i], wrist)
        # Curled if tip is not significantly further from wrist than MCP.
        # Factor 1.1 = tip can be up to 10% further than MCP and still count.
        # Extended finger: tip_d ≈ 2–3× mcp_d (well outside the 1.1 band).
        # Tight fist:      tip_d ≈ 0.5–0.8× mcp_d (well inside the band).
        if tip_d < mcp_d * 1.1:
            curled += 1

    frac    = curled / 4.0
    is_fist = frac >= settings.fist_curl_threshold
    conf    = frac if is_fist else (1.0 - frac)

    # ── 2D hand roll (snap detection) ────────────────────────────────────
    # The lateral angle of the index-to-pinky MCP line in image space.
    # Works at any camera angle — no z-depth required.
    # A wrist snap flips this sign rapidly → large peak_swing in tracker.
    ix, iy = lm[5].x, lm[5].y    # index finger MCP
    px, py = lm[17].x, lm[17].y  # pinky finger MCP
    dx = ix - px
    dy = iy - py
    dist = (dx * dx + dy * dy) ** 0.5
    hand_roll = dx / dist if dist > 0.01 else 0.0

    return is_fist, "camera", hand_roll, conf


def _pick_candidate(
    is_fist: bool,
    peak_vx: float,
    settings,
) -> Optional[Gesture]:
    """
    Map one frame's hand state to a gesture candidate.

    Priority:
      1. FIST            — closed fist (static, highest priority)
      2. WAVE_LEFT/RIGHT — deliberate horizontal sweep exceeding wave_velocity_threshold_px
      3. SNAP            — default for any other non-fist raise

    Snap is the default because the wrist-velocity signal (Pose landmarks) is
    always available and tends to read ~20–30 px even for relatively still raises,
    meaning a low wave threshold causes constant false waves.  By making wave the
    exceptional case (requiring clearly intentional horizontal velocity) and snap
    the fallback, most calm raises correctly fire as snap.
    """
    wave_thresh = float(getattr(settings, 'wave_velocity_threshold_px', 35.0))
    mirror      = bool(getattr(settings,  'mirror_camera', True))

    if is_fist:
        return Gesture.FIST

    # WAVE: only when there is a clear, deliberate horizontal sweep.
    if abs(peak_vx) >= wave_thresh:
        going_right = peak_vx > 0
        if mirror:
            return Gesture.WAVE_LEFT if going_right else Gesture.WAVE_RIGHT
        else:
            return Gesture.WAVE_RIGHT if going_right else Gesture.WAVE_LEFT

    # SNAP: everything else (calm raise, palm turn, quick flick below wave threshold)
    return Gesture.SNAP


# ── Debug overlay ─────────────────────────────────────────────────────────────

def _draw_debug(frame, pose_res, wrist_xy,
                twist_swing, peak_vx, is_fist,
                candidate, consec_raised, min_frames):
    """Draw skeleton + gesture state overlay onto a copy of frame."""
    import mediapipe as mp
    img = frame.copy()
    h, w = img.shape[:2]

    # Draw pose skeleton — or a bright warning if pose was not detected
    if pose_res and pose_res.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            img,
            pose_res.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 200, 0), thickness=2),
        )
    else:
        # Pose model did not detect a body — make this very obvious
        msg = "NO POSE DETECTED"
        fs = max(0.6, w / 800)
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        tx = (w - tw) // 2
        ty = 60
        cv2.rectangle(img, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6),
                      (0, 0, 180), -1)
        cv2.putText(img, msg, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, msg, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)

    # Wrist marker
    if wrist_xy:
        wx, wy = int(wrist_xy[0]), int(wrist_xy[1])
        color = (255, 255, 0) if consec_raised >= min_frames else (0, 128, 255)
        cv2.circle(img, (wx, wy), 12, color, -1)
        cv2.circle(img, (wx, wy), 12, (0, 0, 0), 2)

    # Status panel — bottom-left, scaled to frame size
    ready = consec_raised >= min_frames
    arm_state = "ARM READY" if ready else f"warm {consec_raised}/{min_frames}"
    cand_str = str(candidate) if candidate else "none"
    lines = [
        (arm_state,                               (0, 255, 100) if ready else (0, 165, 255)),
        (f"twist={twist_swing:.3f}",              (255, 255, 255)),
        (f"fist={is_fist}  pvx={peak_vx:.1f}",   (255, 255, 255)),
        (f"cand: {cand_str}",                     (255, 255, 0) if candidate else (160, 160, 160)),
    ]
    fs = max(0.35, w / 1800)          # font scale relative to frame width
    lh = int(fs * 42)
    pad = int(fs * 12)
    panel_h = len(lines) * lh + pad * 2
    panel_w = int(w * 0.30)
    y_start = h - panel_h - 6
    overlay = img.copy()
    cv2.rectangle(overlay, (4, y_start - 2), (4 + panel_w, y_start + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, (line, color) in enumerate(lines):
        y = y_start + pad + i * lh + lh - 4
        cv2.putText(img, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)

    return img
