"""
cameras/analyser.py  — Endora v1.7.22

Hybrid gesture detection: MediaPipe Pose + Hands.

Pipeline per frame:
  1. (Optional) Fisheye dewarp → crop → CLAHE
  2. Pose  → arm raised above head? (wrist above shoulder + margin)
  3. Hands → open vs fist
  4. Classify:
       FIST          — closed fist
       SNAP          — elbow above shoulder (arm raised straight up)
       WAVE_LEFT/RIGHT — elbow below shoulder (arm extended sideways)
  5. Sustain N frames → fire → cooldown
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto
from typing import Callable, Optional

import cv2

log = logging.getLogger(__name__)


# ── Gesture enum ──────────────────────────────────────────────────────────────

class Gesture(Enum):
    SNAP       = auto()  # arm raised straight up, wrist snap
    FIST       = auto()  # arm raised, closed fist (static)
    WAVE_LEFT  = auto()  # arm extended sideways, sweep left
    WAVE_RIGHT = auto()  # arm extended sideways, sweep right

    @property
    def event_name(self) -> str:
        """HA event data value, e.g. 'endora-wave-left'."""
        return f"endora-{self.name.lower().replace('_', '-')}"

    def __str__(self) -> str:
        return self.event_name


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
        # CLAHE cache — object is expensive to create; recreate only when
        # the clip value changes rather than on every frame.
        self._clahe_obj  = None
        self._clahe_clip: float = -1.0

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
            # static_image_mode=True: full detection on every crop (no tracking
            # across crops — the crop is re-centred every frame so tracking breaks).
            static_image_mode=True,
        )

        # ── Gesture state ─────────────────────────────────────────────────
        sustain_counts: dict[Gesture, int] = {g: 0 for g in Gesture}
        # Frames each gesture must appear consecutively before firing.
        # At ~9 fps the arm is often only detected for 1 frame per raise, so
        # SUSTAIN_NEEDED=1 across the board.  Discrimination relies on the
        # gesture classifier, not on sustain count.
        SUSTAIN_NEEDED: dict[Gesture, int] = {
            Gesture.SNAP:       1,
            Gesture.FIST:       1,
            Gesture.WAVE_LEFT:  1,
            Gesture.WAVE_RIGHT: 1,
        }

        last_arm_raised    = False
        arm_must_reset     = False   # True after a gesture fires; blocks re-fire
        _last_gesture_time: float = 0.0

        consecutive_no_pose    = 0
        NO_POSE_TOLERANCE      = 4   # arm-down frames before resetting state
        consecutive_arm_raised = 0
        ARM_RAISE_MIN_FRAMES   = 1   # warm-up frames before gestures fire
        arm_raised_since: float = 0.0
        ARM_HELD_TIMEOUT_S     = 10.0  # reset after arm held motionless this long

        # Anti-furniture / idle-lock reset
        _furniture_rejection_streak = 0
        FURNITURE_RESET_FRAMES = 12
        IDLE_RESET_S   = 60.0
        _last_arm_up_time: float = 0.0
        _idle_reset_done: bool   = False

        # Per-frame classification signals (safe defaults)
        _forearm_dy_norm: float = 0.0  # elbow_y_norm - wrist_y_norm; >0 = wrist above elbow
        _elbow_gap_norm: float  = 0.0  # shoulder_y_norm - elbow_y_norm (debug/log only)
        _wave_dx: float         = 0.0
        _is_fist: bool          = False

        log.info("[%s] Analyser running (v1.7.22 — one-arm gate + forearm classifier)", self.label)

        while not self._stop_evt.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # ── Optional fisheye dewarping ────────────────────────────────
            # Converts raw equidistant fisheye → flat perspective.
            # Maps are built once per unique parameter set (lazy-init).
            if getattr(self.s, 'dewarp_enable', False):
                from cameras.dewarp import build_dewarp_maps, apply_dewarp
                cx_raw = float(getattr(self.s, 'dewarp_cx',          -1.0))
                cy_raw = float(getattr(self.s, 'dewarp_cy',          -1.0))
                dw     = int(getattr(self.s,   'dewarp_out_width',    640))
                dh     = int(getattr(self.s,   'dewarp_out_height',   480))
                fov    = float(getattr(self.s, 'dewarp_fov',        180.0))
                pan    = float(getattr(self.s, 'dewarp_pan',          0.0))
                tilt   = float(getattr(self.s, 'dewarp_tilt',        20.0))
                roll   = float(getattr(self.s, 'dewarp_roll',         0.0))
                vfov   = float(getattr(self.s, 'dewarp_vfov',        75.0))
                _key   = (w, h, dw, dh, fov, pan, tilt, roll, vfov, cx_raw, cy_raw)
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

            # ── Optional 180° image flip ─────────────────────────────────
            # Rotates the frame 180° — useful for cameras mounted upside-down.
            # Equivalent to flipping both axes.  Runs after dewarping so the
            # dewarp geometry is still correct.
            if getattr(self.s, 'flip_image', False):
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                h, w = frame.shape[:2]

            # ── Optional asymmetric crop ──────────────────────────────────
            ct = float(getattr(self.s, 'frame_crop_top',    0))
            cb = float(getattr(self.s, 'frame_crop_bottom', 0))
            cl = float(getattr(self.s, 'frame_crop_left',   0))
            cr = float(getattr(self.s, 'frame_crop_right',  0))
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
            else:
                proc_frame = frame
                ph, pw = h, w

            # ── Optional CLAHE low-light enhancement ──────────────────────
            # CLAHE object is cached and recreated only when clip changes.
            if getattr(self.s, 'low_light_enhance', False):
                clip = float(getattr(self.s, 'low_light_clip', 2.0))
                if clip != self._clahe_clip:
                    self._clahe_obj  = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
                    self._clahe_clip = clip
                lab = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                l_ch = self._clahe_obj.apply(l_ch)
                proc_frame = cv2.cvtColor(
                    cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR
                )

            rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_res = pose.process(rgb)
            rgb.flags.writeable = True

            # ── Furniture / false-pose filter ─────────────────────────────
            # MediaPipe assigns near-zero visibility to furniture false-detections
            # and reasonable scores (~0.5–1.0) to real people.  After
            # FURNITURE_RESET_FRAMES consecutive rejections the model is recreated
            # to break the tracking lock — otherwise MediaPipe keeps re-latching
            # onto the same object.
            if pose_res and pose_res.pose_landmarks:
                _lm_f = pose_res.pose_landmarks.landmark
                _PL_f = mp.solutions.pose.PoseLandmark
                _vis_scores = [
                    _lm_f[_PL_f.LEFT_SHOULDER].visibility,
                    _lm_f[_PL_f.RIGHT_SHOULDER].visibility,
                ]
                _avg_vis = sum(_vis_scores) / len(_vis_scores)
                _min_vis = float(getattr(self.s, 'pose_visibility_min', 0.35))
                if _avg_vis < _min_vis:
                    _furniture_rejection_streak += 1
                    log.debug(
                        "[%s] pose rejected: avg_shoulder_vis=%.2f < min=%.2f "
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
                    pose_res = None
                else:
                    _furniture_rejection_streak = 0

            # ── 1. Arm raised above head? ─────────────────────────────────
            arm_raised, wrist_xy, raised_side = _arm_above_head(
                pose_res, self.s, pw, ph
            )

            # ── Arm-down path ─────────────────────────────────────────────
            if not arm_raised:
                consecutive_no_pose += 1
                if consecutive_no_pose >= NO_POSE_TOLERANCE:
                    if last_arm_raised:
                        log.debug("[%s] arm lowered — resetting sustain counts", self.label)
                        for g in Gesture:
                            sustain_counts[g] = 0
                    last_arm_raised = False
                    # Clear arm_must_reset only once the full cooldown has elapsed.
                    # This prevents re-fire when the arm flickers up/down after a gesture.
                    if arm_must_reset:
                        _elapsed = time.monotonic() - _last_gesture_time
                        if _elapsed >= self.s.cooldown_s:
                            arm_must_reset = False
                            log.debug("[%s] arm_must_reset cleared (%.1fs elapsed)",
                                      self.label, _elapsed)
                        else:
                            log.debug("[%s] arm_must_reset held (%.1f / %.1fs cooldown)",
                                      self.label, _elapsed, self.s.cooldown_s)

                if log.isEnabledFor(logging.DEBUG) and consecutive_no_pose % 10 == 1:
                    if not (pose_res and pose_res.pose_landmarks):
                        log.debug("[%s] NO POSE DETECTED — body not found in frame", self.label)
                    else:
                        log.debug("[%s] arm not raised (pose OK, arm down)", self.label)

                # Idle-lock breaker: after IDLE_RESET_S with no arm raised,
                # recreate the pose model so a different person can be detected.
                now_idle = time.monotonic()
                if _last_arm_up_time == 0.0:
                    _last_arm_up_time = now_idle
                if now_idle - _last_arm_up_time > IDLE_RESET_S and not _idle_reset_done:
                    log.info(
                        "[%s] Idle %.0fs — recreating pose model for fresh detection",
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

                if self.debug_frame_cb is not None:
                    try:
                        dbg = _draw_debug(
                            proc_frame, pose_res, None,
                            0.0, 0.0, 0.0, False, None,
                            consecutive_arm_raised, ARM_RAISE_MIN_FRAMES,
                        )
                        self.debug_frame_cb(self.label, dbg)
                    except Exception as e:
                        log.debug("[%s] debug render error: %s", self.label, e)
                continue

            # ── Arm is up ─────────────────────────────────────────────────
            consecutive_no_pose = 0
            consecutive_arm_raised += 1
            wx, wy = wrist_xy
            _last_arm_up_time = time.monotonic()
            _idle_reset_done  = False

            # Warm-up: discard the first ARM_RAISE_MIN_FRAMES to rule out
            # phantom detections at the moment of raise.
            if consecutive_arm_raised < ARM_RAISE_MIN_FRAMES:
                last_arm_raised = True
                if self.debug_frame_cb is not None:
                    try:
                        dbg = _draw_debug(
                            proc_frame, pose_res, wrist_xy,
                            _forearm_dy_norm, _elbow_gap_norm, _wave_dx,
                            _is_fist, None,
                            consecutive_arm_raised, ARM_RAISE_MIN_FRAMES,
                        )
                        self.debug_frame_cb(self.label, dbg)
                    except Exception as e:
                        log.debug("[%s] debug render error: %s", self.label, e)
                continue

            # arm_must_reset: require arm-down + cooldown before next gesture.
            # Force-clear after 3× cooldown so the user is never permanently locked
            # out if the arm flicker didn't reach NO_POSE_TOLERANCE frames.
            if arm_must_reset:
                _elapsed = time.monotonic() - _last_gesture_time
                if _elapsed >= self.s.cooldown_s * 3.0:
                    arm_must_reset = False
                    log.debug("[%s] arm_must_reset force-cleared (%.1fs elapsed)",
                              self.label, _elapsed)
                else:
                    last_arm_raised = True
                    log.debug("[%s] arm_must_reset — waiting for arm to lower (%.1fs / %.1fs)",
                              self.label, _elapsed, self.s.cooldown_s)
                    # Render debug so the stream doesn't freeze
                    if self.debug_frame_cb is not None:
                        try:
                            dbg = _draw_debug(
                                proc_frame, pose_res, wrist_xy,
                                _elbow_gap_norm, _wave_dx, _is_fist, None,
                                consecutive_arm_raised, ARM_RAISE_MIN_FRAMES,
                            )
                            self.debug_frame_cb(self.label, dbg)
                        except Exception as e:
                            log.debug("[%s] debug render error: %s", self.label, e)
                    continue

            # ── 2. Forearm angle — snap vs wave signal ─────────────────────
            # _forearm_dy_norm = elbow_y_norm − wrist_y_norm
            #   (positive when wrist is ABOVE elbow in the image)
            #
            # SNAP (arm raised upward): forearm points up → wrist clearly
            #   above elbow → forearm_dy ≈ 0.08–0.18
            #
            # WAVE (arm extended sideways): forearm points outward → wrist
            #   at roughly elbow height or below → forearm_dy ≈ −0.05–0.04
            #
            # This works even when elbow_gap is large for BOTH gestures
            # (i.e. both snap and wave are performed with the arm elevated).
            # The forearm orientation is the most position-independent and
            # gesture-specific signal available from a single frame at 9 fps.
            _lm_g = pose_res.pose_landmarks.landmark
            _PL_g = mp.solutions.pose.PoseLandmark
            if raised_side == "RIGHT":
                _elbow_y    = _lm_g[_PL_g.RIGHT_ELBOW].y
                _wrist_y_n  = _lm_g[_PL_g.RIGHT_WRIST].y
                _sh_y_local = _lm_g[_PL_g.RIGHT_SHOULDER].y
            else:
                _elbow_y    = _lm_g[_PL_g.LEFT_ELBOW].y
                _wrist_y_n  = _lm_g[_PL_g.LEFT_WRIST].y
                _sh_y_local = _lm_g[_PL_g.LEFT_SHOULDER].y
            _forearm_dy_norm = _elbow_y - _wrist_y_n  # positive = wrist above elbow
            _elbow_gap_norm  = _sh_y_local - _elbow_y  # positive = elbow above shoulder (debug)

            # Wrist offset from body midline — used only for WAVE direction.
            _mid_x   = ((_lm_g[_PL_g.LEFT_SHOULDER].x + _lm_g[_PL_g.RIGHT_SHOULDER].x)
                        / 2.0) * pw
            _wave_dx = wx - _mid_x   # +ve = wrist right of centre

            if not last_arm_raised:
                log.debug(
                    "[%s] arm raised (%s) wrist=(%.0f,%.0f) "
                    "forearm_dy=%.3f elbow_gap=%.3f wave_dx=%.0f",
                    self.label, raised_side, wx, wy,
                    _forearm_dy_norm, _elbow_gap_norm, _wave_dx,
                )
                arm_raised_since = time.monotonic()

            last_arm_raised = True

            # Reset stale gesture state if arm held motionless for too long
            now = time.monotonic()
            if now - arm_raised_since > ARM_HELD_TIMEOUT_S:
                sustain_counts = {g: 0 for g in Gesture}
                arm_raised_since = now
                log.debug("[%s] arm held still — resetting gesture state", self.label)

            # ── 3. Hand shape ─────────────────────────────────────────────
            # Crop around the raised wrist and run Hands on it.
            # On a 1920-wide dewarped frame the hand is ~50 px across;
            # cropping makes the hand fill the frame and dramatically improves
            # MediaPipe's palm detector accuracy.
            #
            # Fist detection tip: MediaPipe's palm detector is trained on open
            # palms.  For a closed fist, shift the crop centre UPWARD by 60 px
            # from the Pose wrist landmark so the knuckles (higher in frame)
            # sit near the crop centre and the full fist silhouette is visible.
            # Larger crop radius (220 px) ensures the whole hand fits even when
            # the wrist estimate is slightly off.
            _wx_px = int(wx)
            _wy_px = int(wy) - 60   # shift centre up toward knuckles
            _ch    = 220            # half-side of crop in pixels
            _cx1   = max(0,  _wx_px - _ch)
            _cx2   = min(pw, _wx_px + _ch)
            _cy1   = max(0,  _wy_px - _ch)
            _cy2   = min(ph, _wy_px + _ch)
            _hands_crop = rgb[_cy1:_cy2, _cx1:_cx2]
            if _hands_crop.size > 0:
                _hands_rgb = cv2.resize(
                    _hands_crop, (256, 256),
                    interpolation=cv2.INTER_LINEAR,
                )
                hand_res = hands.process(_hands_rgb)
            else:
                hand_res = None

            _is_fist, _hand_conf = _classify_fist(hand_res, self.s)

            # ── 4. Pick candidate ─────────────────────────────────────────
            candidate = _pick_candidate(
                _is_fist, _forearm_dy_norm, _wave_dx, self.s
            )

            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "[%s] arm up | wrist=(%.0f,%.0f) fist=%s "
                    "forearm_dy=%.3f elbow_gap=%.3f wave_dx=%.0f → candidate=%s sustain=%s",
                    self.label, wx, wy,
                    _is_fist, _forearm_dy_norm, _elbow_gap_norm, _wave_dx,
                    str(candidate) if candidate else "none",
                    {g.name: sustain_counts[g] for g in Gesture
                     if sustain_counts[g] > 0},
                )

            # ── 5. Sustain ────────────────────────────────────────────────
            for g in Gesture:
                if g == candidate:
                    sustain_counts[g] += 1
                else:
                    sustain_counts[g] = max(0, sustain_counts[g] - 1)

            needed = SUSTAIN_NEEDED.get(candidate, 1) if candidate else 1

            if candidate and sustain_counts.get(candidate, 0) >= needed:
                confidence = min(1.0, sustain_counts[candidate] / max(1, needed * 2))
                log.debug("[%s] FIRING %s conf=%.2f", self.label, candidate, confidence)
                self.on_candidate(candidate, confidence, self.label)
                sustain_counts         = {g: 0 for g in Gesture}
                consecutive_arm_raised = 0
                arm_must_reset         = True
                _last_gesture_time     = time.monotonic()

            # ── Debug overlay ─────────────────────────────────────────────
            if self.debug_frame_cb is not None:
                try:
                    dbg = _draw_debug(
                        proc_frame, pose_res,
                        wrist_xy if arm_raised else None,
                        _forearm_dy_norm, _elbow_gap_norm, _wave_dx,
                        _is_fist, candidate,
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
    Returns (raised, (wrist_x_px, wrist_y_px), side).

    Arm-raise check for a frontal or fisheye-dewarped camera.
    Only the wrist vs. shoulder comparison is used — an elbow check is not
    needed here because the elbow position is used downstream to discriminate
    wave vs. snap, not as a gate for whether the arm is raised.

    arm_above_head_tolerance (frame fraction):
      0.05 = wrist must be 5 % of frame height above shoulder
      0.15 = stricter; prevents scratching-head from triggering (recommended)
    """
    if not pose_res or not pose_res.pose_landmarks:
        return False, (0.0, 0.0), ""

    import mediapipe as mp
    lm = pose_res.pose_landmarks.landmark
    PL = mp.solutions.pose.PoseLandmark

    margin = float(settings.arm_above_head_tolerance)

    # Body-upright guard: hips must be sufficiently below shoulders.
    # Blocks arm-raise detection when the person is lying down.
    # upright_min < 0 is correct for frontal dewarped fisheye cameras where
    # perspective places hips above shoulders in the image.
    upright_min = float(getattr(settings, 'body_upright_min', -0.15))
    avg_sh_y = (lm[PL.LEFT_SHOULDER].y + lm[PL.RIGHT_SHOULDER].y) / 2.0
    avg_hp_y = (lm[PL.LEFT_HIP].y      + lm[PL.RIGHT_HIP].y)     / 2.0
    if avg_hp_y < avg_sh_y + upright_min:
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "  [arm-check] body not upright — hips=%.3f shoulders=%.3f "
                "(need gap≥%.2f, got %.3f)",
                avg_hp_y, avg_sh_y, upright_min, avg_hp_y - avg_sh_y,
            )
        return False, (0.0, 0.0), ""

    # ── Both-arms guard ───────────────────────────────────────────────────
    # If BOTH wrists are above their respective shoulders, reject — it's a
    # two-handed pose (arms spread wide, T-pose, stretch, "big animal") not
    # a single-arm gesture.  Snap, wave, and fist are always one-handed.
    _rw_raised = lm[PL.RIGHT_WRIST].y < (lm[PL.RIGHT_SHOULDER].y - margin)
    _lw_raised = lm[PL.LEFT_WRIST].y  < (lm[PL.LEFT_SHOULDER].y  - margin)
    if _rw_raised and _lw_raised:
        if log.isEnabledFor(logging.DEBUG):
            log.debug("  [arm-check] both wrists raised — ignoring (two-handed pose)")
        return False, (0.0, 0.0), ""

    pairs = [
        ("RIGHT", PL.RIGHT_SHOULDER, PL.RIGHT_WRIST),
        ("LEFT",  PL.LEFT_SHOULDER,  PL.LEFT_WRIST),
    ]

    for side, sh_id, wr_id in pairs:
        sh = lm[sh_id]
        wr = lm[wr_id]
        # In MediaPipe normalised coords y=0 is the top of frame — "above"
        # means smaller y.  Wrist must clear shoulder by at least `margin`.
        wrist_above_shoulder = wr.y < (sh.y - margin)

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "  [arm-check] %s sh_y=%.3f wr_y=%.3f margin=%.3f "
                "→ wrist_up=%s (gap=%.3f)",
                side, sh.y, wr.y, margin,
                wrist_above_shoulder, sh.y - wr.y,
            )

        if wrist_above_shoulder:
            return True, (wr.x * frame_w, wr.y * frame_h), side

    return False, (0.0, 0.0), ""


def _classify_fist(hand_res, settings) -> tuple[bool, float]:
    """
    Returns (is_fist, confidence).

    Uses rotation-invariant 3D distance: tip vs MCP distance from wrist.
    Extended finger: tip_d ≈ 2–3× mcp_d.
    Curled  finger:  tip_d ≈ 0.5–0.8× mcp_d.

    fist_curl_threshold (0–1): fraction of 4 fingers that must be curled.
    0.85 = 3.4 of 4 fingers curled.
    """
    if not hand_res or not hand_res.multi_hand_landmarks:
        return False, 0.0

    lm = hand_res.multi_hand_landmarks[0].landmark

    TIPS = [8, 12, 16, 20]
    MCPS = [5,  9, 13, 17]
    wrist = lm[0]

    def _d3(a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5

    curled = 0
    for tip_i, mcp_i in zip(TIPS, MCPS):
        tip_d = _d3(lm[tip_i], wrist)
        mcp_d = _d3(lm[mcp_i], wrist)
        if tip_d < mcp_d * 1.1:
            curled += 1

    frac    = curled / 4.0
    is_fist = frac >= float(settings.fist_curl_threshold)
    conf    = frac if is_fist else (1.0 - frac)
    return is_fist, conf


def _pick_candidate(
    is_fist: bool,
    forearm_dy_norm: float,
    wave_dx: float,
    settings,
) -> Optional[Gesture]:
    """
    Map one frame's hand + arm state to a gesture candidate.

    Priority:
      1. FIST            — closed fist, any arm position
      2. SNAP            — forearm points upward (wrist clearly above elbow)
      3. WAVE_LEFT/RIGHT — forearm points sideways (wrist at/below elbow height)

    Discriminator: forearm_dy_norm = elbow_y_norm − wrist_y_norm
      (positive when wrist is ABOVE elbow in normalised coords)

    SNAP: arm raised straight up → forearm vertical → wrist high above elbow
          forearm_dy ≈ 0.08–0.18
    WAVE: arm swept sideways    → forearm horizontal → wrist at elbow height
          forearm_dy ≈ −0.05–0.04

    This works even when the elbow is elevated for BOTH gestures (high waves
    and snaps can both have elbow well above the shoulder).  The forearm
    angle is the most geometry-independent signal at 9 fps.

    snap_forearm_min (default 0.06): crossover threshold.
      Watch forearm_dy in the debug overlay:
        snap  ≈ 0.10+   wave ≈ 0.00 or negative
      Raise toward 0.10 if waves misfire as snap.
      Lower toward 0.03 if snaps misfire as wave.

    wave_dx (wrist − shoulder midline, pixels) sets WAVE_LEFT vs WAVE_RIGHT.
    """
    if is_fist:
        return Gesture.FIST

    snap_forearm_min = float(getattr(settings, 'snap_forearm_min', 0.06))

    if forearm_dy_norm >= snap_forearm_min:
        return Gesture.SNAP

    # Forearm roughly horizontal → WAVE
    mirror      = bool(getattr(settings, 'mirror_camera', False))
    going_right = wave_dx > 0
    if mirror:
        return Gesture.WAVE_LEFT if going_right else Gesture.WAVE_RIGHT
    else:
        return Gesture.WAVE_RIGHT if going_right else Gesture.WAVE_LEFT


# ── Debug overlay ─────────────────────────────────────────────────────────────

def _draw_debug(
    frame, pose_res, wrist_xy,
    forearm_dy_norm, elbow_gap_norm, wave_dx, is_fist,
    candidate, consec_raised, min_frames,
):
    """Draw skeleton + gesture state overlay onto a copy of frame."""
    import mediapipe as mp
    img = frame.copy()
    h, w = img.shape[:2]

    # Pose skeleton
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

    # Status panel
    ready    = consec_raised >= min_frames
    arm_state = "ARM READY" if ready else f"warm {consec_raised}/{min_frames}"
    cand_str  = str(candidate) if candidate else "none"
    lines = [
        (arm_state,                                   (0, 255, 100) if ready else (0, 165, 255)),
        (f"forearm_dy={forearm_dy_norm:.3f}  eg={elbow_gap_norm:.3f}", (255, 255, 255)),
        (f"fist={is_fist}  wave_dx={wave_dx:.0f}",                    (255, 255, 255)),
        (f"cand: {cand_str}",                         (255, 255, 0) if candidate else (160, 160, 160)),
    ]
    fs      = max(0.35, w / 1800)
    lh      = int(fs * 42)
    pad     = int(fs * 12)
    panel_h = len(lines) * lh + pad * 2
    panel_w = int(w * 0.32)
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
