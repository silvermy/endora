"""
cameras/arm_tracker.py

Pure classifier: pose landmarks → ArmState.

No pose-backend dependency in the classifier itself — it takes a dict-like
landmarks object with x/y/visibility per landmark index, so tests can pass
fake landmarks without installing any inference library.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Protocol

import numpy as np


class ArmState(Enum):
    DOWN        = auto()  # neither arm raised
    SINGLE_UP   = auto()  # one arm raised above head
    BOTH_UP     = auto()  # both arms raised above head
    T_POSE      = auto()  # both arms extended horizontal
    CROSS_ARMS  = auto()  # arms crossed in front of chest


class Side(Enum):
    LEFT  = auto()
    RIGHT = auto()


@dataclass
class ArmReading:
    """Output of ArmTracker.classify()."""
    state: ArmState
    # For SINGLE_UP only: which arm is raised + wrist pixel location.
    raised_side: Optional[Side] = None
    wrist_x: float = 0.0
    wrist_y: float = 0.0
    # Forearm verticality (positive = wrist above elbow in frame).
    # Only meaningful when state == SINGLE_UP.
    forearm_dy: float = 0.0
    # True if the body pose is upright enough to trust (hips below shoulders).
    upright: bool = True
    # Palm roll from grlib hand landmarks: (index_mcp.x - pinky_mcp.x) / hand_width.
    # Ranges roughly -1 to +1; only populated when hand_lm is provided to classify().
    snap_roll: float = 0.0
    # True when a wrist is approaching shoulder level but not yet classified
    # as SINGLE_UP — used to fire the chime early to compensate for speaker latency.
    arm_rising: bool = False


# ── Landmark protocol for type hints ──────────────────────────────────────────

class _Point(Protocol):
    x: float
    y: float
    visibility: float


class _Landmarks(Protocol):
    def __getitem__(self, idx: int) -> _Point: ...


# Landmark indices used by ArmTracker.  These match MediaPipe PoseLandmark
# values; the YOLO backend remaps COCO indices to these before calling classify().
NOSE           = 0
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_KNEE      = 25
RIGHT_KNEE     = 26


def _hand_snap_roll(hand_lm: np.ndarray) -> float:
    """Compute palm roll from a flat grlib hand-landmark array (1 hand, 63 floats).

    roll = (index_mcp.x - pinky_mcp.x) / hand_width
    MediaPipe hand indices: INDEX_FINGER_MCP=5, PINKY_MCP=17.
    Ranges roughly -1 to +1; sign depends on which hand is raised.
    """
    if len(hand_lm) < 21 * 3:
        return 0.0
    idx_mcp_x = float(hand_lm[5 * 3])    # INDEX_FINGER_MCP.x
    pnk_mcp_x = float(hand_lm[17 * 3])   # PINKY_MCP.x
    hand_w = abs(idx_mcp_x - pnk_mcp_x)
    if hand_w < 1e-6:
        return 0.0
    return (idx_mcp_x - pnk_mcp_x) / hand_w


# ── Tracker ───────────────────────────────────────────────────────────────────

@dataclass
class ArmTrackerConfig:
    """Thresholds for arm-state classification. All values are frame fractions."""
    arm_above_head_tolerance: float = 0.15
    # Wrist within this distance BELOW shoulder level triggers arm_rising=True
    # for early chime firing (compensates for speaker network latency).
    arm_rising_threshold: float = 0.05
    # When body is NOT upright (reclined/lying down), OR when upright status
    # cannot be determined (hips hidden by blanket), the wrist must clear the
    # shoulder by this larger margin.  Requires a deliberate straight-up arm.
    arm_above_head_tolerance_reclined: float = 0.30
    # Leg-raise guard: if both knees are this far above the average shoulder y
    # (frame fraction, y increases downward so knee_y < shoulder_y means higher),
    # suppress all gesture detection.  Catches legs-in-V while lying down.
    leg_raise_margin: float = 0.05
    body_upright_min: float = -0.15
    # Furniture/ghost rejection: at least ONE shoulder must exceed this
    # confidence.  Uses max (not average) so a real person with one shoulder
    # hidden by a blanket, cushion, or side-on pose is not wrongly rejected.
    pose_visibility_min: float = 0.55
    # Per-keypoint confidence below which a single landmark (shoulder, wrist,
    # elbow) is treated as not-visible.  Drives per-side arm-raise gating so a
    # garbage/occluded keypoint can neither create a false raise nor block a
    # real one on the opposite, visible side.
    keypoint_visibility_min: float = 0.30
    # Forearm-vertical secondary route to SINGLE_UP: if the forearm is at least
    # this vertical (elbow_y - wrist_y, frame fraction) and the wrist is at or
    # above shoulder height, count the arm as raised even when it doesn't clear
    # the full arm_above_head_tolerance.  Makes detection robust to camera
    # angles (e.g. ceiling/high-mount) where a raised arm's wrist doesn't travel
    # far above the shoulder in image space.  Only applies when upright is not
    # confirmed-reclined.  Raise toward 0.15 if hand-near-head misfires.
    forearm_vertical_min: float = 0.10

    # Wrist-near-head exclusion: a raised wrist within this distance (frame
    # fraction, both axes) of the nose keypoint is rejected even if it
    # otherwise passes the raise checks above. Resting/adjusting a hand
    # against your own face (glasses, phone, scratching, chin-on-hand) reads
    # geometrically identical to a raised arm — wrist above shoulder, forearm
    # vertical — but a deliberate raise/snap holds the hand up and away from
    # the head. Only applied when the nose keypoint is confidently visible,
    # so an occluded face never blocks a real raise. Lower toward 0.05 if it
    # ever rejects genuine close-to-head raises; raise toward 0.12 if
    # hand-near-face is still misfiring.
    wrist_head_exclude_dist: float = 0.09

    # Both-arms-up: each wrist must clear its shoulder by this margin.
    both_arms_margin: float = 0.10

    # T-pose: wrists roughly at shoulder height AND clearly lateral from body.
    # wrist.y within this band of shoulder.y (frame fraction) counts as "at shoulder height".
    tpose_wrist_y_band: float = 0.15
    # Wrists must be this far lateral from body midline (frame fraction of width).
    tpose_lateral_min: float = 0.13

    # Cross-arms:
    #   - Each wrist must cross the body midline by min_crossing.
    #   - Wrists must be close together (chest-clasp OR actual crossed position).
    # 0.03 was too sensitive — typing/working with hands in front fires it.
    # 0.08 requires a more deliberate crossing (~8% of frame width past midline).
    cross_arms_min_crossing: float = 0.08
    cross_arms_wrist_proximity: float = 0.22

    # ── Hysteresis ─────────────────────────────────────────────────────────
    # Seconds a new non-DOWN state must be seen before being accepted.
    # Kills single-frame phantoms without adding noticeable latency.
    state_confirm_s: float = 0.20
    # Seconds of contradictory frames before releasing a stable state back to
    # DOWN. Allows brief keypoint dropouts mid-gesture without resetting.
    state_release_s: float = 0.30


class ArmTracker:
    def __init__(self, config: ArmTrackerConfig):
        self.c = config
        self._stable_reading: Optional[ArmReading] = None
        self._pending_state: Optional[ArmState] = None
        self._pending_since: float = 0.0

    def classify(
        self,
        landmarks: Optional[_Landmarks],
        frame_w: int,
        frame_h: int,
        hand_lm: Optional[np.ndarray] = None,
        now: Optional[float] = None,
    ) -> Optional[ArmReading]:
        """Public entry point — time-based hysteresis + optional grlib snap_roll.

        hand_lm: flat numpy array from grlib Pipeline (21*3 floats, 1 hand).
        snap_roll is attached to SINGLE_UP readings only.
        now: monotonic timestamp; defaults to time.monotonic().
        """
        if now is None:
            now = time.monotonic()
        result = self._hyst_classify(landmarks, frame_w, frame_h, now)
        if result is not None and result.state == ArmState.SINGLE_UP and hand_lm is not None:
            result = dataclasses.replace(result, snap_roll=_hand_snap_roll(hand_lm))
        return result

    def _hyst_classify(self, landmarks: Optional[_Landmarks],
                       frame_w: int, frame_h: int, now: float) -> Optional[ArmReading]:
        """Time-based hysteresis: a new state must persist for state_confirm_s
        seconds before being accepted; a stable state requires state_release_s
        seconds of contradictory frames before being released.
        """
        raw = self._classify_raw(landmarks, frame_w, frame_h)
        raw_state = raw.state if raw is not None else ArmState.DOWN

        if self._stable_reading is None:
            if raw_state == ArmState.DOWN:
                self._stable_reading = raw
                self._pending_state = None
                return raw

            if self._pending_state != raw_state:
                self._pending_state = raw_state
                self._pending_since = now

            if (now - self._pending_since) >= self.c.state_confirm_s:
                self._stable_reading = raw
                self._pending_state = None
                return raw

            return ArmReading(state=ArmState.DOWN, arm_rising=raw.arm_rising if raw else False)

        stable_state = self._stable_reading.state
        if raw_state == stable_state:
            self._stable_reading = raw if raw is not None else self._stable_reading
            self._pending_state = None
            return self._stable_reading

        if self._pending_state != raw_state:
            self._pending_state = raw_state
            self._pending_since = now

        needed = self.c.state_release_s if raw_state == ArmState.DOWN else self.c.state_confirm_s

        if (now - self._pending_since) >= needed:
            self._stable_reading = raw if raw is not None else ArmReading(state=ArmState.DOWN)
            self._pending_state = None
            return self._stable_reading

        return self._stable_reading

    def _classify_raw(self, landmarks: Optional[_Landmarks],
                      frame_w: int, frame_h: int) -> Optional[ArmReading]:
        """
        Classify pose landmarks into an ArmReading.
        Returns None if landmarks are missing or visibility is too low.
        """
        if landmarks is None:
            return None

        ls, rs = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
        lh, rh = landmarks[LEFT_HIP], landmarks[RIGHT_HIP]
        le, re = landmarks[LEFT_ELBOW], landmarks[RIGHT_ELBOW]
        lw, rw = landmarks[LEFT_WRIST],  landmarks[RIGHT_WRIST]
        nose = landmarks[NOSE]

        KV = self.c.keypoint_visibility_min

        # Furniture/ghost rejection — require at least ONE confident shoulder.
        # Using max (not average) so a real person with one shoulder hidden by a
        # blanket, cushion, or side-on pose is not rejected outright.  The
        # analyser's >=6-visible-keypoints filter already removes most ghosts
        # before this point.
        if max(ls.visibility, rs.visibility) < self.c.pose_visibility_min:
            return None

        ls_ok = ls.visibility >= KV
        rs_ok = rs.visibility >= KV

        # Body reference from whichever shoulder(s) are actually visible.
        if ls_ok and rs_ok:
            avg_sh_y, avg_sh_x = (ls.y + rs.y) / 2.0, (ls.x + rs.x) / 2.0
        elif ls_ok:
            avg_sh_y, avg_sh_x = ls.y, ls.x
        else:
            avg_sh_y, avg_sh_x = rs.y, rs.x

        # Upright check: only trust when hips are confidently detected.
        # When hips are hidden (blanket, crop) we cannot tell sitting from
        # lying, so upright is set to None and the per-side raise logic below
        # demands a forearm-pointing-up confirmation as a precaution.
        hip_vis = (lh.visibility + rh.visibility) / 2.0
        if hip_vis >= 0.20:
            avg_hp_y = (lh.y + rh.y) / 2.0
            avg_hp_x = (lh.x + rh.x) / 2.0
            torso_dy = avg_hp_y - avg_sh_y  # positive when hips below shoulders
            torso_dx = avg_hp_x - avg_sh_x  # non-zero when body is horizontal
            # Body is reclined when horizontal extent exceeds vertical extent —
            # catches lying-on-couch even when hips appear "below" shoulders due
            # to camera perspective (which would otherwise pass the y-only check).
            body_horizontal = abs(torso_dx) > abs(torso_dy)
            upright: bool | None = (
                not body_horizontal and torso_dy >= self.c.body_upright_min
            )
        else:
            avg_hp_y = avg_sh_y   # fallback for leg-raise guard below
            upright = None  # unknown — hips not visible

        # Leg-raise guard: if both knees are above shoulder level, the person
        # is raising their legs (upside-down V on couch etc.) — suppress to
        # avoid false positives from leg movement being mistaken for arms.
        lk, rk = landmarks[LEFT_KNEE], landmarks[RIGHT_KNEE]
        knee_vis = (lk.visibility + rk.visibility) / 2.0
        if knee_vis >= 0.20:
            avg_knee_y = (lk.y + rk.y) / 2.0
            if avg_knee_y < avg_sh_y - self.c.leg_raise_margin:
                return ArmReading(state=ArmState.DOWN, upright=bool(upright))

        mid_x = avg_sh_x
        both_sh_ok = ls_ok and rs_ok

        # ── Two-handed gestures (need BOTH shoulders confidently visible) ──
        if both_sh_ok:
            # CROSS_ARMS: wrists crossed past midline, at chest height, close together.
            min_cross = self.c.cross_arms_min_crossing
            rw_on_left  = rw.x < mid_x - min_cross
            lw_on_right = lw.x > mid_x + min_cross
            chest_top    = avg_sh_y - 0.02
            chest_bottom = avg_hp_y + 0.02
            rw_at_chest  = chest_top < rw.y < chest_bottom
            lw_at_chest  = chest_top < lw.y < chest_bottom
            wrist_dist = ((rw.x - lw.x) ** 2 + (rw.y - lw.y) ** 2) ** 0.5
            wrists_close = wrist_dist < self.c.cross_arms_wrist_proximity
            if (rw_on_left and lw_on_right
                    and rw_at_chest and lw_at_chest and wrists_close):
                return ArmReading(state=ArmState.CROSS_ARMS, upright=bool(upright))

            # T_POSE: both wrists at shoulder height AND clearly lateral.
            band = self.c.tpose_wrist_y_band
            lat  = self.c.tpose_lateral_min
            lw_at_sh_y = abs(lw.y - ls.y) < band
            rw_at_sh_y = abs(rw.y - rs.y) < band
            lw_lateral = abs(lw.x - mid_x) > lat
            rw_lateral = abs(rw.x - mid_x) > lat
            lw_is_left = lw.x < mid_x
            rw_is_right = rw.x > mid_x
            if (lw_at_sh_y and rw_at_sh_y and lw_lateral and rw_lateral
                    and lw_is_left and rw_is_right):
                return ArmReading(state=ArmState.T_POSE, upright=bool(upright))

            # BOTH_UP: both wrists clearly above their shoulders.
            both_m = max(self.c.arm_above_head_tolerance, self.c.both_arms_margin)
            if upright is None:
                lw_high = lw.y < (ls.y - both_m) and (le.y - lw.y) > 0
                rw_high = rw.y < (rs.y - both_m) and (re.y - rw.y) > 0
            else:
                lw_high = lw.y < (ls.y - both_m)
                rw_high = rw.y < (rs.y - both_m)
            if lw_high and rw_high:
                return ArmReading(state=ArmState.BOTH_UP, upright=bool(upright))

        # ── SINGLE_UP — evaluated per side, works with one visible shoulder ──
        m          = self.c.arm_above_head_tolerance
        m_reclined = self.c.arm_above_head_tolerance_reclined
        fv_min     = self.c.forearm_vertical_min
        whd        = self.c.wrist_head_exclude_dist
        nose_ok    = nose.visibility >= KV

        def _side_raised(wrist, shoulder, elbow) -> bool:
            """Is this arm raised?  Two routes, depending on body posture.

            Primary route — wrist clears the shoulder by the height margin.
            Secondary route — forearm is clearly vertical (wrist well above
            elbow) and the wrist is at/above shoulder height: catches raises
            seen from camera angles where the wrist doesn't travel far above
            the shoulder in image space.  Disabled when confirmed reclined.
            """
            forearm_dy = elbow.y - wrist.y
            forearm_visible = elbow.visibility >= KV
            forearm_vertical = forearm_visible and forearm_dy >= fv_min
            if upright is True:
                raised = (wrist.y < (shoulder.y - m)
                          or (forearm_vertical and wrist.y <= shoulder.y))
            elif upright is False:
                # Lying down — demand a deliberate straight-up arm; no shortcut.
                raised = wrist.y < (shoulder.y - m_reclined)
            else:
                # upright is None (hips hidden): lenient margin requires the
                # forearm to point up; if the elbow isn't visible to confirm
                # that, fall back to the strict reclined margin instead.
                if forearm_visible:
                    raised = wrist.y < (shoulder.y - m) and forearm_dy > 0
                else:
                    raised = wrist.y < (shoulder.y - m_reclined)
            if not raised:
                return False
            # Wrist resting against the face (adjusting glasses, holding a
            # phone, scratching, chin-on-hand) satisfies the height/verticality
            # checks above but is not a deliberate raise — only reject when
            # the nose is confidently visible, so an occluded face can't
            # block a real gesture.
            if nose_ok:
                head_dist = ((wrist.x - nose.x) ** 2 + (wrist.y - nose.y) ** 2) ** 0.5
                if head_dist < whd:
                    return False
            return True

        rw_raised = rs_ok and rw.visibility >= KV and _side_raised(rw, rs, re)
        lw_raised = ls_ok and lw.visibility >= KV and _side_raised(lw, ls, le)

        if rw_raised:
            return ArmReading(
                state=ArmState.SINGLE_UP,
                raised_side=Side.RIGHT,
                wrist_x=rw.x * frame_w,
                wrist_y=rw.y * frame_h,
                forearm_dy=re.y - rw.y,
                upright=bool(upright),
            )
        if lw_raised:
            return ArmReading(
                state=ArmState.SINGLE_UP,
                raised_side=Side.LEFT,
                wrist_x=lw.x * frame_w,
                wrist_y=lw.y * frame_h,
                forearm_dy=le.y - lw.y,
                upright=bool(upright),
            )

        # Detect wrist approaching shoulder — fires chime early to compensate
        # for speaker network latency (e.g. Alexa ~2s delay).
        rise_thresh = self.c.arm_rising_threshold
        lw_rising = ls_ok and lw.visibility >= KV and lw.y < (ls.y + rise_thresh)
        rw_rising = rs_ok and rw.visibility >= KV and rw.y < (rs.y + rise_thresh)
        return ArmReading(state=ArmState.DOWN, upright=bool(upright),
                          arm_rising=(lw_rising or rw_rising))
