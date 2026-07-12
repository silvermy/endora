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
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, Optional, Protocol

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
    # Body-scale factor applied to the geometric thresholds for this person:
    # detected body size / body_scale_reference, clamped. 1.0 = reference size
    # (or size could not be estimated). Consumers that compare raw landmark
    # distances against configured thresholds (e.g. snap_forearm_min in the
    # state machine) should multiply the threshold by this.
    scale_factor: float = 1.0
    # Trajectory evidence, only meaningful for SINGLE_UP (defaults are the
    # permissive value so tests/back-compat callers that build readings by
    # hand are unaffected):
    #   rose_recently — the wrist was seen below shoulder level within the
    #     last raise_travel_window_s, i.e. this raise was an actual upward
    #     motion, not a pose that has simply existed since tracking began
    #     (hand propped against head, furniture ghost, sleeping posture).
    #   wrist_still — the wrist has stayed within wrist_still_max_travel over
    #     the last wrist_still_window_s, i.e. the arm is being HELD up, not
    #     passing through on its way to grab a phone/blanket/glass.
    rose_recently: bool = True
    wrist_still: bool = True
    # SINGLE_UP only: how far the raised wrist actually cleared its shoulder
    # (shoulder_y - wrist_y, frame fraction, positive = above). Logged to
    # feedback.jsonl so threshold tuning can finally see the achieved margin
    # instead of inferring it from forearm_dy (a different quantity) —
    # divide by scale_factor to compare against the configured tolerances.
    raise_margin: float = 0.0


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
    """Palm-orientation signal from a flat grlib hand-landmark array
    (1 hand, 63 floats; MediaPipe indices WRIST=0, INDEX_FINGER_MCP=5,
    MIDDLE_FINGER_MCP=9, PINKY_MCP=17).

    roll = (index_mcp.x - pinky_mcp.x) / hand_size, where hand_size is the
    wrist→middle-MCP distance — an apparent-hand-size proxy that doesn't
    collapse when the knuckle line is edge-on.  |roll| ≈ 0.8–1.1 when the
    palm or back of the hand faces the camera (knuckles spread laterally),
    ≈ 0–0.3 when the hand is edge-on; the sign encodes palm-vs-back /
    left-vs-right hand. Clamped to ±1.5.

    The previous formula divided (index.x − pinky.x) by its own absolute
    value, so every detected hand read exactly ±1.0 — an orientation signal
    in name only, and one that silently armed the snap_roll_threshold OR-
    route once the wrist-crop made hand detection reliable (v1.9.114).
    """
    if len(hand_lm) < 21 * 3:
        return 0.0
    wr_x, wr_y = float(hand_lm[0]), float(hand_lm[1])          # WRIST
    md_x, md_y = float(hand_lm[9 * 3]), float(hand_lm[9 * 3 + 1])  # MIDDLE_MCP
    hand_size = ((md_x - wr_x) ** 2 + (md_y - wr_y) ** 2) ** 0.5
    if hand_size < 1e-6:
        return 0.0
    idx_mcp_x = float(hand_lm[5 * 3])    # INDEX_FINGER_MCP.x
    pnk_mcp_x = float(hand_lm[17 * 3])   # PINKY_MCP.x
    roll = (idx_mcp_x - pnk_mcp_x) / hand_size
    return max(-1.5, min(1.5, roll))


# ── Tracker ───────────────────────────────────────────────────────────────────

# Bounds on the body-scale factor so a bad size estimate (side-on shoulders
# collapsing to a sliver, a partial detection) can at worst halve/double the
# margins, never zero them out or make them unreachable.
_SCALE_FACTOR_MIN = 0.5
_SCALE_FACTOR_MAX = 2.0
# Torso length estimated from shoulder width when the hips are hidden
# (blanket). Matches the ~1.25 torso/biacromial ratio of a typical adult.
_TORSO_PER_SHOULDER_WIDTH = 1.25


@dataclass
class _HistSample:
    """One frame of wrist/shoulder positions for trajectory checks."""
    t: float
    lw_x: float; lw_y: float; lw_ok: bool
    rw_x: float; rw_y: float; rw_ok: bool
    ls_y: float; ls_ok: bool
    rs_y: float; rs_ok: bool


@dataclass
class ArmTrackerConfig:
    """Thresholds for arm-state classification. All values are frame fractions.

    Distance-type thresholds are interpreted at the reference body size
    (body_scale_reference) and scaled per-person: a person whose torso spans
    half the reference gets half the margins, so the same config works for a
    body lying far from the camera and one standing right in front of it.
    """
    arm_above_head_tolerance: float = 0.15
    # Wrist within this distance BELOW shoulder level triggers arm_rising=True
    # for early chime firing (compensates for speaker network latency).
    arm_rising_threshold: float = 0.05
    # When body is NOT upright (reclined/lying down), OR when upright status
    # cannot be determined (hips hidden by blanket), the wrist must clear the
    # shoulder by this larger margin.  Requires a deliberate straight-up arm.
    arm_above_head_tolerance_reclined: float = 0.38
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
    # The forearm-vertical route still requires the wrist to clear the
    # shoulder by this (body-scaled) margin. Feedback data (2026-07-12)
    # showed a day-long false-fire storm through this route with the wrist
    # sitting exactly AT shoulder level (raise_margin 0.000–0.049 on every
    # flagged fire — arm resting on a couch armrest / holding a phone) while
    # every confirmed deliberate raise cleared 0.17+. This margin splits
    # those cleanly while keeping the route's purpose (high-mounted cameras
    # compressing raise height) intact.
    forearm_route_min_margin: float = 0.06

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

    # ── Body scale ─────────────────────────────────────────────────────────
    # Torso length (frame fraction, shoulder-mid to hip-mid) the thresholds
    # above are tuned at. Detected torso / this = the per-person scale factor
    # applied to every distance threshold. When hips are hidden the torso is
    # estimated from shoulder width; if neither is measurable the factor is 1.
    # NOTE: this dataclass default (0.25) matches the unit-test fixtures'
    # torso and keeps them scale-neutral; the PRODUCTION default lives in
    # config/settings.py (0.18, calibrated from live feedback logs) and is
    # always passed in by the analyser.
    body_scale_reference: float = 0.25

    # ── Trajectory (rise / stillness evidence for SNAP) ────────────────────
    # A raise only counts as "rose recently" if the wrist was seen below
    # shoulder level within this many seconds. Blocks re-fires from poses
    # that have simply existed for a while (hand propped against the head,
    # a ghost with a permanently-raised arm). Evidence is only asserted
    # negative once the history buffer actually spans the window, so a
    # freshly-acquired person is never blocked by an empty buffer.
    raise_travel_window_s: float = 2.5
    # The wrist must stay within wrist_still_max_travel (body-scaled frame
    # fraction) of every position sampled in the last wrist_still_window_s
    # for the raise to count as "held still". A reach for a phone/blanket
    # keeps moving through this window and never qualifies.
    wrist_still_window_s: float = 0.30
    wrist_still_max_travel: float = 0.05


class ArmTracker:
    def __init__(self, config: ArmTrackerConfig):
        self.c = config
        self._stable_reading: Optional[ArmReading] = None
        self._pending_state: Optional[ArmState] = None
        self._pending_since: float = 0.0
        # Rolling wrist/shoulder history for the trajectory checks.
        self._hist: Deque[_HistSample] = deque()

    # ── Trajectory history ────────────────────────────────────────────────

    def _record_history(self, landmarks: _Landmarks, now: float) -> None:
        KV = self.c.keypoint_visibility_min
        ls, rs = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
        lw, rw = landmarks[LEFT_WRIST], landmarks[RIGHT_WRIST]
        self._hist.append(_HistSample(
            t=now,
            lw_x=lw.x, lw_y=lw.y, lw_ok=lw.visibility >= KV,
            rw_x=rw.x, rw_y=rw.y, rw_ok=rw.visibility >= KV,
            ls_y=ls.y, ls_ok=ls.visibility >= KV,
            rs_y=rs.y, rs_ok=rs.visibility >= KV,
        ))
        horizon = max(self.c.raise_travel_window_s,
                      self.c.wrist_still_window_s) + 1.0
        while self._hist and now - self._hist[0].t > horizon:
            self._hist.popleft()

    def _rose_recently(self, side: Side, now: float, factor: float) -> bool:
        """Was this side's wrist seen at/near/below shoulder level within the
        raise window?  "Near" (within arm_rising_threshold) matters for the
        reclined case: a resting wrist on a horizontal body sits at almost
        the same image height as the shoulder, so demanding strictly-below
        would leave a lying person with no rise evidence at all.  Unknown
        (buffer doesn't span the window yet — person only just acquired)
        counts as True: only assert "did not rise" when there is enough
        history to actually know. A static pose that persists longer than
        the window loses the benefit of the doubt.
        """
        window = self.c.raise_travel_window_s
        down_line = self.c.arm_rising_threshold * factor
        for s in self._hist:
            if now - s.t > window:
                continue
            if side is Side.LEFT:
                wy, wok, shy, shok = s.lw_y, s.lw_ok, s.ls_y, s.ls_ok
            else:
                wy, wok, shy, shok = s.rw_y, s.rw_ok, s.rs_y, s.rs_ok
            if wok and shok and wy >= shy - down_line:
                return True
        covered = bool(self._hist) and (now - self._hist[0].t) >= window * 0.9
        return not covered

    def _wrist_still(self, side: Side, wx: float, wy: float,
                     now: float, factor: float) -> bool:
        """Has this side's wrist stayed put over the stillness window?
        A sparse/young buffer is permissive — during a real raise the buffer
        fills at frame rate, so motion is always observed when it exists.
        """
        window = self.c.wrist_still_window_s
        max_travel = self.c.wrist_still_max_travel * factor
        for s in self._hist:
            if now - s.t > window:
                continue
            if side is Side.LEFT:
                sx, sy, ok = s.lw_x, s.lw_y, s.lw_ok
            else:
                sx, sy, ok = s.rw_x, s.rw_y, s.rw_ok
            if not ok:
                continue
            if ((wx - sx) ** 2 + (wy - sy) ** 2) ** 0.5 > max_travel:
                return False
        return True

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
        if landmarks is not None:
            self._record_history(landmarks, now)
        raw = self._classify_raw(landmarks, frame_w, frame_h, now)
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
                      frame_w: int, frame_h: int,
                      now: Optional[float] = None) -> Optional[ArmReading]:
        """
        Classify pose landmarks into an ArmReading.
        Returns None if landmarks are missing or visibility is too low.
        now=None (direct/unit-test calls) skips the trajectory checks —
        rose_recently/wrist_still stay at their permissive defaults.
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
        torso_len: Optional[float] = None
        if hip_vis >= 0.20:
            avg_hp_y = (lh.y + rh.y) / 2.0
            avg_hp_x = (lh.x + rh.x) / 2.0
            torso_dy = avg_hp_y - avg_sh_y  # positive when hips below shoulders
            torso_dx = avg_hp_x - avg_sh_x  # non-zero when body is horizontal
            torso_len = (torso_dx ** 2 + torso_dy ** 2) ** 0.5
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

        # Body-scale factor: all distance thresholds below are tuned at
        # body_scale_reference and scale with this person's apparent size, so
        # a body lying far from the camera isn't asked to clear margins sized
        # for one standing right in front of it (and vice versa). The size
        # estimate is the LARGER of torso length and shoulder-width×ratio:
        # each collapses under a different projection — reclining feet-toward-
        # the-camera foreshortens the torso to a sliver while the shoulders
        # stay lateral (live data showed torso-only estimates of 0.5–0.65×
        # for a normal-sized person on the couch, silently shrinking every
        # margin), and a side-on pose collapses shoulder width while the
        # torso stays long. Clamped so whatever survives a degenerate
        # detection can't zero the margins or push them out of reach.
        size_estimates = []
        if torso_len is not None and torso_len > 1e-6:
            size_estimates.append(torso_len)
        if ls_ok and rs_ok:
            shoulder_w = ((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2) ** 0.5
            if shoulder_w > 1e-6:
                size_estimates.append(shoulder_w * _TORSO_PER_SHOULDER_WIDTH)
        if size_estimates and self.c.body_scale_reference > 1e-6:
            f = max(size_estimates) / self.c.body_scale_reference
            f = min(max(f, _SCALE_FACTOR_MIN), _SCALE_FACTOR_MAX)
        else:
            f = 1.0

        # Leg-raise guard: if both knees are above shoulder level, the person
        # is raising their legs (upside-down V on couch etc.) — suppress to
        # avoid false positives from leg movement being mistaken for arms.
        lk, rk = landmarks[LEFT_KNEE], landmarks[RIGHT_KNEE]
        knee_vis = (lk.visibility + rk.visibility) / 2.0
        if knee_vis >= 0.20:
            avg_knee_y = (lk.y + rk.y) / 2.0
            if avg_knee_y < avg_sh_y - self.c.leg_raise_margin * f:
                return ArmReading(state=ArmState.DOWN, upright=bool(upright),
                                  scale_factor=f)

        mid_x = avg_sh_x
        both_sh_ok = ls_ok and rs_ok

        # ── Two-handed gestures (need BOTH shoulders confidently visible) ──
        if both_sh_ok:
            # CROSS_ARMS: wrists crossed past midline, at chest height, close together.
            min_cross = self.c.cross_arms_min_crossing * f
            rw_on_left  = rw.x < mid_x - min_cross
            lw_on_right = lw.x > mid_x + min_cross
            chest_top    = avg_sh_y - 0.02
            chest_bottom = avg_hp_y + 0.02
            rw_at_chest  = chest_top < rw.y < chest_bottom
            lw_at_chest  = chest_top < lw.y < chest_bottom
            wrist_dist = ((rw.x - lw.x) ** 2 + (rw.y - lw.y) ** 2) ** 0.5
            wrists_close = wrist_dist < self.c.cross_arms_wrist_proximity * f
            if (rw_on_left and lw_on_right
                    and rw_at_chest and lw_at_chest and wrists_close):
                return ArmReading(state=ArmState.CROSS_ARMS, upright=bool(upright),
                                  scale_factor=f)

            # T_POSE: both wrists at shoulder height AND clearly lateral.
            band = self.c.tpose_wrist_y_band * f
            lat  = self.c.tpose_lateral_min * f
            lw_at_sh_y = abs(lw.y - ls.y) < band
            rw_at_sh_y = abs(rw.y - rs.y) < band
            lw_lateral = abs(lw.x - mid_x) > lat
            rw_lateral = abs(rw.x - mid_x) > lat
            lw_is_left = lw.x < mid_x
            rw_is_right = rw.x > mid_x
            if (lw_at_sh_y and rw_at_sh_y and lw_lateral and rw_lateral
                    and lw_is_left and rw_is_right):
                return ArmReading(state=ArmState.T_POSE, upright=bool(upright),
                                  scale_factor=f)

            # BOTH_UP: both wrists clearly above their shoulders.
            both_m = max(self.c.arm_above_head_tolerance,
                         self.c.both_arms_margin) * f
            if upright is None:
                lw_high = lw.y < (ls.y - both_m) and (le.y - lw.y) > 0
                rw_high = rw.y < (rs.y - both_m) and (re.y - rw.y) > 0
            else:
                lw_high = lw.y < (ls.y - both_m)
                rw_high = rw.y < (rs.y - both_m)
            if lw_high and rw_high:
                return ArmReading(state=ArmState.BOTH_UP, upright=bool(upright),
                                  scale_factor=f)

        # ── SINGLE_UP — evaluated per side, works with one visible shoulder ──
        m          = self.c.arm_above_head_tolerance * f
        m_reclined = self.c.arm_above_head_tolerance_reclined * f
        fv_min     = self.c.forearm_vertical_min * f
        fr_margin  = self.c.forearm_route_min_margin * f
        whd        = self.c.wrist_head_exclude_dist * f
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
                # Secondary route demands the wrist actually clear the
                # shoulder by fr_margin — wrist merely AT shoulder level with
                # a vertical-ish forearm is the resting-arm/phone posture,
                # not a raise (see forearm_route_min_margin).
                raised = (wrist.y < (shoulder.y - m)
                          or (forearm_vertical
                              and wrist.y <= shoulder.y - fr_margin))
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

        if rw_raised or lw_raised:
            side = Side.RIGHT if rw_raised else Side.LEFT
            wrist, elbow, shoulder = (rw, re, rs) if rw_raised else (lw, le, ls)
            if now is not None:
                rose  = self._rose_recently(side, now, f)
                still = self._wrist_still(side, wrist.x, wrist.y, now, f)
            else:
                rose = still = True
            return ArmReading(
                state=ArmState.SINGLE_UP,
                raised_side=side,
                wrist_x=wrist.x * frame_w,
                wrist_y=wrist.y * frame_h,
                forearm_dy=elbow.y - wrist.y,
                upright=bool(upright),
                scale_factor=f,
                rose_recently=rose,
                wrist_still=still,
                raise_margin=shoulder.y - wrist.y,
            )

        # Detect wrist approaching shoulder — fires chime early to compensate
        # for speaker network latency (e.g. Alexa ~2s delay).
        rise_thresh = self.c.arm_rising_threshold * f
        lw_rising = ls_ok and lw.visibility >= KV and lw.y < (ls.y + rise_thresh)
        rw_rising = rs_ok and rw.visibility >= KV and rw.y < (rs.y + rise_thresh)
        return ArmReading(state=ArmState.DOWN, upright=bool(upright),
                          arm_rising=(lw_rising or rw_rising),
                          scale_factor=f)
