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


# ── Landmark protocol for type hints ──────────────────────────────────────────

class _Point(Protocol):
    x: float
    y: float
    visibility: float


class _Landmarks(Protocol):
    def __getitem__(self, idx: int) -> _Point: ...


# Landmark indices used by ArmTracker.  These match MediaPipe PoseLandmark
# values; the YOLO backend remaps COCO indices to these before calling classify().
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24


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
    body_upright_min: float = -0.15
    pose_visibility_min: float = 0.55

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
    cross_arms_min_crossing: float = 0.03
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

            return ArmReading(state=ArmState.DOWN)

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

        # Visibility filter — four key landmarks
        vis = (ls.visibility + rs.visibility + lh.visibility + rh.visibility) / 4.0
        if vis < self.c.pose_visibility_min:
            return None

        avg_sh_y = (ls.y + rs.y) / 2.0
        avg_hp_y = (lh.y + rh.y) / 2.0
        upright = avg_hp_y >= avg_sh_y + self.c.body_upright_min

        le, re = landmarks[LEFT_ELBOW],  landmarks[RIGHT_ELBOW]
        lw, rw = landmarks[LEFT_WRIST],  landmarks[RIGHT_WRIST]

        mid_x = (ls.x + rs.x) / 2.0

        # ── CROSS_ARMS check (before T_POSE / SINGLE_UP / BOTH_UP) ────────
        # True cross-arms (hands-on-chest or wrists-crossed-on-chest) requires:
        #   1. Wrists crossed past midline (by min_crossing on each side).
        #   2. Both wrists at chest height (between shoulders and hips).
        #   3. Wrists close to each other (wrists near each other in 2D distance).
        #      This handles both clasped-hands-on-chest and crossed-wrists poses.
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
                and rw_at_chest and lw_at_chest
                and wrists_close):
            return ArmReading(state=ArmState.CROSS_ARMS, upright=upright)

        # ── T_POSE check ──────────────────────────────────────────────────
        # Both wrists at shoulder height AND clearly lateral from midline.
        band = self.c.tpose_wrist_y_band
        lat  = self.c.tpose_lateral_min
        lw_at_sh_y   = abs(lw.y - ls.y) < band
        rw_at_sh_y   = abs(rw.y - rs.y) < band
        lw_lateral   = abs(lw.x - mid_x) > lat
        rw_lateral   = abs(rw.x - mid_x) > lat
        # Extended in opposite directions (not both on same side of body).
        lw_is_left   = lw.x < mid_x
        rw_is_right  = rw.x > mid_x
        if (lw_at_sh_y and rw_at_sh_y and lw_lateral and rw_lateral
                and lw_is_left and rw_is_right):
            return ArmReading(state=ArmState.T_POSE, upright=upright)

        # ── BOTH_UP / SINGLE_UP ───────────────────────────────────────────
        m = self.c.arm_above_head_tolerance
        both_m = max(m, self.c.both_arms_margin)
        lw_high = lw.y < (ls.y - both_m)
        rw_high = rw.y < (rs.y - both_m)

        if lw_high and rw_high:
            return ArmReading(state=ArmState.BOTH_UP, upright=upright)

        # Body must be upright for SINGLE_UP to count
        if not upright:
            return ArmReading(state=ArmState.DOWN, upright=upright)

        lw_raised = lw.y < (ls.y - m)
        rw_raised = rw.y < (rs.y - m)

        if rw_raised:
            forearm_dy = re.y - rw.y
            return ArmReading(
                state=ArmState.SINGLE_UP,
                raised_side=Side.RIGHT,
                wrist_x=rw.x * frame_w,
                wrist_y=rw.y * frame_h,
                forearm_dy=forearm_dy,
                upright=upright,
            )
        if lw_raised:
            forearm_dy = le.y - lw.y
            return ArmReading(
                state=ArmState.SINGLE_UP,
                raised_side=Side.LEFT,
                wrist_x=lw.x * frame_w,
                wrist_y=lw.y * frame_h,
                forearm_dy=forearm_dy,
                upright=upright,
            )

        return ArmReading(state=ArmState.DOWN, upright=upright)
