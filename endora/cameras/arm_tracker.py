"""
cameras/arm_tracker.py

Pure classifier: pose landmarks → ArmState.

No MediaPipe dependency in the classifier itself — it takes a dict-like
landmarks object with x/y per PoseLandmark enum value, so tests can pass
fake landmarks without installing MediaPipe.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Protocol


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


# ── Landmark protocol for type hints ──────────────────────────────────────────

class _Point(Protocol):
    x: float
    y: float
    visibility: float


class _Landmarks(Protocol):
    def __getitem__(self, idx: int) -> _Point: ...


# Indices match mediapipe.solutions.pose.PoseLandmark.
# Kept as constants so tests don't need mediapipe installed.
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24


# ── Tracker ───────────────────────────────────────────────────────────────────

@dataclass
class ArmTrackerConfig:
    """Thresholds for arm-state classification. All values are frame fractions."""
    arm_above_head_tolerance: float = 0.15
    body_upright_min: float = -0.15
    pose_visibility_min: float = 0.45

    # Both-arms-up: each wrist must clear its shoulder by this margin.
    both_arms_margin: float = 0.10

    # T-pose: wrists roughly at shoulder height AND clearly lateral from body.
    # wrist.y within this band of shoulder.y (frame fraction) counts as "at shoulder height".
    tpose_wrist_y_band: float = 0.12
    # Wrists must be this far lateral from body midline (frame fraction of width).
    tpose_lateral_min: float = 0.18

    # Cross-arms: wrists within this radius of the *opposite* shoulder (frame fraction).
    cross_arms_radius: float = 0.15


class ArmTracker:
    def __init__(self, config: ArmTrackerConfig):
        self.c = config

    def classify(self, landmarks: Optional[_Landmarks],
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
        # Each wrist must be near the OPPOSITE shoulder.
        r = self.c.cross_arms_radius
        rw_near_left  = ((rw.x - ls.x) ** 2 + (rw.y - ls.y) ** 2) ** 0.5 < r
        lw_near_right = ((lw.x - rs.x) ** 2 + (lw.y - rs.y) ** 2) ** 0.5 < r
        if rw_near_left and lw_near_right:
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
