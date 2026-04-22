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
    # How many consecutive frames a new non-DOWN state must be seen before
    # it's accepted. Kills single-frame phantoms (furniture, flicker, shadows).
    # At 10fps, 3 frames ≈ 0.3s to confirm a real gesture onset.
    state_confirm_frames: int = 3
    # How many consecutive frames of the new state before transitioning AWAY
    # from a stable non-DOWN state back to DOWN. This prevents mid-gesture
    # dropouts when MediaPipe momentarily loses the person.
    state_release_frames: int = 4


class ArmTracker:
    def __init__(self, config: ArmTrackerConfig):
        self.c = config
        # Hysteresis state
        self._stable_reading: Optional[ArmReading] = None
        self._pending_state: Optional[ArmState] = None
        self._pending_count: int = 0

    def classify(self, landmarks: Optional[_Landmarks],
                 frame_w: int, frame_h: int) -> Optional[ArmReading]:
        """
        Public entry point — applies hysteresis to the raw classification.
        A new state must appear for state_confirm_frames consecutive frames
        before being accepted; a stable state is held for state_release_frames
        of contradictory frames before releasing.
        """
        raw = self._classify_raw(landmarks, frame_w, frame_h)
        raw_state = raw.state if raw is not None else ArmState.DOWN

        # First call or we had no stable reading yet
        if self._stable_reading is None:
            # Only accept a non-DOWN state after the confirm window
            if raw_state == ArmState.DOWN:
                self._stable_reading = raw
                self._pending_state = None
                self._pending_count = 0
                return raw

            # Non-DOWN candidate: need state_confirm_frames to confirm
            if self._pending_state == raw_state:
                self._pending_count += 1
            else:
                self._pending_state = raw_state
                self._pending_count = 1

            if self._pending_count >= self.c.state_confirm_frames:
                self._stable_reading = raw
                self._pending_state = None
                self._pending_count = 0
                return raw

            # Not confirmed yet — report DOWN to downstream
            return ArmReading(state=ArmState.DOWN)

        # We have a stable reading — check if a change is underway
        stable_state = self._stable_reading.state
        if raw_state == stable_state:
            # Same state; refresh the stable reading (positions may update)
            self._stable_reading = raw if raw is not None else self._stable_reading
            self._pending_state = None
            self._pending_count = 0
            return self._stable_reading

        # State change candidate
        if self._pending_state == raw_state:
            self._pending_count += 1
        else:
            self._pending_state = raw_state
            self._pending_count = 1

        # Faster to release a stable non-DOWN state to DOWN (release_frames),
        # slower to promote a new non-DOWN state (confirm_frames).
        if raw_state == ArmState.DOWN:
            needed = self.c.state_release_frames
        else:
            needed = self.c.state_confirm_frames

        if self._pending_count >= needed:
            self._stable_reading = raw if raw is not None else ArmReading(state=ArmState.DOWN)
            self._pending_state = None
            self._pending_count = 0
            return self._stable_reading

        # Change not confirmed — keep reporting the previous stable state
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
