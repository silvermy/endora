"""
tests/fake_landmarks.py

Factory functions for building fake pose landmark fixtures.
Each function returns a dict-like landmarks object compatible with the
ArmTracker's expected input.

All coordinates are normalised (0.0 - 1.0) just like MediaPipe's output.
The frame origin (0,0) is top-left; y increases downward.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class Point:
    x: float
    y: float
    visibility: float = 1.0


class Landmarks:
    """Dict-indexable wrapper to mimic mediapipe's landmark list."""
    def __init__(self, points: Dict[int, Point]):
        self._points = points

    def __getitem__(self, idx: int) -> Point:
        return self._points[idx]


# MediaPipe PoseLandmark indices
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW,    RIGHT_ELBOW    = 13, 14
LEFT_WRIST,    RIGHT_WRIST    = 15, 16
LEFT_HIP,      RIGHT_HIP      = 23, 24
LEFT_KNEE,     RIGHT_KNEE     = 25, 26


def _build(**overrides) -> Landmarks:
    """
    Build a landmark set with a reasonable seated-upright default.
    Override any point by passing e.g. left_wrist=Point(0.3, 0.1).
    """
    defaults = {
        LEFT_SHOULDER:  Point(0.40, 0.40),
        RIGHT_SHOULDER: Point(0.60, 0.40),
        LEFT_ELBOW:     Point(0.35, 0.55),
        RIGHT_ELBOW:    Point(0.65, 0.55),
        LEFT_WRIST:     Point(0.32, 0.70),
        RIGHT_WRIST:    Point(0.68, 0.70),
        LEFT_HIP:       Point(0.42, 0.65),
        RIGHT_HIP:      Point(0.58, 0.65),
        LEFT_KNEE:      Point(0.42, 0.80),
        RIGHT_KNEE:     Point(0.58, 0.80),
    }
    name_to_idx = {
        'left_shoulder':  LEFT_SHOULDER,  'right_shoulder': RIGHT_SHOULDER,
        'left_elbow':     LEFT_ELBOW,     'right_elbow':    RIGHT_ELBOW,
        'left_wrist':     LEFT_WRIST,     'right_wrist':    RIGHT_WRIST,
        'left_hip':       LEFT_HIP,       'right_hip':      RIGHT_HIP,
        'left_knee':      LEFT_KNEE,      'right_knee':     RIGHT_KNEE,
    }
    for name, point in overrides.items():
        defaults[name_to_idx[name]] = point
    return Landmarks(defaults)


# ── Pose fixtures ─────────────────────────────────────────────────────────────

def arm_down() -> Landmarks:
    """Seated, both arms at side."""
    return _build()


def right_arm_up_vertical() -> Landmarks:
    """Right arm straight up — good SNAP candidate."""
    return _build(
        right_elbow=Point(0.65, 0.25),  # elbow above shoulder
        right_wrist=Point(0.65, 0.10),  # wrist well above elbow
    )


def right_arm_up_horizontal() -> Landmarks:
    """Right arm extended sideways (not vertical)."""
    return _build(
        right_elbow=Point(0.75, 0.40),
        right_wrist=Point(0.90, 0.40),
    )


def both_arms_up() -> Landmarks:
    """Both arms raised straight up."""
    return _build(
        left_elbow=Point(0.35, 0.25),   left_wrist=Point(0.35, 0.10),
        right_elbow=Point(0.65, 0.25),  right_wrist=Point(0.65, 0.10),
    )


def t_pose() -> Landmarks:
    """Arms extended horizontally to each side."""
    return _build(
        left_elbow=Point(0.25, 0.40),   left_wrist=Point(0.10, 0.40),
        right_elbow=Point(0.75, 0.40),  right_wrist=Point(0.90, 0.40),
    )


def cross_arms() -> Landmarks:
    """Wrists near opposite shoulders."""
    return _build(
        left_elbow=Point(0.50, 0.50),   left_wrist=Point(0.60, 0.40),
        right_elbow=Point(0.50, 0.50),  right_wrist=Point(0.40, 0.40),
    )


def lying_down() -> Landmarks:
    """Hips level with or above shoulders — not upright."""
    return _build(
        left_hip=Point(0.42, 0.30),   right_hip=Point(0.58, 0.30),
    )


def low_visibility() -> Landmarks:
    """Mostly hidden — should be rejected."""
    return _build(
        left_shoulder=Point(0.40, 0.40, visibility=0.1),
        right_shoulder=Point(0.60, 0.40, visibility=0.1),
        left_hip=Point(0.42, 0.65, visibility=0.1),
        right_hip=Point(0.58, 0.65, visibility=0.1),
    )
