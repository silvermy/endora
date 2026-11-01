"""
tests/test_arm_angle.py

An arm counts as raised at raise_elevation_min 0.85 (~58 degrees), not 0.70.

Lying back on the couch with a hand behind the head, the pose model draws
an arm pointing up and back at elevation 0.70-0.83, which passed the old
0.70. On the frame each fired, the snaps labelled intended measured
0.80-1.00, all but one 0.90 or more.

Fixtures in tests/arm_angle/, each ending on the frame that fired live:
five labelled not intended that 0.85 stops, two intended reclined snaps
that must still fire, and intended_at_080 — the one intended snap that
fired at exactly 0.80 and does not fire in replay at 0.85. A recording
ends where it fired, so the replay cannot tell whether it would have fired
a frame later; it is kept to document the cost.
"""
from pathlib import Path

import numpy as np
import pytest

from cameras.arm_tracker import ArmTracker, ArmTrackerConfig
from core.state_machine import Gesture, GestureStateMachine, StateMachineConfig
from tests.test_captured import _YOLOLandmarks

FIXTURES = Path(__file__).parent / "arm_angle"
FALSE = ["reclined_1027_0959", "seated_1029_1132", "reclined_laptop_1030_2134",
         "reclined_hand_at_head_1030_2214", "seated_laptop_stretch_0928"]
INTENDED = ["intended_reclined", "intended_reclined_2"]


def _snaps(name: str, **config) -> list:
    data = np.load(FIXTURES / f"{name}.npz")
    w, h = int(data["frame_w"]), int(data["frame_h"])
    # As deployed: cross-arms and T-pose off. Fresh detections only — the
    # analyser no longer re-feeds a frame (see test_refed_frames.py).
    tracker = ArmTracker(ArmTrackerConfig(detect_cross_arms=False,
                                          detect_t_pose=False, **config))
    sm = GestureStateMachine(StateMachineConfig())
    return [g for k, t in zip(data["keypoints"], data["t_offsets"])
            if (g := sm.tick(tracker.classify(_YOLOLandmarks(k, w, h), w, h,
                                              now=float(t)), now=float(t)))
            is Gesture.SNAP]


def test_the_default_is_058_degrees():
    assert ArmTrackerConfig().raise_elevation_min == 0.85


@pytest.mark.parametrize("name", FALSE)
def test_it_fired_at_the_old_angle(name):
    assert _snaps(name, raise_elevation_min=0.70)


@pytest.mark.parametrize("name", FALSE)
def test_a_hand_behind_the_head_is_not_raised_enough(name):
    assert _snaps(name) == []


@pytest.mark.parametrize("name", INTENDED)
def test_a_real_reclined_snap_still_fires(name):
    assert _snaps(name)


def test_the_cost_one_intended_snap_at_080():
    assert _snaps("intended_at_080", raise_elevation_min=0.70)
    assert _snaps("intended_at_080") == []
