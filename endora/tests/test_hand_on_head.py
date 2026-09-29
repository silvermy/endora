"""
tests/test_hand_on_head.py

The false positives from the easy chair (2026-09-28).

Someone resting a hand on or behind their head fired SNAP again and again,
from the easy chair and from reclining on the couch. The pose model does not
draw that as a hand on a head: it draws a short arm pointing up, which passes
elevation and extension like a real raise. What gives it away is how little
the wrist rises above its shoulder, about one head, measured in shoulder
widths (see ArmTrackerConfig.head_rest_rise_max).

The fixtures in tests/hand_on_head/ are recorded keypoint traces, each ending
at the frame that fired live. The four hand-on-head ones were checked against
the saved capture photos; genuine_flourish.npz is a real raise, seated on the
couch, from the same camera.

A recording stops at the moment it fired, and the guard legitimately delays
a real raise by a pose sample while the arm is still mid-swing, so every
replay holds the last frame for half a second before judging. A hand resting
on a head stays put, so holding it proves nothing in its favour.
"""
from pathlib import Path

import numpy as np
import pytest

from cameras.arm_tracker import ArmTracker, ArmTrackerConfig
from core.state_machine import Gesture, GestureStateMachine, StateMachineConfig
from tests.test_captured import _YOLOLandmarks

FIXTURES = Path(__file__).parent / "hand_on_head"
HAND_ON_HEAD = ["easy_chair_1", "easy_chair_2", "easy_chair_3", "couch_reclined"]


def _replay(name: str, **config) -> list:
    data = np.load(FIXTURES / f"{name}.npz")
    w, h = int(data["frame_w"]), int(data["frame_h"])
    kps, ts = list(data["keypoints"]), [float(t) for t in data["t_offsets"]]
    kps += [kps[-1]] * 5
    ts += [ts[-1] + 0.1 * (i + 1) for i in range(5)]
    # The two-handed poses are off on the live box; left on, they are tested
    # first and can shadow the single-arm raise this is about.
    tracker = ArmTracker(ArmTrackerConfig(detect_cross_arms=False, detect_t_pose=False,
                                          detect_both_up=False, **config))
    sm = GestureStateMachine(StateMachineConfig())
    fired = []
    for k, t in zip(kps, ts):
        gesture = sm.tick(tracker.classify(_YOLOLandmarks(k, w, h), w, h, now=t), now=t)
        if gesture is not None:
            fired.append(gesture)
    return fired


@pytest.mark.parametrize("name", HAND_ON_HEAD)
def test_the_recording_fires_without_the_guard(name):
    # Documents the bug: every one of these really did fire.
    assert Gesture.SNAP in _replay(name, head_rest_rise_max=0.0)


@pytest.mark.parametrize("name", HAND_ON_HEAD)
def test_a_hand_resting_on_the_head_does_not_fire(name):
    assert _replay(name) == []


def test_a_genuine_flourish_still_fires():
    fired = _replay("genuine_flourish")
    assert fired.count(Gesture.SNAP) == 2, fired


def test_the_guard_needs_both_shoulders():
    # With one shoulder hidden there is no width to measure the rise against;
    # an unmeasurable pose must never be rejected.
    tracker = ArmTracker(ArmTrackerConfig())
    assert not tracker._hand_on_head((100.0, 300.0), (100.0, 290.0), 0.0)


def test_zero_disables_the_guard():
    tracker = ArmTracker(ArmTrackerConfig(head_rest_rise_max=0.0))
    assert not tracker._hand_on_head((100.0, 300.0), (100.0, 290.0), 100.0)


def test_a_wrist_a_head_above_the_shoulder_is_rejected():
    tracker = ArmTracker(ArmTrackerConfig())
    # shoulder width 100 px, wrist 80 px above the shoulder: 0.8 widths.
    assert tracker._hand_on_head((100.0, 300.0), (110.0, 220.0), 100.0)
    # 150 px above: an arm really reaching up.
    assert not tracker._hand_on_head((100.0, 300.0), (110.0, 150.0), 100.0)
