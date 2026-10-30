"""
tests/test_other_arm_overhead.py

A one-arm raise does not count while the other hand is clearly overhead too.

Both SNAPs labelled "not intended" on the debug page since 2026-10-21 were
a stretch, seated:

* hands_clasped_overhead — both hands clasped on top of the head (the same
  moment as raise_both/stretch_hands_on_head, which v1.9.200 stopped from
  firing RAISE_BOTH; it still fired SNAP);
* one_held_one_raised — one arm held up for seconds, then the other raised
  to join it;
* reclined_hands_behind_head — 2026-10-26 evening, lying back on the couch
  with both hands behind the head, the other arm at 0.93 but seen at only
  0.64. It got past the first defaults (0.95 / 0.80).
* both_up_elbow_collapsed — 2026-10-29, both hands straight up together.
  On the frame that fired, the model put the right elbow almost on the
  shoulder, the elbow check made that arm unreadable, and the veto, which
  used the zeroed elevation, saw no second hand. The veto now measures the
  other hand from shoulder and wrist alone.

Each has the other arm at elevation 0.98-0.99, bent, so it never counted as
raised and the frame fell through to SINGLE_UP. The genuine fixtures are
labelled intended: one with the other arm hanging down, and one where the
other arm briefly read as high (up to 0.96, at 0.53-0.59 confidence) on the
frame where both arms counted as raised.

See ArmTrackerConfig.other_arm_overhead_max.
"""
from pathlib import Path

import numpy as np
import pytest

from cameras.arm_tracker import ArmTracker, ArmTrackerConfig
from core.state_machine import Gesture, GestureStateMachine, StateMachineConfig
from tests.test_captured import _YOLOLandmarks

FIXTURES = Path(__file__).parent / "stretch"
STRETCHES = ["hands_clasped_overhead", "one_held_one_raised",
             "reclined_hands_behind_head", "both_up_elbow_collapsed"]
GENUINE = ["genuine_other_arm_down", "genuine_other_arm_barely_seen"]


def _replay(name: str, **config) -> list:
    data = np.load(FIXTURES / f"{name}.npz")
    w, h = int(data["frame_w"]), int(data["frame_h"])
    kps, ts = list(data["keypoints"]), [float(t) for t in data["t_offsets"]]
    # A recording stops at the frame that fired; hold it, as the other
    # recorded-trace tests do, so a one-sample delay is not a regression.
    kps += [kps[-1]] * 5
    ts += [ts[-1] + 0.1 * (i + 1) for i in range(5)]
    # As deployed: cross-arms and T-pose off, RAISE_BOTH on.
    tracker = ArmTracker(ArmTrackerConfig(detect_cross_arms=False,
                                          detect_t_pose=False, **config))
    sm = GestureStateMachine(StateMachineConfig())
    fired = []
    for k, t in zip(kps, ts):
        g = sm.tick(tracker.classify(_YOLOLandmarks(k, w, h), w, h, now=t), now=t)
        if g is not None:
            fired.append(g)
    return fired


@pytest.mark.parametrize("name", STRETCHES)
def test_the_recording_fired_without_the_rule(name):
    # Documents the bug: each of these really did fire.
    assert Gesture.SNAP in _replay(name, other_arm_overhead_max=0.0)


@pytest.mark.parametrize("name", STRETCHES)
def test_a_stretch_with_both_hands_up_does_not_snap(name):
    assert Gesture.SNAP not in _replay(name)


@pytest.mark.parametrize("name", GENUINE)
def test_a_genuine_snap_still_fires(name):
    assert Gesture.SNAP in _replay(name)


def test_the_first_defaults_let_the_reclined_stretch_through():
    # Documents why 0.95 / 0.80 was loosened.
    assert Gesture.SNAP in _replay("reclined_hands_behind_head",
                                   other_arm_overhead_max=0.95,
                                   other_arm_overhead_visibility=0.80)


def test_only_the_arm_whose_partner_is_overhead_loses():
    # When both arms read as raised, the confident one keeps its raise
    # because its partner is barely seen (0.53-0.59, under the 0.60 floor);
    # vetoing both dropped this snap.
    assert ArmTrackerConfig().other_arm_overhead_visibility > 0.59
    assert Gesture.SNAP in _replay("genuine_other_arm_barely_seen")


def test_the_veto_does_not_need_the_other_elbow():
    """The firing frame of both_up_elbow_collapsed, alone: the right elbow
    sits on the shoulder, so that arm fails _elbow_is_articulated — but its
    wrist is plainly overhead, and that is all the veto asks."""
    from cameras.arm_tracker import _elbow_is_articulated
    data = np.load(FIXTURES / "both_up_elbow_collapsed.npz")
    last = data["keypoints"][-1]
    s, e, w = (tuple(last[i][:2]) for i in (6, 8, 10))
    assert not _elbow_is_articulated(s, e, w)
    assert (s[1] - w[1]) / np.hypot(s[0] - w[0], s[1] - w[1]) > 0.9
