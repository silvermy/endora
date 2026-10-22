"""
tests/test_raise_both_spread.py

RAISE_BOTH is a two-arm flare: both arms up AND out, hands outside the
elbows. Before 2026-10-22 it only asked whether both wrists were above the
head, and all of the recorded recorded fires was something else:

* stretch_hands_on_head — seated on the couch with a laptop, leaning back
  with both hands clasped on top of the head (labelled not intended);
* one_arm_side_on — standing side-on raising ONE arm, both wrists drawn in
  the same place;
* pid540_a — not photo-checked; the same narrow shape.

See ArmTrackerConfig.both_up_spread_min.
"""
from pathlib import Path

import numpy as np
import pytest

from cameras.arm_tracker import ArmState, ArmTracker, ArmTrackerConfig
from tests.fake_landmarks import SYNTHETIC_BODY, both_arms_up
from tests.test_captured import _YOLOLandmarks

FIXTURES = Path(__file__).parent / "raise_both"
NARROW = ["stretch_hands_on_head", "one_arm_side_on", "pid540_a"]


def _last_state(name: str, **config) -> ArmState:
    data = np.load(FIXTURES / f"{name}.npz")
    w, h = int(data["frame_w"]), int(data["frame_h"])
    tracker = ArmTracker(ArmTrackerConfig(head_rest_rise_max=0.0, **config))
    return tracker._classify_raw(_YOLOLandmarks(data["keypoints"][-1], w, h), w, h).state


@pytest.mark.parametrize("name", NARROW)
def test_the_recording_read_as_both_up_without_the_rule(name):
    assert _last_state(name, both_up_spread_min=0.0) is ArmState.BOTH_UP


@pytest.mark.parametrize("name", NARROW)
def test_hands_closer_than_the_elbows_are_not_both_up(name):
    assert _last_state(name) is not ArmState.BOTH_UP


def test_arms_up_and_out_is_both_up():
    tracker = ArmTracker(ArmTrackerConfig(**SYNTHETIC_BODY))
    assert tracker._classify_raw(both_arms_up(), 1280, 720).state is ArmState.BOTH_UP


def test_the_spread_rule():
    tracker = ArmTracker(ArmTrackerConfig())
    elbows = ((100.0, 200.0), (300.0, 200.0))
    assert tracker._arms_spread((80.0, 50.0), (320.0, 50.0), *elbows)      # a V
    assert not tracker._arms_spread((180.0, 50.0), (220.0, 50.0), *elbows)  # a clasp
    off = ArmTracker(ArmTrackerConfig(both_up_spread_min=0.0))
    assert off._arms_spread((180.0, 50.0), (220.0, 50.0), *elbows)
