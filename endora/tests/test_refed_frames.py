"""
tests/test_refed_frames.py

A person is classified only on frames where they were actually detected.

The analyser used to classify every tracked person on every iteration with
last_lm, the keypoints from their LAST detection, so a dropout replayed a
frozen pose as though it were new frames. One fresh frame with the arm up,
then the same frame again 0.1 s later, passed the "seen on two frames" guard,
and the jump across the gap passed as a sweep.

Fixtures (tests/refed_frames/), each a recording that ends on the frame
that fired live:

* couch_no_pose_2218 — 2026-10-30 22:18, lying on the couch with a laptop,
  labelled not intended. 0.71 s with no detection, then one frame with the
  arm up. Its photo reads NO POSE DETECTED.
* couch_no_pose_1038 — 2026-10-30 10:38, lying on the couch, no arm raised
  in the photo, which also reads NO POSE DETECTED.
* chin_in_hand — 2026-10-03, the phantom tests/test_fast_confirm.py is about,
  labelled not intended.

Holding the last frame stands in for the re-fed frames. Every recording
labelled intended (26 snaps, 7 Jeannies) fires without it.
"""
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import MagicMock

from cameras.analyser import CameraAnalyser, _detected_now
from cameras.arm_tracker import ArmTracker, ArmTrackerConfig
from config.settings import Settings
from core.state_machine import Gesture, GestureStateMachine, StateMachineConfig
from tests.test_captured import _YOLOLandmarks

FIXTURES = Path(__file__).parent / "refed_frames"
REFED = ["couch_no_pose_2218", "couch_no_pose_1038", "chin_in_hand"]


def _snaps(name: str, refed: int, **config) -> list:
    data = np.load(FIXTURES / f"{name}.npz")
    w, h = int(data["frame_w"]), int(data["frame_h"])
    kps, ts = list(data["keypoints"]), [float(t) for t in data["t_offsets"]]
    kps += [kps[-1]] * refed
    ts += [ts[-1] + 0.1 * (i + 1) for i in range(refed)]
    tracker = ArmTracker(ArmTrackerConfig(detect_cross_arms=False, detect_t_pose=False,
                                          **config))
    sm = GestureStateMachine(StateMachineConfig())
    return [g for k, t in zip(kps, ts)
            if (g := sm.tick(tracker.classify(_YOLOLandmarks(k, w, h), w, h, now=t),
                             now=t)) is Gesture.SNAP]


@pytest.mark.parametrize("name", REFED)
def test_it_fired_only_because_a_frame_was_fed_twice(name):
    # At the 0.70 raise angle these fired under; chin_in_hand is no longer a
    # raise at all at 0.85 (see test_arm_angle.py).
    assert _snaps(name, refed=5, raise_elevation_min=0.70)


@pytest.mark.parametrize("name", REFED)
def test_fresh_detections_alone_do_not_fire(name):
    assert _snaps(name, refed=0) == []


def test_a_person_missed_on_a_frame_is_not_classified_on_it():
    a = CameraAnalyser(camera=MagicMock(), settings=Settings(),
                       on_candidate=MagicMock(), label="test")
    a._match_persons([(None, (400.0, 300.0), True, True)], 640, 480, now=1.0)
    entry = next(iter(a._persons.values()))
    assert _detected_now(entry, 1.0)
    a._yolo_runs += 1
    a._match_persons([], 640, 480, now=1.1)          # a run that missed them
    assert not _detected_now(entry, 1.1)
    a._match_persons([(None, (402.0, 301.0), True, True)], 640, 480, now=1.2)
    assert _detected_now(entry, 1.2)
