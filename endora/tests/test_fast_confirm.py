"""
tests/test_fast_confirm.py

A raise on the end of a big sweep skips the state_confirm_s wait, so a fast
flourish is not dropped while it is briefly at the top. It must still be seen
on a second frame.

Live evidence, 2026-10-03: someone reclining with their chin in their hand.
The wrist sits above the shoulder however folded the arm is, so those frames
read as high elevation and made the sweep look progressive; then the pose
model lost the body for one frame and drew a straight, raised arm (extension
0.96, hips at 0.30 confidence). With a zero wait that one frame confirmed the
raise and fired SNAP. The recording ends on that frame.
"""
from pathlib import Path

import numpy as np

from cameras.arm_tracker import ArmTracker, ArmTrackerConfig
from core.state_machine import Gesture, GestureStateMachine, StateMachineConfig
from tests.test_captured import _YOLOLandmarks

PHANTOM = Path(__file__).parent / "hand_on_head" / "chin_in_hand_phantom.npz"


def _snaps(extra_frames: int) -> list:
    data = np.load(PHANTOM)
    w, h = int(data["frame_w"]), int(data["frame_h"])
    kps, ts = list(data["keypoints"]), [float(t) for t in data["t_offsets"]]
    kps += [kps[-1]] * extra_frames
    ts += [ts[-1] + 0.06 * (i + 1) for i in range(extra_frames)]
    tracker = ArmTracker(ArmTrackerConfig(detect_cross_arms=False, detect_t_pose=False,
                                          detect_both_up=False))
    sm = GestureStateMachine(StateMachineConfig())
    fired = []
    for k, t in zip(kps, ts):
        g = sm.tick(tracker.classify(_YOLOLandmarks(k, w, h), w, h, now=t), now=t)
        if g is Gesture.SNAP:
            fired.append(t)
    return fired


def test_one_frame_cannot_confirm_a_raise():
    assert _snaps(extra_frames=0) == []


def test_the_same_raise_seen_again_still_fires():
    """The trade-off: a raise that holds for a second frame is accepted, so
    this only stops phantoms that last a single frame."""
    assert len(_snaps(extra_frames=1)) == 1
