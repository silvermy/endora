"""
tests/test_hand_crop.py

_crop_around_wrist: the hand-detection crop that makes snap_roll usable at
couch distance. Verifies geometry (centred above the wrist, sized from the
forearm, clamped to the frame, upscaled when small) and the fallbacks.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from cameras.analyser import _crop_around_wrist, _HAND_CROP_UPSCALE_PX
from cameras.arm_tracker import (
    ArmReading, ArmState, Side, RIGHT_ELBOW, RIGHT_WRIST,
)
from tests.fake_landmarks import _build, Point


def _reading(wx, wy) -> ArmReading:
    return ArmReading(state=ArmState.SINGLE_UP, raised_side=Side.RIGHT,
                      wrist_x=wx, wrist_y=wy)


def test_crop_is_square_and_upscaled():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    lm = _build(right_elbow=Point(0.50, 0.60), right_wrist=Point(0.50, 0.50))
    crop = _crop_around_wrist(frame, _reading(320, 240), lm)
    assert crop is not None
    assert crop.shape[:2] == (_HAND_CROP_UPSCALE_PX, _HAND_CROP_UPSCALE_PX)


def test_crop_contains_the_wrist_area():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Mark the hand region (just above the wrist) and verify it lands in the crop.
    frame[200:220, 310:330] = 255
    lm = _build(right_elbow=Point(0.50, 0.60), right_wrist=Point(0.50, 0.50))
    crop = _crop_around_wrist(frame, _reading(320, 240), lm)
    assert crop is not None and crop.max() == 255


def test_crop_clamps_at_frame_edge():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    lm = _build(right_elbow=Point(0.02, 0.15), right_wrist=Point(0.02, 0.05))
    crop = _crop_around_wrist(frame, _reading(10, 20), lm)
    assert crop is not None, "edge wrist should still produce a (clamped) crop"


def test_no_raised_side_returns_none():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    lm = _build()
    r = ArmReading(state=ArmState.SINGLE_UP, raised_side=None)
    assert _crop_around_wrist(frame, r, lm) is None


if __name__ == "__main__":
    import traceback
    failed = 0
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception:
            failed += 1
            print(f"  ERROR {t.__name__}")
            traceback.print_exc()
    sys.exit(failed)
