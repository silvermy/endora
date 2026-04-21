"""
tests/test_arm_tracker.py

Unit tests for ArmTracker. Runs without MediaPipe — uses fake landmarks.

Run with: python -m pytest tests/test_arm_tracker.py -v
Or:       python tests/test_arm_tracker.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cameras.arm_tracker import ArmTracker, ArmTrackerConfig, ArmState, Side
from tests.fake_landmarks import (
    arm_down, right_arm_up_vertical, right_arm_up_horizontal,
    both_arms_up, t_pose, cross_arms, lying_down, low_visibility,
)


def _tracker() -> ArmTracker:
    return ArmTracker(ArmTrackerConfig())


def test_arm_down_returns_down_state():
    r = _tracker().classify(arm_down(), 1280, 720)
    assert r.state == ArmState.DOWN


def test_right_arm_vertical_is_single_up_right():
    r = _tracker().classify(right_arm_up_vertical(), 1280, 720)
    assert r.state == ArmState.SINGLE_UP
    assert r.raised_side == Side.RIGHT
    assert r.forearm_dy > 0.10, f"forearm_dy should be vertical, got {r.forearm_dy}"


def test_right_arm_horizontal_is_not_single_up():
    # Wrist at same height as shoulder → not raised above head
    r = _tracker().classify(right_arm_up_horizontal(), 1280, 720)
    assert r.state == ArmState.DOWN


def test_both_arms_up_is_both_up():
    r = _tracker().classify(both_arms_up(), 1280, 720)
    assert r.state == ArmState.BOTH_UP


def test_t_pose_is_t_pose():
    r = _tracker().classify(t_pose(), 1280, 720)
    assert r.state == ArmState.T_POSE


def test_cross_arms_is_cross_arms():
    r = _tracker().classify(cross_arms(), 1280, 720)
    assert r.state == ArmState.CROSS_ARMS


def test_lying_down_single_up_rejected():
    # Even if arm would be up, lying-down body is not upright → DOWN
    from tests.fake_landmarks import _build, Point
    lm = _build(
        # Hips well above shoulders (lying on back, feet toward camera)
        left_hip=Point(0.42, 0.20), right_hip=Point(0.58, 0.20),
        right_elbow=Point(0.65, 0.25), right_wrist=Point(0.65, 0.10),
    )
    r = _tracker().classify(lm, 1280, 720)
    assert r.state == ArmState.DOWN
    assert r.upright is False


def test_low_visibility_returns_none():
    r = _tracker().classify(low_visibility(), 1280, 720)
    assert r is None


def test_none_landmarks_returns_none():
    r = _tracker().classify(None, 1280, 720)
    assert r is None


def test_both_up_prefers_both_over_single():
    # Ambiguity: both arms raised — must return BOTH_UP, not SINGLE_UP
    r = _tracker().classify(both_arms_up(), 1280, 720)
    assert r.state == ArmState.BOTH_UP


def test_t_pose_prefers_t_over_both():
    # T-pose wrists are at shoulder height, not above → not BOTH_UP
    r = _tracker().classify(t_pose(), 1280, 720)
    assert r.state == ArmState.T_POSE


if __name__ == "__main__":
    # Minimal runner without pytest
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
        except Exception as e:
            failed += 1
            print(f"  ERROR {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{'PASSED' if failed == 0 else f'{failed} FAILED'} ({len(tests) - failed}/{len(tests)})")
    sys.exit(failed)
