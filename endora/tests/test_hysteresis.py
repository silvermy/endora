"""
tests/test_hysteresis.py

Tests for ArmTracker's hysteresis layer — the public classify() method.
Ensures phantom detections are filtered and mid-gesture dropouts don't
cause spurious state changes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cameras.arm_tracker import ArmTracker, ArmTrackerConfig, ArmState
from tests.fake_landmarks import arm_down, right_arm_up_vertical


def _tracker(**overrides) -> ArmTracker:
    return ArmTracker(ArmTrackerConfig(**overrides))


def test_single_frame_phantom_is_suppressed():
    """A single frame of SINGLE_UP (phantom / furniture hallucination)
    must not propagate as SINGLE_UP. User sees DOWN."""
    t = _tracker(state_confirm_frames=3)
    r = t.classify(right_arm_up_vertical(), 1280, 720)
    assert r.state == ArmState.DOWN, f"expected DOWN on frame 1, got {r.state}"


def test_three_consecutive_frames_confirm_state():
    """After state_confirm_frames consecutive SINGLE_UP readings,
    public classify should return SINGLE_UP."""
    t = _tracker(state_confirm_frames=3)
    # Frames 1 and 2: reported as DOWN (not confirmed)
    t.classify(right_arm_up_vertical(), 1280, 720)
    t.classify(right_arm_up_vertical(), 1280, 720)
    r = t.classify(right_arm_up_vertical(), 1280, 720)
    assert r.state == ArmState.SINGLE_UP, f"got {r.state}"


def test_flicker_does_not_confirm_state():
    """SINGLE_UP → DOWN → SINGLE_UP → DOWN must not confirm SINGLE_UP."""
    t = _tracker(state_confirm_frames=3)
    for _ in range(5):
        r = t.classify(right_arm_up_vertical(), 1280, 720)
        assert r.state == ArmState.DOWN
        r = t.classify(arm_down(), 1280, 720)
        assert r.state == ArmState.DOWN


def test_mid_gesture_dropout_does_not_release():
    """Once SINGLE_UP is stable, a single-frame DOWN (MediaPipe lost track)
    must NOT flip back to DOWN — stable state persists."""
    t = _tracker(state_confirm_frames=3, state_release_frames=4)
    # Confirm SINGLE_UP
    for _ in range(3):
        t.classify(right_arm_up_vertical(), 1280, 720)
    # Sanity check
    r = t.classify(right_arm_up_vertical(), 1280, 720)
    assert r.state == ArmState.SINGLE_UP
    # One dropout frame — must stay SINGLE_UP
    r = t.classify(arm_down(), 1280, 720)
    assert r.state == ArmState.SINGLE_UP, f"single dropout should be ignored, got {r.state}"


def test_sustained_arm_down_releases_state():
    """After state_release_frames of DOWN, we go back to DOWN."""
    t = _tracker(state_confirm_frames=3, state_release_frames=4)
    # Confirm SINGLE_UP
    for _ in range(3):
        t.classify(right_arm_up_vertical(), 1280, 720)
    # Now sustained arm-down — should release after release_frames
    for _ in range(4):
        t.classify(arm_down(), 1280, 720)
    r = t.classify(arm_down(), 1280, 720)
    assert r.state == ArmState.DOWN


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
        except Exception as e:
            failed += 1
            print(f"  ERROR {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{'PASSED' if failed == 0 else f'{failed} FAILED'} ({len(tests) - failed}/{len(tests)})")
    sys.exit(failed)
