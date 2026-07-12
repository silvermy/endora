"""
tests/test_snap_roll.py

_hand_snap_roll must be a real orientation signal. The pre-v1.9.115 formula
divided (index.x - pinky.x) by its own absolute value, so every detected
hand read exactly ±1.0 — which silently armed the snap_roll_threshold
OR-route once the wrist-crop made hand detection reliable.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from cameras.arm_tracker import _hand_snap_roll


def _hand(wrist=(0.5, 0.9), middle_mcp=(0.5, 0.5),
          index_mcp_x=0.35, pinky_mcp_x=0.65) -> np.ndarray:
    lm = np.zeros(63, dtype=np.float32)
    lm[0], lm[1] = wrist                    # WRIST x,y
    lm[9 * 3], lm[9 * 3 + 1] = middle_mcp   # MIDDLE_FINGER_MCP x,y
    lm[5 * 3] = index_mcp_x                 # INDEX_FINGER_MCP x
    lm[17 * 3] = pinky_mcp_x                # PINKY_MCP x
    return lm


def test_palm_facing_camera_reads_high_magnitude():
    # Knuckles spread laterally ≈ 0.75 of the wrist→middle-MCP length.
    roll = _hand_snap_roll(_hand())
    assert 0.5 < abs(roll) < 1.2, f"got {roll}"


def test_edge_on_hand_reads_low_magnitude():
    roll = _hand_snap_roll(_hand(index_mcp_x=0.52, pinky_mcp_x=0.48))
    assert abs(roll) < 0.3, f"got {roll}"


def test_roll_is_not_saturated_to_unit_value():
    # Regression for the old formula: two different knuckle spreads must give
    # two different magnitudes, not both exactly 1.0.
    wide = _hand_snap_roll(_hand(index_mcp_x=0.30, pinky_mcp_x=0.70))
    narrow = _hand_snap_roll(_hand(index_mcp_x=0.42, pinky_mcp_x=0.58))
    assert abs(wide) != abs(narrow)
    assert abs(abs(wide) - 1.0) > 1e-6 or abs(abs(narrow) - 1.0) > 1e-6


def test_sign_follows_hand_orientation():
    a = _hand_snap_roll(_hand(index_mcp_x=0.35, pinky_mcp_x=0.65))
    b = _hand_snap_roll(_hand(index_mcp_x=0.65, pinky_mcp_x=0.35))
    assert a == -b != 0


def test_degenerate_input_is_zero():
    assert _hand_snap_roll(np.zeros(63, dtype=np.float32)) == 0.0
    assert _hand_snap_roll(np.zeros(10, dtype=np.float32)) == 0.0


def test_extreme_spread_is_clamped():
    roll = _hand_snap_roll(_hand(middle_mcp=(0.5, 0.85),
                                 index_mcp_x=0.1, pinky_mcp_x=0.9))
    assert abs(roll) <= 1.5


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
