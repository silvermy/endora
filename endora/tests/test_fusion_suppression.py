"""
tests/test_fusion_suppression.py

A gesture can fire perfectly in the analyser and still never reach Home
Assistant, because fusion applies cooldowns after the fact. That gap
produced a genuinely confusing symptom in the field: the chime played at
exactly the right moment (it is emitted in the analyser) while nothing
happened in HA, and feedback.jsonl showed no trace of the gesture at all —
indistinguishable from it never having been detected.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Settings
from core.fusion import GestureFusion
from core.state_machine import Gesture


def _fusion(**over):
    s = Settings()
    s.rtsp_url_b = s.rtsp_url_a          # single-camera mode
    for k, v in over.items():
        setattr(s, k, v)
    emitted, suppressed = [], []
    f = GestureFusion(s,
                      on_gesture=lambda g, c, src: emitted.append(g),
                      on_suppressed=lambda g, r: suppressed.append((g, r)))
    return f, emitted, suppressed


def test_a_pose_gesture_does_not_swallow_a_following_snap():
    # The field case: a spurious CROSS_ARMS immediately before a real
    # flourish. At the old shared cooldown the SNAP was dropped for a full
    # two seconds, after the chime had already played.
    f, emitted, _ = _fusion(cooldown_s=2.0, cross_gesture_cooldown_s=0.5)
    f.receive(Gesture.CROSS_ARMS, 1.0, "A")
    import time as _t
    _t.sleep(0.6)                         # longer than the cross-gesture window
    f.receive(Gesture.SNAP, 1.0, "A")
    assert Gesture.SNAP in emitted, f"SNAP was swallowed; emitted={emitted}"


def test_same_gesture_still_respects_the_full_cooldown():
    f, emitted, suppressed = _fusion(cooldown_s=2.0, cross_gesture_cooldown_s=0.5)
    f.receive(Gesture.SNAP, 1.0, "A")
    f.receive(Gesture.SNAP, 1.0, "A")     # immediate repeat
    assert emitted.count(Gesture.SNAP) == 1
    assert suppressed, "the dropped repeat must be reported"


def test_suppression_is_reported_not_silent():
    """The whole point: a dropped gesture must leave a trace, so it can be
    told apart from one that never fired."""
    f, _, suppressed = _fusion(cooldown_s=2.0, cross_gesture_cooldown_s=0.5)
    f.receive(Gesture.SNAP, 1.0, "A")
    f.receive(Gesture.CROSS_ARMS, 1.0, "A")   # inside the cross-gesture window
    assert suppressed, "suppression went unreported"
    name, reason = suppressed[0]
    assert name == "CROSS_ARMS" and "cooldown" in reason


if __name__ == "__main__":
    import traceback
    failed = 0
    for fn in [v for k, v in dict(globals()).items()
               if k.startswith("test_") and callable(v)]:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL  {fn.__name__}: {e}")
        except Exception:
            failed += 1; print(f"  ERROR {fn.__name__}"); traceback.print_exc()
    sys.exit(failed)
