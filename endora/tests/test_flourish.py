"""
tests/test_flourish.py

The gesture Endora actually performs: a theatrical arm sweep, up and
usually straight back down — not a raised hand held still.

This is the discriminator the whole detector now rests on. Every static
false positive we ever fought (an arm on an armrest, draped over a
backrest, holding a phone, a hand at the face) is geometrically similar to
a raise but sweeps at a rate of essentially zero, so requiring the sweep
excludes all of them by construction rather than by accumulated gates.
"""
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cameras.arm_tracker import ArmTracker, ArmTrackerConfig, ArmState
from core.state_machine import Gesture, GestureStateMachine, StateMachineConfig
from tests.fake_landmarks import Landmarks, Point
from tests.fake_landmarks import (
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
)

W, H, TORSO = 1280, 640, 150      # the user's dewarped frame size


def _pose(angle_deg: float, cx: float = 640.0, cy: float = 320.0) -> Landmarks:
    """Right arm at *angle_deg* from hanging-down (0) to straight-up (180),
    swung in the frame plane. 90 is straight out sideways.
    """
    sh = (cx + TORSO * 0.22, cy - TORSO / 2)
    a = math.radians(angle_deg)
    ex, ey = sh[0] + math.sin(a) * TORSO * 0.42, sh[1] + math.cos(a) * TORSO * 0.42
    wx, wy = sh[0] + math.sin(a) * TORSO * 0.85, sh[1] + math.cos(a) * TORSO * 0.85
    P = lambda p: Point(p[0] / W, p[1] / H)
    return Landmarks({
        NOSE: P((cx, cy - TORSO * 0.78)),
        LEFT_SHOULDER: P((cx - TORSO * 0.22, cy - TORSO / 2)),
        RIGHT_SHOULDER: P(sh),
        LEFT_ELBOW: P((cx - TORSO * 0.24, cy - TORSO * 0.08)),
        LEFT_WRIST: P((cx - TORSO * 0.26, cy + TORSO * 0.35)),
        RIGHT_ELBOW: P((ex, ey)), RIGHT_WRIST: P((wx, wy)),
        LEFT_HIP: P((cx - TORSO * 0.18, cy + TORSO / 2)),
        RIGHT_HIP: P((cx + TORSO * 0.18, cy + TORSO / 2)),
        LEFT_KNEE: P((cx - TORSO * 0.18, cy + TORSO * 1.3)),
        RIGHT_KNEE: P((cx + TORSO * 0.18, cy + TORSO * 1.3)),
    })


def _run(angles, fps=10.0, **cfg):
    """Play a sequence of arm angles through the full pipeline."""
    tr = ArmTracker(ArmTrackerConfig())
    sm = GestureStateMachine(StateMachineConfig(snap_sustain_s=0.0, **cfg))
    fired = []
    for i, ang in enumerate(angles):
        t = i / fps          # exact, so frame timings don't drift
        r = tr.classify(_pose(ang), W, H, None, now=t)
        g = sm.tick(r, t) if r is not None else None
        if g is not None:
            fired.append(g)
    return fired


SWEEP_UP = [0, 25, 55, 90, 120, 150, 170, 178]
SWEEP_DOWN = [178, 165, 140, 100, 60, 25, 0]


def test_flourish_fires():
    # The gesture: sweep up and straight back down, never held. Under the
    # previous hold-based logic this fired nothing at all.
    assert Gesture.SNAP in _run(SWEEP_UP + SWEEP_DOWN)


def test_flourish_fires_at_lower_frame_rates():
    for fps in (8.0, 10.0, 15.0):
        assert Gesture.SNAP in _run(SWEEP_UP + SWEEP_DOWN, fps=fps), f"failed at {fps} fps"


def test_flourish_fires_before_the_arm_comes_down():
    # It must fire at the top of the arc — waiting for the descent would put
    # the lights well behind the gesture.
    assert Gesture.SNAP in _run(SWEEP_UP)


def test_flourish_from_an_armrest_fires():
    # Arm starts already part-way up (resting on an armrest, ~elevation 0.3)
    # rather than hanging down. Less climb is available, but the sweep is
    # still unmistakable.
    resting = [70] * 25
    assert Gesture.SNAP in _run(resting + [95, 125, 155, 175] + SWEEP_DOWN)


def test_static_raised_arm_never_fires():
    # An arm parked straight up and left there: geometrically a perfect
    # raise, but it never swept, so it is not a gesture.
    assert not _run([178] * 60)


def test_arm_resting_on_armrest_never_fires():
    assert not _run([70] * 100)


def test_arm_draped_over_backrest_never_fires():
    # Drifts slowly between two raised-looking positions — climb accumulates
    # but far too slowly to be a flourish.
    assert not _run(([100] * 20 + [115] * 20) * 3)


def test_very_slow_arm_raise_does_not_fire():
    # Ten seconds to lift the arm is not a flourish. This is the deliberate
    # trade: the gesture is defined by its speed.
    slow = [i * 2 for i in range(90)]
    assert not _run(slow)


def test_flourish_gate_can_be_disabled_for_hold_style():
    # Escape hatch: turn the flourish requirement off and a held raise fires
    # again, for anyone who prefers the classroom-hand gesture.
    held = SWEEP_UP + [178] * 20
    assert Gesture.SNAP in _run(held, snap_require_flourish=False)


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


# ── Chime gating ──────────────────────────────────────────────────────────────

def test_chime_requires_climb_not_just_rate():
    """Regression: the chime once triggered on sweep_rate alone.

    Rate is climb divided by elapsed time, so a hand's worth of keypoint
    jitter between two adjacent frames divides a tiny climb by a tiny
    interval and yields a large rate. An arm resting near horizontal — where
    elevation is most sensitive to wrist noise — chimed several times a
    minute with no gesture behind it, which is exactly what a user hears as
    "sound effects for no reason".
    """
    from cameras.analyser import _sweep_meets_flourish
    from cameras.arm_tracker import ArmReading, ArmState

    def r(climb, rate):
        return ArmReading(state=ArmState.DOWN, sweep_climb=climb, sweep_rate=rate)

    # Jitter: negligible travel, but a huge rate because the interval is tiny.
    assert not _sweep_meets_flourish(r(0.08, 4.0), 0.60, 0.80)
    assert not _sweep_meets_flourish(r(0.20, 2.0), 0.60, 0.80)
    # A real sweep clears both bars.
    assert _sweep_meets_flourish(r(1.60, 2.40), 0.60, 0.80)
    # A slow drift clears the climb but not the rate.
    assert not _sweep_meets_flourish(r(1.20, 0.30), 0.60, 0.80)
    assert not _sweep_meets_flourish(None, 0.60, 0.80)


def test_resting_arm_never_meets_the_chime_bar():
    """Play a resting arm through the tracker and confirm no frame would chime."""
    from cameras.analyser import _sweep_meets_flourish
    tr = ArmTracker(ArmTrackerConfig())
    for i in range(150):                       # 15 s at rest, arm out horizontal
        r = tr.classify(_pose(90), W, H, None, now=i / 10.0)
        assert not _sweep_meets_flourish(r, 0.60, 0.80), f"would chime at frame {i}"
