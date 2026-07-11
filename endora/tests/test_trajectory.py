"""
tests/test_trajectory.py

Trajectory evidence for SNAP:
  rose_recently — the wrist was seen below shoulder level within
    raise_travel_window_s (a deliberate gesture starts with an actual upward
    motion; a pose that has simply existed since tracking began did not rise).
  wrist_still — the raised wrist has held its position over
    wrist_still_window_s (a reach for a phone/blanket passes through the
    raised zone while still moving).

Plus the GestureStateMachine gates that consume them.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cameras.arm_tracker import (
    ArmReading, ArmState, ArmTracker, ArmTrackerConfig, Side,
)
from core.state_machine import Gesture, GestureStateMachine, StateMachineConfig
from tests.fake_landmarks import _build, Point, arm_down, right_arm_up_vertical


def _tracker() -> ArmTracker:
    return ArmTracker(ArmTrackerConfig())


def _feed(tracker, schedule):
    """Run classify() over [(t, landmarks), ...]; return list of (t, reading)."""
    out = []
    for t, lm in schedule:
        out.append((t, tracker.classify(lm, 1280, 720, None, now=t)))
    return out


# ── Tracker-level trajectory flags ───────────────────────────────────────────

def test_rise_then_hold_becomes_still_and_rose():
    tr = _tracker()
    sched = [(round(0.1 * i, 1), arm_down()) for i in range(5)]           # 0.0–0.4 down
    sched += [(round(0.5 + 0.1 * i, 1), right_arm_up_vertical()) for i in range(11)]  # 0.5–1.5 up
    readings = _feed(tr, sched)

    ups = [(t, r) for t, r in readings if r is not None and r.state == ArmState.SINGLE_UP]
    assert ups, "raise never confirmed"
    # The raise came from below shoulder level moments ago → rose_recently,
    # and a held-up arm reads still (in-transit non-stillness is covered by
    # test_moving_wrist_is_not_still).
    assert all(r.rose_recently for _, r in ups)
    assert ups[-1][1].wrist_still is True, "held arm should read still"


def test_pose_static_since_tracking_start_loses_rise_benefit():
    # Arm up from the very first frame and never anywhere else (chin propped
    # on hand, ghost with a permanently-raised arm). While the history buffer
    # is younger than the window the benefit of the doubt applies; once the
    # buffer spans the window with no below-shoulder sighting, rose_recently
    # goes False.
    tr = _tracker()
    sched = [(round(0.1 * i, 1), right_arm_up_vertical()) for i in range(31)]  # 0.0–3.0
    readings = _feed(tr, sched)

    ups = [(t, r) for t, r in readings if r is not None and r.state == ArmState.SINGLE_UP]
    early = [r for t, r in ups if t < 2.0]
    late = [r for t, r in ups if t >= 2.5]
    assert early and all(r.rose_recently for r in early), \
        "young buffer must keep the benefit of the doubt"
    assert late and all(not r.rose_recently for r in late), \
        "static pose must lose rose_recently once history spans the window"
    # It never moved, so it reads still — rise is the gate that blocks it.
    assert all(r.wrist_still for r in late)


def test_moving_wrist_is_not_still():
    # A reach: the wrist keeps travelling upward through the raised zone.
    tr = _tracker()
    ys = [0.70, 0.58, 0.46, 0.34, 0.22, 0.10, 0.22]   # up and back down
    sched = []
    for i, y in enumerate(ys):
        sched.append((round(0.1 * i, 1), _build(
            right_elbow=Point(0.65, y + 0.14),
            right_wrist=Point(0.65, y),
        )))
    readings = _feed(tr, sched)
    ups = [r for _, r in readings if r is not None and r.state == ArmState.SINGLE_UP]
    assert ups, "moving arm should still classify as SINGLE_UP while high"
    assert all(not r.wrist_still for r in ups), \
        "a wrist in transit must never read as still"


def test_reclined_resting_wrist_still_counts_as_rise_evidence():
    # Lying on a couch the resting wrist sits at almost the same image height
    # as the shoulder (sometimes a hair above it). That must still count as
    # "the arm was down" — otherwise a reclined person could never produce
    # rise evidence and reclined SNAPs would be permanently blocked.
    from tests.fake_landmarks import Landmarks
    from tests.fake_landmarks import (
        NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    )

    def _reclined(rw_y, re_y):
        return Landmarks({
            NOSE:           Point(0.28, 0.50),
            LEFT_SHOULDER:  Point(0.34, 0.49),
            RIGHT_SHOULDER: Point(0.36, 0.51),
            LEFT_HIP:       Point(0.59, 0.49),
            RIGHT_HIP:      Point(0.61, 0.51),
            LEFT_KNEE:      Point(0.70, 0.52),
            RIGHT_KNEE:     Point(0.72, 0.52),
            LEFT_ELBOW:     Point(0.47, 0.53),
            LEFT_WRIST:     Point(0.50, 0.52),
            RIGHT_ELBOW:    Point(0.40, re_y),
            RIGHT_WRIST:    Point(0.40, rw_y),
        })

    resting = _reclined(rw_y=0.49, re_y=0.52)   # wrist 0.02 ABOVE shoulder line
    raised  = _reclined(rw_y=0.10, re_y=0.30)   # deliberate straight-up arm

    tr = _tracker()
    # Long rest — buffer fully spans the raise window, so no benefit of the
    # doubt is in play; the evidence must come from the resting samples.
    sched = [(round(0.1 * i, 1), resting) for i in range(30)]     # 0.0–2.9
    sched += [(round(3.0 + 0.1 * i, 1), raised) for i in range(11)]  # 3.0–4.0
    readings = _feed(tr, sched)

    ups = [r for _, r in readings if r is not None and r.state == ArmState.SINGLE_UP]
    assert ups, "reclined raise never confirmed"
    assert ups[0].upright is False
    assert all(r.rose_recently for r in ups), \
        "near-shoulder resting wrist must count as rise evidence"
    assert ups[-1].wrist_still is True


# ── State-machine gates ──────────────────────────────────────────────────────
# Configs mirror the production settings defaults (snap_sustain_s=0.2,
# snap_forearm_min=0.06) rather than the StateMachineConfig dataclass
# defaults, which are stricter.

def _machine(**overrides) -> GestureStateMachine:
    cfg = dict(snap_sustain_s=0.2, snap_forearm_min=0.06)
    cfg.update(overrides)
    return GestureStateMachine(StateMachineConfig(**cfg))


def _up_reading(**kw) -> ArmReading:
    defaults = dict(state=ArmState.SINGLE_UP, raised_side=Side.RIGHT,
                    wrist_x=800, wrist_y=100, forearm_dy=0.15)
    defaults.update(kw)
    return ArmReading(**defaults)


def test_snap_blocked_without_rise():
    m = _machine()
    fired = [m.tick(_up_reading(rose_recently=False), t / 10) for t in range(15)]
    assert not any(fired), "SNAP must not fire without rise evidence"


def test_snap_blocked_while_wrist_moving_then_fires_when_still():
    m = _machine()
    fired = [m.tick(_up_reading(wrist_still=False), t / 10) for t in range(5)]
    assert not any(fired), "SNAP must not fire while the wrist is moving"
    g = m.tick(_up_reading(wrist_still=True), 0.5)
    assert g == Gesture.SNAP, f"got {g}"


def test_gates_can_be_disabled():
    m = _machine(snap_require_rise=False, snap_require_still=False)
    fired = [m.tick(_up_reading(rose_recently=False, wrist_still=False), t / 10)
             for t in range(5)]
    assert Gesture.SNAP in fired, "disabled gates must not block SNAP"


def test_snap_forearm_min_scales_with_body():
    # forearm_dy 0.04 is under the unscaled snap_forearm_min (0.06) but over
    # the threshold scaled for a half-size body (0.03).
    m_ref = _machine()
    fired_ref = [m_ref.tick(_up_reading(forearm_dy=0.04, scale_factor=1.0), t / 10)
                 for t in range(5)]
    assert not any(fired_ref), "full-size body: 0.04 forearm must not snap"

    m_small = _machine()
    fired_small = [m_small.tick(_up_reading(forearm_dy=0.04, scale_factor=0.5), t / 10)
                   for t in range(5)]
    assert Gesture.SNAP in fired_small, "half-size body: 0.04 forearm should snap"


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
