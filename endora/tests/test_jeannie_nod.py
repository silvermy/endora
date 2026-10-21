"""
tests/test_jeannie_nod.py

Jeannie is folded arms with a nod, sitting or standing.

The pose alone is a resting position. Every one of the 13 FOLDED_ARMS that
reached Home Assistant between 2026-09-30 and 10-03 was someone at rest —
lying with hands on the chest, sitting with hands together, a visitor
standing with arms crossed — and none moved their head by more than 0.05
shoulder widths while folded. So FOLDED_ARMS now also needs a nod: the head
dipping towards the shoulders and coming back up, on frames where the arms
are folded.

The resting fixtures in tests/jeannie_resting/ are three of those 13 fires.
No real nod has been recorded yet (the old recordings end when the pose
fired, before anyone nodded), so the nod side is synthetic.
"""
from pathlib import Path

import numpy as np
import pytest

from cameras.arm_tracker import ArmTracker, ArmTrackerConfig
from core.state_machine import Gesture, GestureStateMachine, StateMachineConfig
from tests.fake_landmarks import LEFT_WRIST, NOSE, RIGHT_WRIST, Point
from tests.test_captured import _YOLOLandmarks
from tests.test_folded_arms import SHOULDER_W, H, W, _pose

FPS = 10.0
REST = Path(__file__).parent / "jeannie_resting"


def _nose_drop(lm, drop_sw: float):
    """The same pose with the head lowered by drop_sw shoulder widths."""
    p = lm._points[NOSE]
    lm._points[NOSE] = Point(p.x, p.y + drop_sw * SHOULDER_W / H, p.visibility)
    return lm


def _run(drops, folded=None):
    """Feed a sequence of head drops (one per frame); returns fire times."""
    tr = ArmTracker(ArmTrackerConfig())
    sm = GestureStateMachine(StateMachineConfig())
    fired = []
    for i, drop in enumerate(drops):
        t = i / FPS
        is_folded = folded[i] if folded is not None else True
        lm = _pose() if is_folded else _pose(wrist_gap=SHOULDER_W * 1.5)
        g = sm.tick(tr.classify(_nose_drop(lm, drop), W, H, None, now=t), t)
        if g is Gesture.FOLDED_ARMS:
            fired.append(t)
    return fired


def _nod(depth=0.2):
    return [depth / 2, depth, depth, depth / 2]


# ── the pose alone ────────────────────────────────────────────────────────────

def test_folded_arms_without_a_nod_do_not_fire():
    assert _run([0.0] * 40) == []


def test_waiting_for_the_nod_is_reported_once(caplog):
    import logging
    with caplog.at_level(logging.INFO):
        _run([0.0] * 40)
    waits = [r for r in caplog.records if "no_nod" in r.getMessage()]
    assert len(waits) == 1


# ── the gesture ───────────────────────────────────────────────────────────────

def test_fold_then_nod_fires():
    drops = [0.0] * 10 + _nod() + [0.0] * 20
    fired = _run(drops)
    assert len(fired) == 1
    assert 1.0 <= fired[0] <= 1.8, fired


def test_a_nod_anywhere_during_a_long_fold_fires():
    drops = [0.0] * 30 + _nod() + [0.0] * 10
    assert len(_run(drops)) == 1


def test_a_twitch_is_not_a_nod():
    drops = [0.0] * 10 + _nod(depth=0.05) + [0.0] * 20
    assert _run(drops) == []


def test_the_head_dropping_and_staying_down_is_not_a_nod():
    """Slumping, or settling lower in a chair: the head goes down and does
    not come back."""
    drops = [0.0] * 10 + [0.1, 0.2] + [0.25] * 25
    assert _run(drops) == []


def test_a_nod_before_folding_does_not_count():
    """Only frames with the arms folded are watched, so walking up, sitting
    down or talking before folding cannot pre-load a nod."""
    drops = [0.0] * 5 + _nod() + [0.0] * 30
    folded = [False] * 12 + [True] * 27
    assert _run(drops, folded) == []


# ── the recorded resting poses ───────────────────────────────────────────────

RESTING = ["lying_hands_on_chest", "sitting_hands_together", "visitor_arms_crossed"]


def _replay_resting(name, folder=REST, tracker_cfg=None, **sm_cfg):
    data = np.load(folder / f"{name}.npz")
    w, h = int(data["frame_w"]), int(data["frame_h"])
    kps, ts = list(data["keypoints"]), [float(t) for t in data["t_offsets"]]
    kps += [kps[-1]] * 20
    ts += [ts[-1] + 0.1 * (i + 1) for i in range(20)]
    tr = ArmTracker(ArmTrackerConfig(**(tracker_cfg or {})))
    sm = GestureStateMachine(StateMachineConfig(**sm_cfg))
    return [t for k, t in zip(kps, ts)
            if sm.tick(tr.classify(_YOLOLandmarks(k, w, h), w, h, now=t), now=t)
            is Gesture.FOLDED_ARMS]


@pytest.mark.parametrize("name", RESTING)
def test_the_recording_fires_without_the_nod(name):
    # Documents the bug: each of these really did fire.
    assert _replay_resting(name, folded_require_nod=False)


@pytest.mark.parametrize("name", RESTING)
def test_resting_with_folded_arms_does_not_fire(name):
    """Replayed, and then held still for two more seconds, none fires now."""
    assert _replay_resting(name) == []


# ── the head moves, the arms stay still ──────────────────────────────────────
# 2026-10-04: three FOLDED_ARMS fired while the user sat on the couch with a
# mug at their chin — hands together near the chest, head dipping to the cup
# and back. Depth, speed and hand height all failed to separate them from real
# nods; which part moved did. Hands/head travel: 0.26-0.49 for the three real
# Jeannies in tests/jeannie_real/, 1.08-1.68 for the three drinks.

REAL = Path(__file__).parent / "jeannie_real"
MUGS = ["mug_1", "mug_2", "mug_3"]
REAL_NODS = ["standing", "seated_1", "seated_2"]


OLD_SHALLOW_NOD = {"folded_nod_min_dip": 0.10}


@pytest.mark.parametrize("name", MUGS)
def test_a_drink_fired_while_any_small_dip_counted_as_a_nod(name):
    # Documents the bug: at the first threshold, with the arms rule off.
    assert _replay_resting(name, tracker_cfg={"folded_nod_arms_ratio_max": 0.0,
                                              **OLD_SHALLOW_NOD})


# ── a nod is a clear nod ─────────────────────────────────────────────────────
# 2026-10-20: sitting with a phone in the lap — hands together at the waist,
# glancing down at the screen and back — fired FOLDED_ARMS twice on "nods" of
# 0.10-0.13. The user's deliberate Jeannies that morning dipped 0.24-0.32.

PHONES = ["phone_in_lap_1", "phone_in_lap_2"]
ALL_REAL = REAL_NODS + ["live_seated", "live_standing_1", "live_standing_2"]


@pytest.mark.parametrize("name", PHONES)
def test_glancing_at_a_phone_fired_at_the_old_threshold(name):
    assert _replay_resting(name, tracker_cfg=OLD_SHALLOW_NOD)


@pytest.mark.parametrize("name", PHONES + MUGS)
def test_a_glance_or_a_sip_is_not_a_nod(name):
    assert _replay_resting(name) == []


@pytest.mark.parametrize("name", ALL_REAL)
def test_every_recorded_real_jeannie_still_fires(name):
    assert len(_replay_resting(name, folder=REAL)) == 1


ARMS_RULE_ON = {"folded_nod_arms_ratio_max": 0.6}


def test_the_arms_rule_is_off_by_default():
    """2026-10-20: at 0.6 it rejected all four of the user's real seated
    Jeannies (hands/head 1.03-2.08, inside the drinks' range). Off until it
    can be refitted on the nod recordings that now get saved."""
    assert ArmTrackerConfig().folded_nod_arms_ratio_max == 0.0


@pytest.mark.parametrize("name", MUGS)
def test_with_the_arms_rule_on_a_drink_is_not_a_jeannie(name):
    assert _replay_resting(name, tracker_cfg=ARMS_RULE_ON) == []


@pytest.mark.parametrize("name", REAL_NODS)
def test_a_real_jeannie_still_fires(name):
    assert len(_replay_resting(name, folder=REAL)) == 1


def _hands_drop(lm, drop_sw: float):
    """Move both wrists down with the head, as hands holding a cup do."""
    for idx in (LEFT_WRIST, RIGHT_WRIST):
        p = lm._points[idx]
        lm._points[idx] = Point(p.x, p.y + drop_sw * SHOULDER_W / H, p.visibility)
    return lm


def test_with_the_arms_rule_on_a_nod_with_the_hands_moving_does_not_fire():
    tr = ArmTracker(ArmTrackerConfig(**ARMS_RULE_ON))
    sm = GestureStateMachine(StateMachineConfig())
    drops = [0.0] * 10 + _nod() + [0.0] * 20
    fired = []
    for i, drop in enumerate(drops):
        t = i / FPS
        lm = _hands_drop(_nose_drop(_pose(), drop), drop)
        if sm.tick(tr.classify(lm, W, H, None, now=t), t) is Gesture.FOLDED_ARMS:
            fired.append(t)
    assert fired == []


# ── saying why an attempt was not accepted ───────────────────────────────────
# Sitting on the couch, attempts produced nothing at all — not even a fold to
# wait for a nod on — and nothing in the log said why.

def _misses(caplog, lm, frames=1):
    import logging
    tr = ArmTracker(ArmTrackerConfig())
    with caplog.at_level(logging.INFO, logger="cameras.arm_tracker"):
        for i in range(frames):
            tr._classify_raw(lm, W, H, now=i / FPS)
    return [r.getMessage() for r in caplog.records if "near-miss" in r.getMessage()]


def test_a_leaning_fold_says_how_far_it_leaned(caplog):
    [msg] = _misses(caplog, _pose(lean_deg=60), frames=10)
    assert "torso lean 60 deg (max 45)" in msg


def test_hands_nowhere_near_each_other_are_not_an_attempt(caplog):
    assert _misses(caplog, _pose(wrist_gap=SHOULDER_W * 3), frames=10) == []


def test_an_accepted_fold_reports_nothing(caplog):
    assert _misses(caplog, _pose(), frames=10) == []


def test_a_passing_moment_is_not_reported(caplog):
    """Arms on their way somewhere pass through the shape; only a pose held
    for folded_near_miss_settle_s is an attempt worth describing."""
    assert _misses(caplog, _pose(lean_deg=60), frames=3) == []


def test_resting_for_a_long_time_is_reported_once(caplog):
    """Lying with hands on the chest is a near-miss for hours. One line."""
    assert len(_misses(caplog, _pose(lean_deg=60), frames=600)) == 1


def test_each_separate_attempt_is_reported(caplog):
    import logging
    tr = ArmTracker(ArmTrackerConfig())
    with caplog.at_level(logging.INFO, logger="cameras.arm_tracker"):
        for start in (0.0, 5.0):                 # two attempts, 4 s apart
            for i in range(10):
                tr._classify_raw(_pose(lean_deg=60), W, H, now=start + i / FPS)
    assert len([r for r in caplog.records if "near-miss" in r.getMessage()]) == 2


# ── every nod leaves a recording ─────────────────────────────────────────────
# Recordings were only saved when a gesture fired, so the four real Jeannies
# rejected on 2026-10-20 left nothing to replay.

def test_the_tracker_reports_each_nod_it_decides():
    tr = ArmTracker(ArmTrackerConfig(**ARMS_RULE_ON))
    drops = [0.0] * 10 + _nod() + [0.0] * 5        # head only: accepted
    for i, d in enumerate(drops):
        tr.classify(_nose_drop(_pose(), d), W, H, None, now=i / FPS)
    moving = [0.0] * 5 + _nod() + [0.0] * 5        # hands with the head: rejected
    for i, d in enumerate(moving):
        t = 10.0 + i / FPS
        tr.classify(_hands_drop(_nose_drop(_pose(), d), d), W, H, None, now=t)
    kinds = [k for k, _r, _d in tr.nod_events]
    assert kinds == ["accepted", "rejected"]


def test_the_analyser_saves_a_recording_for_each_nod():
    from unittest.mock import MagicMock
    from cameras.analyser import CameraAnalyser
    from config.settings import Settings
    a = CameraAnalyser(camera=MagicMock(), settings=Settings(),
                       on_candidate=MagicMock(), label="A")
    a._recorder = MagicMock()
    entry = MagicMock()
    entry.arm_tracker.nod_events = [("rejected", 1.36, 0.19), ("accepted", None, 0.24)]
    a._record_nods(entry, 97)
    calls = a._recorder.save.call_args_list
    assert [c.kwargs["label"] for c in calls] == ["A_nod_rejected_pid97",
                                                   "A_nod_accepted_pid97"]
    assert all(c.kwargs["gesture_name"] == "MANUAL" for c in calls)
    assert "hands/head 1.36" in calls[0].kwargs["note"]
    assert entry.arm_tracker.nod_events == []          # consumed, not re-saved
