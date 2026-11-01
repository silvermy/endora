"""
tests/test_fold_no_nod_recording.py

A fold that ends without the nod it was waiting for saves a recording.

On 2026-11-01 seated Jeannies registered the fold but never the nod — over a
hundred "no nod yet" lines at 0.00-0.03 shoulder widths — and none could be
studied, because recordings were saved only when a gesture fired or a nod
was decided.
"""
from unittest.mock import MagicMock

from cameras.analyser import CameraAnalyser, _NO_NOD_SAVE_GAP_S
from cameras.arm_tracker import ArmReading, ArmState
from config.settings import Settings
from core.state_machine import Gesture


def _setup(waiting=True):
    a = CameraAnalyser(camera=MagicMock(), settings=Settings(),
                       on_candidate=MagicMock(), label="A")
    a._recorder = MagicMock()
    a._match_persons([(None, (400.0, 300.0), True, True)], 640, 480, now=0.0)
    entry = next(iter(a._persons.values()))
    entry.state_machine._sustain.nod_wait_logged = waiting
    return a, entry


FOLDED = ArmReading(state=ArmState.FOLDED_ARMS, nod_dip=0.02)
DOWN = ArmReading(state=ArmState.DOWN)


def _run(a, entry, frames):
    for t, reading, gesture in frames:
        a._record_fold_without_nod(entry, 0, reading, gesture, t)


def test_a_fold_that_ends_without_a_nod_is_saved_once_it_is_over():
    a, e = _setup()
    _run(a, e, [(1.0, FOLDED, None), (1.1, DOWN, None), (1.5, DOWN, None)])
    a._recorder.save.assert_not_called()          # within sustain_gap_s
    _run(a, e, [(2.0, DOWN, None), (2.1, DOWN, None)])
    a._recorder.save.assert_called_once()
    kw = a._recorder.save.call_args.kwargs
    assert kw["gesture_name"] == "MANUAL"
    assert kw["label"] == "A_fold_no_nod_pid0"
    assert "largest dip 0.02" in kw["note"]


def test_a_fold_that_fires_is_not_saved():
    a, e = _setup()
    _run(a, e, [(1.0, FOLDED, None), (1.2, FOLDED, Gesture.FOLDED_ARMS),
                (2.5, DOWN, None)])
    a._recorder.save.assert_not_called()


def test_a_fold_too_short_to_wait_for_a_nod_is_not_saved():
    a, e = _setup(waiting=False)
    _run(a, e, [(1.0, FOLDED, None), (2.5, DOWN, None)])
    a._recorder.save.assert_not_called()


def test_a_flickering_fold_saves_at_most_one_per_gap():
    a, e = _setup()
    t, frames = 1.0, []
    while t < 1.0 + _NO_NOD_SAVE_GAP_S:
        frames += [(t, FOLDED, None), (t + 1.0, DOWN, None)]
        t += 2.0
    _run(a, e, frames)
    assert a._recorder.save.call_count == 1
