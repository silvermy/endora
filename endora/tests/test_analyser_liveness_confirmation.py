"""
tests/test_analyser_liveness_confirmation.py

Regression coverage for a real bug: once a static ghost (e.g. a framed
picture) got tracked as a pid via a single lucky noise-driven wrist-
liveness pass, it exempted itself from the check forever afterward simply
by matching its own unchanging position on every later frame — since the
"known centroids" exemption list was built from every currently-tracked
pid, which trivially includes the ghost's own history.

Fix: only CONFIRMED-human pids (two genuine passes within
_LIVENESS_CONFIRM_WINDOW_S of each other) contribute to that list. A
single lucky pass creates a pid but does not confirm it, so it cannot
self-exempt on the next frame; without a second genuine pass in time, it
never becomes confirmed at all.

Uses CameraAnalyser._match_persons directly with a real Settings()
instance (not a bare MagicMock — _make_person_entry constructs a real
ArmTracker/GestureStateMachine from dozens of getattr(settings, key,
default) calls, which silently return auto-generated MagicMock children
instead of the intended defaults on a bare mock).
"""
from unittest.mock import MagicMock

from cameras.analyser import CameraAnalyser, _LIVENESS_CONFIRM_WINDOW_S
from config.settings import Settings

CENTROID = (400.0, 300.0)


def _analyser() -> CameraAnalyser:
    return CameraAnalyser(
        camera=MagicMock(), settings=Settings(), on_candidate=MagicMock(), label="test",
    )


def _only_pid(a: CameraAnalyser) -> int:
    assert len(a._persons) == 1
    return next(iter(a._persons))


def test_single_lucky_pass_creates_a_pid_but_does_not_confirm_it():
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)
    pid = _only_pid(a)
    assert a._persons[pid].confirmed_human is False


def test_ghost_never_self_exempts_without_a_second_genuine_pass():
    """The exact regression: after the single lucky pass, the ghost keeps
    matching its own position every frame (a static object always
    re-detects at the identical spot) but never independently shows
    genuine motion again. It must never become confirmed, no matter how
    many frames pass — so known_centroids built from confirmed_human-only
    entries never includes it, and a fresh detection at that same spot
    with no genuine motion this frame is correctly excluded upstream by
    _all_valid_landmarks (not exercised directly here, but this is the
    invariant that check depends on).
    """
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)
    pid = _only_pid(a)

    # Many subsequent frames where this pid is (hypothetically) re-matched
    # with raw_live=False — as would happen if it were still being fed in
    # via some other path. confirmed_human must stay False throughout.
    for i in range(50):
        a._match_persons([(None, CENTROID, False)], 640, 480, now=1000.0 + i)
        assert a._persons[pid].confirmed_human is False

    known_centroids = [e.centroid for e in a._persons.values() if e.confirmed_human]
    assert known_centroids == []


def test_two_genuine_passes_within_window_confirms():
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)
    pid = _only_pid(a)
    assert a._persons[pid].confirmed_human is False

    a._match_persons(
        [(None, CENTROID, True)], 640, 480,
        now=1000.0 + _LIVENESS_CONFIRM_WINDOW_S - 1,
    )
    assert a._persons[pid].confirmed_human is True


def test_two_genuine_passes_outside_window_does_not_confirm():
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)
    pid = _only_pid(a)

    a._match_persons(
        [(None, CENTROID, True)], 640, 480,
        now=1000.0 + _LIVENESS_CONFIRM_WINDOW_S + 1,
    )
    assert a._persons[pid].confirmed_human is False


def test_confirmation_is_sticky_after_resting():
    """Once confirmed, a real person can go still indefinitely afterward
    without losing confirmed status — this is the original v1.9.106 intent,
    which the self-exemption fix must not regress.
    """
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)
    pid = _only_pid(a)
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1010.0)
    assert a._persons[pid].confirmed_human is True

    for i in range(50):
        a._match_persons([(None, CENTROID, False)], 640, 480, now=1010.0 + i * 10)
        assert a._persons[pid].confirmed_human is True
