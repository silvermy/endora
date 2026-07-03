"""
tests/test_analyser_liveness_confirmation.py

Regression coverage for two real, sequential bugs in the same feature:

1. Once a static ghost (e.g. a framed picture) got tracked as a pid via a
   single lucky noise-driven wrist-liveness pass, it exempted itself from
   the check forever afterward simply by matching its own unchanging
   position on every later frame — the "known centroids" exemption list
   was built from every currently-tracked pid, which trivially includes
   the ghost's own history. Fixed by only letting CONFIRMED-human pids
   (two genuine passes within _LIVENESS_CONFIRM_WINDOW_S of each other)
   contribute to that list.

2. The window alone was not enough: confirmed live on-device, a ghost
   still got fully confirmed. Real-world lighting noise (a flicker, an
   exposure adjustment) does not produce independent single-frame flukes —
   it produces a BURST of several correlated frames that all read as
   "changed" together, easily landing two "genuine" passes within any 60s
   window for a completely static object. Fixed by additionally requiring
   the centroid to have actually moved by a meaningful amount between the
   two passes (see _CENTROID_MOVED_MIN_FRAC) — something no amount of
   foreground-mask noise can fake, since it comes from the pose model's
   own keypoint coordinates, not the background model.

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
# >= _CENTROID_MOVED_MIN_FRAC (0.02) of the 800px diagonal for a 640x480
# frame is 16px; 100px is unambiguously a real displacement, not jitter.
MOVED_CENTROID = (500.0, 300.0)


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
    """After the single lucky pass, the ghost keeps matching its own
    position every frame (a static object always re-detects at the
    identical spot) but never independently shows genuine motion again.
    It must never become confirmed, no matter how many frames pass.
    """
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)
    pid = _only_pid(a)

    for i in range(50):
        a._match_persons([(None, CENTROID, False)], 640, 480, now=1000.0 + i)
        assert a._persons[pid].confirmed_human is False

    known_centroids = [e.centroid for e in a._persons.values() if e.confirmed_human]
    assert known_centroids == []


def test_two_genuine_passes_at_the_same_spot_does_not_confirm():
    """The exact live regression: a static ghost's foreground-mask check
    can pass twice within the confirm window (correlated lighting noise,
    not independent flukes) while its reported position never moves at
    all. Without also requiring displacement, this used to confirm it.
    """
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)
    pid = _only_pid(a)

    a._match_persons(
        [(None, CENTROID, True)], 640, 480,
        now=1000.0 + _LIVENESS_CONFIRM_WINDOW_S - 1,
    )
    assert a._persons[pid].confirmed_human is False


def test_two_genuine_moved_passes_within_window_confirms():
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)
    pid = _only_pid(a)
    assert a._persons[pid].confirmed_human is False

    a._match_persons(
        [(None, MOVED_CENTROID, True)], 640, 480,
        now=1000.0 + _LIVENESS_CONFIRM_WINDOW_S - 1,
    )
    assert a._persons[pid].confirmed_human is True


def test_two_genuine_moved_passes_outside_window_does_not_confirm():
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)
    pid = _only_pid(a)

    a._match_persons(
        [(None, MOVED_CENTROID, True)], 640, 480,
        now=1000.0 + _LIVENESS_CONFIRM_WINDOW_S + 1,
    )
    assert a._persons[pid].confirmed_human is False


def test_confirmation_is_sticky_after_resting():
    """Once confirmed, a real person can go still indefinitely afterward
    without losing confirmed status — this is the original v1.9.106 intent,
    which the self-exemption and moved-check fixes must not regress.
    """
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)
    pid = _only_pid(a)
    a._match_persons([(None, MOVED_CENTROID, True)], 640, 480, now=1010.0)
    assert a._persons[pid].confirmed_human is True

    for i in range(50):
        a._match_persons([(None, MOVED_CENTROID, False)], 640, 480, now=1010.0 + i * 10)
        assert a._persons[pid].confirmed_human is True
