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

from cameras.analyser import CameraAnalyser, _LIVENESS_CONFIRM_WINDOW_S, _known_centroids
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


# ── Regression: a real person who hasn't yet earned their second genuine
# pass must not be pruned the instant they hold still — see _known_centroids.
# Without this grace period, anyone who sits down and doesn't move their
# wrist within a couple of seconds vanishes from tracking (and the debug
# overlay) and has to start over as a brand-new pid, resetting the
# confirmation clock, the next time they move.

def test_unconfirmed_pid_is_self_exempt_within_grace_period():
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)
    pid = _only_pid(a)
    assert a._persons[pid].confirmed_human is False

    known = _known_centroids(a._persons, now=1000.0 + _LIVENESS_CONFIRM_WINDOW_S - 1)
    assert known == [CENTROID]


def test_unconfirmed_pid_loses_exemption_after_grace_period_elapses():
    """A static ghost gets the same grace window as a real person, not a
    free pass forever — if it never earns a genuine, moved second pass
    before the window elapses, it drops back out of known_centroids and is
    pruned exactly as before this fix.
    """
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)

    known = _known_centroids(a._persons, now=1000.0 + _LIVENESS_CONFIRM_WINDOW_S + 1)
    assert known == []


def test_confirmed_pid_stays_exempt_regardless_of_grace_period():
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)
    pid = _only_pid(a)
    a._match_persons([(None, MOVED_CENTROID, True)], 640, 480, now=1010.0)
    assert a._persons[pid].confirmed_human is True

    known = _known_centroids(a._persons, now=1010.0 + 10 * _LIVENESS_CONFIRM_WINDOW_S)
    assert known == [MOVED_CENTROID]


# ── the window needs a floor as well as a ceiling ────────────────────────────

def test_two_adjacent_frames_do_not_confirm():
    """"Two genuine passes within 60 s" was only ever an upper bound, so two
    frames 0.06 s apart satisfied it — which is not the sustained evidence
    the name implies.

    Recorded 2026-09-26: a painting of a figure was confirmed human 0.56 s
    after its track was born and fired FOLDED_ARMS off the artwork eleven
    seconds later, in an empty room.
    """
    from cameras.analyser import _LIVENESS_CONFIRM_MIN_GAP_S
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)
    pid = _only_pid(a)
    t = 1000.0
    for i in range(1, 8):                       # ~0.5 s of jittering frames
        t = 1000.0 + i * 0.06
        a._match_persons([(None, MOVED_CENTROID if i % 2 else CENTROID, True)],
                         640, 480, now=t)
    assert t - 1000.0 < _LIVENESS_CONFIRM_MIN_GAP_S, "fixture must stay inside the gap"
    assert a._persons[pid].confirmed_human is False


def test_a_pass_after_the_minimum_gap_still_confirms():
    """A real person is in view for many seconds, so the floor costs them
    nothing."""
    from cameras.analyser import _LIVENESS_CONFIRM_MIN_GAP_S
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)
    pid = _only_pid(a)
    a._match_persons([(None, MOVED_CENTROID, True)], 640, 480,
                     now=1000.0 + _LIVENESS_CONFIRM_MIN_GAP_S + 0.1)
    assert a._persons[pid].confirmed_human is True


def test_the_floor_is_below_the_ceiling():
    from cameras.analyser import (_LIVENESS_CONFIRM_MIN_GAP_S,
                                  _LIVENESS_CONFIRM_WINDOW_S)
    assert 0 < _LIVENESS_CONFIRM_MIN_GAP_S < _LIVENESS_CONFIRM_WINDOW_S


def test_a_long_run_of_jitter_still_confirms_eventually():
    """The floor delays confirmation; it must not prevent it. A pid feeding
    genuine moved passes continuously is confirmed once two of them are far
    enough apart."""
    a = _analyser()
    a._match_persons([(None, CENTROID, True)], 640, 480, now=1000.0)
    pid = _only_pid(a)
    for i in range(1, 60):
        a._match_persons([(None, MOVED_CENTROID if i % 2 else CENTROID, True)],
                         640, 480, now=1000.0 + i * 0.1)
    assert a._persons[pid].confirmed_human is True
