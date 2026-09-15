"""
tests/test_person_prune_rate.py

Keeping a tracked person alive across a momentary detection dropout, at any
pose sample rate.

This rule has now been wrong at both ends. A 2.0 s wall-clock timeout pruned
people on a Pi whose idle heartbeat ran every ~2.1 s (fixed by counting
missed runs instead). Counting missed runs alone then pruned people on a
Jetson sampling at 17.6/s, where three misses is 0.17 s — roughly 0.6 new
pids per second, each starting with the empty history a sweep is measured
against, and the resulting churn held the confidence hysteresis on its
permissive threshold so a 0.25-confidence junk detection fired a snap.

Requiring both conditions is what makes the rule rate-independent, so both
regimes are tested here rather than one.
"""
from cameras.analyser import _PERSON_PRUNE_MISSES, _PERSON_PRUNE_MIN_S


class _FakeEntry:
    def __init__(self, last_seen, last_seen_run):
        self.last_seen = last_seen
        self.last_seen_run = last_seen_run


class _Pruner:
    """The prune rule alone, free of camera, model and threads."""

    def __init__(self):
        self._persons = {}
        self._yolo_runs = 0
        self.lost = []

    # mirrors Analyser._prune_persons
    def prune(self, now):
        stale = [pid for pid, e in self._persons.items()
                 if self._yolo_runs - e.last_seen_run >= _PERSON_PRUNE_MISSES
                 and now - e.last_seen >= _PERSON_PRUNE_MIN_S]
        for pid in stale:
            self.lost.append(pid)
            del self._persons[pid]

    def run_without_seeing_anyone(self, n, rate_hz, now):
        """Advance n pose samples at *rate_hz*, detecting nobody."""
        for _ in range(n):
            self._yolo_runs += 1
            now += 1.0 / rate_hz
            self.prune(now)
        return now


def _tracked(rate_hz):
    p = _Pruner()
    p._persons[1] = _FakeEntry(last_seen=0.0, last_seen_run=0)
    return p


def test_a_brief_dropout_at_jetson_rates_does_not_lose_the_person():
    # 17.6 samples/s: three missed runs is 0.17 s. A person who flickers out
    # for a few frames mid-gesture must keep their pid, and with it the
    # elevation history the climb is measured from.
    p = _tracked(17.6)
    p.run_without_seeing_anyone(10, 17.6, now=0.0)     # 0.57 s of dropout
    assert not p.lost, "person pruned after half a second at 17.6 Hz"


def test_a_long_absence_at_jetson_rates_does_prune():
    p = _tracked(17.6)
    p.run_without_seeing_anyone(60, 17.6, now=0.0)     # 3.4 s
    assert p.lost == [1]


def test_slow_sampling_still_needs_the_missed_runs():
    # 0.5 samples/s: two missed runs is already 4 s, well past the time
    # floor, but the missed-run condition must still hold the person — this
    # is the Pi regression the misses rule was introduced for.
    p = _tracked(0.5)
    p.run_without_seeing_anyone(2, 0.5, now=0.0)
    assert not p.lost, "pruned on two missed runs at 0.5 Hz"


def test_slow_sampling_prunes_once_both_conditions_are_met():
    p = _tracked(0.5)
    p.run_without_seeing_anyone(_PERSON_PRUNE_MISSES, 0.5, now=0.0)
    assert p.lost == [1]


def test_a_person_seen_every_run_is_never_pruned():
    p = _tracked(17.6)
    now = 0.0
    for _ in range(200):
        p._yolo_runs += 1
        now += 1 / 17.6
        p._persons[1].last_seen = now
        p._persons[1].last_seen_run = p._yolo_runs
        p.prune(now)
    assert not p.lost
