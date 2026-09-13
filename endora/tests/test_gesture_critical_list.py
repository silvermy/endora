"""
tests/test_gesture_critical_list.py

main._GESTURE_CRITICAL is the startup diagnostic: the settings that decide
whether a gesture fires, each logged with the file its value came from. It
exists because a stale entry in settings.yaml or runtime_overrides.yaml
silently outranks a shipped default, and three debugging sessions were spent
chasing behaviour that no longer matched the code.

A name in that list that is not a real Settings field is worse than useless
— it reports nothing while looking like coverage. This test caught exactly
that: `hold_after_snap_s`, written from memory, when the field is
`hold_duration_s`.
"""
from config.settings import Settings

import main


def test_every_critical_name_is_a_real_setting():
    s = Settings()
    missing = [k for k in main._GESTURE_CRITICAL if not hasattr(s, k)]
    assert not missing, f"not Settings fields: {missing}"


def test_no_duplicates():
    dupes = {k for k in main._GESTURE_CRITICAL
             if main._GESTURE_CRITICAL.count(k) > 1}
    assert not dupes, f"listed twice: {sorted(dupes)}"


def test_every_gesture_enable_flag_is_listed():
    """The flags decide whether a gesture can fire at all, so leaving one out
    means the log cannot explain its behaviour. HOLD was the one missing —
    it fires from the same raised arm as a SNAP, hold_duration_s later, with
    its own event and chime, and reads as one gesture firing twice.
    """
    s = Settings()
    flags = [f for f in vars(s) if f.startswith("gesture_") and f.endswith("_enable")]
    assert flags, "no gesture enable flags found — has the naming changed?"
    missing = [f for f in flags if f not in main._GESTURE_CRITICAL]
    assert not missing, f"gesture flags absent from the startup log: {missing}"


def test_the_list_reports_a_source_for_every_entry():
    # effective() must return (value, source) for each name; a typo'd name
    # would otherwise surface as a silent omission at startup.
    s = Settings()
    eff = s.effective(main._GESTURE_CRITICAL)
    assert set(eff) == set(main._GESTURE_CRITICAL)
    for key, (_value, source) in eff.items():
        assert isinstance(source, str) and source, key
