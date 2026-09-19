"""
tests/test_gesture_critical_list.py

GESTURE_CRITICAL is the startup diagnostic: the settings that decide
whether a gesture fires, each logged with the file its value came from. It
exists because a stale entry in settings.yaml or runtime_overrides.yaml
silently outranks a shipped default, and three debugging sessions were spent
chasing behaviour that no longer matched the code.

A name in that list that is not a real Settings field is worse than useless
— it reports nothing while looking like coverage. This test caught exactly
that: `hold_after_snap_s`, written from memory, when the field is
`hold_duration_s`.
"""
from pathlib import Path

from config.registry import GESTURE_CRITICAL
from config.settings import Settings


def test_every_critical_name_is_a_real_setting():
    s = Settings()
    missing = [k for k in GESTURE_CRITICAL if not hasattr(s, k)]
    assert not missing, f"not Settings fields: {missing}"


def test_no_duplicates():
    dupes = {k for k in GESTURE_CRITICAL
             if GESTURE_CRITICAL.count(k) > 1}
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
    missing = [f for f in flags if f not in GESTURE_CRITICAL]
    assert not missing, f"gesture flags absent from the startup log: {missing}"


def test_there_is_exactly_one_copy_of_the_list():
    """The startup log and the debug page's /effective both read this.

    They used to hold separate literals, and drifted: gesture_hold_enable was
    added to one in v1.9.149 so the log could explain HOLD, and never reached
    the other, so the page a user actually reads went on omitting it. This
    test passed throughout, because it only ever checked one of them.
    """
    import re
    root = Path(__file__).resolve().parent.parent
    literals = [
        py for py in root.rglob("*.py")
        if ".venv" not in py.parts and py.parent.name != "tests"
        and re.search(r"^GESTURE_CRITICAL\b.*=\s*\[", py.read_text(), re.M)
    ]
    assert len(literals) == 1, f"defined in {[str(p) for p in literals]}"


def test_the_list_reports_a_source_for_every_entry():
    # effective() must return (value, source) for each name; a typo'd name
    # would otherwise surface as a silent omission at startup.
    s = Settings()
    eff = s.effective(GESTURE_CRITICAL)
    assert set(eff) == set(GESTURE_CRITICAL)
    for key, (_value, source) in eff.items():
        assert isinstance(source, str) and source, key
