"""
tests/test_debug_server_unsaved.py

The debug page's sliders apply live, so a change looked done while it was
only in memory and vanished on the next restart. The page now highlights
every setting changed since the last save, from /unsaved.
"""
from cameras import debug_server as ds
from config.settings import Settings


def setup_function(_):
    ds._settings = Settings()
    ds._touched = set()
    ds._saved_snapshot = None


def test_nothing_is_unsaved_at_first():
    assert ds._unsaved_keys() == []


def test_a_changed_setting_is_unsaved():
    before = ds._settings.raise_elevation_min
    assert ds._apply_setting("raise_elevation_min", str(before + 0.05))
    assert ds._unsaved_keys() == ["raise_elevation_min"]


def test_putting_it_back_clears_it():
    before = ds._settings.raise_elevation_min
    ds._apply_setting("raise_elevation_min", str(before + 0.05))
    ds._apply_setting("raise_elevation_min", str(before))
    assert ds._unsaved_keys() == []


def test_saving_clears_it():
    ds._apply_setting("raise_elevation_min", "0.77")
    ds._mark_saved()                      # what /save does once the files are written
    assert ds._unsaved_keys() == []


def test_the_baseline_is_taken_before_the_first_change():
    """A change made before anything asked for /unsaved must still count."""
    ds._apply_setting("raise_elevation_min", "0.77")
    assert "raise_elevation_min" in ds._unsaved_keys()


def test_toggles_count_too():
    key = ds._TOGGLES[0][0]                 # whichever toggle the page shows first
    on = getattr(ds._settings, key)
    ds._apply_setting(key, "false" if on else "true")
    assert ds._unsaved_keys() == [key]
