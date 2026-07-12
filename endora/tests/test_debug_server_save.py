"""Tests for the debug page's Save button (cameras/debug_server.py).

Regression coverage for a bug where every Save silently snapshotted every
slider/toggle — including ones the user never touched — into
runtime_overrides.yaml, permanently locking those params out of the HA
Configuration tab even though the user never meant to override them there.
"""
from pathlib import Path

from cameras import debug_server as ds
from config.settings import Settings


def _redirect_data_dir(monkeypatch, tmp_path):
    """Make Path("/data/...") resolve under tmp_path instead of the real /data."""
    real_path = Path

    def fake_path(p, *args, **kwargs):
        p = str(p)
        if p.startswith("/data/") or p == "/data":
            return real_path(tmp_path, p.lstrip("/"))
        return real_path(p, *args, **kwargs)

    monkeypatch.setattr(ds, "Path", fake_path)


def setup_function(_):
    ds._settings = Settings()
    ds._touched = set()


def test_save_only_persists_touched_and_preexisting_keys(tmp_path, monkeypatch):
    _redirect_data_dir(monkeypatch, tmp_path)

    overrides_path = tmp_path / "data" / "runtime_overrides.yaml"
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    overrides_path.write_text("snap_forearm_min: 0.06\nsnap_sustain_s: 0.2\n")
    # In production settings.py loads runtime_overrides.yaml at startup, so
    # the in-memory values match the file. Mirror that here — Save re-
    # serializes pre-existing keys from the live settings object, and this
    # fixture's 0.06 deliberately differs from the current default (0.05).
    ds._settings.snap_forearm_min = 0.06
    ds._settings.snap_sustain_s = 0.2

    # Simulate the user only touching one unrelated slider this session.
    assert ds._apply_setting("cooldown_s", "3.0")

    ok, err = ds._save_to_yaml()
    assert ok, err

    saved = overrides_path.read_text()
    assert "cooldown_s: 3" in saved
    assert "snap_forearm_min: 0.06" in saved
    assert "snap_sustain_s: 0.2" in saved
    # low_light_enhance was never touched and never previously saved —
    # it must NOT be pinned just because Save was clicked.
    assert "low_light_enhance" not in saved
    assert "arm_above_head_tolerance" not in saved


def test_untouched_toggle_survives_across_saves_until_actually_changed(tmp_path, monkeypatch):
    _redirect_data_dir(monkeypatch, tmp_path)

    overrides_path = tmp_path / "data" / "runtime_overrides.yaml"
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    overrides_path.write_text("low_light_enhance: false\n")

    # A later session touches a different param and saves again.
    assert ds._apply_setting("snap_forearm_min", "0.05")
    ok, err = ds._save_to_yaml()
    assert ok, err
    assert "low_light_enhance: false" in overrides_path.read_text()

    # Now the user actually flips the toggle on the debug page — it should update.
    assert ds._apply_setting("low_light_enhance", "true")
    ok, err = ds._save_to_yaml()
    assert ok, err
    assert "low_light_enhance: true" in overrides_path.read_text()


def test_reset_overrides_deletes_file_and_reverts_settings(tmp_path, monkeypatch):
    _redirect_data_dir(monkeypatch, tmp_path)

    overrides_path = tmp_path / "data" / "runtime_overrides.yaml"
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    overrides_path.write_text("snap_forearm_min: 0.09\ncooldown_s: 7\n")
    # Simulate a session where those overrides were live and one more slider
    # was touched but not yet saved.
    ds._settings.snap_forearm_min = 0.09
    ds._settings.cooldown_s = 7.0
    assert ds._apply_setting("snap_sustain_s", "0.9")

    ok, err = ds._reset_overrides()
    assert ok, err
    assert not overrides_path.exists(), "overrides file must be deleted"
    # Live settings revert to defaults (no options.json in the test env),
    # including the touched-but-unsaved one, and the touched set is cleared
    # so a subsequent Save doesn't resurrect anything.
    defaults = Settings()
    assert ds._settings.snap_forearm_min == defaults.snap_forearm_min
    assert ds._settings.cooldown_s == defaults.cooldown_s
    assert ds._settings.snap_sustain_s == defaults.snap_sustain_s
    assert ds._touched == set()


def test_reset_overrides_ok_when_no_file(tmp_path, monkeypatch):
    _redirect_data_dir(monkeypatch, tmp_path)
    ok, err = ds._reset_overrides()
    assert ok, err
