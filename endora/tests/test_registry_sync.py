"""
tests/test_registry_sync.py

Enforces that config/registry.py (the single source of truth for every
setting) stays in sync with everything derived from or checked against it:
the Settings dataclass, config.json's options/schema blocks, and
debug_server.py's slider/joystick/toggle lists.

This is what actually blocks a bad push (there's no CI here — just the
pre-push pytest hook) — a script can regenerate config.json from the
registry, but only this test catches someone editing the registry and
forgetting to re-run it.
"""
import dataclasses
import json
from pathlib import Path

from config.registry import REGISTRY
from config.settings import Settings
from scripts.gen_config_json import build_options_and_schema

CONFIG_JSON_PATH = Path(__file__).parent.parent / "config.json"


def test_every_registry_field_is_a_settings_field():
    settings_fields = {f.name: f for f in dataclasses.fields(Settings)}
    missing = [r.key for r in REGISTRY if r.key not in settings_fields]
    assert not missing, f"registry keys missing from Settings dataclass: {missing}"


def test_every_settings_field_is_in_the_registry():
    registry_keys = {r.key for r in REGISTRY}
    settings_keys = {f.name for f in dataclasses.fields(Settings)}
    extra = settings_keys - registry_keys
    assert not extra, f"Settings fields not declared in the registry: {extra}"


def test_settings_defaults_match_registry():
    defaults = Settings()
    mismatches = []
    for r in REGISTRY:
        actual = getattr(defaults, r.key)
        if actual != r.default:
            mismatches.append((r.key, actual, r.default))
    assert not mismatches, f"Settings default != registry default: {mismatches}"


def test_settings_field_types_match_registry():
    settings_fields = {f.name: f for f in dataclasses.fields(Settings)}
    mismatches = []
    for r in REGISTRY:
        field = settings_fields[r.key]
        if field.type != r.type.__name__:
            mismatches.append((r.key, field.type, r.type.__name__))
    assert not mismatches, f"Settings field type != registry type: {mismatches}"


def test_config_json_matches_registry():
    """Fails if config.json is stale — run scripts/gen_config_json.py to fix."""
    expected_options, expected_schema = build_options_and_schema()
    data = json.loads(CONFIG_JSON_PATH.read_text())
    assert data["options"] == expected_options, (
        "config.json 'options' is out of sync with the registry — "
        "run scripts/gen_config_json.py"
    )
    assert data["schema"] == expected_schema, (
        "config.json 'schema' is out of sync with the registry — "
        "run scripts/gen_config_json.py"
    )


def test_debug_page_lists_derived_from_registry_match_known_layout():
    """Locks in the exact slider/joystick/toggle order the debug page has
    always rendered — a UIMeta.order collision or a missing order= on a new
    field would silently scramble this without failing any other test.
    """
    from cameras.debug_server import _PARAMS, _JOY_PARAMS, _TOGGLES

    assert [p[0] for p in _PARAMS] == [
        "arm_above_head_tolerance", "snap_forearm_min", "snap_sustain_s",
        "cooldown_s", "bg_subtract_min_foreground", "wrist_head_exclude_dist",
        "arm_above_head_tolerance_reclined",
        "yolo_conf", "dewarp_vfov", "frame_crop_bottom", "pose_visibility_min",
    ]
    assert [p[5] for p in _PARAMS] == [
        "Gesture", "Gesture", "Gesture", "Gesture", "Gesture", "Gesture", "Gesture",
        "Body", "View", "View", "View",
    ]
    assert [p[0] for p in _JOY_PARAMS] == ["dewarp_pan", "dewarp_tilt"]
    assert [p[0] for p in _TOGGLES] == ["low_light_enhance", "bg_subtract_enable"]
