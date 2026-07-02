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

from config.registry import REGISTRY
from config.settings import Settings


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
