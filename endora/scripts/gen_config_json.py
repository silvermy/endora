#!/usr/bin/env python3
"""
scripts/gen_config_json.py

Regenerates config.json's "options" and "schema" blocks from
config/registry.py. Run this after adding/editing a user_facing
SettingField; tests/test_registry_sync.py fails the build if you forget.

Only the "options" and "schema" object VALUES are replaced, via targeted
text splicing rather than a full json.load/json.dump round-trip — the rest
of config.json (name, version, arch, ports, map, etc.) is left byte-for-byte
untouched, since a full re-dump would also reformat unrelated arrays/objects
elsewhere in the file.

Usage: .venv/bin/python scripts/gen_config_json.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.registry import REGISTRY

CONFIG_JSON_PATH = Path(__file__).parent.parent / "config.json"

_TYPE_NAMES = {str: "str", int: "int", float: "float", bool: "bool"}


def schema_type(field) -> str:
    if field.enum:
        return f"list({'|'.join(field.enum)})"
    return _TYPE_NAMES[field.type]


def build_options_and_schema() -> tuple[dict, dict]:
    options: dict = {}
    schema: dict = {}
    for f in REGISTRY:
        if not f.user_facing or f.deprecated:
            continue
        options[f.key] = f.default
        schema[f.key] = schema_type(f)
    return options, schema


def _find_object_span(text: str, key: str) -> tuple[int, int]:
    """Return (start, end) char offsets of the `{...}` value for `"key": {`,
    with `end` pointing just past the matching closing brace. Tracks string
    literals so braces inside string values do not confuse the brace count.
    """
    needle = f'"{key}":'
    key_idx = text.index(needle)
    brace_start = text.index("{", key_idx + len(needle))

    depth = 0
    in_string = False
    escaped = False
    i = brace_start
    while i < len(text):
        c = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return brace_start, i + 1
        i += 1
    raise ValueError(f"unbalanced braces while scanning for {key!r} object")


def _render_object(obj: dict, indent_spaces: int) -> str:
    body = json.dumps(obj, indent=2, ensure_ascii=False)
    pad = " " * indent_spaces
    lines = body.splitlines()
    return "\n".join(lines[:1] + [pad + line for line in lines[1:]])


def main() -> None:
    text = CONFIG_JSON_PATH.read_text()
    options, schema = build_options_and_schema()

    for key, obj in (("options", options), ("schema", schema)):
        start, end = _find_object_span(text, key)
        new_block = _render_object(obj, indent_spaces=2)
        text = text[:start] + new_block + text[end:]

    CONFIG_JSON_PATH.write_text(text)
    print(f"Wrote {len(options)} options / {len(schema)} schema entries to {CONFIG_JSON_PATH}")


if __name__ == "__main__":
    main()
