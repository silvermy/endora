# Endora — Claude Code Guidelines

## Before every commit / push

1. **Bump the version** in both:
   - `endora/version.py` — `__version__ = "1.9.X"`
   - `endora/config.json` — `"version": "1.9.X"`
   Use the next patch number (check `git log --oneline -1` for current).

2. **Run the test suite** and confirm all pass:
   ```
   cd endora && .venv/bin/python -m pytest tests/ -q
   ```
   Fix any failures before committing.

## Project layout

- `endora/core/` — state machine, fusion, feedback logger
- `endora/cameras/` — YOLO pose model, arm tracker, analyser
- `endora/config/settings.py` — all tunable thresholds
- `endora/tests/` — pytest suite (no MediaPipe/YOLO needed)

## Python environment

Use `.venv` inside `endora/`: `endora/.venv/bin/python`

## Commit message style

`v1.9.X: short description of change`
