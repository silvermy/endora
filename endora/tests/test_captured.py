"""
tests/test_captured.py

Regression tests that replay captured YOLO keypoints through the full
ArmTracker → GestureStateMachine pipeline and assert the expected gesture fires.

Captures are .npz files produced by cameras/recorder.py (either auto-saved
when a gesture fires with ENDORA_RECORD_TESTS=1, or via the debug page
"Capture test case" button).

Search path (first directory found that contains .npz files):
  1. tests/captures/          — committed fixtures (checked into git)
  2. /data/test_captures/     — live captures from the add-on on the Pi

Run locally:
  pytest tests/test_captured.py -v

If no .npz files are found the test suite is skipped (not failed), so CI
always passes even on a clean checkout without captured data.

Format of each .npz:
  keypoints  float32 [N, 17, 3]  — COCO keypoints (x_px, y_px, conf)
  t_offsets  float64 [N]         — seconds, zero-based
  frame_w    int32
  frame_h    int32
  label      str
  gesture    str                 — expected Gesture enum name (e.g. "SNAP")
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import numpy as np

try:
    import pytest
    _PYTEST = True
except ImportError:
    # Allow standalone execution without pytest installed
    _PYTEST = False
    class _PytestStub:
        class mark:
            @staticmethod
            def parametrize(*a, **kw):
                return lambda f: f
        class skip:
            class Exception(Exception):
                pass
            @staticmethod
            def __call__(msg=""):
                raise _PytestStub.skip.Exception(msg)
        @staticmethod
        def fail(msg=""):
            raise AssertionError(msg)
    pytest = _PytestStub()  # type: ignore

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from cameras.arm_tracker import ArmTracker, ArmTrackerConfig
from core.state_machine import GestureStateMachine, StateMachineConfig, Gesture


# ── Lightweight COCO→MediaPipe adapter (no cv2 / ultralytics needed) ──────────
# Mirrors cameras/analyser.py:_YOLOLandmarks exactly.
_COCO_TO_MP = {5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 23, 12: 24}


class _KP:
    __slots__ = ("x", "y", "visibility")

    def __init__(self, x: float, y: float, visibility: float):
        self.x, self.y, self.visibility = x, y, visibility


class _YOLOLandmarks:
    """Map a [17, 3] COCO keypoint array to ArmTracker's MediaPipe indices."""

    def __init__(self, kps: np.ndarray, frame_w: int, frame_h: int) -> None:
        self._pts = {
            mp_idx: _KP(
                x=float(kps[ci, 0]) / frame_w,
                y=float(kps[ci, 1]) / frame_h,
                visibility=float(kps[ci, 2]),
            )
            for ci, mp_idx in _COCO_TO_MP.items()
        }

    def __getitem__(self, idx: int) -> _KP:
        return self._pts[idx]


# ── Fixture discovery ─────────────────────────────────────────────────────────

def _find_captures() -> List[Path]:
    candidates = [
        Path(__file__).parent / "captures",
        Path("/data/test_captures"),
    ]
    for d in candidates:
        if d.is_dir():
            files = sorted(d.glob("*.npz"))
            if files:
                return files
    return []


_CAPTURES = _find_captures()


def pytest_configure(config):
    if not _CAPTURES:
        print("\n[test_captured] No .npz captures found — "
              "set ENDORA_RECORD_TESTS=1 and perform gestures, "
              "or commit .npz files to tests/captures/")


# ── Parametrized replay test ──────────────────────────────────────────────────

@pytest.mark.parametrize("capture_path", _CAPTURES,
                         ids=[p.stem for p in _CAPTURES])
def test_gesture_fires_for_capture(capture_path: Path):
    """Replay a captured keypoint sequence and assert the expected gesture fires."""
    data = np.load(capture_path, allow_pickle=True)

    keypoints  = data["keypoints"]           # [N, 17, 3]
    t_offsets  = data["t_offsets"]           # [N]
    frame_w    = int(data["frame_w"])
    frame_h    = int(data["frame_h"])
    expected   = str(data["gesture"])        # e.g. "SNAP"
    label      = str(data.get("label", capture_path.stem))

    if expected == "MANUAL":
        pytest.skip(f"{label}: manual capture with no expected gesture")

    try:
        expected_gesture = Gesture[expected]
    except KeyError:
        pytest.fail(f"Unknown gesture '{expected}' in {capture_path.name}")

    tracker = ArmTracker(ArmTrackerConfig())
    sm      = GestureStateMachine(StateMachineConfig())

    fired: List[Gesture] = []
    for kps, t in zip(keypoints, t_offsets):
        lm      = _YOLOLandmarks(kps, frame_w, frame_h)
        reading = tracker.classify(lm, frame_w, frame_h, now=float(t))
        gesture = sm.tick(reading, now=float(t))
        if gesture is not None:
            fired.append(gesture)

    assert expected_gesture in fired, (
        f"Expected {expected_gesture.name} to fire for '{label}', "
        f"but got: {[g.name for g in fired] or 'nothing'}"
    )


# ── Timing test: gesture fires quickly ───────────────────────────────────────

@pytest.mark.parametrize("capture_path", _CAPTURES,
                         ids=[p.stem for p in _CAPTURES])
def test_gesture_fires_within_1s_of_first_motion(capture_path: Path):
    """The gesture should fire quickly — within 1 s of the sequence start."""
    data = np.load(capture_path, allow_pickle=True)
    expected = str(data["gesture"])
    if expected in ("MANUAL", "UNKNOWN"):
        pytest.skip("no expected gesture")

    try:
        expected_gesture = Gesture[expected]
    except KeyError:
        pytest.skip(f"unknown gesture {expected}")

    keypoints = data["keypoints"]
    t_offsets = data["t_offsets"]
    frame_w   = int(data["frame_w"])
    frame_h   = int(data["frame_h"])

    tracker = ArmTracker(ArmTrackerConfig())
    sm      = GestureStateMachine(StateMachineConfig())

    fired_at: float | None = None
    for kps, t in zip(keypoints, t_offsets):
        lm      = _YOLOLandmarks(kps, frame_w, frame_h)
        reading = tracker.classify(lm, frame_w, frame_h, now=float(t))
        gesture = sm.tick(reading, now=float(t))
        if gesture == expected_gesture:
            fired_at = float(t)
            break

    assert fired_at is not None, f"{expected_gesture.name} never fired"
    assert fired_at <= 1.0, (
        f"{expected_gesture.name} fired too late: {fired_at:.2f}s "
        f"(threshold 1.0s)"
    )


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _CAPTURES:
        print("No captures found. Nothing to test.")
        sys.exit(0)

    import traceback
    failed = 0
    for path in _CAPTURES:
        try:
            test_gesture_fires_for_capture(path)
            print(f"  PASS  {path.stem}")
        except pytest.skip.Exception as e:
            print(f"  SKIP  {path.stem}: {e}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {path.stem}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {path.stem}: {e}")
            traceback.print_exc()

    print(f"\n{'PASSED' if failed == 0 else f'{failed} FAILED'} "
          f"({len(_CAPTURES) - failed}/{len(_CAPTURES)})")
    sys.exit(failed)
