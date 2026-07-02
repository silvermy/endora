"""
tests/test_pose_model_resolve.py

Regression test for resolve_model_path's fallback warning: a format-string
argument-count mismatch there made the log call raise TypeError every time
a requested imgsz had no bundled/cached model (e.g. 480, since only 320 and
640 are ever bundled — see Dockerfile). Python's logging module swallows
that error internally (prints a traceback, doesn't propagate), so the
function still returned the right fallback values, but the warning
explaining why never actually appeared anywhere — which is what made this
session's CPU investigation take so long to pin down.

IMPORTANT: _export_at_size is unconditionally patched out in every test
here. On non-aarch64 machines (any dev laptop) it performs a REAL
ultralytics ONNX export as a side effect, which writes its output to a
fixed path derived from the .pt filename — colliding with and overwriting
the real bundled yolo11n-pose.onnx in the repo root. That happened once
already while writing this test and corrupted the local model file.
Never call resolve_model_path in a test without patching this out first.
"""
import logging
from pathlib import Path
from unittest.mock import patch

from cameras.pose_model import resolve_model_path

REPO_ROOT = Path(__file__).parent.parent
NANO_640 = REPO_ROOT / "yolo11n-pose.onnx"


def _resolve_without_exporting(*args, **kwargs):
    with patch("cameras.pose_model._export_at_size", return_value=False):
        return resolve_model_path(*args, **kwargs)


def test_fallback_returns_native_size_when_requested_size_unavailable():
    assert NANO_640.exists(), "expected yolo11n-pose.onnx at repo root for this test"
    path, actual_imgsz = _resolve_without_exporting(str(NANO_640), imgsz=480)
    assert path == str(NANO_640)
    assert actual_imgsz == 640


def test_fallback_warning_message_formats_without_raising(caplog):
    """This is the exact bug: the warning call itself used to raise
    TypeError('not enough arguments for format string') when this branch
    ran. caplog captures raw LogRecords; record.getMessage() is what
    actually performs %-substitution, so this reproduces the crash site.
    """
    with caplog.at_level(logging.WARNING, logger="cameras.pose_model"):
        _resolve_without_exporting(str(NANO_640), imgsz=480)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a fallback warning to be logged"
    for record in warnings:
        message = record.getMessage()  # raises if the format string is broken
        assert "480" in message
        assert "640" in message


def test_bundled_320_variant_is_used_when_available():
    path, actual_imgsz = _resolve_without_exporting(str(NANO_640), imgsz=320)
    assert path.endswith("yolo11n-pose-320.onnx")
    assert actual_imgsz == 320


def test_matching_native_size_is_used_directly():
    path, actual_imgsz = _resolve_without_exporting(str(NANO_640), imgsz=640)
    assert path == str(NANO_640)
    assert actual_imgsz == 640
