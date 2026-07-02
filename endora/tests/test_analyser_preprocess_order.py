"""
tests/test_analyser_preprocess_order.py

Regression coverage for a false-positive cause: CLAHE (low_light_enhance)
used to run inside _preprocess, before the background-subtraction ghost
filter ever saw the frame. CLAHE boosts local contrast hardest in dim
regions, which can amplify sensor noise into something that reads as
motion — letting a static object (e.g. a framed picture in a shadowed
corner) intermittently clear the liveness check with nothing actually
moving. CLAHE must now run as a separate step, AFTER the frame the
background subtractor samples.
"""
from unittest.mock import MagicMock

import numpy as np

from cameras.analyser import CameraAnalyser


def _analyser(low_light_enhance: bool) -> CameraAnalyser:
    settings = MagicMock()
    settings.low_light_enhance = low_light_enhance
    settings.low_light_clip = 2.0
    settings.dewarp_enable = False
    settings.flip_image = False
    settings.frame_crop_top = 0
    settings.frame_crop_bottom = 0
    settings.frame_crop_left = 0
    settings.frame_crop_right = 0
    return CameraAnalyser(
        camera=MagicMock(), settings=settings, on_candidate=MagicMock(), label="test",
    )


def _noisy_frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)


def test_preprocess_never_applies_clahe_even_when_enabled():
    a = _analyser(low_light_enhance=True)
    frame = _noisy_frame()
    proc, pw, ph = a._preprocess(frame)
    assert np.array_equal(proc, frame)


def test_low_light_enhance_is_noop_when_disabled():
    a = _analyser(low_light_enhance=False)
    frame = _noisy_frame()
    out = a._apply_low_light_enhance(frame)
    assert np.array_equal(out, frame)


def test_low_light_enhance_changes_pixels_when_enabled():
    a = _analyser(low_light_enhance=True)
    frame = _noisy_frame()
    out = a._apply_low_light_enhance(frame)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)
