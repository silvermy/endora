"""
tests/test_capture_frames.py

Each camera frame is analysed exactly once.

On the Jetson the analyser loop ran the pose model 16.6 times a second on a
camera delivering 10: get_frame returns the newest frame however often it is
called, so four inferences in ten were repeats. That wasted GPU, and a repeat
is a second, identical observation of the same moment — which a check that
wants a raise "seen on two frames" (v1.9.192) cannot afford.
"""
import threading
import time

import numpy as np

from cameras.capture import RtspCapture


def _cam():
    return RtspCapture("rtsp://example.invalid/stream", name="test")


def _frame(v):
    return np.full((4, 4, 3), v, np.uint8)


def test_nothing_until_the_first_frame():
    cam = _cam()
    frame, seq = cam.wait_new_frame(0, timeout=0.01)
    assert frame is None and seq == 0


def test_a_new_frame_is_returned_once():
    cam = _cam()
    cam._publish(_frame(1))
    frame, seq = cam.wait_new_frame(0, timeout=0.01)
    assert frame[0, 0, 0] == 1 and seq == 1
    # Asking again before anything new arrives gets nothing, not a repeat.
    again, seq2 = cam.wait_new_frame(seq, timeout=0.01)
    assert again is None and seq2 == seq


def test_a_slow_reader_skips_to_the_newest_frame():
    cam = _cam()
    for v in (1, 2, 3):
        cam._publish(_frame(v))
    frame, seq = cam.wait_new_frame(0, timeout=0.01)
    assert frame[0, 0, 0] == 3 and seq == 3


def test_the_reader_wakes_as_soon_as_a_frame_arrives():
    cam = _cam()
    threading.Timer(0.05, cam._publish, args=(_frame(7),)).start()
    t0 = time.monotonic()
    frame, _ = cam.wait_new_frame(0, timeout=2.0)
    assert frame[0, 0, 0] == 7
    assert time.monotonic() - t0 < 1.0


def test_stopping_releases_a_waiting_reader():
    cam = _cam()
    threading.Timer(0.05, cam.stop).start()
    t0 = time.monotonic()
    frame, _ = cam.wait_new_frame(0, timeout=2.0)
    assert frame is None and time.monotonic() - t0 < 1.0


def test_the_returned_frame_is_a_copy():
    cam = _cam()
    cam._publish(_frame(1))
    frame, _ = cam.wait_new_frame(0, timeout=0.01)
    frame[:] = 9
    again = cam.get_frame()
    assert again[0, 0, 0] == 1
