"""
tests/test_feedback_logger.py

Unit tests for FeedbackLogger persistence + rotate-on-download.
Pure stdlib — no cv2 / YOLO / MediaPipe needed.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import core.feedback_logger as fl


def _logger(tmp_path, monkeypatch):
    monkeypatch.setattr(fl, "_LOG_PATH", tmp_path / "feedback.jsonl")
    return fl.FeedbackLogger()


def test_mark_no_pose_persists(tmp_path, monkeypatch):
    logger = _logger(tmp_path, monkeypatch)
    assert logger.mark_no_pose() is True
    lines = (tmp_path / "feedback.jsonl").read_bytes()
    assert lines.count(b"\n") == 1
    assert b"no_pose" in lines


def test_rotate_backs_up_and_clears(tmp_path, monkeypatch):
    logger = _logger(tmp_path, monkeypatch)
    logger.mark_no_pose()
    logger.mark_false_negative(gesture_hint="SNAP")

    data = logger.rotate()
    assert data.count(b"\n") == 2                      # both entries returned
    assert (tmp_path / "feedback.jsonl.prev").read_bytes() == data
    assert (tmp_path / "feedback.jsonl").read_bytes() == b""   # live cleared

    # Fresh writes after rotation land in the new (empty) live log.
    logger.mark_no_pose()
    assert (tmp_path / "feedback.jsonl").read_bytes().count(b"\n") == 1


def test_rotate_empty_does_not_clobber_backup(tmp_path, monkeypatch):
    logger = _logger(tmp_path, monkeypatch)
    logger.mark_no_pose()
    logger.rotate()                                    # prev now has 1 entry
    prev = (tmp_path / "feedback.jsonl.prev").read_bytes()
    assert prev.count(b"\n") == 1

    # Rotating an already-empty live log must keep the previous backup intact.
    assert logger.rotate() == b""
    assert (tmp_path / "feedback.jsonl.prev").read_bytes() == prev
