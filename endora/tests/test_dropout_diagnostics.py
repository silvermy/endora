"""
tests/test_dropout_diagnostics.py

Sitting on the couch on 2026-10-03, folding the arms made the person vanish
from the detector altogether. Whether they fell just under the confidence
threshold or out of the model's view entirely needs opposite fixes, so a
dropout is logged once, with the strongest below-threshold detection seen
near where the person was.
"""
import logging
from unittest.mock import MagicMock

import numpy as np

from cameras.analyser import CameraAnalyser
from config.settings import Settings

CENTROID = (400.0, 300.0)


def _analyser():
    a = CameraAnalyser(camera=MagicMock(), settings=Settings(),
                       on_candidate=MagicMock(), label="test")
    a._match_persons([(None, CENTROID, True, True)], 640, 480, now=0.0)
    return a, next(iter(a._persons))


def _missed_run(a, weak, now):
    a._yolo_runs += 1                       # a run in which nobody matched
    a._note_dropouts(weak, 0.29, 640, 480, now)


def _seen_run(a, now):
    a._yolo_runs += 1
    a._match_persons([(None, CENTROID, True, True)], 640, 480, now=now)
    a._note_dropouts(np.empty((0, 3), np.float32), 0.29, 640, 480, now)


def _lines(caplog):
    return [r.getMessage() for r in caplog.records if "not detected" in r.getMessage()]


def test_a_dropout_reports_the_best_candidate_near_the_person(caplog):
    a, pid = _analyser()
    near = np.array([[405.0, 302.0, 0.18], [410.0, 295.0, 0.22]], np.float32)
    far = np.array([[50.0, 40.0, 0.27]], np.float32)       # someone else
    with caplog.at_level(logging.INFO):
        _missed_run(a, np.vstack([near, far]), now=1.0)
        _missed_run(a, near[:1], now=1.5)
        _seen_run(a, now=2.0)
    [line] = _lines(caplog)
    assert "not detected for 1.0s (threshold 0.29)" in line
    assert "best candidate there 0.22" in line
    assert "found again" in line


def test_vanishing_completely_says_so(caplog):
    a, pid = _analyser()
    with caplog.at_level(logging.INFO):
        _missed_run(a, np.empty((0, 3), np.float32), now=1.0)
        _seen_run(a, now=2.0)
    [line] = _lines(caplog)
    assert "no candidate there at all" in line


def test_a_single_missed_run_is_not_reported(caplog):
    a, pid = _analyser()
    with caplog.at_level(logging.INFO):
        _missed_run(a, np.empty((0, 3), np.float32), now=1.0)
        _seen_run(a, now=1.1)
    assert _lines(caplog) == []


def test_a_dropout_that_ends_in_losing_the_person_is_reported(caplog):
    a, pid = _analyser()
    with caplog.at_level(logging.INFO):
        for i in range(1, 6):
            _missed_run(a, np.empty((0, 3), np.float32), now=float(i))
            a._prune_persons(float(i))
    [line] = _lines(caplog)
    assert "then lost" in line


def test_the_model_keeps_only_below_threshold_candidates():
    from cameras.pose_model import PoseModel
    m = PoseModel.__new__(PoseModel)
    m.conf = 0.29
    preds = np.zeros((4, 56), np.float32)
    preds[:, 0], preds[:, 1] = 100.0, 50.0
    preds[:, 4] = [0.02, 0.10, 0.28, 0.50]     # noise, weak, weak, kept
    weak = m._weak_candidates(preds, ratio=0.5, pad_w=0.0, pad_h=10.0)
    assert np.allclose(sorted(weak[:, 2]), [0.10, 0.28])
    assert weak[0, 0] == 200.0 and weak[0, 1] == 80.0   # back in frame pixels


def _anchor(cx, cy, conf, size=100.0):
    row = np.zeros(56, np.float32)
    row[:5] = [cx, cy, size, size, conf]
    return row


def test_an_anchor_overlapping_a_real_detection_is_not_a_candidate():
    # A person seen at 0.6 is ringed by overlapping anchors at every lower
    # confidence. Reporting those made every rejection look like a near miss.
    from cameras.pose_model import PoseModel
    m = PoseModel.__new__(PoseModel)
    m.conf = 0.22
    preds = np.stack([_anchor(300, 300, 0.60), _anchor(305, 302, 0.21),
                      _anchor(900, 300, 0.15)])
    weak = m._weak_candidates(preds, ratio=1.0, pad_w=0.0, pad_h=0.0)
    assert np.allclose(weak[:, 2], [0.15])           # only the separate one


def test_a_weak_detection_on_its_own_is_still_a_candidate():
    from cameras.pose_model import PoseModel
    m = PoseModel.__new__(PoseModel)
    m.conf = 0.22
    preds = np.stack([_anchor(300, 300, 0.21), _anchor(302, 301, 0.12)])
    weak = m._weak_candidates(preds, ratio=1.0, pad_w=0.0, pad_h=0.0)
    assert np.allclose(weak[:, 2], [0.21])


def test_a_dropout_reports_detections_that_were_rejected_near_the_person(caplog):
    a, pid = _analyser()
    near = [(402.0, 298.0, "no sign of life")]
    far = [(50.0, 40.0, "too few keypoints")]
    with caplog.at_level(logging.INFO):
        a._yolo_runs += 1
        a._note_dropouts(None, 0.22, 640, 480, 1.0, rejected=near + far)
        a._yolo_runs += 1
        a._note_dropouts(None, 0.22, 640, 480, 1.5, rejected=near)
        _seen_run(a, now=2.0)
    [line] = _lines(caplog)
    assert "detected but rejected: no sign of life x2" in line
    assert "too few keypoints" not in line


def test_rejections_are_collected_with_their_reason():
    from cameras.analyser import _all_valid_landmarks
    kps = np.zeros((1, 17, 3), np.float32)
    kps[0, :3] = [[100, 100, 0.9], [110, 100, 0.9], [90, 100, 0.9]]  # 3 keypoints
    boxes = np.array([[80, 80, 120, 140, 0.5]], np.float32)
    rejected = []
    assert _all_valid_landmarks(kps, 640, 480, boxes=boxes, rejected=rejected) == []
    assert rejected == [(100.0, 110.0, "too few keypoints")]
