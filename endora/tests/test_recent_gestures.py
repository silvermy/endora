"""
tests/test_recent_gestures.py

The strip under the debug video: the last few gestures that fired, each with
its photo and an Intended / Not intended label on its recording. It replaced
four buttons that marked "the last gesture" in a file nothing read.
"""
import json

import numpy as np
import pytest

from cameras import debug_server as ds
from cameras import labeler


@pytest.fixture
def store(tmp_path, monkeypatch):
    caps = tmp_path / "captures"
    caps.mkdir()
    monkeypatch.setattr(labeler, "CAPTURE_DIR", caps)
    monkeypatch.setattr(labeler, "LABELS_PATH", tmp_path / "labels.json")

    def rec(name):
        np.savez_compressed(caps / name, keypoints=np.zeros((1, 17, 3), np.float32))

    class Photos:
        items = []
        def list_captures(self):
            return list(self.items)

    photos = Photos()
    monkeypatch.setattr(ds, "_frame_capture", photos)
    return rec, photos


def test_newest_first_and_only_fired_gestures(store):
    rec, _ = store
    rec("1000_A_snap_pid1.npz")
    rec("1010_A_folded_arms_pid2.npz")
    rec("1020_A_nod_accepted_pid2.npz")       # a nod, not a fired gesture
    rec("1030_missed.npz")                      # a manual save
    names = [r["name"] for r in ds._recent_gestures()]
    assert names == ["1010_A_folded_arms_pid2.npz", "1000_A_snap_pid1.npz"]


def test_the_label_comes_from_the_label_store(store):
    rec, _ = store
    rec("1000_A_snap_pid1.npz")
    labeler.set_label("1000_A_snap_pid1.npz", "not_intended")
    [row] = ds._recent_gestures()
    assert row["label"] == "not_intended"


def test_the_photo_is_matched_by_gesture_and_time(store):
    rec, photos = store
    rec("1000_A_snap_pid1.npz")
    photos.items = [
        {"event_type": "gesture_FOLDED_ARMS", "timestamp": 1000.2, "filename": "wrong_kind.jpg"},
        {"event_type": "gesture_SNAP", "timestamp": 1000.4, "filename": "right.jpg"},
        {"event_type": "gesture_SNAP", "timestamp": 1002.9, "filename": "further.jpg"},
    ]
    [row] = ds._recent_gestures()
    assert row["photo"] == "right.jpg"


def test_no_photo_when_none_is_close_enough(store):
    rec, photos = store
    rec("1000_A_snap_pid1.npz")
    photos.items = [{"event_type": "gesture_SNAP", "timestamp": 1010.0, "filename": "late.jpg"}]
    [row] = ds._recent_gestures()
    assert row["photo"] is None


def test_at_most_six(store):
    rec, _ = store
    for i in range(9):
        rec(f"{1000 + 5 * i}_A_snap_pid1.npz")
    assert len(ds._recent_gestures()) == 6


def test_one_action_seen_by_two_tracks_is_one_card(store):
    """Two tracks on one body each leave a recording of the same SNAP."""
    rec, _ = store
    rec("1000_A_snap_pid956.npz")
    rec("1000_A_snap_pid961.npz")
    rec("1006_A_snap_pid961.npz")              # a separate SNAP, 6 s later
    rows = ds._recent_gestures()
    assert [sorted(r["names"]) for r in rows] == [
        ["1006_A_snap_pid961.npz"],
        ["1000_A_snap_pid956.npz", "1000_A_snap_pid961.npz"]]


def test_a_grouped_card_shows_a_label_from_any_of_its_recordings(store):
    rec, _ = store
    rec("1000_A_snap_pid956.npz")
    rec("1001_A_snap_pid961.npz")
    labeler.set_label("1000_A_snap_pid956.npz", "intended")
    [row] = ds._recent_gestures()
    assert row["label"] == "intended"


def test_a_jeannie_card_shows_the_pose_held_not_the_end_of_it(store):
    """Held poses fire at the end of the hold, by when the arms may already
    be down; the photo from when the fold was recognised shows it held."""
    rec, photos = store
    rec("1000_A_folded_arms_pid1.npz")
    photos.items = [
        {"event_type": "state_FOLDED_ARMS", "timestamp": 995.0, "filename": "too_early.jpg"},
        {"event_type": "state_FOLDED_ARMS", "timestamp": 999.4, "filename": "held.jpg"},
        {"event_type": "gesture_FOLDED_ARMS", "timestamp": 1000.4, "filename": "arms_down.jpg"},
    ]
    [row] = ds._recent_gestures()
    assert row["photo"] == "held.jpg"


def test_a_jeannie_without_a_pose_photo_falls_back_to_the_gesture_photo(store):
    rec, photos = store
    rec("1000_A_folded_arms_pid1.npz")
    photos.items = [{"event_type": "gesture_FOLDED_ARMS", "timestamp": 1000.4,
                     "filename": "arms_down.jpg"}]
    [row] = ds._recent_gestures()
    assert row["photo"] == "arms_down.jpg"
