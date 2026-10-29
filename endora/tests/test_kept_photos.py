"""
tests/test_kept_photos.py

Gesture photos outlive the rolling debug-frame set.

State changes filled the 200-frame set in about an hour and a half on the
Jetson, so the photo of a fire was gone before it was labelled. A gesture
frame and the state frame just before it now move beside the keypoint traces
and are never evicted.
"""
import numpy as np
import pytest

from cameras import frame_capture as fc


@pytest.fixture
def cap(tmp_path, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(fc.time, "time", lambda: clock["t"])
    c = fc.FrameCapture(capture_dir=tmp_path / "frames", max_frames=3,
                        keep_dir=tmp_path / "captures" / "photos")

    def save(event, at):
        clock["t"] = at
        return c.save(np.zeros((8, 8, 3), np.uint8), event, camera="A")

    return c, save, tmp_path


def test_a_gesture_photo_survives_eviction(cap):
    c, save, _ = cap
    kept = save("gesture_SNAP", 1000.0)
    for i in range(10):
        save("state_DOWN", 1010.0 + i)
    assert kept.parent.name == "photos" and kept.exists()
    assert c.get_jpeg(kept.name) is not None
    names = [d["filename"] for d in c.list_captures()]
    assert kept.name in names
    assert len([n for n in names if "state_" in n]) == 3     # the rolling cap


def test_the_state_frame_before_a_held_pose_is_kept_with_it(cap):
    c, save, tmp = cap
    save("state_FOLDED_ARMS", 997.0)       # older than the window: rolls away
    held = save("state_FOLDED_ARMS", 998.5)
    save("gesture_FOLDED_ARMS", 1000.0)
    keep = tmp / "captures" / "photos"
    assert (keep / held.name).exists()
    assert (keep / held.with_suffix(".json").name).exists()
    assert len(list(keep.glob("*.jpg"))) == 2


def test_a_state_frame_after_the_gesture_is_not_kept(cap):
    _, save, tmp = cap
    save("gesture_SNAP", 1000.0)
    save("state_DOWN", 1000.5)
    assert len(list((tmp / "captures" / "photos").glob("*.jpg"))) == 1


def test_list_is_newest_first_across_both_places(cap):
    c, save, _ = cap
    save("gesture_SNAP", 1000.0)
    save("state_DOWN", 1001.0)
    ts = [d["timestamp"] for d in c.list_captures()]
    assert ts == sorted(ts, reverse=True)


def test_kept_photos_do_not_look_like_recordings(cap, monkeypatch):
    """The label store lists *.npz in the capture dir; a photos/ folder in it
    must not show up as a capture."""
    from cameras import labeler
    _, save, tmp = cap
    monkeypatch.setattr(labeler, "CAPTURE_DIR", tmp / "captures")
    save("gesture_SNAP", 1000.0)
    assert labeler.list_captures() == []


# ── expiry ───────────────────────────────────────────────────────────────────

DAY = 86400.0


@pytest.fixture
def labels(tmp_path, monkeypatch):
    from cameras import labeler
    monkeypatch.setattr(labeler, "CAPTURE_DIR", tmp_path / "captures")
    monkeypatch.setattr(labeler, "LABELS_PATH", tmp_path / "labels.json")
    return labeler


def test_old_unlabelled_photos_expire_and_recent_ones_stay(cap, labels):
    c, save, tmp = cap
    old = save("gesture_SNAP", 1000.0)
    new = save("gesture_SNAP", 1000.0 + 20 * DAY)
    assert c.expire_kept(now=1000.0 + 31 * DAY) == 1
    assert not old.exists() and not old.with_suffix(".json").exists()
    assert new.exists()


def test_a_labelled_fire_keeps_both_its_photos(cap, labels):
    c, save, tmp = cap
    held = save("state_FOLDED_ARMS", 998.0)
    fire = save("gesture_FOLDED_ARMS", 1001.4)
    labels.set_label("1001_A_folded_arms_pid7.npz", "not_intended")
    assert c.expire_kept(now=1000.0 + 365 * DAY) == 0
    keep = tmp / "captures" / "photos"
    assert (keep / held.name).exists() and fire.exists()


def test_clearing_a_label_lets_the_photo_expire(cap, labels):
    c, save, _ = cap
    fire = save("gesture_SNAP", 1000.2)
    labels.set_label("1000_A_snap_pid1.npz", "intended")
    labels.set_label("1000_A_snap_pid1.npz", None)
    assert c.expire_kept(now=1000.0 + 31 * DAY) == 1
    assert not fire.exists()


def test_a_labelled_fire_does_not_shield_its_neighbours(cap, labels):
    c, save, _ = cap
    labelled = save("gesture_SNAP", 1000.0)
    other = save("gesture_SNAP", 1060.0)
    labels.set_label("1000_A_snap_pid1.npz", "intended")
    c.expire_kept(now=1000.0 + 31 * DAY)
    assert labelled.exists() and not other.exists()
