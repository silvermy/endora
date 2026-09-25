"""
tests/test_recorder_per_person.py

The test recorder keeps one rolling buffer per tracked person.

It used to keep a single buffer fed with `valid[0]` — the first detection of
each frame by array index. That index is not stable between frames and is not
necessarily the person who fires, and one occupant routinely produces two
concurrent detections on this camera. So a saved trace could interleave two
different bodies frame by frame.

That is not a theoretical complaint. Captures recorded on 2026-09-25 alternate
between shoulders of 22 px and 78 px, hip confidence of 0.31 and 0.96, and a
torso leaning 12 degrees and 60 degrees — one occupant on a couch, recorded as
two. Replaying those files reproduced the gesture they were supposed to
explain in neither case, which cost an evening of debugging the wrong thing.

Every capture is evidence, and evidence that silently mixes two subjects is
worse than none.
"""
import numpy as np
import pytest

from cameras.recorder import TestRecorder as Recorder, _PID_IDLE_DROP_S


def _kps(shoulder_px: float) -> np.ndarray:
    """A [17, 3] COCO row whose shoulder width identifies which body it is."""
    k = np.zeros((17, 3), dtype=np.float32)
    k[5] = (400 + shoulder_px / 2, 350, 0.99)     # left shoulder
    k[6] = (400 - shoulder_px / 2, 350, 0.99)     # right shoulder
    return k


def _rec(tmp_path) -> Recorder:
    return Recorder(window_s=5.0, save_dir=tmp_path)


def _width(path) -> float:
    z = np.load(path, allow_pickle=True)
    k = z["keypoints"]
    return float(k[0][5][0] - k[0][6][0])


# ── one body per file ─────────────────────────────────────────────────────────

def test_two_concurrent_people_do_not_share_a_buffer(tmp_path):
    r = _rec(tmp_path)
    for i in range(10):                       # both seen on every frame
        r.on_frame(_kps(78), 1280, 640, i / 10.0, pid=0)
        r.on_frame(_kps(22), 1280, 640, i / 10.0, pid=6)

    wide = r.save(gesture_name="FOLDED_ARMS", label="a", pid=0)
    narrow = r.save(gesture_name="FOLDED_ARMS", label="b", pid=6)

    assert _width(wide) == pytest.approx(78, abs=1)
    assert _width(narrow) == pytest.approx(22, abs=1)


def test_every_frame_in_a_file_is_the_same_body(tmp_path):
    """The actual defect: a trace that changes subject mid-capture."""
    r = _rec(tmp_path)
    for i in range(10):
        r.on_frame(_kps(78), 1280, 640, i / 10.0, pid=0)
        r.on_frame(_kps(22), 1280, 640, i / 10.0, pid=6)
    k = np.load(r.save(label="x", pid=0), allow_pickle=True)["keypoints"]
    widths = {round(float(f[5][0] - f[6][0])) for f in k}
    assert widths == {78}, f"trace mixes bodies: {sorted(widths)}"


def test_the_gesture_saves_the_firing_persons_buffer(tmp_path):
    from core.state_machine import Gesture
    r = _rec(tmp_path)
    for i in range(10):
        r.on_frame(_kps(78), 1280, 640, i / 10.0, pid=0)
        r.on_frame(_kps(22), 1280, 640, i / 10.0, pid=6)
    r.on_gesture(Gesture.FOLDED_ARMS, "A", pid=6)
    saved = list(tmp_path.glob("*.npz"))
    assert len(saved) == 1
    assert _width(saved[0]) == pytest.approx(22, abs=1)
    assert "pid6" in saved[0].name, saved[0].name


# ── filenames ─────────────────────────────────────────────────────────────────

def test_two_people_firing_in_the_same_second_get_two_files(tmp_path):
    """The filename is built from int(time.time()), so simultaneous fires
    collided and one silently overwrote the other — losing exactly the
    second trace these per-person buffers exist to provide. Observed live:
    two pids fired FOLDED_ARMS 1 ms apart and both 'saved' to
    1790297391_A_folded_arms.npz.
    """
    from core.state_machine import Gesture
    r = _rec(tmp_path)
    for i in range(10):
        r.on_frame(_kps(78), 1280, 640, i / 10.0, pid=0)
        r.on_frame(_kps(22), 1280, 640, i / 10.0, pid=6)
    r.on_gesture(Gesture.FOLDED_ARMS, "A", pid=0)
    r.on_gesture(Gesture.FOLDED_ARMS, "A", pid=6)
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_the_pid_is_recorded_inside_the_file(tmp_path):
    r = _rec(tmp_path)
    r.on_frame(_kps(78), 1280, 640, 0.0, pid=6)
    z = np.load(r.save(label="x", pid=6), allow_pickle=True)
    assert str(z["pid"]) == "6"


# ── housekeeping ──────────────────────────────────────────────────────────────

def test_buffers_for_departed_people_are_dropped(tmp_path):
    """Person ids churn — ten new ones in 159 seconds was measured on the
    deployed camera — so the buffer dict must not grow without limit."""
    r = _rec(tmp_path)
    for pid in range(40):
        r.on_frame(_kps(78), 1280, 640, pid * 2.0, pid=pid)
    r.on_frame(_kps(78), 1280, 640, 40 * 2.0, pid=999)
    assert len(r._bufs) < 40, f"{len(r._bufs)} buffers retained"


def test_a_still_active_person_is_never_dropped(tmp_path):
    r = _rec(tmp_path)
    t = 0.0
    for i in range(int(_PID_IDLE_DROP_S * 4)):
        t = i * 0.5
        r.on_frame(_kps(78), 1280, 640, t, pid=0)     # continuously present
        r.on_frame(_kps(22), 1280, 640, t, pid=1)
    assert r.save(label="x", pid=0) is not None
    assert r.save(label="y", pid=1) is not None


def test_a_caller_with_no_person_tracking_still_works(tmp_path):
    """pid defaults to None, which is simply its own buffer — the debug
    page's manual capture has no particular person in mind."""
    r = _rec(tmp_path)
    for i in range(5):
        r.on_frame(_kps(78), 1280, 640, i / 10.0)
    assert r.manual_capture(label="manual") is not None


def test_a_manual_capture_picks_the_busiest_buffer(tmp_path):
    r = _rec(tmp_path)
    for i in range(3):
        r.on_frame(_kps(22), 1280, 640, i / 10.0, pid=6)
    for i in range(10):
        r.on_frame(_kps(78), 1280, 640, i / 10.0, pid=0)
    assert _width(r.manual_capture(label="m")) == pytest.approx(78, abs=1)


def test_saving_an_unknown_person_is_harmless(tmp_path):
    r = _rec(tmp_path)
    r.on_frame(_kps(78), 1280, 640, 0.0, pid=0)
    assert r.save(label="x", pid=999) is None


def test_the_window_is_still_trimmed_per_person(tmp_path):
    r = _rec(tmp_path)
    for i in range(200):                      # 20 s at 10 Hz, window is 5 s
        r.on_frame(_kps(78), 1280, 640, i / 10.0, pid=0)
    z = np.load(r.save(label="x", pid=0), allow_pickle=True)
    assert z["t_offsets"][-1] <= 5.1, z["t_offsets"][-1]


# ── the analyser feeds it per person ──────────────────────────────────────────

def test_the_analyser_records_each_tracked_person(tmp_path):
    """A regression guard on the call site, not the recorder: the old code
    picked valid[0] out of the raw detection array instead of iterating the
    tracked persons."""
    import inspect
    from cameras import analyser
    src = inspect.getsource(analyser.CameraAnalyser._run)
    # Comments in that function describe the old approach by name, so match
    # on code only — otherwise the explanation trips the test.
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "for pid, e in self._persons.items()" in code, \
        "the recorder must be fed per tracked person"
    assert "valid[0]" not in code, "valid[0] is the bug this replaces"
    assert "on_frame(kps_row, pw, ph, now, pid=pid)" in code
