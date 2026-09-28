"""
tests/test_labeler.py

Marking captures with what was actually intended.

The filenames record what the state machine DECIDED, which is the thing
under review — so they are the hypothesis, not the labels. Three fixes this
week stopped short for want of the difference: the folded-arms span ceiling
was fitted against two known-false captures out of seventy-seven,
flourish_require_progressive shipped switched off because enabling it would
suppress five of fourteen snaps with only two confirmed bad, and the recline
work could only be judged against captures whose status was guessed from
when they happened.

What matters most here is that labelling can never damage the evidence. The
captures are read-only, opinions live in a separate file, and losing that
file loses opinions and nothing else.
"""
import json

import numpy as np
import pytest

from cameras import labeler


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setattr(labeler, "CAPTURE_DIR", tmp_path / "caps")
    monkeypatch.setattr(labeler, "LABELS_PATH", tmp_path / "labels.json")
    (tmp_path / "caps").mkdir()
    return labeler


def _capture(lab, name, frames=10, gesture="SNAP", pid="7"):
    k = np.zeros((frames, 17, 3), dtype=np.float32)
    for f in range(frames):
        k[f, 5] = (450, 300, 0.9)
        k[f, 6] = (350, 300, 0.9)
        k[f, 9] = (460, 300 - f * 10, 0.8)      # a wrist rising
    np.savez_compressed(
        lab.CAPTURE_DIR / name, keypoints=k,
        t_offsets=np.arange(frames, dtype=np.float64) / 10.0,
        frame_w=np.int32(1280), frame_h=np.int32(640),
        label=np.array(name), gesture=np.array(gesture), pid=np.array(pid))
    return name


# ── the evidence is never touched ────────────────────────────────────────────

def test_labelling_does_not_modify_the_capture(lab):
    n = _capture(lab, "1790000000_A_snap_pid7.npz")
    before = (lab.CAPTURE_DIR / n).read_bytes()
    lab.set_label(n, "not_intended", "arm on the sofa back")
    assert (lab.CAPTURE_DIR / n).read_bytes() == before


def test_losing_the_label_file_loses_only_opinions(lab):
    n = _capture(lab, "1790000000_A_snap_pid7.npz")
    lab.set_label(n, "intended")
    lab.LABELS_PATH.unlink()
    assert lab.list_captures()[0]["label"] is None
    assert lab.load_trace(n) is not None


def test_a_corrupt_label_file_does_not_take_the_captures_down(lab):
    n = _capture(lab, "1790000000_A_snap_pid7.npz")
    lab.LABELS_PATH.write_text("{ this is not json")
    assert lab.list_captures()[0]["name"] == n
    assert lab.set_label(n, "intended") is True
    assert lab.get_labels()[n]["label"] == "intended"


# ── labels ───────────────────────────────────────────────────────────────────

def test_a_label_round_trips(lab):
    n = _capture(lab, "1790000000_A_snap_pid7.npz")
    lab.set_label(n, "not_intended", "stretching")
    row = lab.list_captures()[0]
    assert (row["label"], row["note"]) == ("not_intended", "stretching")


def test_a_label_can_be_changed_and_cleared(lab):
    n = _capture(lab, "1790000000_A_snap_pid7.npz")
    lab.set_label(n, "intended")
    lab.set_label(n, "unsure")
    assert lab.list_captures()[0]["label"] == "unsure"
    lab.set_label(n, None)
    assert lab.list_captures()[0]["label"] is None


def test_an_unknown_label_is_refused(lab):
    """The vocabulary is three words on purpose — a guessed label is worse
    than an absent one, because it looks like data."""
    n = _capture(lab, "1790000000_A_snap_pid7.npz")
    assert lab.set_label(n, "probably") is False
    assert lab.list_captures()[0]["label"] is None


def test_notes_are_bounded(lab):
    n = _capture(lab, "1790000000_A_snap_pid7.npz")
    lab.set_label(n, "unsure", "x" * 5000)
    assert len(lab.get_labels()[n]["note"]) <= 500


# ── names arrive from a URL ──────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [
    "../../etc/passwd", "../labels.json", "sub/dir.npz", "", "notanpz.txt",
])
def test_a_name_outside_the_capture_directory_is_refused(lab, bad):
    assert lab.load_trace(bad) is None
    assert lab.set_label(bad, "intended") is False


def test_a_missing_capture_reads_as_absent_not_an_error(lab):
    assert lab.load_trace("1790000000_A_snap_pid9.npz") is None


# ── listing and parsing ──────────────────────────────────────────────────────

def test_the_filename_is_parsed_into_its_parts(lab):
    _capture(lab, "1790612472_A_folded_arms_pid240.npz")
    row = lab.list_captures()[0]
    assert (row["ts"], row["gesture"], row["pid"]) == (1790612472, "folded_arms", "240")


def test_older_filenames_without_a_pid_still_parse(lab):
    """Captures recorded before per-person buffers existed are still
    evidence and must remain labellable."""
    _capture(lab, "1790267818_A_folded_arms.npz")
    row = lab.list_captures()[0]
    assert (row["gesture"], row["pid"]) == ("folded_arms", "")


def test_captures_are_listed_newest_first(lab):
    _capture(lab, "1790000001_A_snap_pid1.npz")
    _capture(lab, "1790000003_A_snap_pid3.npz")
    _capture(lab, "1790000002_A_snap_pid2.npz")
    assert [r["ts"] for r in lab.list_captures()] == [1790000003, 1790000002, 1790000001]


def test_an_empty_directory_is_not_an_error(lab):
    assert lab.list_captures() == []
    assert lab.summary()["total"] == 0


# ── the trace the page draws ─────────────────────────────────────────────────

def test_a_trace_carries_keypoints_and_geometry(lab):
    n = _capture(lab, "1790000000_A_snap_pid7.npz", frames=12)
    t = lab.load_trace(n)
    assert len(t["frames"]) == 12 and len(t["frames"][0]) == 17
    assert (t["frame_w"], t["frame_h"]) == (1280, 640)
    assert len(t["t"]) == 12


def test_a_trace_keeps_confidences(lab):
    """The page draws joints at an opacity set by confidence. Dropping it
    would make a guessed joint look exactly as certain as a seen one, which
    is the illusion that made several of these captures hard to judge."""
    n = _capture(lab, "1790000000_A_snap_pid7.npz")
    t = lab.load_trace(n)
    assert t["frames"][0][5][2] == pytest.approx(0.9, abs=0.01)
    assert t["frames"][0][1][2] == 0.0          # a joint that was never seen


def test_a_trace_is_json_serialisable(lab):
    n = _capture(lab, "1790000000_A_snap_pid7.npz")
    json.dumps(lab.load_trace(n))               # must not raise on numpy types


# ── the summary that says whether a conclusion is safe ───────────────────────

def test_the_summary_counts_by_gesture_and_label(lab):
    a = _capture(lab, "1790000001_A_snap_pid1.npz", gesture="SNAP")
    b = _capture(lab, "1790000002_A_snap_pid2.npz", gesture="SNAP")
    c = _capture(lab, "1790000003_A_folded_arms_pid3.npz", gesture="FOLDED_ARMS")
    lab.set_label(a, "intended")
    lab.set_label(b, "not_intended")
    s = lab.summary()
    assert s["total"] == 3 and s["labelled"] == 2
    assert s["by_gesture"]["snap"]["intended"] == 1
    assert s["by_gesture"]["snap"]["not_intended"] == 1
    assert s["by_gesture"]["folded_arms"]["unlabelled"] == 1


# ── the page ─────────────────────────────────────────────────────────────────

def test_the_page_renders_with_a_real_skeleton():
    html = labeler.page().decode()
    assert "__SKELETON__" not in html, "template placeholder left unreplaced"
    assert json.dumps(labeler.SKELETON) in html


def test_the_page_offers_exactly_the_three_labels():
    html = labeler.page().decode()
    for lbl in labeler.LABELS:
        assert f'data-l="{lbl}"' in html


def test_the_page_uses_relative_urls():
    """It is reachable both directly and behind Home Assistant's ingress
    prefix; an absolute /label/... would 404 through the proxy."""
    html = labeler.page().decode()
    assert '"/label/list"' not in html and '"label/list"' in html
