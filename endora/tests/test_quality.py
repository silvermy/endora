"""
tests/test_quality.py

Hour-by-hour detection quality, measured on every tracked detection rather
than only on recordings — which exist only when something fired, and so are
biased towards the moments detection worked.
"""
import datetime
import json

import numpy as np

from cameras.quality import HourlyQuality, read_recent


class Clock:
    def __init__(self):
        self.t = datetime.datetime(2026, 10, 21, 19, 10)
    def __call__(self):
        return self.t


def _kps(conf=0.9, wrists=0.9):
    k = np.zeros((17, 3), np.float32)
    k[:, 2] = conf
    k[[9, 10], 2] = wrists
    return k


def test_nothing_is_reported_until_the_hour_turns(tmp_path):
    clock = Clock()
    q = HourlyQuality(clock=clock, path=tmp_path / "q.jsonl")
    q.add_frame(np.full((10, 10, 3), 100, np.uint8))
    assert q.tick() is None


def test_the_finished_hour_is_summarised_and_written(tmp_path):
    clock = Clock()
    path = tmp_path / "q.jsonl"
    q = HourlyQuality(clock=clock, path=path)
    q.add_frame(np.full((10, 10, 3), 100, np.uint8))          # grey: IR mode
    colour = np.zeros((10, 10, 3), np.uint8); colour[..., 2] = 200
    q.add_frame(colour)                                         # colour
    q.add_detection(_kps(0.9, 0.9), 0.80)
    q.add_detection(_kps(0.7, 0.2), 0.40)                       # wrists not seen
    q.note_dropout()
    q.note_lost()
    clock.t = datetime.datetime(2026, 10, 21, 20, 0, 5)
    s = q.tick()
    assert s["hour"] == "2026-10-21 19:00"
    assert s["ir_fraction"] == 0.5
    assert s["tracked_samples"] == 2
    assert s["wrists_seen"] == 0.5
    assert s["box_conf"] == 0.6
    assert (s["dropouts"], s["lost"]) == (1, 1)
    assert json.loads(path.read_text())["hour"] == "2026-10-21 19:00"


def test_the_next_hour_starts_empty(tmp_path):
    clock = Clock()
    q = HourlyQuality(clock=clock, path=tmp_path / "q.jsonl")
    q.add_frame(np.full((10, 10, 3), 100, np.uint8))
    q.note_dropout()
    clock.t = datetime.datetime(2026, 10, 21, 20, 0, 5)
    q.tick()
    assert q.summary()["dropouts"] == 0 and q.summary()["frames"] == 0


def test_an_hour_with_no_frames_is_not_written(tmp_path):
    clock = Clock()
    path = tmp_path / "q.jsonl"
    q = HourlyQuality(clock=clock, path=path)
    clock.t = datetime.datetime(2026, 10, 21, 20, 0, 5)
    q.tick()
    assert not path.exists()


def test_reading_back(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"hour": "a"}\nnot json\n{"hour": "b"}\n')
    assert [r["hour"] for r in read_recent(path=path)] == ["a", "b"]


def test_a_muted_colour_frame_is_not_mistaken_for_infrared(tmp_path):
    """This camera's colour is muted: lamplit evening frames measure only
    2.5-3.0, against 0 for its infrared night mode."""
    clock = Clock()
    q = HourlyQuality(clock=clock, path=tmp_path / "q.jsonl")
    dim = np.full((10, 10, 3), 70, np.uint8)
    dim[..., 2] = 75                               # a faint warm cast
    q.add_frame(dim)
    assert q.summary()["ir_fraction"] == 0.0


def test_a_few_coloured_pixels_do_not_make_infrared_look_like_colour(tmp_path):
    clock = Clock()
    q = HourlyQuality(clock=clock, path=tmp_path / "q.jsonl")
    ir = np.full((10, 10, 3), 40, np.uint8)
    ir[0, :3] = (0, 255, 0)                        # an overlay box, say
    q.add_frame(ir)
    assert q.summary()["ir_fraction"] == 1.0
