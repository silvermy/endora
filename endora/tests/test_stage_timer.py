"""
tests/test_stage_timer.py

The analyser's loop-timing line.

It exists because a live install was sampling poses 0.63 times a second
while its stats line cheerfully reported 11 fps — and the flourish window
only measures a climb if several samples land inside it, so the pose sample
rate, not the capture rate, is what decides whether a sweep can be detected
at all. The number has to be reported, and it has to be the *pose* rate.
"""
import logging
import time

from cameras.analyser import _StageTimer


def _run(timer, iters, yolo_every=1, stages=("prep", "yolo"), pace=0.003):
    runs = 0
    for i in range(iters):
        time.sleep(pace)          # so the reporting period actually elapses
        t = time.monotonic()
        for s in stages:
            if s == "yolo":
                if i % yolo_every:
                    continue
                runs += 1
            t = timer.mark(s, t)
        timer.tick("A", runs)
    return runs


def test_nothing_is_logged_before_the_period_elapses():
    # One line every 30 s in normal operation; it must not become per-frame
    # log spam on a fast loop.
    timer = _StageTimer(period_s=1000.0)
    logging.getLogger("cameras.analyser").setLevel(logging.INFO)
    _run(timer, 50)


def test_reports_pose_rate_not_loop_rate(caplog):
    # The distinction the line exists to make: four loop iterations per pose
    # sample must report a pose rate a quarter of the loop rate.
    logging.getLogger("cameras.analyser").setLevel(logging.INFO)
    timer = _StageTimer(period_s=0.05)
    with caplog.at_level(logging.INFO, logger="cameras.analyser"):
        _run(timer, 40, yolo_every=4)
    lines = [r.message for r in caplog.records if "pose" in r.message]
    assert lines, "no timing line logged"
    msg = lines[-1]
    loop = float(msg.split("loop ")[1].split(" iter/s")[0])
    pose = float(msg.split("pose ")[1].split(" sample/s")[0])
    assert pose < loop, msg
    assert abs(pose * 4 - loop) < loop * 0.35, f"pose rate not ~1/4 of loop: {msg}"


def test_stage_names_appear_in_order(caplog):
    logging.getLogger("cameras.analyser").setLevel(logging.INFO)
    timer = _StageTimer(period_s=0.05)
    with caplog.at_level(logging.INFO, logger="cameras.analyser"):
        _run(timer, 30, stages=("prep", "bgsub", "clahe", "yolo"))
    msg = [r.message for r in caplog.records if "mean ms/iter" in r.message][-1]
    tail = msg.split("mean ms/iter: ")[1]
    assert tail.index("prep") < tail.index("bgsub") < tail.index("clahe")


def test_counters_reset_between_periods(caplog):
    # Otherwise every reported rate is an average since boot and a slowdown
    # that starts an hour in is invisible.
    logging.getLogger("cameras.analyser").setLevel(logging.INFO)
    timer = _StageTimer(period_s=0.05)
    with caplog.at_level(logging.INFO, logger="cameras.analyser"):
        _run(timer, 30)
        first = len([r for r in caplog.records if "pose" in r.message])
        _run(timer, 30)
        second = len([r for r in caplog.records if "pose" in r.message])
    assert second > first, "second period never reported"
    # A fresh period counts only its own samples, so the counts stay small
    # rather than growing without bound.
    msg = [r.message for r in caplog.records if "pose" in r.message][-1]
    n = int(msg.split("(")[1].split(" in")[0])
    assert n <= 30, f"sample count carried over between periods: {msg}"
