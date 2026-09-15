"""
tests/test_debug_idle_cost.py

The debug overlay is drawn only while someone is looking at it.

Rendering it copies the frame and draws skeletons, boxes and a status panel,
which measured ~8% of the analyser loop on the Jetson — and it was paid on
every iteration whether or not a browser was open, because nothing tracked
viewers. Since the chime is now served by the same server, turning the debug
port off to reclaim that would also silence the sound, so the idle cost has
to go instead of the feature.

What must NOT be gated is the other half: the log ring buffer and the
gesture frame captures earn their keep precisely when nobody is watching,
since their value is having the record from before the page was opened.
Every intermittent bug in this project was found that way.
"""
import time

import pytest

from cameras import debug_server


@pytest.fixture(autouse=True)
def _reset_viewer():
    debug_server._last_viewed = 0.0
    yield
    debug_server._last_viewed = 0.0


def test_nobody_watching_means_nothing_to_render():
    assert not debug_server.is_being_viewed()


def test_requesting_a_frame_marks_a_viewer():
    debug_server.note_viewer()
    assert debug_server.is_being_viewed()


def test_the_mark_expires():
    debug_server.note_viewer()
    debug_server._last_viewed = time.monotonic() - debug_server._VIEWER_TIMEOUT_S - 1
    assert not debug_server.is_being_viewed()


def test_the_window_outlasts_a_slow_poll():
    # The page polls /frame; a backgrounded tab or a slow link must not make
    # the renderer idle out underneath a viewer who is still there.
    debug_server.note_viewer()
    debug_server._last_viewed = time.monotonic() - 2.0
    assert debug_server.is_being_viewed(), "gave up on a viewer after 2s"


# ── the analyser's side of the gate ───────────────────────────────────────────

class _Analyser:
    """Just the render decision from CameraAnalyser._run."""

    def __init__(self, frame_cb, wanted_cb):
        self.debug_frame_cb = frame_cb
        self.debug_wanted_cb = wanted_cb

    def would_render(self):
        return self.debug_frame_cb is not None and (
            self.debug_wanted_cb is None or self.debug_wanted_cb()
        )


def test_render_is_skipped_when_unwatched():
    assert not _Analyser(lambda *_: None, lambda: False).would_render()


def test_render_happens_when_watched():
    assert _Analyser(lambda *_: None, lambda: True).would_render()


def test_no_predicate_means_always_render():
    # Back-compatibility: every caller that does not pass one — including the
    # tests — must behave exactly as before.
    assert _Analyser(lambda *_: None, None).would_render()


def test_no_callback_means_never_render():
    assert not _Analyser(None, lambda: True).would_render()


def test_captures_and_logs_are_not_gated_on_viewers():
    """Guard the intent, not just the mechanism.

    A future change that routes frame captures or the log handler through the
    viewer check would remove exactly the evidence this project relies on.
    """
    import inspect
    from cameras import analyser

    src = inspect.getsource(analyser.CameraAnalyser._run)
    capture_block = src.split("_frame_capture is not None")[1].split("_prev_primary_state =")[0]
    assert "debug_wanted_cb" not in capture_block, \
        "gesture frame captures must be saved whether or not anyone is watching"
