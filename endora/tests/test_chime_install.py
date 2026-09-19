"""
tests/test_chime_install.py

_install_chime_wav decides how a speaker gets hold of the chime audio, and
the answer differs per deployment: the add-on copies it into HA's /media,
while the Jetson has no access to that folder and must serve the clip over
its own debug server instead.

The standalone route was described in the function's docstring but never
implemented — it returned "" whenever /media was absent, which silently
disabled the chime on exactly the deployment the docstring was about.

The digest assertions are not cosmetic: a fixed URL is what made a replaced
chime keep playing from HA's and the speaker's caches (v1.9.140), so a URL
that does not move with the audio is the bug, not a detail.
"""
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from core.system import _install_chime_wav

WAV = Path(__file__).parent.parent / "cameras" / "static" / "chime.wav"


@pytest.fixture
def digest():
    return hashlib.sha256(WAV.read_bytes()).hexdigest()[:8]


@pytest.fixture
def addon():
    """Run as though under the Supervisor."""
    with patch("core.system.deployment.is_addon", return_value=True):
        yield


# ── add-on route ──────────────────────────────────────────────────────────

def test_addon_copies_into_media_and_returns_a_media_source_url(addon, tmp_path, digest):
    url = _install_chime_wav(media_dir=tmp_path)
    assert url == f"media-source://media_source/local/endora_chime_{digest}.wav"
    assert (tmp_path / f"endora_chime_{digest}.wav").exists()


def test_addon_route_wins_even_when_a_debug_port_is_available(addon, tmp_path):
    """/media works regardless of debug_port, so it stays the preferred route
    on the add-on — it does not depend on the debug page being switched on."""
    url = _install_chime_wav("10.0.0.5", 8765, media_dir=tmp_path)
    assert url.startswith("media-source://")


def test_superseded_clips_are_removed(addon, tmp_path, digest):
    stale = tmp_path / "endora_chime_deadbeef.wav"
    stale.write_bytes(b"old")
    _install_chime_wav(media_dir=tmp_path)
    assert not stale.exists()
    assert (tmp_path / f"endora_chime_{digest}.wav").exists()


def test_standalone_ignores_a_media_directory_that_happens_to_exist(tmp_path, digest):
    """The bug this guards against, seen on a live Jetson.

    The route used to be chosen by whether /media was a directory. The L4T
    base image has one, so a standalone install copied the clip into its own
    container's throwaway /media and handed Home Assistant a media-source://
    URL for a file only the HA machine could serve. It played only because an
    earlier add-on install had left that file behind, and would have gone
    silent the moment anyone tidied up — with nothing in the log to say why.
    """
    url = _install_chime_wav("10.0.0.5", 8765, media_dir=tmp_path)
    assert url == f"http://10.0.0.5:8765/sound/chime.wav?v={digest}"
    assert not list(tmp_path.glob("endora_chime*.wav")), \
        "wrote into a /media that Home Assistant cannot read"


# ── standalone route ──────────────────────────────────────────────────────

def test_standalone_serves_the_clip_from_the_debug_server(tmp_path, digest):
    missing = tmp_path / "no-media-here"
    url = _install_chime_wav("10.0.0.5", 8765, media_dir=missing)
    assert url == f"http://10.0.0.5:8765/sound/chime.wav?v={digest}"


def test_standalone_url_carries_the_audio_digest(tmp_path, digest):
    """The debug server matches on path alone, so the query string is free to
    carry the cache key — the path still serves, but the URL moves when the
    audio does.

    The path became /sound/<name> when gestures gained their own sounds; the
    bare /chime.wav endpoint is still served for anything holding the old URL.
    """
    url = _install_chime_wav("10.0.0.5", 8765, media_dir=tmp_path / "nope")
    assert url.endswith(f"?v={digest}")
    assert "/sound/chime.wav?" in url


def test_standalone_without_a_debug_port_gives_up_loudly(tmp_path, caplog):
    """The chime depends on the debug server off the Supervisor. Returning ""
    is correct, but it must say why — this combination is otherwise a chime
    that simply never sounds."""
    with caplog.at_level("WARNING", logger="core.system"):
        url = _install_chime_wav("10.0.0.5", 0, media_dir=tmp_path / "nope")
    assert url == ""
    assert any("debug_port" in r.getMessage() for r in caplog.records)


def test_standalone_without_a_host_ip_gives_up(tmp_path):
    url = _install_chime_wav("", 8765, media_dir=tmp_path / "nope")
    assert url == ""


# ── fallthrough ───────────────────────────────────────────────────────────

def test_unwritable_media_falls_through_to_the_http_route(addon, tmp_path, digest):
    """A mounted but unwritable /media used to return "" outright. If the
    debug server can serve the clip, that is a working chime going unused."""
    with patch("shutil.copy2", side_effect=PermissionError):
        url = _install_chime_wav("10.0.0.5", 8765, media_dir=tmp_path)
    assert url == f"http://10.0.0.5:8765/sound/chime.wav?v={digest}"


def test_missing_source_audio_returns_empty(tmp_path):
    with patch.object(Path, "exists", return_value=False):
        assert _install_chime_wav("10.0.0.5", 8765, media_dir=tmp_path) == ""
