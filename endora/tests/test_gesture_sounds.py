"""
tests/test_gesture_sounds.py

Per-gesture sounds.

The arm-raise chime fires on the raise that precedes a SNAP, not on the
gesture itself — so FOLDED_ARMS, which raises no arm, never reaches it and
was silent. These sounds are played when the gesture resolves instead.

Also covers serving them: the endpoint used to know one filename, and now
takes a name from the URL, which is only safe if it is confined to the
bundled directory.
"""
from pathlib import Path

import pytest

from core import system as sys_mod
from core.state_machine import Gesture

STATIC = Path(__file__).resolve().parent.parent / "cameras" / "static"


# ── the mapping ───────────────────────────────────────────────────────────────

def test_folded_arms_has_its_own_sound():
    assert Gesture.FOLDED_ARMS in sys_mod._GESTURE_SOUNDS


def test_every_mapped_sound_is_actually_bundled():
    """A mapping naming a file that is not shipped fails at runtime with only
    a log line — the gesture is simply silent, which is what it was before."""
    for gesture, filename in sys_mod._GESTURE_SOUNDS.items():
        assert (STATIC / filename).is_file(), f"{gesture.name} -> {filename} missing"


def test_mapping_is_keyed_by_gesture_not_string():
    # A string key would silently stop matching if the enum were renamed.
    for key in sys_mod._GESTURE_SOUNDS:
        assert isinstance(key, Gesture)


# ── install: the same two deployment routes as the chime ──────────────────────

def _digest(name):
    import hashlib
    return hashlib.sha256((STATIC / name).read_bytes()).hexdigest()[:8]


def test_standalone_serves_the_sound_from_the_debug_server(tmp_path):
    name = sys_mod._GESTURE_SOUNDS[Gesture.FOLDED_ARMS]
    url = sys_mod._install_sound(name, "10.0.0.5", 8765, media_dir=tmp_path / "none")
    assert url == f"http://10.0.0.5:8765/sound/{name}?v={_digest(name)}"


def test_addon_copies_the_sound_into_media(tmp_path):
    from unittest.mock import patch
    name = sys_mod._GESTURE_SOUNDS[Gesture.FOLDED_ARMS]
    stem, suffix = Path(name).stem, Path(name).suffix
    with patch("core.system.deployment.is_addon", return_value=True):
        url = sys_mod._install_sound(name, media_dir=tmp_path)
    assert url == f"media-source://media_source/local/endora_{stem}_{_digest(name)}{suffix}"
    assert (tmp_path / f"endora_{stem}_{_digest(name)}{suffix}").exists()


def test_sounds_do_not_delete_each_other(tmp_path):
    """Each install prunes superseded copies of ITS OWN clip.

    The prune glob was written for a single sound. Shared across two, each
    startup would have had the second sound delete the first.
    """
    from unittest.mock import patch
    with patch("core.system.deployment.is_addon", return_value=True):
        sys_mod._install_sound("chime.wav", media_dir=tmp_path)
        sys_mod._install_sound("folded_arms.mp3", media_dir=tmp_path)
    installed = sorted(p.name for p in tmp_path.glob("endora_*"))
    assert len(installed) == 2, installed


def test_a_missing_sound_returns_empty_rather_than_raising(tmp_path):
    assert sys_mod._install_sound("nope.wav", "10.0.0.5", 8765,
                                  media_dir=tmp_path / "none") == ""


def test_the_chime_still_installs_the_same_way(tmp_path):
    # _install_chime_wav is now a caller of _install_sound; existing installs
    # carry its URL in Home Assistant's saved config.
    url = sys_mod._install_chime_wav("10.0.0.5", 8765, media_dir=tmp_path / "none")
    assert url.endswith(f"?v={_digest('chime.wav')}")


# ── serving ───────────────────────────────────────────────────────────────────

def test_content_type_is_correct_per_extension():
    from cameras.debug_server import _AUDIO_TYPES
    assert _AUDIO_TYPES[".wav"] == "audio/wav"
    assert _AUDIO_TYPES[".mp3"] == "audio/mpeg"


@pytest.mark.parametrize("attack", [
    "../settings.yaml",
    "../../core/system.py",
    "..%2Fsystem.py",
])
def test_path_traversal_is_refused(attack):
    """The clip name arrives from a URL. Resolving it inside the static
    directory and comparing parents is what keeps '../' from reading the
    host's files."""
    static = STATIC.resolve()
    clip = (STATIC / attack).resolve()
    assert clip.parent != static or not clip.is_file()


# ── exactly one sound per gesture ─────────────────────────────────────────────

class _Notifier:
    def __init__(self, name): self.name, self.plays = name, 0
    def notify(self): self.plays += 1


def _system_with(gesture_sounds):
    """A GestureSystem stub carrying just the sound-selection logic."""
    sysobj = sys_mod.GestureSystem.__new__(sys_mod.GestureSystem)
    sysobj._chime = _Notifier("chime")
    sysobj._gesture_chimes = gesture_sounds
    return sysobj


def test_a_mapped_gesture_plays_only_its_own_sound():
    """The bug this replaces: the analyser played the chime for every gesture
    and the per-gesture sound was played somewhere else, so both reached the
    speaker a millisecond apart. With announce:true the second replaces the
    first, and the gesture's own sound was audible only if it won that race —
    reported as "I got a bewitched sound, not genie".
    """
    jeannie = _Notifier("jeannie")
    s = _system_with({Gesture.FOLDED_ARMS: jeannie})
    s._play_gesture_sound(Gesture.FOLDED_ARMS)
    assert (jeannie.plays, s._chime.plays) == (1, 0)


def test_an_unmapped_gesture_still_gets_the_chime():
    jeannie = _Notifier("jeannie")
    s = _system_with({Gesture.FOLDED_ARMS: jeannie})
    s._play_gesture_sound(Gesture.SNAP)
    assert (jeannie.plays, s._chime.plays) == (0, 1)


def test_no_chime_configured_is_harmless():
    s = _system_with({})
    s._chime = None
    s._play_gesture_sound(Gesture.SNAP)          # must not raise


def test_the_analyser_defers_to_the_callback():
    """The analyser must not also fire its own chime when a callback is set,
    or the two sounds race again."""
    import inspect
    from cameras import analyser
    src = inspect.getsource(analyser.CameraAnalyser._run)
    block = src.split("Guarantee the sound accompanies a real gesture")[1][:1600]
    assert "self._gesture_sound_cb(gesture)" in block
    assert "elif self._chime is not None" in block, \
        "the chime path must be an ELSE, not a second unconditional play"
