"""Tests for output/chime.py — no network calls made."""
import time
import threading
import unittest
from unittest.mock import MagicMock, patch

from output.chime import HAChimeBackend, SonosDirectBackend, make_chime_notifier


def _settings(**kw):
    defaults = dict(
        chime_enable=True, chime_entity_id="media_player.living_room",
        chime_volume=40, chime_debounce_s=0.0,
        sonos_ip="", sonos_player_id="",
        sonos_enable=False, sonos_volume=40, sonos_debounce_s=4.0,
        ha_url="http://supervisor/core/api",
    )
    defaults.update(kw)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


# ── HAChimeBackend ────────────────────────────────────────────────────────────

class TestHAChimeBackend(unittest.TestCase):

    def _make(self, **kw):
        with patch.dict("os.environ", {"SUPERVISOR_TOKEN": "test-token"}):
            return HAChimeBackend(_settings(**kw), "http://host/chime.wav")

    def test_entity_id_stored(self):
        b = self._make()
        self.assertEqual(b._entity_id, "media_player.living_room")

    def test_play_posts_correct_payload(self):
        b = self._make()
        captured = []
        def fake_urlopen(req, timeout=None):
            captured.append(json_body(req))
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            b._play()
        self.assertEqual(len(captured), 1)
        import json
        body = json.loads(captured[0])
        self.assertEqual(body["entity_id"], "media_player.living_room")
        self.assertEqual(body["media_content_id"], "http://host/chime.wav")
        self.assertTrue(body["announce"])

    def test_missing_entity_id_skips_play(self):
        b = self._make(chime_entity_id="")
        calls = []
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: calls.append(1)):
            b._play()
        self.assertEqual(calls, [])

    def test_debounce_suppresses_rapid_notify(self):
        b = self._make(chime_debounce_s=10.0)
        fired = []
        with patch.object(b, "_play", side_effect=lambda: fired.append(1)):
            b.notify()
            b.notify()
            b.notify()
        time.sleep(0.05)
        self.assertEqual(len(fired), 1)

    def test_zero_debounce_allows_repeated_notify(self):
        b = self._make(chime_debounce_s=0.0)
        fired = []
        with patch.object(b, "_play", side_effect=lambda: fired.append(1)):
            b.notify(); time.sleep(0.05)
            b.notify(); time.sleep(0.05)
        self.assertEqual(len(fired), 2)


def json_body(req):
    import json
    return req.data.decode()


# ── SonosDirectBackend ────────────────────────────────────────────────────────

class TestSonosDirectBackend(unittest.TestCase):

    def _make(self, **kw):
        kw.setdefault("sonos_ip", "192.168.1.42")
        kw.setdefault("sonos_player_id", "RINCON_TEST")
        return SonosDirectBackend(_settings(**kw), "http://host/chime.wav")

    def test_resolved_when_both_configured(self):
        b = self._make()
        self.assertTrue(b._resolved)

    def test_unresolved_skips_play(self):
        with patch("output.chime._ssdp_find_sonos", return_value=None):
            b = SonosDirectBackend(_settings(sonos_ip="", sonos_player_id=""),
                                   "http://host/chime.wav")
        b._resolved = False; b._ip = None; b._player_id = None
        calls = []
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: calls.append(1)):
            b._play()
        self.assertEqual(calls, [])


# ── Factory ───────────────────────────────────────────────────────────────────

class TestMakeChimeNotifier(unittest.TestCase):

    def test_disabled_returns_none(self):
        s = _settings(chime_enable=False, sonos_enable=False)
        self.assertIsNone(make_chime_notifier(s, "http://host/chime.wav"))

    def test_entity_id_gives_ha_backend(self):
        s = _settings(chime_enable=True, chime_entity_id="media_player.foo")
        with patch.dict("os.environ", {"SUPERVISOR_TOKEN": "tok"}):
            n = make_chime_notifier(s, "http://host/chime.wav")
        self.assertIsInstance(n, HAChimeBackend)

    def test_sonos_ip_gives_sonos_backend(self):
        s = _settings(chime_enable=True, chime_entity_id="",
                      sonos_ip="192.168.1.10", sonos_player_id="RINCON_X")
        n = make_chime_notifier(s, "http://host/chime.wav")
        self.assertIsInstance(n, SonosDirectBackend)

    def test_neither_set_gives_ha_backend(self):
        s = _settings(chime_enable=True, chime_entity_id="", sonos_ip="")
        with patch.dict("os.environ", {"SUPERVISOR_TOKEN": "tok"}):
            n = make_chime_notifier(s, "http://host/chime.wav")
        self.assertIsInstance(n, HAChimeBackend)

    def test_chime_enable_true_returns_notifier(self):
        s = _settings(chime_enable=True, chime_entity_id="media_player.foo")
        with patch.dict("os.environ", {"SUPERVISOR_TOKEN": "tok"}):
            n = make_chime_notifier(s, "http://host/chime.wav")
        self.assertIsNotNone(n)


if __name__ == "__main__":
    unittest.main()
