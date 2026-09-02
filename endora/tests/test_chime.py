"""Tests for output/chime.py — no network calls made."""
import json
import time
import threading
import unittest
from unittest.mock import MagicMock, patch

from output.chime import ChimeNotifier, make_chime_notifier


def _settings(**kw):
    defaults = dict(
        chime_enable=True, chime_entity_id="media_player.living_room",
        chime_volume=40, chime_debounce_s=0.0,
        ha_url="http://supervisor/core/api",
    )
    defaults.update(kw)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


class TestChimeNotifier(unittest.TestCase):

    def _make(self, **kw):
        with patch.dict("os.environ", {"SUPERVISOR_TOKEN": "test-token"}):
            return ChimeNotifier(_settings(**kw), "http://host/chime.wav")

    def test_entity_id_stored(self):
        n = self._make()
        self.assertEqual(n._entity_id, "media_player.living_room")

    def test_post_sends_correct_payload(self):
        n = self._make()
        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(req.data.decode())
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            n._post()

        self.assertEqual(len(captured), 1)
        body = json.loads(captured[0])
        self.assertEqual(body["entity_id"], "media_player.living_room")
        self.assertEqual(body["media_content_id"], "http://host/chime.wav")
        self.assertTrue(body["announce"])

    def test_missing_entity_id_skips_post(self):
        n = self._make(chime_entity_id="")
        calls = []
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: calls.append(1)):
            n._post()
        self.assertEqual(calls, [])

    def test_missing_chime_url_skips_post(self):
        with patch.dict("os.environ", {"SUPERVISOR_TOKEN": "test-token"}):
            n = ChimeNotifier(_settings(), "")
        calls = []
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: calls.append(1)):
            n._post()
        self.assertEqual(calls, [])

    def test_debounce_suppresses_rapid_notify(self):
        n = self._make(chime_debounce_s=10.0)
        fired = []
        with patch.object(n, "_post", side_effect=lambda: fired.append(1)):
            n.notify()
            n.notify()
            n.notify()
        time.sleep(0.05)
        self.assertEqual(len(fired), 1)

    def test_zero_debounce_allows_repeated_notify(self):
        n = self._make(chime_debounce_s=0.0)
        fired = []
        with patch.object(n, "_post", side_effect=lambda: fired.append(1)):
            n.notify(); time.sleep(0.05)
            n.notify(); time.sleep(0.05)
        self.assertEqual(len(fired), 2)

    def test_volume_included_in_payload(self):
        n = self._make(chime_volume=60)
        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(req.data.decode())
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            n._post()

        body = json.loads(captured[0])
        self.assertAlmostEqual(body["extra"]["volume"], 0.6)


class TestMakeChimeNotifier(unittest.TestCase):

    def test_disabled_returns_none(self):
        s = _settings(chime_enable=False)
        self.assertIsNone(make_chime_notifier(s, "http://host/chime.wav"))

    def test_enabled_returns_notifier(self):
        s = _settings(chime_enable=True, chime_entity_id="media_player.foo")
        with patch.dict("os.environ", {"SUPERVISOR_TOKEN": "tok"}):
            n = make_chime_notifier(s, "http://host/chime.wav")
        self.assertIsInstance(n, ChimeNotifier)

    def test_enabled_no_entity_id_still_returns_notifier(self):
        s = _settings(chime_enable=True, chime_entity_id="")
        with patch.dict("os.environ", {"SUPERVISOR_TOKEN": "tok"}):
            n = make_chime_notifier(s, "http://host/chime.wav")
        self.assertIsInstance(n, ChimeNotifier)


if __name__ == "__main__":
    unittest.main()
