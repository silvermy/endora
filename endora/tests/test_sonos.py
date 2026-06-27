"""Tests for SonosNotifier — no network calls made."""
import time
import unittest
from unittest.mock import MagicMock, patch

from output.sonos import SonosNotifier


def _settings(ip="192.168.1.42", player_id="RINCON_TEST", volume=30, debounce=0.0, enable=True):
    s = MagicMock()
    s.sonos_ip = ip
    s.sonos_player_id = player_id
    s.sonos_volume = volume
    s.sonos_debounce_s = debounce
    s.sonos_enable = enable
    return s


class TestSonosNotifier(unittest.TestCase):

    def _make(self, **kw):
        return SonosNotifier(_settings(**kw), chime_url="http://host/chime.wav")

    def test_resolved_when_both_configured(self):
        n = self._make()
        self.assertTrue(n._resolved)
        self.assertEqual(n._ip, "192.168.1.42")
        self.assertEqual(n._player_id, "RINCON_TEST")

    def test_debounce_suppresses_rapid_calls(self):
        n = self._make(debounce=10.0)
        posted = []
        with patch.object(n, "_post_clip", side_effect=lambda: posted.append(1)):
            n.notify()
            n.notify()
            n.notify()
        # Only the first call should have advanced past debounce
        time.sleep(0.05)
        self.assertEqual(len(posted), 1)

    def test_zero_debounce_allows_repeated_calls(self):
        n = self._make(debounce=0.0)
        posted = []
        with patch.object(n, "_post_clip", side_effect=lambda: posted.append(1)):
            n.notify()
            time.sleep(0.05)
            n.notify()
            time.sleep(0.05)
        self.assertEqual(len(posted), 2)

    def test_not_resolved_skips_post(self):
        """_post_clip should bail out early when device is unresolved."""
        with patch("output.sonos._ssdp_discover_sonos", return_value=None):
            n = SonosNotifier(_settings(ip="", player_id=""), chime_url="http://host/chime.wav")
        # Force unresolved state
        n._resolved = False
        n._ip = None
        n._player_id = None
        calls = []
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: calls.append(1)):
            n._post_clip()
        self.assertEqual(calls, [])

    def test_chime_url_stored(self):
        n = SonosNotifier(_settings(), chime_url="http://192.168.1.50:8765/chime.wav")
        self.assertEqual(n._chime_url, "http://192.168.1.50:8765/chime.wav")


if __name__ == "__main__":
    unittest.main()
