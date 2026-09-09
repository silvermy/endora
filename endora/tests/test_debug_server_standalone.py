"""
tests/test_debug_server_standalone.py

The debug UI is the only way to see what Endora is doing, and moving it to a
Jetson takes away both of the add-on's routes to it — the sidebar "Open Web
UI" entry and the Configuration tab that sets debug_port. What is left is
the direct port, so these tests pin the two things that decide whether it
comes up there at all.

start() binds two listeners: the direct debug port, and an "ingress" port
that exists solely for HA's sidebar proxy. Standalone there is nothing
proxying the second one, and the deployment runs with network_mode: host —
so binding it would claim a real port on the Jetson to serve a page nothing
links to.

Sockets and threads are faked; this is about which ports start() decides to
bind, not about serving HTTP.
"""
import logging
from unittest.mock import patch

import pytest

from cameras import debug_server


class _FakeServer:
    """Stands in for HTTPServer — records the address instead of binding."""

    def __init__(self, address, handler):
        self.address = address
        self.handler = handler

    def serve_forever(self):  # pragma: no cover - never run, threads are faked
        raise AssertionError("serve_forever should not run in tests")


class _FakeThread:
    def __init__(self, *a, **kw):
        self.kwargs = kw

    def start(self):
        pass


@pytest.fixture
def bound_ports():
    """Run start() with sockets/threads faked; yield a recorder of ports.

    start() attaches a handler to the root logger, which would otherwise
    accumulate across tests and leak captured records into unrelated ones.
    """
    ports: list[int] = []

    class _Recording(_FakeServer):
        def __init__(self, address, handler):
            super().__init__(address, handler)
            ports.append(address[1])

    root = logging.getLogger()
    before = list(root.handlers)
    with patch.object(debug_server, "HTTPServer", _Recording), \
         patch.object(debug_server.threading, "Thread", _FakeThread):
        yield ports
    for h in root.handlers:
        if h not in before:
            root.removeHandler(h)


def _start_as(bound_ports, *, addon: bool, port=8765, ingress_port=8766):
    with patch.object(debug_server.deployment, "is_addon", return_value=addon):
        debug_server.start(port, ingress_port=ingress_port)
    return bound_ports


def test_addon_binds_both_the_debug_port_and_ingress(bound_ports):
    assert _start_as(bound_ports, addon=True) == [8765, 8766]


def test_standalone_binds_only_the_debug_port(bound_ports):
    """Nothing proxies the ingress port off the Supervisor, and with
    network_mode: host that bind would be on the Jetson's real interface."""
    assert _start_as(bound_ports, addon=False) == [8765]


def test_standalone_still_serves_the_full_ui_on_the_debug_port(bound_ports):
    """The UI must not be ingress-only — it is the sole route in standalone."""
    _start_as(bound_ports, addon=False)
    assert bound_ports == [8765]


def test_debug_port_is_bound_on_all_interfaces(bound_ports):
    """0.0.0.0, not localhost: the browser is on another machine entirely."""
    captured = {}

    class _Recording(_FakeServer):
        def __init__(self, address, handler):
            super().__init__(address, handler)
            captured.setdefault("host", address[0])

    root = logging.getLogger()
    before = list(root.handlers)
    with patch.object(debug_server, "HTTPServer", _Recording), \
         patch.object(debug_server.threading, "Thread", _FakeThread), \
         patch.object(debug_server.deployment, "is_addon", return_value=False):
        debug_server.start(8765)
    for h in root.handlers:
        if h not in before:
            root.removeHandler(h)

    assert captured["host"] == "0.0.0.0"
