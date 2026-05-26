#!/usr/bin/env python3
"""
Endora — Home Assistant Add-on entry point

When running as a HA add-on, settings come from /data/options.json
(written by the Supervisor).  The SUPERVISOR_TOKEN env var is injected
automatically by the Supervisor for API access.
"""

import faulthandler
import logging
import os
import signal
import sys

faulthandler.enable()  # dump native stack trace on SIGSEGV/SIGFPE/etc.

from config.settings import Settings
from core.system import GestureSystem
from version import __version__


def setup_logging(level_str: str):
    # Force line-buffering so log messages appear immediately in the container
    # without needing PYTHONUNBUFFERED=1 (which would invalidate slow Docker layers).
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass  # Python < 3.7 fallback; not expected
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,  # reconfigure even if a library already added handlers
    )
    logging.getLogger().setLevel(level)
    # Silence noisy third-party loggers regardless of our log level
    for _noisy in ("matplotlib", "PIL", "ultralytics", "urllib3",
                   "absl", "tensorflow"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)


def main():
    settings = Settings.load()
    setup_logging(settings.log_level)

    log = logging.getLogger("main")
    log.info("Endora v%s starting (HA add-on mode)", __version__)
    log.info("RTSP A: %s", _mask(settings.rtsp_url_a))
    log.info("RTSP B: %s", _mask(settings.rtsp_url_b))
    log.info("HA event: %s → %s/events/%s",
             settings.ha_event_name, settings.ha_url, settings.ha_event_name)
    if settings.debug_port > 0:
        import socket as _sock
        try:
            _s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
            _s.connect(("8.8.8.8", 80))
            _ip = _s.getsockname()[0]
            _s.close()
        except Exception:
            _ip = "homeassistant.local"
        log.info("Debug stream: http://%s:%d/", _ip, settings.debug_port)
    else:
        log.warning("Debug stream DISABLED (debug_port=0). "
                    "Set debug_port=8765 in the add-on Configuration tab to enable.")

    system = GestureSystem(settings)

    def _shutdown(sig, frame):
        log.info("Shutdown signal received")
        system.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    system.run()


def _mask(url: str) -> str:
    import re
    return re.sub(r"(rtsp://[^:]+:)[^@]+(@)", r"\1****\2", url)


if __name__ == "__main__":
    main()
