#!/usr/bin/with-contenv bashio
# ── Endora — S6 service run script ──────────────────────────────────────
# This file lives at /etc/services.d/endora/run
# S6 calls it to start (and restart on crash) the endora process.
# with-contenv imports the container environment so RTSP URLs etc. are visible.

set -e

bashio::log.info "Endora v1.6.11 starting..."

if ! bashio::config.has_value "rtsp_url_a"; then
    bashio::log.fatal "rtsp_url_a is not configured"
    exit 1
fi

if ! bashio::config.has_value "rtsp_url_b"; then
    bashio::log.fatal "rtsp_url_b is not configured"
    exit 1
fi

bashio::log.info "Camera A: $(bashio::config 'rtsp_url_a' | sed 's|//[^:]*:[^@]*@|//****:****@|g')"
bashio::log.info "Camera B: $(bashio::config 'rtsp_url_b' | sed 's|//[^:]*:[^@]*@|//****:****@|g')"
bashio::log.info "HA event: $(bashio::config 'ha_event_name')"

# Export add-on options as environment variables so Python settings.py
# can read them directly regardless of /data/options.json parsing order.
export DEBUG_PORT="$(bashio::config 'debug_port')"
export LOG_LEVEL="$(bashio::config 'log_level')"

if [ "${DEBUG_PORT}" != "0" ] && [ -n "${DEBUG_PORT}" ]; then
    bashio::log.info "Debug stream: http://<ha-ip>:${DEBUG_PORT}/"
fi

exec /opt/venv/bin/python3 /app/main.py
