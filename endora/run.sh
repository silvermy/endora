#!/usr/bin/with-contenv bashio
# ── Endora — S6 service run script ──────────────────────────────────────
# This file lives at /etc/services.d/endora/run
# S6 calls it to start (and restart on crash) the endora process.
# with-contenv imports the container environment so RTSP URLs etc. are visible.

set -e

bashio::log.info "Endora v1.5.9 starting..."

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

exec /opt/venv/bin/python3 /app/main.py
