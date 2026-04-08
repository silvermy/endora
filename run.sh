#!/usr/bin/with-contenv bashio
# ── Gesture Cam add-on startup script ────────────────────────────────────────
#
# bashio is available in all HA base images.
# `bashio::config` reads values from /data/options.json (the add-on Options).
#
# The Supervisor injects SUPERVISOR_TOKEN automatically — we don't need to
# read it here; the Python code picks it up from the environment.

set -e

bashio::log.info "Gesture Cam starting…"

# Validate required options
if ! bashio::config.has_value "rtsp_url_a"; then
    bashio::log.fatal "rtsp_url_a is required — set it in the add-on configuration"
    exit 1
fi
if ! bashio::config.has_value "rtsp_url_b"; then
    bashio::log.fatal "rtsp_url_b is required — set it in the add-on configuration"
    exit 1
fi

bashio::log.info "Camera A: $(bashio::config 'rtsp_url_a' | sed 's|//[^:]*:[^@]*@|//****:****@|')"
bashio::log.info "Camera B: $(bashio::config 'rtsp_url_b' | sed 's|//[^:]*:[^@]*@|//****:****@|')"
bashio::log.info "HA event name: $(bashio::config 'ha_event_name')"

# Run the Python application
# /data/options.json is automatically available; Python reads it directly.
exec python3 /app/main.py
