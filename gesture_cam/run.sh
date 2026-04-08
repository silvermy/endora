#!/usr/bin/with-contenv bashio
# ── Gesture Cam — S6 Stage 2 run script ──────────────────────────────────────
#
# /usr/bin/with-contenv: provided by the HA base image.
# It imports the container environment (env vars passed via Docker/Supervisor)
# into this shell, making RTSP URLs and tokens available to Python.
#
# S6_KEEP_ENV=1 in the Dockerfile ensures env vars survive the S6 init stages.

set -e

bashio::log.info "Gesture Cam v1.0.4 starting..."

# ── Validate required options ─────────────────────────────────────────────────
if ! bashio::config.has_value "rtsp_url_a"; then
    bashio::log.fatal "rtsp_url_a is not set — configure it in the add-on options"
    exit 1
fi

if ! bashio::config.has_value "rtsp_url_b"; then
    bashio::log.fatal "rtsp_url_b is not set — configure it in the add-on options"
    exit 1
fi

# Log camera URLs with credentials masked
bashio::log.info "Camera A: $(bashio::config 'rtsp_url_a' | sed 's|//[^:]*:[^@]*@|//****:****@|g')"
bashio::log.info "Camera B: $(bashio::config 'rtsp_url_b' | sed 's|//[^:]*:[^@]*@|//****:****@|g')"
bashio::log.info "HA event: $(bashio::config 'ha_event_name')"
bashio::log.info "Log level: $(bashio::config 'log_level')"

# ── Launch Python app ─────────────────────────────────────────────────────────
# /data/options.json is written by the Supervisor before this script runs.
# Python reads it directly — no need to pass options as env vars.
# exec replaces this shell with Python so S6 tracks the correct PID.
exec /opt/venv/bin/python3 /app/main.py
