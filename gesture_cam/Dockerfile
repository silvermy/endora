# ── Gesture Cam — Home Assistant Add-on ──────────────────────────────────────
#
# Multi-arch: builds for aarch64 (Pi 4/5, most HA hardware) and amd64.
#
# Base: ghcr.io/home-assistant/{arch}-base-python:3.11-alpine3.19
# This is the official HA add-on base — slim, pre-wired for the Supervisor.
#
# We install OpenCV's headless build and MediaPipe's arm64 wheel.
# The image is ~900 MB after layer squash (MediaPipe + OpenCV are large).

ARG BUILD_FROM=ghcr.io/home-assistant/base-debian:bookworm
FROM $BUILD_FROM

LABEL \
  io.hass.name="Gesture Cam" \
  io.hass.description="Dual RTSP camera gesture detection" \
  io.hass.version="1.0.0" \
  io.hass.type="addon" \
  maintainer="your@email.com"

# ── System deps ───────────────────────────────────────────────────────────────
# ffmpeg  : RTSP decoding backend for OpenCV
# libgl1  : OpenCV runtime dep (even headless build needs libGL.so.1)
# glib    : mediapipe runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Use pip3 to install your requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
WORKDIR /app
COPY cameras/   cameras/
COPY config/    config/
COPY core/      core/
COPY output/    output/
COPY main.py    .

# Ensure S6 is used
ENV S6_KEEP_ENV=1

# ── Add-on run script ─────────────────────────────────────────────────────────
COPY run.sh /run.sh
RUN chmod +x /run.sh

# The Supervisor calls /run.sh to start the add-on
CMD ["/run.sh"]
