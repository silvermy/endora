# ── Gesture Cam — Home Assistant Add-on ──────────────────────────────────────
#
# Multi-arch: aarch64 (Pi 4/5, most HA hardware) and amd64.
# Base: ghcr.io/home-assistant/base-debian:bookworm
# Debian is required for MediaPipe's pre-built wheels (glibc dependency).
# Alpine (musl libc) is not compatible with MediaPipe wheels.

FROM ghcr.io/home-assistant/base-debian:bookworm

# ── S6-Overlay config ─────────────────────────────────────────────────────────
# S6_KEEP_ENV     : passes env vars (RTSP URLs, HA_TOKEN) through to Python
# S6_BEHAVIOUR_IF_STAGE2_FAILS=2 : stops container immediately on crash
#                   (so Docker/HA restarts it rather than silently hanging)
ENV S6_KEEP_ENV=1 \
    S6_BEHAVIOUR_IF_STAGE2_FAILS=2 \
    LANG=C.UTF-8

# ── System dependencies ───────────────────────────────────────────────────────
# ffmpeg       : RTSP decode backend for OpenCV (CAP_FFMPEG)
# libgl1-mesa-glx : OpenCV runtime — needs libGL.so.1 even headless
# libglib2.0-0 : MediaPipe runtime dependency
# libstdc++6   : MediaPipe C++ runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# ── Python virtualenv + dependencies ─────────────────────────────────────────
# Using a venv avoids conflicts with the base image's system Python packages.
# --prefer-binary: use pre-built wheels where available (much faster build,
#   avoids compiling MediaPipe/OpenCV from source on arm64).
COPY requirements.txt /tmp/requirements.txt

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip3 install --no-cache-dir --prefer-binary -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
WORKDIR /app
COPY cameras/  cameras/
COPY config/   config/
COPY core/     core/
COPY output/   output/
COPY main.py   .

# ── Run script ────────────────────────────────────────────────────────────────
# Strip Windows line endings (CRLF → LF) — common source of "not found" errors
# when the file is edited on Windows before being committed.
COPY run.sh /run.sh
RUN chmod a+x /run.sh \
    && sed -i 's/\r$//' /run.sh

# S6 entrypoint is provided by the base image (/init).
# CMD is the Stage 2 script S6 runs once the system is ready.
CMD ["/run.sh"]
