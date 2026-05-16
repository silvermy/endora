# ── Endora — Home Assistant Add-on ───────────────────────────────────────────
# Base: ghcr.io/home-assistant/base-debian:bookworm
# S6-overlay is built into this base image. /init is the entrypoint and
# becomes PID 1. We register our app as an S6 service under /etc/services.d/
# rather than using CMD, which would bypass S6 entirely.

FROM ghcr.io/home-assistant/base-debian:bookworm

ENV S6_KEEP_ENV=1 \
    S6_BEHAVIOUR_IF_STAGE2_FAILS=2 \
    LANG=C.UTF-8

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# ── Python virtualenv ─────────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --prefer-binary \
       -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
ENV PATH="/opt/venv/bin:$PATH"

# ── Pre-download YOLO pose model ──────────────────────────────────────────────
# Downloads yolo11n-pose.pt (~7 MB) so the container starts without internet.
# Change the model name here if you override yolo_pose_model in settings.yaml.
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"

# ── Application code ──────────────────────────────────────────────────────────
WORKDIR /app
COPY cameras/    cameras/
COPY config/     config/
COPY core/       core/
COPY output/     output/
COPY main.py     .
COPY version.py  .
COPY config.json .

# ── S6 service definition ─────────────────────────────────────────────────────
# S6 looks for services in /etc/services.d/<name>/run
# The 'run' script is exec'd by S6 as the service process.
# No CMD or ENTRYPOINT — the base image's /init (S6) handles that.
RUN mkdir -p /etc/services.d/endora
COPY run.sh /etc/services.d/endora/run
RUN chmod a+x /etc/services.d/endora/run \
    && sed -i 's/\r$//' /etc/services.d/endora/run
