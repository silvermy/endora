# ── Endora — Home Assistant Add-on ───────────────────────────────────────────
# BUILD_FROM is passed by the HA Supervisor at build time (replaces build.json).
# S6-overlay is built into this base image. /init is the entrypoint and
# becomes PID 1. We register our app as an S6 service under /etc/services.d/
# rather than using CMD, which would bypass S6 entirely.

ARG BUILD_FROM=ghcr.io/home-assistant/base-debian:bookworm
FROM ${BUILD_FROM}

ENV S6_KEEP_ENV=1 \
    S6_BEHAVIOUR_IF_STAGE2_FAILS=2 \
    LANG=C.UTF-8 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

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
    && /opt/venv/bin/pip install --no-cache-dir --no-deps grlib==1.0.12 \
    && rm /tmp/requirements.txt
ENV PATH="/opt/venv/bin:$PATH"

# ── Application code ──────────────────────────────────────────────────────────
# yolo11n-pose.onnx must be generated on an x86/macOS machine first:
#   python3 -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt').export(format='onnx', opset=12)"
# then committed to the repo alongside this Dockerfile.
WORKDIR /app
COPY cameras/           cameras/
COPY config/            config/
COPY core/              core/
COPY output/            output/
COPY main.py            .
COPY version.py         .
COPY config.json        .
COPY yolo11n-pose.onnx  .

# ── Validate ONNX model (no PyTorch inference — safe on all ARM64 CPUs) ───────
RUN python3 -c "import onnxruntime as ort; s = ort.InferenceSession('yolo11n-pose.onnx', providers=['CPUExecutionProvider']); print('ONNX model OK:', s.get_inputs()[0].shape)"

# ── S6 service definition ─────────────────────────────────────────────────────
# S6 looks for services in /etc/services.d/<name>/run
# The 'run' script is exec'd by S6 as the service process.
# No CMD or ENTRYPOINT — the base image's /init (S6) handles that.
RUN mkdir -p /etc/services.d/endora
COPY run.sh /etc/services.d/endora/run
RUN chmod a+x /etc/services.d/endora/run \
    && sed -i 's/\r$//' /etc/services.d/endora/run

# ── HA Supervisor metadata (replaces build.json) ──────────────────────────────
ARG BUILD_VERSION
LABEL \
    io.hass.version="${BUILD_VERSION}" \
    io.hass.type="addon" \
    io.hass.arch="aarch64|amd64"
