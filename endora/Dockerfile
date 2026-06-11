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

# ── Models — kept before application code so this layer is only invalidated ───
# when the model file itself changes, not on every code commit.
#
# 320×320 fast model: PyTorch/ultralytics causes SIGILL on aarch64 (Pi 4/5),
# so we cannot export at build time. Generate it once on an x86/macOS machine:
#
#   python3 -c "from ultralytics import YOLO; \
#               YOLO('yolo11n-pose.pt').export(format='onnx', imgsz=320, simplify=True)"
#
# Then commit the resulting yolo11n-pose-320.onnx alongside yolo11n-pose.onnx.
# Uncomment the COPY below once the file exists in the repo.
#
WORKDIR /app

# Remove any stale .pt that a previous build may have downloaded into /app/
# (it would trigger a runtime export attempt → SIGILL on aarch64).
RUN rm -f yolo11n-pose.pt

COPY yolo11n-pose.onnx      .
COPY yolo11n-pose-320.onnx  .
COPY yolo11s-pose.onnx      .
COPY yolo11s-pose-320.onnx  .

# ── Application code (changes frequently — only invalidates these COPY layers)
COPY cameras/           cameras/
COPY config/            config/
COPY core/              core/
COPY output/            output/
COPY main.py            .
COPY version.py         .
COPY config.json        .

# ── Validate ONNX models ─────────────────────────────────────────────────────
RUN python3 -c "import onnxruntime as ort; s = ort.InferenceSession('yolo11n-pose.onnx',     providers=['CPUExecutionProvider']); print('ONNX nano-640 OK:', s.get_inputs()[0].shape)"
RUN python3 -c "import onnxruntime as ort; s = ort.InferenceSession('yolo11n-pose-320.onnx', providers=['CPUExecutionProvider']); print('ONNX nano-320 OK:', s.get_inputs()[0].shape)"
RUN python3 -c "import onnxruntime as ort; s = ort.InferenceSession('yolo11s-pose.onnx',     providers=['CPUExecutionProvider']); print('ONNX small-640 OK:', s.get_inputs()[0].shape)"
RUN python3 -c "import onnxruntime as ort; s = ort.InferenceSession('yolo11s-pose-320.onnx', providers=['CPUExecutionProvider']); print('ONNX small-320 OK:', s.get_inputs()[0].shape)"

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
