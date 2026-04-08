# Use the Home Assistant Debian-based multi-arch image
FROM ghcr.io/home-assistant/base-debian:bookworm

# ── 1. S6-Overlay Configuration ───────────────────────────────────────────────
# S6_KEEP_ENV: Ensures your RTSP URLs from HA are visible to Python
# S6_BEHAVIOUR_IF_STAGE2_FAILS: Stops container immediately if run.sh crashes
ENV S6_KEEP_ENV=1 \
    S6_BEHAVIOUR_IF_STAGE2_FAILS=2 \
    LANG=C.UTF-8

# ── 2. Install System Dependencies ───────────────────────────────────────────
# We install Python, Pip, and the specific libraries MediaPipe/OpenCV need
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# ── 3. Install Python Requirements ───────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
# --break-system-packages is required on Debian Bookworm to bypass PEP 668
# --prefer-binary prevents pip from trying to build MediaPipe from source
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir \
       --break-system-packages \
       --prefer-binary \
       -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ── 4. Final Setup ────────────────────────────────────────────────────────────
WORKDIR /app
COPY . .

# Move run.sh to root and fix line endings (prevents "not found" errors)
RUN mv run.sh /run.sh \
    && chmod a+x /run.sh \
    && sed -i 's/\r$//' /run.sh

# Do NOT use ENTRYPOINT. The base image already has /init as the entrypoint.
# CMD is the "Stage 2" script that S6 will run once the system is ready.
CMD [ "/run.sh" ]

