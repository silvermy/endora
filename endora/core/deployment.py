"""
core/deployment.py

Which of the two supported deployments is this process running as?

Endora ships two ways out of one codebase:

  * **HA add-on** — a Supervisor-managed container on the Home Assistant
    host (a Pi). Config arrives as /data/options.json, auth as
    SUPERVISOR_TOKEN, inference on the CPU. Built from `Dockerfile`.

  * **Standalone container** — a plain container anywhere on the LAN,
    typically a Jetson doing GPU inference, reporting to HA with a
    Long-Lived Access Token. Built from `Dockerfile.jetson` (or the generic
    `Dockerfile` via docker-compose.yml).

Note what this module deliberately does NOT do: nothing in the pipeline
branches on the answer. Where config comes from is resolved by
Settings.load(), which auth path to use by output/backends.py, and which
execution provider to use by cameras/pose_model.py — each from the evidence
it actually cares about, not from a global mode flag. Adding such a flag is
how two deployments turn into two code paths and then into two behaviours.

This exists so the startup banner can say which deployment it is. With two
of them in the wild, "which one is this and what is it actually running" is
the first question of every debugging session, and until v1.9.145 the banner
announced "HA add-on mode" unconditionally — including on a Jetson.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

HA_OPTIONS_PATH = Path("/data/options.json")

ADDON = "HA add-on"
STANDALONE = "standalone container"


def is_addon(
    env: Optional[Mapping[str, str]] = None,
    options_path: Path = HA_OPTIONS_PATH,
) -> bool:
    """True when running under the Home Assistant Supervisor.

    SUPERVISOR_TOKEN is the reliable signal — the Supervisor injects it and
    nothing else does. options.json is checked as a fallback because a user
    debugging an add-on by hand (`docker exec`, or a container started
    outside the Supervisor against the same /data) still has add-on-shaped
    config, and reporting that as "standalone" would send them looking in
    the wrong file.
    """
    env = os.environ if env is None else env
    if env.get("SUPERVISOR_TOKEN"):
        return True
    return options_path.exists()


def mode(
    env: Optional[Mapping[str, str]] = None,
    options_path: Path = HA_OPTIONS_PATH,
) -> str:
    return ADDON if is_addon(env, options_path) else STANDALONE


def accelerators(available: Optional[list[str]] = None) -> str:
    """Human-readable summary of the GPU providers this build could use.

    Reports what is *available*, not what is in use — PoseModel logs the
    provider ONNX Runtime actually accepted once a session exists, and the
    two disagreeing is itself the diagnosis (a GPU present but not selected
    means a misconfigured provider setting; a GPU absent on a Jetson means
    the CPU-only onnxruntime wheel got installed).
    """
    if available is None:
        try:
            import onnxruntime as ort
            available = list(ort.get_available_providers())
        except Exception:
            return "unknown (onnxruntime not importable)"

    gpu = [p for p in available
           if p in ("TensorrtExecutionProvider", "CUDAExecutionProvider")]
    return ", ".join(gpu) if gpu else "none (CPU inference)"


def describe(
    env: Optional[Mapping[str, str]] = None,
    options_path: Path = HA_OPTIONS_PATH,
    available: Optional[list[str]] = None,
) -> str:
    """One-line deployment summary for the startup banner."""
    return f"{mode(env, options_path)}, accelerators: {accelerators(available)}"
