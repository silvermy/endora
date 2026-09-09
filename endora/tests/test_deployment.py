"""
tests/test_deployment.py

core/deployment.py answers "which of the two deployments is this?" for the
startup banner. Before v1.9.145 the banner said "HA add-on mode"
unconditionally, which was simply false on a Jetson — and a log line that
misreports the deployment sends every subsequent debugging step to the
wrong config file.

Both inputs are injectable (environment mapping, options.json path) so the
detection can be tested without a Supervisor and without touching /data.
"""
from pathlib import Path

from core import deployment
from core.deployment import ADDON, STANDALONE

CUDA = "CUDAExecutionProvider"
TRT = "TensorrtExecutionProvider"
CPU = "CPUExecutionProvider"


def _missing(tmp_path: Path) -> Path:
    return tmp_path / "options.json"


def _present(tmp_path: Path) -> Path:
    p = tmp_path / "options.json"
    p.write_text("{}")
    return p


# ── detection ─────────────────────────────────────────────────────────────

def test_supervisor_token_means_addon(tmp_path):
    assert deployment.is_addon({"SUPERVISOR_TOKEN": "x"}, _missing(tmp_path))


def test_options_json_alone_still_means_addon(tmp_path):
    """A hand-started container against an add-on's /data has add-on-shaped
    config; calling that standalone would point the user at settings.yaml,
    which is not where its values are coming from."""
    assert deployment.is_addon({}, _present(tmp_path))


def test_neither_signal_means_standalone(tmp_path):
    assert not deployment.is_addon({}, _missing(tmp_path))


def test_ha_token_alone_is_not_an_addon(tmp_path):
    """The Jetson deployment authenticates with a Long-Lived Access Token —
    that is the standalone path, not the Supervisor one."""
    assert not deployment.is_addon({"HA_TOKEN": "x"}, _missing(tmp_path))


def test_empty_supervisor_token_is_not_an_addon(tmp_path):
    """docker-compose passes through unset variables as empty strings."""
    assert not deployment.is_addon({"SUPERVISOR_TOKEN": ""}, _missing(tmp_path))


def test_mode_names_each_deployment(tmp_path):
    assert deployment.mode({"SUPERVISOR_TOKEN": "x"}, _missing(tmp_path)) == ADDON
    assert deployment.mode({}, _missing(tmp_path)) == STANDALONE


# ── accelerator reporting ─────────────────────────────────────────────────

def test_accelerators_lists_gpu_providers():
    assert deployment.accelerators([TRT, CUDA, CPU]) == f"{TRT}, {CUDA}"


def test_accelerators_on_a_pi_says_cpu():
    assert deployment.accelerators([CPU]) == "none (CPU inference)"


def test_accelerators_ignores_non_gpu_providers():
    """CoreML and Azure show up on a dev laptop and are not what this line
    is reporting on."""
    assert deployment.accelerators(
        ["CoreMLExecutionProvider", "AzureExecutionProvider", CPU]
    ) == "none (CPU inference)"


# ── banner ────────────────────────────────────────────────────────────────

def test_describe_reports_deployment_and_accelerators(tmp_path):
    line = deployment.describe({}, _missing(tmp_path), [TRT, CPU])
    assert STANDALONE in line
    assert TRT in line


def test_describe_on_the_pi_addon(tmp_path):
    line = deployment.describe({"SUPERVISOR_TOKEN": "x"}, _missing(tmp_path), [CPU])
    assert line == f"{ADDON}, accelerators: none (CPU inference)"
