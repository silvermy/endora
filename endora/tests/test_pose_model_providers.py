"""
tests/test_pose_model_providers.py

Covers select_providers — the mapping from the yolo_execution_provider
setting to an ONNX Runtime providers list.

This logic is only ever exercised for real on a Jetson, where the failure
mode it guards against is silent: if the GPU provider is missing (the
CPU-only onnxruntime wheel got installed, or the nvidia container runtime
is not in use) ONNX Runtime does not raise — it drops the provider and runs
on the CPU, and the only symptom is that the expensive board performs like
the Pi it replaced. So the selection is a pure function of an *injected*
provider list, testable on a dev laptop with no GPU and no onnxruntime-gpu.

The CPU tail on every non-"cpu" preference is deliberate and load-bearing:
ORT assigns nodes to the first provider that can run each one, so dropping
CPU from the list turns an op TensorRT cannot handle into a hard failure.
"""
import logging

from cameras.pose_model import (
    CPU_EP,
    CUDA_EP,
    TRT_EP,
    select_providers,
)

ALL_EPS = [TRT_EP, CUDA_EP, CPU_EP]
CPU_ONLY = [CPU_EP]


def _names(providers):
    """Provider names only — entries are either a string or (name, options)."""
    return [p[0] if isinstance(p, tuple) else p for p in providers]


# ── auto ──────────────────────────────────────────────────────────────────

def test_auto_on_a_pi_is_cpu_only():
    assert _names(select_providers("auto", available=CPU_ONLY)) == [CPU_EP]


def test_auto_prefers_tensorrt_then_cuda_then_cpu():
    assert _names(select_providers("auto", available=ALL_EPS)) == ALL_EPS


def test_auto_falls_through_to_cuda_when_tensorrt_is_absent():
    assert _names(select_providers("auto", available=[CUDA_EP, CPU_EP])) == [CUDA_EP, CPU_EP]


def test_auto_does_not_warn_about_a_missing_gpu(caplog):
    """A Pi has no GPU provider and that is not a problem — warning on every
    boot would be pure noise. Only an *explicit* request warns."""
    with caplog.at_level(logging.WARNING, logger="cameras.pose_model"):
        select_providers("auto", available=CPU_ONLY)
    assert caplog.records == []


# ── explicit choices ──────────────────────────────────────────────────────

def test_cpu_is_honoured_even_when_a_gpu_is_available():
    """This is the A/B-comparison escape hatch; it has to actually pin."""
    assert _names(select_providers("cpu", available=ALL_EPS)) == [CPU_EP]


def test_cuda_excludes_tensorrt_even_when_available():
    assert _names(select_providers("cuda", available=ALL_EPS)) == [CUDA_EP, CPU_EP]


def test_explicitly_requested_provider_warns_when_missing(caplog):
    with caplog.at_level(logging.WARNING, logger="cameras.pose_model"):
        providers = select_providers("tensorrt", available=CPU_ONLY)
    assert _names(providers) == [CPU_EP]
    assert any(TRT_EP in r.getMessage() for r in caplog.records)


def test_preference_is_case_and_whitespace_insensitive():
    """Values arrive from a YAML file or an env var, both hand-edited."""
    assert _names(select_providers("  TensorRT ", available=ALL_EPS)) == ALL_EPS


def test_unknown_preference_warns_and_behaves_as_auto(caplog):
    with caplog.at_level(logging.WARNING, logger="cameras.pose_model"):
        providers = select_providers("gpu", available=ALL_EPS)
    assert _names(providers) == ALL_EPS
    assert any("gpu" in r.getMessage() for r in caplog.records)


def test_empty_available_list_still_returns_cpu():
    """ORT raises on an empty providers list — a config typo should degrade
    to slow, not to a container that will not start."""
    assert _names(select_providers("cuda", available=[])) == [CPU_EP]


# ── TensorRT options ──────────────────────────────────────────────────────

def test_tensorrt_entry_carries_fp16_and_a_writable_engine_cache(tmp_path):
    cache = tmp_path / "trt_cache"
    providers = select_providers("auto", available=ALL_EPS, cache_dir=str(cache))
    name, opts = providers[0]
    assert name == TRT_EP
    assert opts["trt_fp16_enable"] is True
    assert opts["trt_engine_cache_enable"] is True
    assert opts["trt_engine_cache_path"] == str(cache)
    assert cache.is_dir(), "cache dir must exist before TensorRT writes to it"


def test_unwritable_cache_dir_disables_caching_but_still_runs_on_gpu(tmp_path, caplog):
    """Better a slow start than no start: an unwritable /data must not stop
    inference, and must not let TensorRT default to writing into /app."""
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    with caplog.at_level(logging.WARNING, logger="cameras.pose_model"):
        providers = select_providers(
            "auto", available=ALL_EPS, cache_dir=str(blocker / "trt_cache")
        )
    name, opts = providers[0]
    assert name == TRT_EP
    assert opts["trt_fp16_enable"] is True
    assert "trt_engine_cache_enable" not in opts
    assert "trt_engine_cache_path" not in opts
    assert any("cache" in r.getMessage() for r in caplog.records)


def test_cpu_preference_has_no_options_dict():
    assert select_providers("cpu", available=ALL_EPS) == [CPU_EP]
