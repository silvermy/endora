"""
tests/test_deployment_parity.py

Endora ships from one codebase as two deployments — the HA add-on
(Dockerfile, CPU, Supervisor-managed) and the standalone Jetson container
(Dockerfile.jetson, GPU, Long-Lived Access Token). Two build files that
must stay in step is exactly the arrangement that quietly drifts: add a
bundled model size to one and `yolo_imgsz` silently falls back on the
other; add a Python package to one and the other crashes on import at
runtime, on a machine you are not sitting in front of.

Nothing here builds an image — these are text assertions over the two
Dockerfiles and the two requirements files, in the same spirit as
test_registry_sync.py: the thing that actually blocks a bad push, since
there is no CI and neither image can be built on a dev laptop anyway.

What is deliberately NOT asserted: that the two files are similar. They
should not be. Base image, Python install location, entrypoint (S6 vs
CMD) and the ONNX Runtime wheel all differ on purpose. Only the payload —
models and application code — has to match.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
ADDON_DOCKERFILE = REPO_ROOT / "Dockerfile"
JETSON_DOCKERFILE = REPO_ROOT / "Dockerfile.jetson"
COMPOSE_FILES = [REPO_ROOT / "docker-compose.yml",
                 REPO_ROOT / "docker-compose.jetson.yml"]
ADDON_REQS = REPO_ROOT / "requirements.txt"
JETSON_REQS = REPO_ROOT / "requirements.jetson.txt"

# COPY sources that are meant to differ between the two images: the
# requirements file each one installs, and the S6 service script, which only
# exists because the add-on base image runs S6 as PID 1.
_DEPLOYMENT_SPECIFIC_COPIES = {
    "requirements.txt", "requirements.jetson.txt", "run.sh",
}


def _copy_sources(dockerfile: Path) -> set[str]:
    """Source paths of every COPY instruction, normalised without trailing /."""
    return {
        m.group(1).rstrip("/")
        for m in re.finditer(r"^COPY\s+(\S+)\s+\S+", dockerfile.read_text(), re.M)
    }


def _app_payload(dockerfile: Path) -> set[str]:
    return _copy_sources(dockerfile) - _DEPLOYMENT_SPECIFIC_COPIES


def _models_tag(dockerfile: Path) -> str:
    m = re.search(r"^ARG\s+MODELS_TAG=(\S+)", dockerfile.read_text(), re.M)
    assert m, f"no ARG MODELS_TAG in {dockerfile.name}"
    return m.group(1)


def _model_files(dockerfile: Path) -> set[str]:
    return set(re.findall(r"\$\{MODELS_BASE\}/(\S+?\.onnx)", dockerfile.read_text()))


def _requirements(path: Path) -> dict[str, str]:
    """{package: full requirement line} for uncommented entries."""
    reqs = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip().lower()
        reqs[name] = line
    return reqs


# ── application payload ───────────────────────────────────────────────────

def test_both_images_ship_the_same_application_code():
    """A new package directory added to one Dockerfile and not the other is
    an ImportError that only appears on the deployment you did not test."""
    assert _app_payload(ADDON_DOCKERFILE) == _app_payload(JETSON_DOCKERFILE)


def test_application_payload_covers_every_source_package():
    """Guards the other direction: a newly created top-level package that
    was never added to either Dockerfile."""
    payload = _app_payload(ADDON_DOCKERFILE)
    packages = {
        d.name for d in REPO_ROOT.iterdir()
        if d.is_dir() and (d / "__init__.py").exists() and d.name != "tests"
    }
    missing = packages - payload
    assert not missing, f"source packages shipped in neither image: {missing}"


# ── models ────────────────────────────────────────────────────────────────

def test_both_images_pin_the_same_models_release():
    assert _models_tag(ADDON_DOCKERFILE) == _models_tag(JETSON_DOCKERFILE)


def test_both_images_bundle_the_same_model_files():
    """yolo_imgsz only takes effect at a size with a matching bundled .onnx
    (see resolve_model_path). A size bundled in one image and not the other
    means the same setting silently means two different resolutions."""
    assert _model_files(ADDON_DOCKERFILE) == _model_files(JETSON_DOCKERFILE)


def test_bundled_models_cover_every_offered_imgsz():
    """Every value in the yolo_imgsz dropdown needs a real model behind it."""
    from config.registry import REGISTRY

    field = next(f for f in REGISTRY if f.key == "yolo_imgsz")
    bundled = _model_files(ADDON_DOCKERFILE)
    for size in field.enum:
        # 640 is the native export and ships unsuffixed.
        expected = "yolo11s-pose.onnx" if size == "640" else f"yolo11s-pose-{size}.onnx"
        assert expected in bundled, f"yolo_imgsz={size} offered but {expected} not bundled"


# ── dependencies ──────────────────────────────────────────────────────────

def test_shared_dependencies_agree_on_version():
    """Packages present in both files must be pinned identically — a version
    skew between deployments makes a bug reproduce on one and not the other."""
    addon, jetson = _requirements(ADDON_REQS), _requirements(JETSON_REQS)
    mismatched = {
        name: (addon[name], jetson[name])
        for name in addon.keys() & jetson.keys()
        if addon[name] != jetson[name]
    }
    assert not mismatched, f"same package pinned differently per deployment: {mismatched}"


def test_jetson_requirements_exclude_the_cpu_onnxruntime():
    """`onnxruntime` from PyPI would satisfy the import while silently
    replacing the JetPack-built GPU wheel installed separately in
    Dockerfile.jetson — turning the whole board into an expensive Pi."""
    assert "onnxruntime" not in _requirements(JETSON_REQS)


def test_jetson_requirements_exclude_torch_and_ultralytics():
    """PyPI has no working torch for Jetson, and the only ultralytics import
    is the ONNX export path, which returns early on aarch64 anyway."""
    jetson = _requirements(JETSON_REQS)
    assert "ultralytics" not in jetson
    assert "torch" not in jetson


def test_jetson_installs_a_gpu_onnxruntime_from_a_jetpack_index():
    text = JETSON_DOCKERFILE.read_text()
    assert "onnxruntime-gpu" in text
    assert "--index-url" in text, "the JetPack wheel index must be explicit"


# ── standalone debug access ───────────────────────────────────────────────

def test_standalone_compose_enables_the_debug_ui():
    """debug_port ships as 0 (off), and add-on users turn it on in the
    Configuration tab. A standalone deployment has no such tab and no
    options.json, so unless compose passes a non-zero DEBUG_PORT the debug
    page — the only window into what the thing is doing — never starts."""
    for compose in COMPOSE_FILES:
        text = compose.read_text()
        m = re.search(r"DEBUG_PORT=\$\{DEBUG_PORT:-(\d+)\}", text)
        assert m, f"{compose.name} does not pass DEBUG_PORT with a default"
        assert int(m.group(1)) > 0, f"{compose.name} defaults DEBUG_PORT to off"


def test_standalone_compose_uses_host_networking():
    """The debug page and the RTSP cameras are both reached on the LAN, and
    the logged URL assumes the container's IP is the host's."""
    for compose in COMPOSE_FILES:
        assert "network_mode: host" in compose.read_text(), compose.name


def test_jetson_compose_requests_the_nvidia_runtime():
    """Without it the container starts happily and runs on the CPU."""
    assert "runtime: nvidia" in (REPO_ROOT / "docker-compose.jetson.yml").read_text()
