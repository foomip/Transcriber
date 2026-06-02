"""
backend — hardware detection, backend configuration, and model path helpers.

Responsibilities:
    - Detect the best analysis backend (CPU, CUDA GPU, ROCm GPU)
    - Configure model-loading parameters based on available hardware resources
    - Manage llama.cpp GGUF model paths and downloads
    - Expose detect_analysis_backend() as the single entry point for callers
"""

import importlib.util
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from .utils import AnalysisModelError
from lib.hardware import parse_rocm_product_name
from lib.progress import ProgressTimer

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------

ANALYSIS_BACKEND_ENV = "TRANSCRIBER_ANALYSIS_BACKEND"

LLAMA_CPP_MODEL_PATH_ENV = "TRANSCRIBER_LLAMA_CPP_MODEL_PATH"
LLAMA_CPP_MODEL_REPO_ENV = "TRANSCRIBER_LLAMA_CPP_MODEL_REPO"
LLAMA_CPP_CACHE_DIR_ENV = "TRANSCRIBER_GGUF_CACHE_DIR"
LLAMA_CPP_CONTEXT_SIZE_ENV = "TRANSCRIBER_LLAMA_CPP_CONTEXT_SIZE"
LLAMA_CPP_BATCH_SIZE_ENV = "TRANSCRIBER_LLAMA_CPP_BATCH_SIZE"
LLAMA_CPP_GPU_LAYERS_ENV = "TRANSCRIBER_LLAMA_CPP_GPU_LAYERS"
LLAMA_CPP_GPU_HEADROOM_ENV = "TRANSCRIBER_LLAMA_CPP_GPU_HEADROOM_GIB"
LLAMA_CPP_LAYER_COUNT_ENV = "TRANSCRIBER_LLAMA_CPP_LAYER_COUNT"

# ---------------------------------------------------------------------------
# Model / backend defaults
# ---------------------------------------------------------------------------

DEFAULT_LLAMA_CPP_MODEL_REPO_ID = "ggml-org/gemma-4-E4B-it-GGUF"
DEFAULT_LLAMA_CPP_MODEL_FILENAME = "gemma-4-E4B-it-Q4_K_M.gguf"

LLAMA_CPP_BACKEND_NAME = "llama_cpp"

# ---------------------------------------------------------------------------
# System / memory constants
# ---------------------------------------------------------------------------

_CPUINFO_PATH = "/proc/cpuinfo"
_MEMINFO_PATH = "/proc/meminfo"
_GIB = 1024 ** 3

# ---------------------------------------------------------------------------
# llama.cpp defaults
# ---------------------------------------------------------------------------

DEFAULT_LLAMA_CPP_CONTEXT_SIZE = 4096
DEFAULT_LLAMA_CPP_BATCH_SIZE = 256
DEFAULT_LLAMA_CPP_LAYER_COUNT = 42
DEFAULT_LLAMA_CPP_GPU_HEADROOM_GIB = 3.0
DEFAULT_LLAMA_CPP_KV_CACHE_GIB = 0.75

# The context window must hold the whole transcript prompt plus the generated
# report, so it is sized from the transcript character budget rather than a
# fixed value. gemma-4-E4B is trained for 131072 tokens, which is the ceiling.
DEFAULT_LLAMA_CPP_MAX_CONTEXT_SIZE = 131072
LLAMA_CPP_CHARS_PER_TOKEN = 3.0
LLAMA_CPP_CONTEXT_MARGIN_TOKENS = 512
LLAMA_CPP_CONTEXT_ALIGNMENT = 256


# ---------------------------------------------------------------------------
# AnalysisBackend dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalysisBackend:
    name: str
    device_name: str
    model_id: str
    model_kwargs: dict[str, Any]
    notes: tuple[str, ...] = ()
    max_new_tokens: int = 2048


# ---------------------------------------------------------------------------
# Generic environment-variable helpers
# ---------------------------------------------------------------------------

def _positive_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if not value:
        return None

    try:
        parsed = float(value.replace("_", ""))
    except ValueError:
        print(f"  ⚠️  Ignoring invalid {name}={value!r}")
        return None

    if parsed <= 0:
        print(f"  ⚠️  Ignoring non-positive {name}={value!r}")
        return None
    return parsed


def _positive_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if not value:
        return None

    try:
        parsed = int(value.replace("_", ""))
    except ValueError:
        print(f"  ⚠️  Ignoring invalid {name}={value!r}")
        return None

    if parsed <= 0:
        print(f"  ⚠️  Ignoring non-positive {name}={value!r}")
        return None
    return parsed


def _analysis_backend_preference() -> str:
    configured_backend = os.environ.get(ANALYSIS_BACKEND_ENV, "auto").strip().lower()
    if configured_backend in {"", "auto"}:
        return "auto"
    if configured_backend == LLAMA_CPP_BACKEND_NAME:
        return configured_backend

    print(f"  ⚠️  Ignoring invalid {ANALYSIS_BACKEND_ENV}={configured_backend!r}")
    return "auto"


# ---------------------------------------------------------------------------
# System info helpers
# ---------------------------------------------------------------------------

def _available_ram_bytes() -> int | None:
    """Return available system RAM in bytes from /proc/meminfo, or None on error."""
    try:
        with open(_MEMINFO_PATH, encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _detect_gpu() -> tuple[str, str]:
    """Return (kind, device_name). kind is 'cuda', 'rocm', or 'cpu'."""
    # NVIDIA check
    try:
        import pynvml # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        return "cuda", name
    except (ImportError, Exception) as exc:
        # Log failure for easier debugging in Docker
        if os.environ.get("DEBUG") == "1":
            print(f"  DEBUG: NVIDIA detection failed: {exc}")
        
        # Fallback: check for device node
        if os.path.exists("/dev/nvidia0"):
            return "cuda", "NVIDIA GPU (detected via /dev/nvidia0)"

    # AMD check
    if os.path.exists("/dev/kfd"):
        try:
            res = subprocess.check_output(
                ["rocm-smi", "--showproductname"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if name := parse_rocm_product_name(res):
                return "rocm", name
            if os.environ.get("DEBUG") == "1":
                print("  DEBUG: Could not parse a ROCm GPU name from rocm-smi output")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            if os.environ.get("DEBUG") == "1":
                print(f"  DEBUG: ROCm SMI failed: {exc}")
        return "rocm", "AMD GPU"

    return "cpu", "CPU"


def _nvidia_free_vram_bytes() -> int | None:
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.free
    except (ImportError, Exception):
        return None


def _amd_free_vram_bytes() -> int | None:
    # Primary method: use rocm-smi if available
    try:
        res = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram"], stderr=subprocess.DEVNULL, text=True)
        # Expected format:
        # GPU  VRAM Total  VRAM Used  VRAM %
        # 0    12288MB    1024MB     8%
        lines = res.strip().splitlines()
        if len(lines) > 1:
            parts = lines[1].split()
            total_mb = int(parts[1].replace("MB", ""))
            used_mb = int(parts[2].replace("MB", ""))
            return (total_mb - used_mb) * 1024 * 1024
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError):
        pass
    # Fallback: read sysfs entries if rocm-smi is not present
    try:
        total_path = "/sys/class/drm/card0/device/mem_info_vram_total"
        used_path = "/sys/class/drm/card0/device/mem_info_vram_used"
        with open(total_path, encoding="utf-8") as f:
            total_kb = int(f.read().strip())
        with open(used_path, encoding="utf-8") as f:
            used_kb = int(f.read().strip())
        # Values are in kilobytes
        return (total_kb - used_kb) * 1024
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# llama.cpp path / download helpers
# ---------------------------------------------------------------------------

def _llama_cpp_is_available() -> bool:
    return importlib.util.find_spec("llama_cpp") is not None


def _llama_cpp_model_path() -> str:
    configured_model_path = os.environ.get(LLAMA_CPP_MODEL_PATH_ENV)
    if configured_model_path:
        return configured_model_path

    cache_dir = os.environ.get(
        LLAMA_CPP_CACHE_DIR_ENV,
        os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "transcriber", "gguf",
        ),
    )
    return os.path.join(cache_dir, DEFAULT_LLAMA_CPP_MODEL_FILENAME)


def _llama_cpp_model_exists(model_path: str | None = None) -> bool:
    return os.path.exists(model_path or _llama_cpp_model_path())


def _llama_cpp_model_repo_id() -> str:
    return os.environ.get(
        LLAMA_CPP_MODEL_REPO_ENV,
        DEFAULT_LLAMA_CPP_MODEL_REPO_ID,
    )


def _llama_cpp_display_name(model_path: str) -> str:
    return os.path.basename(model_path) or model_path


def _download_llama_cpp_model(model_path: str) -> None:
    repo_id = _llama_cpp_model_repo_id()
    filename = os.path.basename(model_path)
    model_dir = os.path.dirname(model_path) or "."

    if not filename:
        raise AnalysisModelError(f"Could not infer a GGUF filename from {model_path!r}.")

    os.makedirs(model_dir, exist_ok=True)

    with ProgressTimer(
        f"  Downloading {filename} from {repo_id}...",
        done_message=f"Downloaded {filename}",
    ):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise AnalysisModelError(
                "Could not download the GGUF analysis model because huggingface_hub "
                "is not installed. Install huggingface-hub or set "
                f"{LLAMA_CPP_MODEL_PATH_ENV} to a local GGUF file."
            ) from exc

        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=model_dir,
            )
        except Exception as exc:
            raise AnalysisModelError(
                f"Could not download GGUF analysis model {filename} from {repo_id}: {exc}. "
                f"Set {LLAMA_CPP_MODEL_PATH_ENV} to a local GGUF file or set "
                f"{LLAMA_CPP_MODEL_REPO_ENV} to a Hugging Face repo containing {filename}."
            ) from exc

        if os.path.abspath(downloaded_path) != os.path.abspath(model_path):
            shutil.copyfile(downloaded_path, model_path)


def _ensure_llama_cpp_model(model_path: str) -> str:
    if _llama_cpp_model_exists(model_path):
        return model_path

    _download_llama_cpp_model(model_path)
    if _llama_cpp_model_exists(model_path):
        return model_path

    raise AnalysisModelError(
        f"Could not find the GGUF analysis model at {model_path} after download. "
        f"Set {LLAMA_CPP_MODEL_PATH_ENV} to a local GGUF file or set "
        f"{LLAMA_CPP_MODEL_REPO_ENV} to a Hugging Face repo containing "
        f"{_llama_cpp_display_name(model_path)}."
    )


# ---------------------------------------------------------------------------
# llama.cpp parameter helpers
# ---------------------------------------------------------------------------

def _required_llama_cpp_context_size(transcript_chars: int | None = None) -> int:
    """Size the context window to hold the transcript prompt plus generation.

    When the transcript length is known, size the window from the actual prompt
    payload instead of the maximum transcript budget. This avoids reserving a
    massive KV cache for short recordings, which can otherwise prevent any GPU
    layer offload on cards with moderate VRAM.

    When ``transcript_chars`` is omitted, fall back to ``transcript_char_budget()``
    so existing callers and tests retain the previous conservative behaviour.
    The result is clamped to the model's trained context window.
    """
    from lib.report import transcript_char_budget

    if transcript_chars is None:
        transcript_chars = transcript_char_budget()
    else:
        transcript_chars = max(0, transcript_chars)

    prompt_tokens = math.ceil(transcript_chars / LLAMA_CPP_CHARS_PER_TOKEN)
    required = prompt_tokens + 2048 + LLAMA_CPP_CONTEXT_MARGIN_TOKENS
    aligned = math.ceil(required / LLAMA_CPP_CONTEXT_ALIGNMENT) * LLAMA_CPP_CONTEXT_ALIGNMENT
    return max(
        DEFAULT_LLAMA_CPP_CONTEXT_SIZE,
        min(DEFAULT_LLAMA_CPP_MAX_CONTEXT_SIZE, aligned),
    )


def _llama_cpp_context_size(transcript_chars: int | None = None) -> int:
    return _positive_int_env(LLAMA_CPP_CONTEXT_SIZE_ENV) or _required_llama_cpp_context_size(
        transcript_chars
    )


def _llama_cpp_batch_size() -> int:
    return _positive_int_env(LLAMA_CPP_BATCH_SIZE_ENV) or DEFAULT_LLAMA_CPP_BATCH_SIZE


def _llama_cpp_layer_count() -> int:
    return _positive_int_env(LLAMA_CPP_LAYER_COUNT_ENV) or DEFAULT_LLAMA_CPP_LAYER_COUNT


def _llama_cpp_gpu_layers(
    model_path: str,
    transcript_chars: int | None = None,
) -> tuple[int, tuple[str, ...]]:
    configured_gpu_layers = _positive_int_env(LLAMA_CPP_GPU_LAYERS_ENV)
    if configured_gpu_layers is not None:
        return configured_gpu_layers, (
            f"Using {LLAMA_CPP_GPU_LAYERS_ENV}={configured_gpu_layers} GPU layers",
        )

    kind, _ = _detect_gpu()
    if kind == "cuda":
        free_bytes = _nvidia_free_vram_bytes()
    elif kind == "rocm":
        free_bytes = _amd_free_vram_bytes()
    else:
        free_bytes = None

    if free_bytes is None:
        return 0, (
            "Could not inspect GPU VRAM; llama.cpp will keep all layers in system RAM",
        )

    try:
        model_size_bytes = os.path.getsize(model_path)
    except OSError:
        return 0, (f"GGUF model file not found at {model_path}",)

    layer_count = _llama_cpp_layer_count()
    context_size = _llama_cpp_context_size(transcript_chars)
    headroom_gib = (
        _positive_float_env(LLAMA_CPP_GPU_HEADROOM_ENV)
        or DEFAULT_LLAMA_CPP_GPU_HEADROOM_GIB
    )
    kv_cache_gib = max(
        DEFAULT_LLAMA_CPP_KV_CACHE_GIB,
        ((context_size + 2048) / 4096)
        * DEFAULT_LLAMA_CPP_KV_CACHE_GIB,
    )
    reserve_bytes = int((headroom_gib + kv_cache_gib) * _GIB)
    safe_gpu_bytes = max(0, free_bytes - reserve_bytes)
    bytes_per_layer = model_size_bytes / max(1, layer_count)
    gpu_layers = min(layer_count, int(safe_gpu_bytes / max(1, int(bytes_per_layer))))

    note = (
        f"Estimated {gpu_layers}/{layer_count} llama.cpp layers on GPU "
        f"from {free_bytes / _GIB:.1f} GiB free VRAM with "
        f"{headroom_gib + kv_cache_gib:.1f} GiB reserved"
    )
    return gpu_layers, (note,)


def _llama_cpp_model_kwargs(
    model_path: str,
    transcript_chars: int | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    gpu_layers, notes = _llama_cpp_gpu_layers(model_path, transcript_chars)
    model_kwargs: dict[str, Any] = {
        "model_path": model_path,
        "n_ctx": _llama_cpp_context_size(transcript_chars),
        "n_batch": _llama_cpp_batch_size(),
        "n_gpu_layers": gpu_layers,
        "verbose": False,
    }
    return model_kwargs, notes


# ---------------------------------------------------------------------------
# Per-backend configuration builders
# ---------------------------------------------------------------------------

def _llama_cpp_analysis_backend(
    kind: str,
    device_name: str,
    transcript_chars: int | None = None,
) -> AnalysisBackend:
    model_path = _llama_cpp_model_path()
    if _llama_cpp_is_available():
        model_path = _ensure_llama_cpp_model(model_path)
    model_kwargs, model_notes = _llama_cpp_model_kwargs(model_path, transcript_chars)
    return AnalysisBackend(
        name=kind,
        device_name=device_name,
        model_id=model_path,
        model_kwargs=model_kwargs,
        notes=(
            f"Using llama.cpp model {_llama_cpp_display_name(model_path)}",
            "llama.cpp dynamically splits layers between GPU VRAM and system RAM",
        ) + model_notes,
        max_new_tokens=2048,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_analysis_backend(transcript_chars: int | None = None) -> AnalysisBackend:
    """Return the best analysis backend for summarization."""
    backend_preference = _analysis_backend_preference()
    kind, device_name = _detect_gpu()

    if kind == "cpu":
        # In a llama.cpp-only world, CPU is just llama.cpp with 0 layers on GPU.
        return _llama_cpp_analysis_backend(kind, device_name, transcript_chars)

    if backend_preference == LLAMA_CPP_BACKEND_NAME or backend_preference == "auto":
        return _llama_cpp_analysis_backend(kind, device_name, transcript_chars)

    # Since we are removing Transformers, any other preference just falls back to llama.cpp
    return _llama_cpp_analysis_backend(kind, device_name, transcript_chars)


def _display_model_name(backend: AnalysisBackend) -> str:
    return _llama_cpp_display_name(backend.model_id)
