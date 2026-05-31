"""
backend — hardware detection, backend configuration, and model path helpers.

Responsibilities:
    - Detect the best analysis backend (CPU, CUDA GPU, ROCm GPU)
    - Configure model-loading parameters based on available hardware resources
    - Manage llama.cpp GGUF model paths and downloads
    - Expose detect_analysis_backend() as the single entry point for callers
"""

import importlib
import importlib.util
import math
import os
import shutil
from dataclasses import dataclass
from typing import Any

import torch

from .utils import AnalysisModelError
from lib.progress import ProgressTimer

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------

ANALYSIS_BACKEND_ENV = "TRANSCRIBER_ANALYSIS_BACKEND"
ANALYSIS_MODEL_ENV = "TRANSCRIBER_ANALYSIS_MODEL"
GPU_HEADROOM_ENV = "TRANSCRIBER_ANALYSIS_GPU_HEADROOM_GIB"
GPU_MAX_MEMORY_ENV = "TRANSCRIBER_ANALYSIS_GPU_MAX_MEMORY_GIB"

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

DEFAULT_ANALYSIS_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_ROCM_ANALYSIS_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ROCM_LLAMA_CPP_MODEL_REPO_ID = "ggml-org/gemma-4-E4B-it-GGUF"
DEFAULT_ROCM_LLAMA_CPP_MODEL_FILENAME = "gemma-4-E4B-it-Q4_K_M.gguf"

ANALYSIS_MAX_NEW_TOKENS = 4096
ROCM_ANALYSIS_MAX_NEW_TOKENS = 1024

TRANSFORMERS_BACKEND_NAME = "transformers"
LLAMA_CPP_BACKEND_NAME = "llama_cpp"
ROCM_ATTENTION_IMPLEMENTATION = "eager"

# ---------------------------------------------------------------------------
# System / memory constants
# ---------------------------------------------------------------------------

_CPUINFO_PATH = "/proc/cpuinfo"
_MEMINFO_PATH = "/proc/meminfo"
_GIB = 1024 ** 3
FLOAT32_MIN_RAM_GIB = 32   # float32 weights ~30 GB + headroom for KV cache and OS
DEFAULT_GPU_HEADROOM_GIB = 2
CPU_OFFLOAD_HEADROOM_GIB = 8

# ---------------------------------------------------------------------------
# llama.cpp defaults
# ---------------------------------------------------------------------------

DEFAULT_ROCM_LLAMA_CPP_CONTEXT_SIZE = 4096
DEFAULT_ROCM_LLAMA_CPP_BATCH_SIZE = 256
DEFAULT_ROCM_LLAMA_CPP_LAYER_COUNT = 42
DEFAULT_ROCM_LLAMA_CPP_GPU_HEADROOM_GIB = 3.0
DEFAULT_ROCM_LLAMA_CPP_KV_CACHE_GIB = 0.75

# The context window must hold the whole transcript prompt plus the generated
# report, so it is sized from the transcript character budget rather than a
# fixed value. gemma-4-E4B is trained for 131072 tokens, which is the ceiling.
DEFAULT_ROCM_LLAMA_CPP_MAX_CONTEXT_SIZE = 131072
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
    engine: str = TRANSFORMERS_BACKEND_NAME
    use_plain_prompt: bool = False
    max_new_tokens: int = ANALYSIS_MAX_NEW_TOKENS


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
    if configured_backend in {TRANSFORMERS_BACKEND_NAME, LLAMA_CPP_BACKEND_NAME}:
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


def _cpu_supports_avx512_bf16() -> bool:
    """Return True if the CPU advertises avx512_bf16 in /proc/cpuinfo.

    AVX-512 BF16 is required for native bfloat16 matmuls on x86.  Without it,
    PyTorch silently upcasts every BF16 operand to float32 before each matmul
    and casts back, adding overhead with no quality benefit.
    """
    try:
        with open(_CPUINFO_PATH, encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags") and "avx512_bf16" in line.split():
                    return True
    except OSError:
        pass
    return False


def _memory_gib_string(gib: float) -> str:
    return f"{max(1, int(gib))}GiB"


def _gpu_max_memory() -> tuple[dict[Any, str], tuple[str, ...]]:
    """Return a conservative Accelerate max_memory map for GPU inference."""
    try:
        _free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return {}, ()

    total_gib = total_bytes / _GIB
    requested_max_gib = _positive_float_env(GPU_MAX_MEMORY_ENV)
    if requested_max_gib is not None:
        gpu_limit_gib = min(requested_max_gib, total_gib)
        source_note = f"using {GPU_MAX_MEMORY_ENV}={requested_max_gib:g} GiB"
    else:
        headroom_gib = _positive_float_env(GPU_HEADROOM_ENV) or DEFAULT_GPU_HEADROOM_GIB
        gpu_limit_gib = total_gib - headroom_gib
        source_note = (
            f"leaving {headroom_gib:g} GiB headroom for offloaded weights "
            "and generation cache"
        )

    max_memory: dict[Any, str] = {0: _memory_gib_string(gpu_limit_gib)}

    available_ram = _available_ram_bytes()
    if available_ram is not None:
        cpu_limit_gib = (available_ram / _GIB) - CPU_OFFLOAD_HEADROOM_GIB
        max_memory["cpu"] = _memory_gib_string(cpu_limit_gib)

    notes = (f"GPU memory capped at {max_memory[0]} ({source_note})",)
    return max_memory, notes


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
    return os.path.join(cache_dir, DEFAULT_ROCM_LLAMA_CPP_MODEL_FILENAME)


def _llama_cpp_model_exists(model_path: str | None = None) -> bool:
    return os.path.exists(model_path or _llama_cpp_model_path())


def _llama_cpp_model_repo_id() -> str:
    return os.environ.get(
        LLAMA_CPP_MODEL_REPO_ENV,
        DEFAULT_ROCM_LLAMA_CPP_MODEL_REPO_ID,
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

def _required_llama_cpp_context_size() -> int:
    """Size the context window to hold the transcript prompt plus generation.

    The transcript fed to analysis is bounded by ``transcript_char_budget()``,
    so the context window is derived from the same budget to guarantee the
    prompt fits. The result is clamped to the model's trained context window.
    """
    from lib.report import transcript_char_budget

    budget_chars = transcript_char_budget()
    prompt_tokens = math.ceil(budget_chars / LLAMA_CPP_CHARS_PER_TOKEN)
    required = (
        prompt_tokens + ROCM_ANALYSIS_MAX_NEW_TOKENS + LLAMA_CPP_CONTEXT_MARGIN_TOKENS
    )
    aligned = math.ceil(required / LLAMA_CPP_CONTEXT_ALIGNMENT) * LLAMA_CPP_CONTEXT_ALIGNMENT
    return max(
        DEFAULT_ROCM_LLAMA_CPP_CONTEXT_SIZE,
        min(DEFAULT_ROCM_LLAMA_CPP_MAX_CONTEXT_SIZE, aligned),
    )


def _llama_cpp_context_size() -> int:
    return _positive_int_env(LLAMA_CPP_CONTEXT_SIZE_ENV) or _required_llama_cpp_context_size()


def _llama_cpp_batch_size() -> int:
    return _positive_int_env(LLAMA_CPP_BATCH_SIZE_ENV) or DEFAULT_ROCM_LLAMA_CPP_BATCH_SIZE


def _llama_cpp_layer_count() -> int:
    return _positive_int_env(LLAMA_CPP_LAYER_COUNT_ENV) or DEFAULT_ROCM_LLAMA_CPP_LAYER_COUNT


def _rocm_llama_cpp_gpu_layers(model_path: str) -> tuple[int, tuple[str, ...]]:
    configured_gpu_layers = _positive_int_env(LLAMA_CPP_GPU_LAYERS_ENV)
    if configured_gpu_layers is not None:
        return configured_gpu_layers, (
            f"Using {LLAMA_CPP_GPU_LAYERS_ENV}={configured_gpu_layers} GPU layers",
        )

    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return 0, (
            "Could not inspect ROCm VRAM; llama.cpp will keep all layers in system RAM",
        )

    try:
        model_size_bytes = os.path.getsize(model_path)
    except OSError:
        return 0, (f"GGUF model file not found at {model_path}",)

    layer_count = _llama_cpp_layer_count()
    context_size = _llama_cpp_context_size()
    headroom_gib = (
        _positive_float_env(LLAMA_CPP_GPU_HEADROOM_ENV)
        or DEFAULT_ROCM_LLAMA_CPP_GPU_HEADROOM_GIB
    )
    kv_cache_gib = max(
        DEFAULT_ROCM_LLAMA_CPP_KV_CACHE_GIB,
        ((context_size + ROCM_ANALYSIS_MAX_NEW_TOKENS) / 4096)
        * DEFAULT_ROCM_LLAMA_CPP_KV_CACHE_GIB,
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


def _rocm_llama_cpp_model_kwargs(model_path: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    gpu_layers, notes = _rocm_llama_cpp_gpu_layers(model_path)
    model_kwargs: dict[str, Any] = {
        "model_path": model_path,
        "n_ctx": _llama_cpp_context_size(),
        "n_batch": _llama_cpp_batch_size(),
        "n_gpu_layers": gpu_layers,
        "verbose": False,
    }
    return model_kwargs, notes


# ---------------------------------------------------------------------------
# Per-backend configuration builders
# ---------------------------------------------------------------------------

def _analysis_model_id(backend_name: str) -> tuple[str, tuple[str, ...]]:
    configured_model = os.environ.get(ANALYSIS_MODEL_ENV)
    if configured_model:
        return configured_model, (f"Using {ANALYSIS_MODEL_ENV}={configured_model}",)

    if backend_name == "rocm":
        return DEFAULT_ROCM_ANALYSIS_MODEL_ID, (
            f"Using ROCm analysis model {DEFAULT_ROCM_ANALYSIS_MODEL_ID}",
        )

    return DEFAULT_ANALYSIS_MODEL_ID, ()


def _cpu_analysis_backend() -> AnalysisBackend:
    model_id, model_notes = _analysis_model_id("cpu")

    if _cpu_supports_avx512_bf16():
        return AnalysisBackend(
            name="cpu",
            device_name="CPU",
            model_id=model_id,
            model_kwargs={
                "device_map": "auto",
                "torch_dtype": "auto",
            },
            notes=model_notes + (
                "CPU supports AVX-512 BF16 — loading in BF16 (~15 GB RAM)",
            ),
        )

    available_gib = (_available_ram_bytes() or 0) / _GIB
    if available_gib >= FLOAT32_MIN_RAM_GIB:
        return AnalysisBackend(
            name="cpu",
            device_name="CPU",
            model_id=model_id,
            model_kwargs={
                "device_map": "auto",
                "torch_dtype": torch.float32,
            },
            notes=model_notes + (
                f"CPU lacks AVX-512 BF16; {available_gib:.0f} GB RAM available"
                f" — loading in float32 for native AVX2 matmuls (~30 GB RAM)",
            ),
        )

    return AnalysisBackend(
        name="cpu",
        device_name="CPU",
        model_id=model_id,
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": "auto",
        },
        notes=model_notes + (
            f"CPU lacks AVX-512 BF16 and only {available_gib:.0f} GB RAM available"
            f" (need {FLOAT32_MIN_RAM_GIB} GB for float32)"
            f" — falling back to BF16 (~15 GB RAM)",
        ),
    )


def _rocm_transformers_analysis_backend(
    device_name: str,
    model_id: str,
    model_notes: tuple[str, ...],
) -> AnalysisBackend:
    return AnalysisBackend(
        name="rocm",
        device_name=device_name,
        model_id=model_id,
        model_kwargs={
            "device_map": {"": "cuda"},
            "torch_dtype": torch.float16,
            "attn_implementation": ROCM_ATTENTION_IMPLEMENTATION,
        },
        notes=model_notes + (
            "⚠️ DEPRECATED: PyTorch/Transformers ROCm path is deprecated. Use transcriber:rocm-llama instead.",
            "ROCm loads the analysis model fully on GPU to avoid CPU/GPU offload faults",
            "ROCm uses float16 with eager attention for generation stability",
        ),
        use_plain_prompt=True,
        max_new_tokens=ROCM_ANALYSIS_MAX_NEW_TOKENS,
    )


def _rocm_llama_cpp_analysis_backend(device_name: str) -> AnalysisBackend:
    model_path = _llama_cpp_model_path()
    if _llama_cpp_is_available():
        model_path = _ensure_llama_cpp_model(model_path)
    model_kwargs, model_notes = _rocm_llama_cpp_model_kwargs(model_path)
    return AnalysisBackend(
        name="rocm",
        device_name=device_name,
        model_id=model_path,
        model_kwargs=model_kwargs,
        notes=(
            f"Using ROCm llama.cpp model {_llama_cpp_display_name(model_path)}",
            "ROCm llama.cpp dynamically splits layers between AMD VRAM and system RAM",
        ) + model_notes,
        engine=LLAMA_CPP_BACKEND_NAME,
        use_plain_prompt=True,
        max_new_tokens=ROCM_ANALYSIS_MAX_NEW_TOKENS,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_analysis_backend() -> AnalysisBackend:
    """Return the best analysis backend for summarization."""
    hip_version = getattr(torch.version, "hip", None)
    backend_preference = _analysis_backend_preference()

    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "GPU"

        backend_name = "rocm" if hip_version else "cuda"
        model_id, model_notes = _analysis_model_id(backend_name)

        if backend_name == "rocm":
            if backend_preference == LLAMA_CPP_BACKEND_NAME:
                return _rocm_llama_cpp_analysis_backend(device_name)

            if (
                backend_preference == "auto"
                and not os.environ.get(ANALYSIS_MODEL_ENV)
                and _llama_cpp_is_available()
            ):
                return _rocm_llama_cpp_analysis_backend(device_name)

            print(
                "  ⚠️ DEPRECATED: Using the old PyTorch/Transformers ROCm path."
                " This backend is deprecated and will be removed in a future release."
                " Use transcriber:rocm-llama (llama.cpp/GGUF) instead for AMD GPUs."
            )
            return _rocm_transformers_analysis_backend(device_name, model_id, model_notes)

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": "auto",
        }
        max_memory, notes = _gpu_max_memory()
        if max_memory:
            model_kwargs["max_memory"] = max_memory

        return AnalysisBackend(
            name=backend_name,
            device_name=device_name,
            model_id=model_id,
            model_kwargs=model_kwargs,
            notes=model_notes + notes,
        )

    # No GPU — determine the best dtype for CPU inference.
    # Without AVX-512 BF16, PyTorch upcasts BF16 operands to float32 before
    # each matmul then casts back — overhead with no quality benefit.  Loading
    # in float32 avoids that and uses the faster native AVX2 float32 path, but
    # only when enough RAM is available (~30 GB for weights + headroom).
    return _cpu_analysis_backend()


def _display_model_name(backend: AnalysisBackend) -> str:
    if backend.engine == LLAMA_CPP_BACKEND_NAME:
        return _llama_cpp_display_name(backend.model_id)
    return backend.model_id
