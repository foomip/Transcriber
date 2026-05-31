"""
analysis — local Hugging Face model loading and summary generation.

Responsibilities:
    - Orchestrate backend detection, model loading, and report generation
    - Re-export the public API and internal symbols needed by callers and tests

Implementation is split across focused sub-modules:
    lib/analysis/backend.py  — hardware detection and backend configuration
    lib/analysis/model.py    — model loading and inference execution
    lib/analysis/prompt.py   — prompt construction and summary task definitions
    lib/analysis/utils.py    — grounding validation and report parsing
"""

# Re-exported so that callers (transcribe.py, youtube-summarize.py) and tests
# can continue to access everything via `from lib import analysis`.
import torch  # noqa: F401  (tests access analysis.torch.cuda / analysis.torch.version)

from .backend import (  # noqa: F401
    ANALYSIS_BACKEND_ENV,
    ANALYSIS_MAX_NEW_TOKENS,
    ANALYSIS_MODEL_ENV,
    CPU_OFFLOAD_HEADROOM_GIB,
    DEFAULT_ANALYSIS_MODEL_ID,
    DEFAULT_GPU_HEADROOM_GIB,
    DEFAULT_ROCM_ANALYSIS_MODEL_ID,
    DEFAULT_ROCM_LLAMA_CPP_BATCH_SIZE,
    DEFAULT_ROCM_LLAMA_CPP_CONTEXT_SIZE,
    DEFAULT_ROCM_LLAMA_CPP_GPU_HEADROOM_GIB,
    DEFAULT_ROCM_LLAMA_CPP_KV_CACHE_GIB,
    DEFAULT_ROCM_LLAMA_CPP_LAYER_COUNT,
    DEFAULT_ROCM_LLAMA_CPP_MAX_CONTEXT_SIZE,
    DEFAULT_ROCM_LLAMA_CPP_MODEL_FILENAME,
    DEFAULT_ROCM_LLAMA_CPP_MODEL_REPO_ID,
    FLOAT32_MIN_RAM_GIB,
    GPU_HEADROOM_ENV,
    GPU_MAX_MEMORY_ENV,
    LLAMA_CPP_BACKEND_NAME,
    LLAMA_CPP_BATCH_SIZE_ENV,
    LLAMA_CPP_CACHE_DIR_ENV,
    LLAMA_CPP_CHARS_PER_TOKEN,
    LLAMA_CPP_CONTEXT_SIZE_ENV,
    LLAMA_CPP_GPU_HEADROOM_ENV,
    LLAMA_CPP_GPU_LAYERS_ENV,
    LLAMA_CPP_LAYER_COUNT_ENV,
    LLAMA_CPP_MODEL_PATH_ENV,
    LLAMA_CPP_MODEL_REPO_ENV,
    ROCM_ANALYSIS_MAX_NEW_TOKENS,
    ROCM_ATTENTION_IMPLEMENTATION,
    TRANSFORMERS_BACKEND_NAME,
    AnalysisBackend,
    _GIB,
    _CPUINFO_PATH,
    _MEMINFO_PATH,
    _analysis_model_id,
    _available_ram_bytes,
    _cpu_analysis_backend,
    _cpu_supports_avx512_bf16,
    _display_model_name,
    _download_llama_cpp_model,
    _ensure_llama_cpp_model,
    _llama_cpp_display_name,
    _llama_cpp_is_available,
    _llama_cpp_model_exists,
    _llama_cpp_model_path,
    _llama_cpp_context_size,
    _required_llama_cpp_context_size,
    _rocm_llama_cpp_gpu_layers,
    _rocm_llama_cpp_model_kwargs,
    detect_analysis_backend,
)
from .model import (  # noqa: F401
    ChatTokenizer,
    GenerativeModel,
    LlamaCppModel,
    _fit_prompt_to_context,
    _generate_report_with_llama_cpp,
    _generate_report_with_transformers,
    _load_llama_cpp_model,
    _query,
    _query_llama_cpp,
)
from .prompt import (  # noqa: F401
    ANALYSIS_SYSTEM_PROMPT,
    SUMMARY_TASKS,
    _build_compact_user_message,
    _build_prompt_for_backend,
    _build_user_message,
)
from .utils import (  # noqa: F401
    GROUNDING_MIN_RATIO,
    GROUNDING_MIN_TOP_TERM_OVERLAP,
    AnalysisGroundingError,
    AnalysisModelError,
    _content_words,
    _parse_report_sections,
    _validate_grounding,
)


def generate_summaries(
    transcript_body: str,
    meta: dict[str, str],
) -> list[tuple[str, str]]:
    """
    Load the configured analysis model and generate the report sections.

    Returns a list of (markdown_heading, generated_text) pairs in the
    same order as SUMMARY_TASKS, ready to be handed to report.compile().

    device_map="auto" lets HuggingFace Accelerate place the model on the
    best available PyTorch backend. ROCm GPUs appear through torch.cuda.
    """
    backend = detect_analysis_backend()
    print(f"\n🤖 Loading {_display_model_name(backend)}...")
    if backend.engine == LLAMA_CPP_BACKEND_NAME:
        print("   (Missing default GGUF files download to the local GGUF cache.)\n")
    else:
        print("   (First run downloads model files to the HuggingFace cache.)\n")

    if backend.name == "rocm":
        print(f"  ✅ ROCm GPU detected for summarization: {backend.device_name}")
        for note in backend.notes:
            print(f"  ℹ️  {note}")
    elif backend.name == "cuda":
        print(f"  ✅ CUDA GPU detected for summarization: {backend.device_name}")
        for note in backend.notes:
            print(f"  ℹ️  {note}")
    else:
        print("  ℹ️  Using CPU for summarization")
        for note in backend.notes:
            icon = "⚠️ " if "lacks" in note else "ℹ️ "
            print(f"  {icon} {note}")

    if backend.engine == LLAMA_CPP_BACKEND_NAME:
        generated_report = _generate_report_with_llama_cpp(backend, transcript_body, meta)
    else:
        generated_report = _generate_report_with_transformers(backend, transcript_body, meta)

    _validate_grounding(generated_report, transcript_body)
    return _parse_report_sections(generated_report)
