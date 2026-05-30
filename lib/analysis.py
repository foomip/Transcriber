"""
analysis.py — local Hugging Face model loading and summary generation.

Responsibilities:
    - Load the configured analysis model (auto-placed on GPU when available)
    - Format transcript text into a grounded meeting-analysis prompt
    - Run report generation and return (heading, text) pairs
"""

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.progress import ProgressTimer

ANALYSIS_MODEL_ENV = "TRANSCRIBER_ANALYSIS_MODEL"
DEFAULT_ANALYSIS_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_ROCM_ANALYSIS_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert meeting analyst. Analyze the transcript carefully "
    "and provide clear, accurate information based only on the transcript. "
    "Do not invent facts, names, dates, events, decisions, or background details. "
    "If the transcript does not contain information for a requested section, say so explicitly."
)
ANALYSIS_MAX_NEW_TOKENS = 4096
ROCM_ANALYSIS_MAX_NEW_TOKENS = 1024
GROUNDING_MIN_RATIO = 0.16
GROUNDING_MIN_TOP_TERM_OVERLAP = 2

_STOP_WORDS = {
    "about", "above", "after", "again", "against", "also", "another", "because",
    "before", "being", "between", "could", "during", "every", "first", "from",
    "have", "having", "into", "itself", "just", "like", "major", "meeting",
    "more", "most", "other", "over", "really", "should", "some", "specific",
    "still", "their", "there", "these", "thing", "things", "think", "this",
    "those", "through", "today", "under", "using", "where", "which", "while",
    "with", "would", "youre",
}

# Each tuple: (markdown heading, analysis instruction)
SUMMARY_TASKS: list[tuple[str, str]] = [
    (
        "## Executive Summary",
        "Provide a brief executive summary (2-3 sentences) of the key outcomes and decisions from this transcript.",
    ),
    (
        "## Detailed Summary",
        "Provide a detailed summary of the transcript, covering all major topics, discussions, and outcomes in paragraph form.",
    ),
    (
        "## Action Items",
        "List the specific action items that were assigned during this meeting. Include who is responsible for each item when mentioned.",
    ),
    (
        "## Key Decisions",
        "List the key decisions that were made during this meeting. Focus on concrete decisions and outcomes.",
    ),
    (
        "## Topics Discussed",
        "List the main topics and subjects that were discussed in this meeting.",
    ),
]


class AnalysisGroundingError(RuntimeError):
    """Raised when generated analysis is not sufficiently grounded in the transcript."""


class AnalysisModelError(RuntimeError):
    """Raised when the configured analysis model cannot be loaded or initialized."""


class ChatTokenizer(Protocol):
    eos_token_id: int | None

    def __call__(self, text: str, *, return_tensors: str) -> Any: ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        return_tensors: str,
        return_dict: bool,
    ) -> Any: ...

    def decode(self, token_ids: Any, *, skip_special_tokens: bool) -> str: ...


class GenerativeModel(Protocol):
    device: torch.device

    def eval(self) -> Any: ...

    def generate(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class AnalysisBackend:
    name: str
    device_name: str
    model_id: str
    model_kwargs: dict[str, Any]
    notes: tuple[str, ...] = ()


_CPUINFO_PATH = "/proc/cpuinfo"
_MEMINFO_PATH = "/proc/meminfo"
_GIB = 1024 ** 3
FLOAT32_MIN_RAM_GIB = 32   # float32 weights ~30 GB + headroom for KV cache and OS
DEFAULT_GPU_HEADROOM_GIB = 7
CPU_OFFLOAD_HEADROOM_GIB = 8
GPU_HEADROOM_ENV = "TRANSCRIBER_ANALYSIS_GPU_HEADROOM_GIB"
GPU_MAX_MEMORY_ENV = "TRANSCRIBER_ANALYSIS_GPU_MAX_MEMORY_GIB"
ROCM_ATTENTION_IMPLEMENTATION = "eager"


def _analysis_model_id(backend_name: str) -> tuple[str, tuple[str, ...]]:
    configured_model = os.environ.get(ANALYSIS_MODEL_ENV)
    if configured_model:
        return configured_model, (f"Using {ANALYSIS_MODEL_ENV}={configured_model}",)

    if backend_name == "rocm":
        return DEFAULT_ROCM_ANALYSIS_MODEL_ID, (
            f"Using ROCm analysis model {DEFAULT_ROCM_ANALYSIS_MODEL_ID}",
        )

    return DEFAULT_ANALYSIS_MODEL_ID, ()


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


def detect_analysis_backend() -> AnalysisBackend:
    """Return the best PyTorch backend for summarization."""
    hip_version = getattr(torch.version, "hip", None)

    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "GPU"

        backend_name = "rocm" if hip_version else "cuda"
        model_id, model_notes = _analysis_model_id(backend_name)

        if backend_name == "rocm":
            return AnalysisBackend(
                name=backend_name,
                device_name=device_name,
                model_id=model_id,
                model_kwargs={
                    "device_map": {"": "cuda"},
                    "torch_dtype": torch.float16,
                    "attn_implementation": ROCM_ATTENTION_IMPLEMENTATION,
                },
                notes=model_notes + (
                    "ROCm loads the analysis model fully on GPU to avoid CPU/GPU offload faults",
                    "ROCm uses float16 with eager attention for generation stability",
                ),
            )

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


def _build_user_message(
    transcript_body: str,
    meta: dict[str, str],
) -> str:
    """
    Assemble a user message for the analysis model:

        <instructions>

        Title: …
        Date:  …
        …
        ----------

        <transcript text>
    """
    section_instructions = "\n".join(
        f"{heading}\n{instruction}" for heading, instruction in SUMMARY_TASKS
    )

    return (
        "Create a Markdown meeting report using exactly the sections listed below. "
        "Use only facts present in the transcript. Do not add external context, examples, "
        "biographical details, sports results, historical background, or generic explanations. "
        "If there are no action items or no key decisions, write that none were explicitly stated. "
        "Return only the requested Markdown sections and their content.\n\n"
        f"{section_instructions}\n\n"
        f"Title: {meta['title']}\n"
        f"Date: {meta['date']}\n"
        f"Time: {meta['time']}\n"
        f"Duration: {meta['duration']}\n"
        f"Requested transcription language: {meta.get('requested_language', 'Auto-detect')}\n"
        f"Detected transcription language: {meta.get('detected_language', 'Unknown')}\n"
        f"Language detection confidence: {meta.get('language_probability', 'Unknown')}\n"
        f"Transcription model: {meta.get('transcription_model', 'Unknown')}\n"
        f"Participants: Unknown (no speaker diarization)\n\n"
        "----------\n\n"
        f"{transcript_body}"
    )


def _build_compact_user_message(
    transcript_body: str,
    _meta: dict[str, str],
) -> str:
    return (
        "You are an expert meeting analyst. Use only the transcript. "
        "Do not invent facts. Return a Markdown report with these exact headings:\n"
        "## Executive Summary\n"
        "## Detailed Summary\n"
        "## Action Items\n"
        "## Key Decisions\n"
        "## Topics Discussed\n"
        "If a section has no explicit information, say none was explicitly stated.\n\n"
        "Transcript:\n"
        f"{transcript_body}"
    )


def _content_words(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z][a-z0-9']+", text.lower())
        if len(token) >= 4 and token not in _STOP_WORDS
    ]


def _validate_grounding(generated_report: str, transcript_body: str) -> None:
    transcript_words = _content_words(transcript_body)
    generated_words = _content_words(generated_report)
    if not transcript_words or not generated_words:
        return

    transcript_terms = set(transcript_words)
    grounded_count = sum(1 for word in generated_words if word in transcript_terms)
    grounded_ratio = grounded_count / len(generated_words)

    top_transcript_terms = {
        term for term, _ in Counter(transcript_words).most_common(20)
    }
    top_term_overlap = top_transcript_terms.intersection(generated_words)

    if (
        grounded_ratio < GROUNDING_MIN_RATIO
        or len(top_term_overlap) < GROUNDING_MIN_TOP_TERM_OVERLAP
    ):
        raise AnalysisGroundingError(
            "Generated report appears unrelated to the transcript "
            f"(grounded word ratio {grounded_ratio:.0%}, "
            f"top-term overlap {len(top_term_overlap)}). "
            "Refusing to save hallucinated analysis."
        )


def _parse_report_sections(generated_report: str) -> list[tuple[str, str]]:
    headings = [heading for heading, _ in SUMMARY_TASKS]
    pattern = re.compile(
        r"(?m)^\s*(## (?:"
        + "|".join(re.escape(heading.removeprefix("## ")) for heading in headings)
        + r"))\s*$"
    )
    matches = list(pattern.finditer(generated_report))
    found: dict[str, str] = {}

    for index, match in enumerate(matches):
        heading = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(generated_report)
        text = generated_report[start:end].strip()
        if heading not in found:
            found[heading] = text

    return [
        (heading, found.get(heading, "No information was generated for this section."))
        for heading in headings
    ]


def _query(
    model: GenerativeModel,
    tokenizer: ChatTokenizer,
    user_message: str,
    *,
    max_new_tokens: int = ANALYSIS_MAX_NEW_TOKENS,
    use_plain_prompt: bool = False,
) -> str:
    """Run a single analysis-model forward pass and return only the generated text."""
    if use_plain_prompt:
        inputs = tokenizer(user_message, return_tensors="pt").to(model.device)
    else:
        messages = [
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True
        ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt tokens so we decode only the newly generated output.
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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
    print(f"\n🤖 Loading {backend.model_id}...")
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

    tokenizer_factory: Any = AutoTokenizer
    model_factory: Any = AutoModelForCausalLM

    with ProgressTimer(
        "  Loading analysis tokenizer...",
        done_message="Tokenizer loaded",
    ):
        try:
            tokenizer = cast(ChatTokenizer, tokenizer_factory.from_pretrained(backend.model_id))
        except Exception as exc:
            raise AnalysisModelError(
                f"Could not load tokenizer for {backend.model_id}: {exc}"
            ) from exc

    with ProgressTimer(
        "  Loading analysis model and placing weights...",
        done_message=f"Model ready on {backend.name.upper()}",
    ):
        try:
            model = cast(GenerativeModel, model_factory.from_pretrained(
                backend.model_id,
                **backend.model_kwargs,
            ))
        except Exception as exc:
            raise AnalysisModelError(
                f"Could not load analysis model {backend.model_id}: {exc}"
            ) from exc
    model.eval()

    use_rocm_prompt = backend.name == "rocm"
    user_msg = (
        _build_compact_user_message(transcript_body, meta)
        if use_rocm_prompt
        else _build_user_message(transcript_body, meta)
    )
    max_new_tokens = (
        ROCM_ANALYSIS_MAX_NEW_TOKENS
        if backend.name == "rocm"
        else ANALYSIS_MAX_NEW_TOKENS
    )
    with ProgressTimer(
        "  Generating meeting summary report...",
        done_message="Generated meeting summary report",
    ):
        try:
            generated_report = _query(
                model,
                tokenizer,
                user_msg,
                max_new_tokens=max_new_tokens,
                use_plain_prompt=use_rocm_prompt,
            )
        except torch.OutOfMemoryError as exc:
            raise AnalysisModelError(
                "Analysis model ran out of GPU memory during generation. "
                "Try lowering TRANSCRIBER_MAX_TRANSCRIPT_CHARS or setting "
                f"{ANALYSIS_MODEL_ENV} to a smaller local model."
            ) from exc

    _validate_grounding(generated_report, transcript_body)
    return _parse_report_sections(generated_report)
