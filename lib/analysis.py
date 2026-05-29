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

DEFAULT_ANALYSIS_MODEL_ID = "google/gemma-4-E4B-it"
ANALYSIS_MODEL_ID = os.environ.get("TRANSCRIBER_ANALYSIS_MODEL", DEFAULT_ANALYSIS_MODEL_ID)
ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert meeting analyst. Analyze the transcript carefully "
    "and provide clear, accurate information based only on the transcript. "
    "Do not invent facts, names, dates, events, decisions, or background details. "
    "If the transcript does not contain information for a requested section, say so explicitly."
)
ANALYSIS_MAX_NEW_TOKENS = 4096
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
    model_kwargs: dict[str, Any]


def detect_analysis_backend() -> AnalysisBackend:
    """Return the best PyTorch backend for summarization."""
    hip_version = getattr(torch.version, "hip", None)

    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "GPU"

        backend_name = "rocm" if hip_version else "cuda"
        return AnalysisBackend(
            name=backend_name,
            device_name=device_name,
            model_kwargs={
                "device_map": "auto",
                "torch_dtype": "auto",
            },
        )

    return AnalysisBackend(
        name="cpu",
        device_name="CPU",
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": "auto",
        },
    )


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
        r"(?m)^(## (?:"
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
) -> str:
    """Run a single analysis-model forward pass and return only the generated text."""
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
            max_new_tokens=ANALYSIS_MAX_NEW_TOKENS,
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
    print(f"\n🤖 Loading {ANALYSIS_MODEL_ID}...")
    print("   (First run downloads model files to the HuggingFace cache.)\n")

    backend = detect_analysis_backend()
    if backend.name == "rocm":
        print(f"  ✅ ROCm GPU detected for summarization: {backend.device_name}")
    elif backend.name == "cuda":
        print(f"  ✅ CUDA GPU detected for summarization: {backend.device_name}")
    else:
        print("  ℹ️  No PyTorch GPU detected for summarization — using CPU")

    tokenizer_factory: Any = AutoTokenizer
    model_factory: Any = AutoModelForCausalLM

    with ProgressTimer(
        "  Loading analysis tokenizer...",
        done_message="Tokenizer loaded",
    ):
        try:
            tokenizer = cast(ChatTokenizer, tokenizer_factory.from_pretrained(ANALYSIS_MODEL_ID))
        except Exception as exc:
            raise AnalysisModelError(
                f"Could not load tokenizer for {ANALYSIS_MODEL_ID}: {exc}"
            ) from exc

    with ProgressTimer(
        "  Loading analysis model and placing weights...",
        done_message=f"Model ready on {backend.name.upper()}",
    ):
        try:
            model = cast(GenerativeModel, model_factory.from_pretrained(
                ANALYSIS_MODEL_ID,
                **backend.model_kwargs,
            ))
        except Exception as exc:
            raise AnalysisModelError(
                f"Could not load analysis model {ANALYSIS_MODEL_ID}: {exc}"
            ) from exc
    model.eval()

    user_msg = _build_user_message(transcript_body, meta)
    with ProgressTimer(
        "  Generating meeting summary report...",
        done_message="Generated meeting summary report",
    ):
        generated_report = _query(model, tokenizer, user_msg)

    _validate_grounding(generated_report, transcript_body)
    return _parse_report_sections(generated_report)
