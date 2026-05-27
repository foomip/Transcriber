"""
analysis.py — LFM2-2.6B-Transcript model loading and summary generation.

Responsibilities:
    - Load LiquidAI/LFM2-2.6B-Transcript (auto-placed on GPU when available)
    - Format transcript text into the LFM2 prompting schema
    - Run each of the five summary passes and return (heading, text) pairs
"""

from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.progress import ProgressTimer

LFM2_MODEL_ID = "LiquidAI/LFM2-2.6B-Transcript"
LFM2_SYSTEM_PROMPT = (
    "You are an expert meeting analyst. Analyze the transcript carefully "
    "and provide clear, accurate information based on the content."
)
LFM2_TEMPERATURE    = 0.3
LFM2_MAX_NEW_TOKENS = 800

# Each tuple: (markdown heading, LFM2 instruction)
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
    """Return the best PyTorch backend for LFM2 summarization."""
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
    instruction: str,
    transcript_body: str,
    meta: dict[str, str],
) -> str:
    """
    Assemble a user message in the LFM2 prompting schema:

        <instruction>

        Title: …
        Date:  …
        …
        ----------

        <transcript text>
    """
    return (
        f"{instruction}\n\n"
        f"Title: {meta['title']}\n"
        f"Date: {meta['date']}\n"
        f"Time: {meta['time']}\n"
        f"Duration: {meta['duration']}\n"
        f"Participants: Unknown (no speaker diarization)\n\n"
        "----------\n\n"
        f"{transcript_body}"
    )


def _query(
    model: GenerativeModel,
    tokenizer: ChatTokenizer,
    user_message: str,
) -> str:
    """Run a single LFM2 forward pass and return only the generated text."""
    messages = [
        {"role": "system", "content": LFM2_SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=LFM2_MAX_NEW_TOKENS,
            temperature=LFM2_TEMPERATURE,
            do_sample=True,
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
    Load LFM2-2.6B-Transcript and run all five summary passes.

    Returns a list of (markdown_heading, generated_text) pairs in the
    same order as SUMMARY_TASKS, ready to be handed to report.compile().

    device_map="auto" lets HuggingFace Accelerate place the model on the
    best available PyTorch backend. ROCm GPUs appear through torch.cuda.
    """
    print(f"\n🤖 Loading {LFM2_MODEL_ID}...")
    print("   (First run downloads ~5 GB to the HuggingFace cache.)\n")

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
        "  Loading LFM2 tokenizer...",
        done_message="Tokenizer loaded",
    ):
        tokenizer = cast(ChatTokenizer, tokenizer_factory.from_pretrained(LFM2_MODEL_ID))

    with ProgressTimer(
        "  Loading LFM2 model and placing weights...",
        done_message=f"Model ready on {backend.name.upper()}",
    ):
        model = cast(GenerativeModel, model_factory.from_pretrained(
            LFM2_MODEL_ID,
            **backend.model_kwargs,
        ))
    model.eval()

    sections: list[tuple[str, str]] = []
    for heading, instruction in SUMMARY_TASKS:
        section_name = heading.lstrip("# ")
        user_msg = _build_user_message(instruction, transcript_body, meta)
        with ProgressTimer(
            f"  Generating: {section_name}...",
            done_message=f"Generated {section_name}",
        ):
            result = _query(model, tokenizer, user_msg)
        sections.append((heading, result))

    return sections
