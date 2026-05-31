"""
model — model loading and inference execution.

Responsibilities:
    - Define typed Protocols for tokenizer, generative model, and llama.cpp model
    - Load Transformers or llama.cpp models from disk / HuggingFace cache
    - Run a single forward pass and return the generated text
    - Orchestrate the full report-generation round-trip for each engine
"""

import importlib
from typing import Any, Protocol, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .backend import (
    ANALYSIS_BACKEND_ENV,
    ANALYSIS_MAX_NEW_TOKENS,
    ANALYSIS_MODEL_ENV,
    LLAMA_CPP_BACKEND_NAME,
    TRANSFORMERS_BACKEND_NAME,
    AnalysisBackend,
    _ensure_llama_cpp_model,
    _llama_cpp_model_exists,
)
from .prompt import ANALYSIS_SYSTEM_PROMPT, _build_prompt_for_backend
from .utils import AnalysisModelError
from lib.progress import ProgressTimer


# ---------------------------------------------------------------------------
# Typed Protocols
# ---------------------------------------------------------------------------

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


class LlamaCppModel(Protocol):
    def create_completion(self, **kwargs: Any) -> Any: ...

# ---------------------------------------------------------------------------
# Transformers inference
# ---------------------------------------------------------------------------

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


def _generate_report_with_transformers(
    backend: AnalysisBackend,
    transcript_body: str,
    meta: dict[str, str],
) -> str:
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

    user_msg = _build_prompt_for_backend(backend, transcript_body, meta)
    with ProgressTimer(
        "  Generating meeting summary report...",
        done_message="Generated meeting summary report",
    ):
        try:
            return _query(
                model,
                tokenizer,
                user_msg,
                max_new_tokens=backend.max_new_tokens,
                use_plain_prompt=backend.use_plain_prompt,
            )
        except torch.OutOfMemoryError as exc:
            raise AnalysisModelError(
                "Analysis model ran out of GPU memory during generation. "
                "Try lowering TRANSCRIBER_MAX_TRANSCRIPT_CHARS or setting "
                f"{ANALYSIS_MODEL_ENV} to a smaller local model."
            ) from exc


# ---------------------------------------------------------------------------
# llama.cpp inference
# ---------------------------------------------------------------------------

# Tokens kept free between the prompt and the model's context limit as a final
# safety net against char-to-token estimation error (e.g. token-dense scripts).
LLAMA_CPP_CONTEXT_GUARD_MARGIN = 64


def _fit_prompt_to_context(
    model: LlamaCppModel,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, int]:
    """Trim the prompt so prompt + generation fits the model context window.

    The context window is normally sized to hold the whole transcript, so this
    only acts as a safety net. It no-ops when the model does not expose the
    llama.cpp ``n_ctx``/``tokenize``/``detokenize`` helpers (e.g. mock models).
    """
    n_ctx_fn = getattr(model, "n_ctx", None)
    tokenize_fn = getattr(model, "tokenize", None)
    detokenize_fn = getattr(model, "detokenize", None)
    if not (callable(n_ctx_fn) and callable(tokenize_fn) and callable(detokenize_fn)):
        return prompt, max_new_tokens

    try:
        context_size = int(n_ctx_fn())
        tokens = list(tokenize_fn(prompt.encode("utf-8")))
    except Exception:
        return prompt, max_new_tokens

    allowed_prompt_tokens = context_size - max_new_tokens - LLAMA_CPP_CONTEXT_GUARD_MARGIN
    if allowed_prompt_tokens < 1:
        # Generation budget itself is too large; shrink it to leave room.
        max_new_tokens = max(1, context_size - len(tokens) - LLAMA_CPP_CONTEXT_GUARD_MARGIN)
        allowed_prompt_tokens = context_size - max_new_tokens - LLAMA_CPP_CONTEXT_GUARD_MARGIN

    if len(tokens) <= allowed_prompt_tokens:
        return prompt, max_new_tokens

    try:
        trimmed = detokenize_fn(tokens[:allowed_prompt_tokens]).decode("utf-8", "ignore")
    except Exception:
        return prompt, max_new_tokens
    print(
        "  ⚠️  Transcript prompt exceeded the llama.cpp context window; "
        f"trimmed to {allowed_prompt_tokens} tokens."
    )
    return trimmed, max_new_tokens


def _query_llama_cpp(
    model: LlamaCppModel,
    user_message: str,
    *,
    max_new_tokens: int,
) -> str:
    user_message, max_new_tokens = _fit_prompt_to_context(
        model, user_message, max_new_tokens
    )
    response = model.create_completion(
        prompt=user_message,
        max_tokens=max_new_tokens,
        temperature=0,
    )
    choices = response.get("choices", [])
    if not choices:
        return ""

    choice = choices[0]
    text = choice.get("text")
    if isinstance(text, str):
        return text.strip()

    message = choice.get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    return ""


def _load_llama_cpp_model(backend: AnalysisBackend) -> LlamaCppModel:
    try:
        llama_cpp_module = importlib.import_module("llama_cpp")
    except ImportError as exc:
        raise AnalysisModelError(
            "Could not import llama.cpp support. Install llama-cpp-python or "
            f"set {ANALYSIS_BACKEND_ENV}={TRANSFORMERS_BACKEND_NAME} to use the "
            "Transformers backend."
        ) from exc

    model_path = cast(str, backend.model_kwargs.get("model_path", backend.model_id))
    if not _llama_cpp_model_exists(model_path):
        model_path = _ensure_llama_cpp_model(model_path)
        backend.model_kwargs["model_path"] = model_path

    try:
        return cast(LlamaCppModel, llama_cpp_module.Llama(**backend.model_kwargs))
    except Exception as exc:
        raise AnalysisModelError(
            f"Could not load llama.cpp analysis model {model_path}: {exc}"
        ) from exc


def _generate_report_with_llama_cpp(
    backend: AnalysisBackend,
    transcript_body: str,
    meta: dict[str, str],
) -> str:
    with ProgressTimer(
        "  Loading llama.cpp analysis model...",
        done_message=f"Model ready on {backend.name.upper()}",
    ):
        model = _load_llama_cpp_model(backend)

    user_msg = _build_prompt_for_backend(backend, transcript_body, meta)
    with ProgressTimer(
        "  Generating meeting summary report...",
        done_message="Generated meeting summary report",
    ):
        return _query_llama_cpp(
            model,
            user_msg,
            max_new_tokens=backend.max_new_tokens,
        )
