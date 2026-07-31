"""
transcription.py — Audio device detection and Faster-Whisper transcription.

Responsibilities:
  - Probe for a CUDA GPU and select the best CTranslate2 compute backend
  - Format raw second offsets into human-readable HH:MM:SS.xx timestamps
  - Run Faster-Whisper against a WAV file and return timestamped lines
"""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import ctranslate2
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo

from lib.hardware import detect_gpu
from lib.progress import ProgressTimer
from lib.languages import SUPPORTED_LANGUAGES

# Swap for "small" / "medium" / "large-v3" to trade speed for accuracy.
WHISPER_MODEL_SIZE = "small"


@dataclass(frozen=True)
class TranscriptionResult:
    lines: list[str]
    requested_language: str | None
    requested_language_description: str | None
    detected_language: str
    detected_language_description: str
    language_probability: float
    model_size: str
    engine: str = "faster-whisper"


class WhisperTranscriber(Protocol):
    def transcribe(
        self, audio: str, *, beam_size: int = 5, language: str | None = None
    ) -> tuple[Iterable[Segment], TranscriptionInfo]: ...


def normalize_language_code(language_code: str) -> str:
    return language_code.strip().lower()


def language_description(language_code: str | None) -> str | None:
    if language_code is None:
        return None
    return SUPPORTED_LANGUAGES.get(normalize_language_code(language_code))


def format_supported_languages() -> str:
    language_lines = sorted(
        SUPPORTED_LANGUAGES.items(), key=lambda item: item[1].casefold()
    )
    return "\n".join(
        f"  {language_code} - {description}"
        for language_code, description in language_lines
    )


# Preferred CTranslate2 compute-type order, most efficient first.
# CTranslate2 uses this same string notation for both CUDA and ROCm devices.
_GPU_COMPUTE_TYPE_PREFERENCE = ["float16", "int8_float16", "float32", "int8"]
_CPU_COMPUTE_TYPE = "int8"


def _best_gpu_compute_type() -> str:
    """Ask CTranslate2 which compute types the current GPU supports and return
    the best one according to ``_GPU_COMPUTE_TYPE_PREFERENCE``.

    Falls back to ``float16`` when the query is unavailable (older builds).
    """
    get_supported = getattr(ctranslate2, "get_supported_compute_types", None)
    if not callable(get_supported):
        return "float16"
    try:
        supported: set[str] = set(cast(Any, get_supported)("cuda"))
    except Exception:  # noqa: BLE001
        return "float16"
    for candidate in _GPU_COMPUTE_TYPE_PREFERENCE:
        if candidate in supported:
            return candidate
    return "float16"


def detect_device() -> tuple[str, str]:
    """
    Return ``(device, compute_type)`` for Faster-Whisper / CTranslate2.

    Detection uses the shared ``lib.hardware.detect_gpu()`` helper so the
    same logic covers both NVIDIA and AMD ROCm.  CTranslate2 uses the
    device string ``"cuda"`` for both GPU families; callers need not
    distinguish between them at the CTranslate2 API level.

    The compute type for GPU paths is chosen dynamically by querying
    CTranslate2 for what the current device actually supports, using the
    preference order: float16 → int8_float16 → float32 → int8.
    """
    kind, device_name = detect_gpu()

    if kind == "cuda":
        compute_type = _best_gpu_compute_type()
        print(f"  ✅ CUDA GPU detected for transcription: {device_name}")
        return "cuda", compute_type

    if kind == "rocm":
        # CTranslate2 uses device="cuda" for ROCm builds too.
        compute_type = _best_gpu_compute_type()
        print(f"  ✅ ROCm GPU detected for transcription: {device_name}")
        if os.environ.get("DEBUG") == "1":
            print(
                "  DEBUG: Using CTranslate2 device='cuda' for ROCm "
                "(ROCm builds still use CUDA device string)"
            )
        return "cuda", compute_type

    print("  ℹ️  No GPU detected — running on CPU")
    return "cpu", _CPU_COMPUTE_TYPE


def fmt_ts(seconds: float) -> str:
    """
    Convert a raw second offset to HH:MM:SS.xx notation.

    Examples
    --------
    fmt_ts(4.5)      →  '00:00:04.50'
    fmt_ts(723.1)    →  '00:12:03.10'
    fmt_ts(3723.45)  →  '01:02:03.45'
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def transcribe_audio(
    audio_path: str,
    device: str,
    compute_type: str,
    language: str | None = None,
) -> TranscriptionResult:
    """
    Run Faster-Whisper on *audio_path* and return a list of timestamped lines.

    Each line looks like:
        [00:00:04.50 -> 00:00:12.30]  Hey everyone, let's look at the schema.
    """
    print(
        f"\n📦 Loading Faster-Whisper ({WHISPER_MODEL_SIZE}) "
        f"on [{device.upper()}] [{compute_type}]..."
    )
    with ProgressTimer(
        "  Preparing Faster-Whisper model...",
        done_message="Faster-Whisper model ready",
    ):
        model = cast(
            WhisperTranscriber,
            WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type),
        )

    if language is not None:
        language_name = language_description(language) or "Unknown"
        print(f"🌐 Requested language: {language_name} ({language})")

    print(f"📝 Transcribing '{os.path.basename(audio_path)}'...")
    with ProgressTimer(
        "  Running speech recognition...",
        done_message="Speech recognition complete",
    ):
        segments, info = model.transcribe(audio_path, beam_size=5, language=language)

    detected_language_description = language_description(info.language) or "Unknown"

    print(
        f"🌐 Detected language: {detected_language_description} ({info.language}) "
        f"({info.language_probability:.0%} confidence)\n"
    )

    lines: list[str] = []
    with ProgressTimer(
        "  Collecting transcript segments...",
        done_message="Transcript segments collected",
    ):
        for seg in segments:
            line = f"[{fmt_ts(seg.start)} -> {fmt_ts(seg.end)}]  {seg.text.strip()}"
            lines.append(line)

    for line in lines:
        print(line)

    return TranscriptionResult(
        lines=lines,
        requested_language=language,
        requested_language_description=language_description(language),
        detected_language=info.language,
        detected_language_description=detected_language_description,
        language_probability=info.language_probability,
        model_size=WHISPER_MODEL_SIZE,
        engine="faster-whisper",
    )


def _faster_whisper_fallback(
    audio_path: str,
    language: str | None,
    *,
    force_cpu: bool,
) -> TranscriptionResult:
    if force_cpu:
        device, compute_type = "cpu", _CPU_COMPUTE_TYPE
        print("  ℹ️  Using Faster-Whisper CPU fallback")
    else:
        device, compute_type = detect_device()
    return transcribe_audio(
        audio_path,
        device,
        compute_type,
        language=language,
    )


def transcribe_audio_auto(
    audio_path: str,
    language: str | None = None,
) -> TranscriptionResult:
    """Select whisper.cpp Vulkan or Faster-Whisper without changing the CLI.

    Docker profiles are explicit: the CPU image always uses Faster-Whisper on
    CPU, while the Vulkan image attempts whisper.cpp and falls back to CPU for
    expected runtime/model failures. Native execution keeps the existing
    Faster-Whisper CUDA/ROCm/CPU detection unless whisper.cpp is requested.
    """
    profile = os.environ.get("TRANSCRIBER_RUNTIME_PROFILE", "native").strip().lower()
    preference = os.environ.get("TRANSCRIBER_TRANSCRIPTION_BACKEND", "auto").strip().lower()
    if preference not in {"auto", "faster_whisper", "whisper_cpp"}:
        print(
            "  ⚠️  Ignoring invalid TRANSCRIBER_TRANSCRIPTION_BACKEND="
            f"{preference!r}"
        )
        preference = "auto"

    force_cpu = profile in {"cpu", "vulkan"}
    use_whisper_cpp = preference == "whisper_cpp" or (
        preference == "auto" and profile == "vulkan"
    )
    if not use_whisper_cpp:
        return _faster_whisper_fallback(
            audio_path,
            language,
            force_cpu=force_cpu,
        )

    # Imports are lazy so native/CPU users do not pay for Vulkan adapter setup.
    from lib.vulkan import probe_vulkan
    from lib.whisper_cpp import WhisperCppError, ensure_model, transcribe

    probe = probe_vulkan()
    if not probe.available or probe.selected_device is None:
        print(f"  ⚠️  Vulkan transcription unavailable: {probe.reason}; using CPU")
        return _faster_whisper_fallback(audio_path, language, force_cpu=True)

    device = probe.selected_device
    print(f"  ✅ Vulkan GPU detected for transcription: {device.name}")
    try:
        print(f"\n📦 Preparing whisper.cpp ({WHISPER_MODEL_SIZE}) on [VULKAN]...")
        model_path = ensure_model()
        print(f"📝 Transcribing '{os.path.basename(audio_path)}' with whisper.cpp...")
        raw_result = transcribe(
            audio_path,
            model_path=model_path,
            device=device,
            language=language,
        )
    except WhisperCppError as exc:
        print(f"  ⚠️  Vulkan transcription failed: {exc}")
        return _faster_whisper_fallback(audio_path, language, force_cpu=True)

    detected_description = language_description(raw_result.detected_language) or "Unknown"
    lines = [
        f"[{fmt_ts(start)} -> {fmt_ts(end)}]  {text}"
        for start, end, text in raw_result.segments
    ]
    for line in lines:
        print(line)
    print(
        f"🌐 Detected language: {detected_description} "
        f"({raw_result.detected_language}) "
        f"({raw_result.language_probability:.0%} confidence)\n"
    )
    return TranscriptionResult(
        lines=lines,
        requested_language=language,
        requested_language_description=language_description(language),
        detected_language=raw_result.detected_language,
        detected_language_description=detected_description,
        language_probability=raw_result.language_probability,
        model_size=WHISPER_MODEL_SIZE,
        engine="whisper.cpp",
    )
