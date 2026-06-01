"""
transcription.py — Audio device detection and Faster-Whisper transcription.

Responsibilities:
  - Probe for a CUDA GPU and select the best CTranslate2 compute backend
  - Format raw second offsets into human-readable HH:MM:SS.xx timestamps
  - Run Faster-Whisper against a WAV file and return timestamped lines
"""

import os
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, cast

import ctranslate2
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo

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


def detect_device() -> tuple[str, str]:
    """
    Return (device, compute_type) for Faster-Whisper / CTranslate2.

    Uses CTranslate2's own CUDA probe so PyTorch is not needed for
    detection. Falls back to CPU int8 when no GPU is found.
    """
    get_cuda_device_count = cast(
        Callable[[], int], getattr(ctranslate2, "get_cuda_device_count")
    )
    cuda_count = get_cuda_device_count()

    if cuda_count > 0:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8")
            print(f"  ✅ GPU detected: {gpu_name}")
        except Exception as exc:
            if os.environ.get("DEBUG") == "1":
                print(f"  DEBUG: NVIDIA name lookup failed: {exc}")
            print(f"  ✅ {cuda_count} CUDA GPU(s) detected")
        return "cuda", "float16"

    if os.environ.get("DEBUG") == "1" and os.path.exists("/dev/nvidia0"):
        try:
            nvidia_devices = sorted(
                entry for entry in os.listdir("/dev") if entry.startswith("nvidia")
            )
            print(
                "  DEBUG: CUDA runtime reported 0 visible devices, but NVIDIA device nodes exist: "
                + ", ".join(f"/dev/{entry}" for entry in nvidia_devices)
            )
        except OSError as exc:
            print(f"  DEBUG: Could not inspect /dev for NVIDIA device nodes: {exc}")

    # AMD / ROCm check
    if os.path.exists("/dev/kfd"):
        try:
            res = subprocess.check_output(["rocm-smi", "--showproductname"], stderr=subprocess.DEVNULL, text=True)
            lines = res.strip().splitlines()
            if len(lines) > 1:
                gpu_name = lines[1].split(None, 1)[-1].strip()
                print(f"  ℹ️  ROCm GPU detected for summarization: {gpu_name}")
            else:
                print("  ℹ️  ROCm GPU detected for summarization")
        except Exception as exc:
            if os.environ.get("DEBUG") == "1":
                print(f"  DEBUG: ROCm name lookup failed: {exc}")
            print("  ℹ️  ROCm GPU detected for summarization")
        print("  ℹ️  Faster-Whisper does not expose ROCm here — transcribing on CPU")
        return "cpu", "int8"

    print("  ℹ️  No GPU detected — running on CPU")
    return "cpu", "int8"


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
    )
