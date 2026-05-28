"""
transcription.py — Audio device detection and Faster-Whisper transcription.

Responsibilities:
  - Probe for a CUDA GPU and select the best CTranslate2 compute backend
  - Format raw second offsets into human-readable HH:MM:SS.xx timestamps
  - Run Faster-Whisper against a WAV file and return timestamped lines
"""

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, cast

import ctranslate2
import torch
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo

from lib.progress import ProgressTimer

# Swap for "small" / "medium" / "large-v3" to trade speed for accuracy.
WHISPER_MODEL_SIZE = "small"

SUPPORTED_LANGUAGES: dict[str, str] = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "as": "Assamese",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fo": "Faroese",
    "fr": "French",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jw": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "la": "Latin",
    "lb": "Luxembourgish",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Myanmar",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tr": "Turkish",
    "tt": "Tatar",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "yue": "Cantonese",
    "zh": "Chinese",
}


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

    Uses CTranslate2's own CUDA probe so PyTorch is only needed for
    reporting the GPU name — not for the detection itself. Falls back
    to CPU int8 when no GPU is found.
    """
    get_cuda_device_count = cast(
        Callable[[], int], getattr(ctranslate2, "get_cuda_device_count")
    )
    cuda_count = get_cuda_device_count()

    if cuda_count > 0:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✅ GPU detected: {gpu_name}")
        except Exception:
            print(f"  ✅ {cuda_count} CUDA GPU(s) detected")
        return "cuda", "float16"

    if getattr(torch.version, "hip", None) and torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ℹ️  ROCm GPU detected for summarization: {gpu_name}")
        except Exception:
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
