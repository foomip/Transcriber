"""whisper.cpp transcription adapter used by the Vulkan Docker profile."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lib.vulkan import VulkanDevice

WHISPER_CPP_BIN_ENV = "TRANSCRIBER_WHISPER_CPP_BIN"
WHISPER_CPP_MODEL_PATH_ENV = "TRANSCRIBER_WHISPER_CPP_MODEL_PATH"
WHISPER_CPP_CACHE_DIR_ENV = "TRANSCRIBER_WHISPER_CPP_CACHE_DIR"
DEFAULT_WHISPER_CPP_BIN = "whisper-cli"
DEFAULT_MODEL_REPO = "ggerganov/whisper.cpp"
DEFAULT_MODEL_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"
DEFAULT_MODEL_FILENAME = "ggml-small.bin"
DEFAULT_MODEL_SHA256 = "edd29d67e70b000132af65205b99bb774b77abc13d10103e14f80ce2242913e1"
_MAX_JSON_BYTES = 64 * 1024 * 1024


class WhisperCppError(RuntimeError):
    """An expected whisper.cpp preparation or inference failure."""


@dataclass(frozen=True)
class WhisperCppResult:
    segments: tuple[tuple[float, float, str], ...]
    detected_language: str
    language_probability: float


def _cache_dir() -> Path:
    configured = os.environ.get(WHISPER_CPP_CACHE_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return xdg_cache / "transcriber" / "whisper"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_marker(marker_path: Path) -> None:
    temporary = marker_path.with_suffix(marker_path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(DEFAULT_MODEL_SHA256 + "\n", encoding="utf-8")
    os.replace(temporary, marker_path)


def _verify_default_model(model_path: Path) -> None:
    marker_path = model_path.with_suffix(model_path.suffix + ".sha256")
    try:
        if marker_path.read_text(encoding="utf-8").strip() == DEFAULT_MODEL_SHA256:
            return
    except OSError:
        pass

    actual = _sha256(model_path)
    if actual != DEFAULT_MODEL_SHA256:
        raise WhisperCppError(
            f"whisper.cpp model checksum mismatch for {model_path}: "
            f"expected {DEFAULT_MODEL_SHA256}, got {actual}"
        )
    _write_marker(marker_path)


def ensure_model() -> Path:
    configured_path = os.environ.get(WHISPER_CPP_MODEL_PATH_ENV)
    if configured_path:
        model_path = Path(configured_path).expanduser()
        if not model_path.is_file():
            raise WhisperCppError(f"Configured whisper.cpp model was not found: {model_path}")
        return model_path

    model_dir = _cache_dir()
    model_path = model_dir / DEFAULT_MODEL_FILENAME
    if model_path.is_file():
        _verify_default_model(model_path)
        return model_path

    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(
                repo_id=DEFAULT_MODEL_REPO,
                filename=DEFAULT_MODEL_FILENAME,
                revision=DEFAULT_MODEL_REVISION,
                local_dir=str(model_dir),
            )
        )
    except Exception as exc:  # Hugging Face exposes multiple transport exceptions.
        raise WhisperCppError(
            f"Could not download {DEFAULT_MODEL_FILENAME} from {DEFAULT_MODEL_REPO}: {exc}"
        ) from exc

    if downloaded.resolve() != model_path.resolve():
        try:
            os.replace(downloaded, model_path)
        except OSError as exc:
            raise WhisperCppError(f"Could not install whisper.cpp model: {exc}") from exc
    if not model_path.is_file():
        raise WhisperCppError(f"Downloaded whisper.cpp model is missing: {model_path}")
    _verify_default_model(model_path)
    return model_path


def _normalise_audio(audio_path: str, output_path: Path) -> None:
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                audio_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise WhisperCppError("FFmpeg is required by the whisper.cpp backend") from exc
    except OSError as exc:
        raise WhisperCppError(f"Could not start FFmpeg: {exc}") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise WhisperCppError(f"FFmpeg could not normalise the audio: {detail}")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WhisperCppError(f"whisper.cpp JSON field {field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise WhisperCppError(f"whisper.cpp JSON field {field} must be finite")
    return parsed


def parse_result(path: Path) -> WhisperCppResult:
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            raise WhisperCppError("whisper.cpp JSON output exceeded the size limit")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WhisperCppError("whisper.cpp did not produce JSON output") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise WhisperCppError(f"Could not read whisper.cpp JSON output: {exc}") from exc

    if not isinstance(payload, dict):
        raise WhisperCppError("whisper.cpp JSON output must be an object")
    result = payload.get("result")
    raw_segments = payload.get("transcription")
    if not isinstance(result, dict) or not isinstance(raw_segments, list):
        raise WhisperCppError("whisper.cpp JSON output has an unsupported schema")

    language = result.get("language")
    if not isinstance(language, str) or not language.strip():
        raise WhisperCppError("whisper.cpp JSON did not contain a detected language")
    probability = _finite_number(result.get("language_probability"), "language_probability")
    if not 0.0 <= probability <= 1.0:
        raise WhisperCppError("whisper.cpp language probability must be between 0 and 1")

    segments: list[tuple[float, float, str]] = []
    previous_start = 0.0
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise WhisperCppError(f"whisper.cpp segment {index} must be an object")
        offsets = raw_segment.get("offsets")
        text = raw_segment.get("text")
        if not isinstance(offsets, dict) or not isinstance(text, str):
            raise WhisperCppError(f"whisper.cpp segment {index} has an unsupported schema")
        start_ms = _finite_number(offsets.get("from"), f"segment {index} start")
        end_ms = _finite_number(offsets.get("to"), f"segment {index} end")
        start = start_ms / 1000.0
        end = end_ms / 1000.0
        if start < 0 or end < start or start < previous_start:
            raise WhisperCppError(f"whisper.cpp segment {index} has invalid timestamps")
        previous_start = start
        segments.append((start, end, text.strip()))

    return WhisperCppResult(tuple(segments), language.strip().lower(), probability)


def transcribe(
    audio_path: str,
    *,
    model_path: Path,
    device: VulkanDevice,
    language: str | None,
) -> WhisperCppResult:
    whisper_bin = os.environ.get(WHISPER_CPP_BIN_ENV, DEFAULT_WHISPER_CPP_BIN)
    with tempfile.TemporaryDirectory(prefix="transcriber-whisper-") as temp_dir:
        temp_path = Path(temp_dir)
        normalised_audio = temp_path / "audio.wav"
        output_prefix = temp_path / "transcript"
        _normalise_audio(audio_path, normalised_audio)

        command = [
            whisper_bin,
            "--model",
            str(model_path),
            "--file",
            str(normalised_audio),
            "--language",
            language or "auto",
            "--beam-size",
            "5",
            "--output-json-full",
            "--output-file",
            str(output_prefix),
            "--device",
            str(device.index),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise WhisperCppError(f"whisper.cpp executable was not found: {whisper_bin}") from exc
        except OSError as exc:
            raise WhisperCppError(f"Could not start whisper.cpp: {exc}") from exc

        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit status {completed.returncode}"
            raise WhisperCppError(f"whisper.cpp transcription failed: {detail}")
        return parse_result(output_prefix.with_suffix(".json"))
