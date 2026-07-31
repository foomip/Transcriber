"""
Tests for lib/transcription.py.

Hardware, filesystem, and Whisper boundaries are all mocked so these tests
run without a GPU, without model files, and without real audio.
"""

import os
import sys
from types import SimpleNamespace

import ctranslate2

from lib import transcription
from lib.hardware import parse_rocm_product_name


class NullProgressTimer:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        return None


# ---------------------------------------------------------------------------
# fmt_ts
# ---------------------------------------------------------------------------


def test_fmt_ts_formats_hours_minutes_and_fractional_seconds():
    assert transcription.fmt_ts(4.5) == "00:00:04.50"
    assert transcription.fmt_ts(723.1) == "00:12:03.10"
    assert transcription.fmt_ts(3723.45) == "01:02:03.45"


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------


def test_language_helpers_normalize_lookup_and_format_languages():
    assert transcription.normalize_language_code(" EN ") == "en"
    assert transcription.language_description(" PT ") == "Portuguese"
    assert transcription.language_description("zz") is None

    supported = transcription.format_supported_languages().splitlines()
    assert "  af - Afrikaans" in supported
    assert "  en - English" in supported
    assert supported == sorted(supported, key=lambda line: line.split(" - ", 1)[1].casefold())


# ---------------------------------------------------------------------------
# detect_device — NVIDIA CUDA path
# ---------------------------------------------------------------------------


def test_detect_device_uses_cuda_when_nvidia_gpu_detected(monkeypatch):
    """NVIDIA GPU → device='cuda', compute type from CTranslate2 query."""
    monkeypatch.setattr(transcription, "detect_gpu", lambda: ("cuda", "NVIDIA RTX 3060"))
    monkeypatch.setattr(
        ctranslate2, "get_supported_compute_types", lambda _device: ["float16", "int8"]
    )

    device, compute_type = transcription.detect_device()
    assert device == "cuda"
    assert compute_type == "float16"


def test_detect_device_cuda_falls_back_to_int8_float16_when_float16_unsupported(monkeypatch):
    monkeypatch.setattr(transcription, "detect_gpu", lambda: ("cuda", "NVIDIA GTX 750"))
    monkeypatch.setattr(
        ctranslate2, "get_supported_compute_types", lambda _device: ["int8_float16", "int8"]
    )

    device, compute_type = transcription.detect_device()
    assert device == "cuda"
    assert compute_type == "int8_float16"


def test_detect_device_cuda_falls_back_to_float32_when_only_float32_and_int8_available(monkeypatch):
    monkeypatch.setattr(transcription, "detect_gpu", lambda: ("cuda", "NVIDIA GPU"))
    monkeypatch.setattr(
        ctranslate2, "get_supported_compute_types", lambda _device: ["float32", "int8"]
    )

    device, compute_type = transcription.detect_device()
    assert device == "cuda"
    assert compute_type == "float32"


# ---------------------------------------------------------------------------
# detect_device — AMD ROCm path
# ---------------------------------------------------------------------------


def test_detect_device_uses_cuda_device_string_for_rocm_gpu(monkeypatch):
    """ROCm GPU → device='cuda' (CTranslate2 ROCm uses the CUDA device string),
    compute type chosen dynamically."""
    monkeypatch.setattr(transcription, "detect_gpu", lambda: ("rocm", "Radeon RX 7800 XT"))
    monkeypatch.setattr(
        ctranslate2, "get_supported_compute_types", lambda _device: ["float16", "int8"]
    )

    device, compute_type = transcription.detect_device()
    assert device == "cuda"
    assert compute_type == "float16"


def test_detect_device_rocm_selects_int8_float16_when_float16_unavailable(monkeypatch):
    monkeypatch.setattr(transcription, "detect_gpu", lambda: ("rocm", "AMD GPU"))
    monkeypatch.setattr(
        ctranslate2, "get_supported_compute_types", lambda _device: ["int8_float16", "int8"]
    )

    device, compute_type = transcription.detect_device()
    assert device == "cuda"
    assert compute_type == "int8_float16"


def test_detect_device_rocm_falls_back_to_float16_when_get_supported_compute_types_missing(
    monkeypatch,
):
    """Older CTranslate2 builds may not expose get_supported_compute_types."""
    monkeypatch.setattr(transcription, "detect_gpu", lambda: ("rocm", "AMD GPU"))
    # Remove the attribute entirely to simulate an older build.
    monkeypatch.delattr(ctranslate2, "get_supported_compute_types", raising=False)

    device, compute_type = transcription.detect_device()
    assert device == "cuda"
    assert compute_type == "float16"


def test_detect_device_rocm_falls_back_to_float16_when_get_supported_raises(monkeypatch):
    monkeypatch.setattr(transcription, "detect_gpu", lambda: ("rocm", "AMD GPU"))
    monkeypatch.setattr(
        ctranslate2, "get_supported_compute_types", lambda _device: (_ for _ in ()).throw(RuntimeError("no device"))
    )

    device, compute_type = transcription.detect_device()
    assert device == "cuda"
    assert compute_type == "float16"


# ---------------------------------------------------------------------------
# detect_device — CPU fallback
# ---------------------------------------------------------------------------


def test_detect_device_uses_cpu_when_no_gpu_is_available(monkeypatch):
    monkeypatch.setattr(transcription, "detect_gpu", lambda: ("cpu", "CPU"))

    device, compute_type = transcription.detect_device()
    assert device == "cpu"
    assert compute_type == "int8"


def test_detect_device_does_not_fall_back_to_cpu_for_rocm(monkeypatch):
    """ROCm must no longer fall back to CPU — it should use device='cuda'."""
    monkeypatch.setattr(transcription, "detect_gpu", lambda: ("rocm", "Radeon RX 6800 XT"))
    monkeypatch.setattr(
        ctranslate2, "get_supported_compute_types", lambda _device: ["float16"]
    )

    device, _ = transcription.detect_device()
    assert device == "cuda", "ROCm GPU should use CTranslate2 device='cuda', not 'cpu'"


# ---------------------------------------------------------------------------
# _best_gpu_compute_type
# ---------------------------------------------------------------------------


def test_best_gpu_compute_type_follows_preference_order(monkeypatch):
    monkeypatch.setattr(
        ctranslate2, "get_supported_compute_types", lambda _device: ["int8", "float16"]
    )
    assert transcription._best_gpu_compute_type() == "float16"


def test_best_gpu_compute_type_skips_unavailable_types(monkeypatch):
    # Only int8 and float32 available — int8_float16 and float16 absent.
    monkeypatch.setattr(
        ctranslate2, "get_supported_compute_types", lambda _device: ["int8", "float32"]
    )
    assert transcription._best_gpu_compute_type() == "float32"


def test_best_gpu_compute_type_returns_float16_when_api_absent(monkeypatch):
    monkeypatch.delattr(ctranslate2, "get_supported_compute_types", raising=False)
    assert transcription._best_gpu_compute_type() == "float16"


# ---------------------------------------------------------------------------
# parse_rocm_product_name (hardware utility)
# ---------------------------------------------------------------------------


def test_parse_rocm_product_name_supports_verbose_rocm_smi_output():
    output = """
====================    ROCm System Management Interface    ====================
======================================== Product Info ========================================
GPU[0]          : Card series: Radeon RX 7800 XT
GPU[0]          : Card model: 0x747e
"""

    assert parse_rocm_product_name(output) == "Radeon RX 7800 XT"


# ---------------------------------------------------------------------------
# transcribe_audio
# ---------------------------------------------------------------------------


def test_transcribe_audio_formats_segments_and_language_metadata(monkeypatch):
    created_models = []

    class FakeWhisperModel:
        def __init__(self, model_size, *, device, compute_type):
            self.model_size = model_size
            self.device = device
            self.compute_type = compute_type
            self.transcribe_calls = []
            created_models.append(self)

        def transcribe(self, audio, *, beam_size=5, language=None):
            self.transcribe_calls.append((audio, beam_size, language))
            segments = [
                SimpleNamespace(start=0.0, end=4.5, text=" Hello there "),
                SimpleNamespace(start=4.5, end=65.25, text="Next topic"),
            ]
            info = SimpleNamespace(language="en", language_probability=0.94)
            return segments, info

    monkeypatch.setattr(transcription, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(transcription, "ProgressTimer", NullProgressTimer)

    result = transcription.transcribe_audio(
        "meeting_20260528_1030.wav",
        "cpu",
        "int8",
        language="en",
    )

    assert created_models[0].model_size == transcription.WHISPER_MODEL_SIZE
    assert created_models[0].device == "cpu"
    assert created_models[0].compute_type == "int8"
    assert created_models[0].transcribe_calls == [("meeting_20260528_1030.wav", 5, "en")]
    assert result.lines == [
        "[00:00:00.00 -> 00:00:04.50]  Hello there",
        "[00:00:04.50 -> 00:01:05.25]  Next topic",
    ]
    assert result.requested_language == "en"
    assert result.requested_language_description == "English"
    assert result.detected_language == "en"
    assert result.detected_language_description == "English"
    assert result.language_probability == 0.94


def test_transcribe_audio_passes_cuda_device_for_rocm_gpu(monkeypatch):
    """When ROCm is detected, detect_device() returns device='cuda'.
    transcribe_audio must forward that string unchanged to WhisperModel."""
    created_models = []

    class FakeWhisperModel:
        def __init__(self, model_size, *, device, compute_type):
            created_models.append((device, compute_type))

        def transcribe(self, audio, *, beam_size=5, language=None):
            info = SimpleNamespace(language="en", language_probability=0.99)
            return [], info

    monkeypatch.setattr(transcription, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(transcription, "ProgressTimer", NullProgressTimer)

    # Simulate detect_device() having returned ("cuda", "float16") for a ROCm GPU.
    transcription.transcribe_audio("audio.wav", "cuda", "float16")

    assert created_models[0] == ("cuda", "float16")


# ---------------------------------------------------------------------------
# transcribe_audio_auto backend routing
# ---------------------------------------------------------------------------


def test_transcribe_audio_auto_cpu_profile_forces_faster_whisper_cpu(monkeypatch):
    calls = []
    monkeypatch.setenv("TRANSCRIBER_RUNTIME_PROFILE", "cpu")
    monkeypatch.setenv("TRANSCRIBER_TRANSCRIPTION_BACKEND", "auto")
    monkeypatch.setattr(
        transcription,
        "transcribe_audio",
        lambda audio, device, compute_type, language=None: calls.append(
            (audio, device, compute_type, language)
        )
        or transcription.TranscriptionResult(
            lines=[],
            requested_language=language,
            requested_language_description=None,
            detected_language="en",
            detected_language_description="English",
            language_probability=1.0,
            model_size="small",
        ),
    )

    transcription.transcribe_audio_auto("audio.wav", language="en")

    assert calls == [("audio.wav", "cpu", "int8", "en")]


def test_transcribe_audio_auto_uses_whisper_cpp_for_vulkan(monkeypatch, tmp_path):
    from lib import vulkan, whisper_cpp
    from lib.vulkan import VulkanDevice, VulkanProbeResult

    device = VulkanDevice(1, "Radeon", "discrete", 0x1002, 8, 7)
    monkeypatch.setenv("TRANSCRIBER_RUNTIME_PROFILE", "vulkan")
    monkeypatch.setenv("TRANSCRIBER_TRANSCRIPTION_BACKEND", "auto")
    monkeypatch.setattr(
        vulkan,
        "probe_vulkan",
        lambda: VulkanProbeResult(True, device, (device,)),
    )
    monkeypatch.setattr(whisper_cpp, "ensure_model", lambda: tmp_path / "model.bin")
    monkeypatch.setattr(
        whisper_cpp,
        "transcribe",
        lambda *args, **kwargs: whisper_cpp.WhisperCppResult(
            ((0.0, 1.25, "Hello"),), "en", 0.95
        ),
    )

    result = transcription.transcribe_audio_auto("audio.wav")

    assert result.engine == "whisper.cpp"
    assert result.lines == ["[00:00:00.00 -> 00:00:01.25]  Hello"]
    assert result.language_probability == 0.95


def test_transcribe_audio_auto_falls_back_once_after_whisper_cpp_error(monkeypatch):
    from lib import vulkan, whisper_cpp
    from lib.vulkan import VulkanDevice, VulkanProbeResult

    device = VulkanDevice(0, "GPU", "discrete", 0, None, None)
    fallback_calls = []
    monkeypatch.setenv("TRANSCRIBER_RUNTIME_PROFILE", "vulkan")
    monkeypatch.setattr(
        vulkan,
        "probe_vulkan",
        lambda: VulkanProbeResult(True, device, (device,)),
    )
    monkeypatch.setattr(whisper_cpp, "ensure_model", lambda: None)
    monkeypatch.setattr(
        whisper_cpp,
        "transcribe",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            whisper_cpp.WhisperCppError("device lost")
        ),
    )
    monkeypatch.setattr(
        transcription,
        "transcribe_audio",
        lambda audio, device, compute_type, language=None: fallback_calls.append(
            (device, compute_type)
        )
        or transcription.TranscriptionResult(
            lines=[],
            requested_language=None,
            requested_language_description=None,
            detected_language="en",
            detected_language_description="English",
            language_probability=1.0,
            model_size="small",
        ),
    )

    result = transcription.transcribe_audio_auto("audio.wav")

    assert result.engine == "faster-whisper"
    assert fallback_calls == [("cpu", "int8")]
