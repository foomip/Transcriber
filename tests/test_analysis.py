import os
import subprocess
import sys
import types
from unittest.mock import MagicMock

import pytest

from lib import analysis
import lib.analysis.backend as analysis_backend


@pytest.fixture(autouse=True)
def native_runtime_profile(monkeypatch):
    """Keep backend unit tests independent of the enclosing Docker profile."""
    monkeypatch.setenv("TRANSCRIBER_RUNTIME_PROFILE", "native")


@pytest.fixture(autouse=True)
def cpu_only_llama_cpp_build(monkeypatch):
    """Default tests to a CPU-only llama.cpp build (no GPU-backend routing).

    The dev machine may have a GPU-enabled llama.cpp build installed; without
    this, the hardware-routing tests would depend on the host's build.
    """
    monkeypatch.setattr(analysis_backend, "_llama_cpp_compiled_libs", lambda: ())


BASE_META = {
    "title": "Meeting Recording",
    "date": "May 28, 2026",
    "time": "09:30 AM",
    "duration": "12 minutes",
    "requested_language": "English (en)",
    "detected_language": "English (en)",
    "language_probability": "94%",
    "transcription_model": "Faster-Whisper small",
}


class FakeLlamaCppModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_detect_gpu_nvidia(monkeypatch):
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA RTX 3060"
    monkeypatch.setitem(sys.modules, "pynvml", mock_pynvml)
    monkeypatch.setattr(os.path, "exists", lambda path: path != "/dev/kfd")
    
    kind, name = analysis_backend._detect_gpu()
    assert kind == "cuda"
    assert name == "NVIDIA RTX 3060"


def test_detect_gpu_amd(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: path == "/dev/kfd")

    def fake_check_output(args, **kwargs):
        if args == ["rocm-smi", "--showproductname"]:
            return "GPU  Product Name\n0    Navi 21"
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    kind, name = analysis._detect_gpu()
    assert kind == "rocm"
    assert name == "Navi 21"


def test_detect_gpu_amd_parses_verbose_rocm_smi_output(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: path == "/dev/kfd")

    def fake_check_output(args, **kwargs):
        if args == ["rocm-smi", "--showproductname"]:
            return """
====================    ROCm System Management Interface    ====================
======================================== Product Info ========================================
GPU[0]          : Card series: Radeon RX 7800 XT
GPU[0]          : Card model: 0x747e
"""
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    kind, name = analysis._detect_gpu()
    assert kind == "rocm"
    assert name == "Radeon RX 7800 XT"


def test_detect_gpu_cpu(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    # Inject stub pynvml module that raises on usage
    stub = types.SimpleNamespace(
        nvmlInit=lambda: (_ for _ in ()).throw(RuntimeError("NVML not available")),
        nvmlDeviceGetCount=lambda: (_ for _ in ()).throw(RuntimeError("NVML not available")),
    )
    monkeypatch.setitem(sys.modules, "pynvml", stub)

    kind, name = analysis._detect_gpu()
    assert kind == "cpu"
    assert name == "CPU"


def test_nvidia_free_vram_bytes(monkeypatch):
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = types.SimpleNamespace(free=1024 * 1024 * 1024, total=2048 * 1024 * 1024)
    monkeypatch.setitem(sys.modules, "pynvml", mock_pynvml)
    
    assert analysis_backend._nvidia_free_vram_bytes() == 1024 * 1024 * 1024


def test_amd_free_vram_bytes(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: True)  # Not used by this func but good practice

    def fake_check_output(args, **kwargs):
        if args == ["rocm-smi", "--showmeminfo", "vram"]:
            return "GPU  VRAM Total  VRAM Used  VRAM %\n0    12288MB    2288MB     18%"
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    assert analysis._amd_free_vram_bytes() == (12288 - 2288) * 1024 * 1024



def test_amd_free_vram_bytes_parses_verbose_byte_output(monkeypatch):
    def fake_check_output(args, **kwargs):
        if args == ["rocm-smi", "--showmeminfo", "vram"]:
            return """
====================    ROCm System Management Interface    ====================
================================ Memory Usage (Bytes) ================================
GPU[0]          : VRAM Total Memory (B): 17163091968
GPU[0]          : VRAM Total Used Memory (B): 1073741824
================================================================================
"""
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    assert analysis._amd_free_vram_bytes() == 16089350144



def test_amd_free_vram_bytes_uses_amdsmi_when_available(monkeypatch):
    fake_amdsmi = types.SimpleNamespace(
        amdsmi_init=MagicMock(),
        amdsmi_get_processor_handles=lambda: ["gpu0"],
        amdsmi_get_gpu_vram_usage=lambda handle: {
            "vram_total": 16 * analysis._GIB,
            "vram_used": 2 * analysis._GIB,
        },
        amdsmi_shut_down=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "amdsmi", fake_amdsmi)

    assert analysis._amd_free_vram_bytes() == 14 * analysis._GIB
    fake_amdsmi.amdsmi_init.assert_called_once_with()
    fake_amdsmi.amdsmi_shut_down.assert_called_once_with()



def test_amd_free_vram_bytes_falls_back_to_sysfs_bytes(tmp_path, monkeypatch):
    sysfs_dir = tmp_path / "card1" / "device"
    sysfs_dir.mkdir(parents=True)
    (sysfs_dir / "mem_info_vram_total").write_text(str(16 * analysis._GIB), encoding="utf-8")
    (sysfs_dir / "mem_info_vram_used").write_text(str(2 * analysis._GIB), encoding="utf-8")

    monkeypatch.delitem(sys.modules, "amdsmi", raising=False)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(analysis_backend.glob, "glob", lambda pattern: [str(sysfs_dir / "mem_info_vram_total")])

    assert analysis._amd_free_vram_bytes() == 14 * analysis._GIB


def test_detect_analysis_backend_returns_llama_cpp_on_cuda(monkeypatch):
    monkeypatch.setattr(analysis_backend, "_detect_gpu", lambda: ("cuda", "RTX 3060"))
    monkeypatch.setattr(analysis_backend, "_llama_cpp_is_available", lambda: True)
    monkeypatch.setattr(analysis_backend, "_llama_cpp_model_exists", lambda path: True)
    monkeypatch.setattr(analysis_backend, "_nvidia_free_vram_bytes", lambda: 8 * analysis._GIB)

    backend = analysis.detect_analysis_backend()

    assert backend.name == "cuda"
    assert backend.device_name == "RTX 3060"
    assert "llama.cpp" in backend.notes[0]
    assert backend.max_new_tokens == 2048


def test_detect_analysis_backend_returns_llama_cpp_on_rocm(monkeypatch):
    monkeypatch.setattr(analysis_backend, "_detect_gpu", lambda: ("rocm", "Navi 21"))
    monkeypatch.setattr(analysis_backend, "_llama_cpp_is_available", lambda: True)
    monkeypatch.setattr(analysis_backend, "_llama_cpp_model_exists", lambda path: True)
    monkeypatch.setattr(analysis_backend, "_amd_free_vram_bytes", lambda: 8 * analysis._GIB)

    backend = analysis.detect_analysis_backend()

    assert backend.name == "rocm"
    assert backend.device_name == "Navi 21"
    assert "llama.cpp" in backend.notes[0]
    assert backend.max_new_tokens == 2048


def test_detect_analysis_backend_preserves_rocm_kind_when_name_is_generic(monkeypatch):
    monkeypatch.setattr(analysis_backend, "_detect_gpu", lambda: ("rocm", "Radeon RX 7800 XT"))
    monkeypatch.setattr(analysis_backend, "_llama_cpp_is_available", lambda: True)
    monkeypatch.setattr(analysis_backend, "_llama_cpp_model_exists", lambda path: True)
    monkeypatch.setattr(analysis_backend, "_amd_free_vram_bytes", lambda: 8 * analysis._GIB)

    backend = analysis.detect_analysis_backend()

    assert backend.name == "rocm"
    assert backend.device_name == "Radeon RX 7800 XT"


def test_detect_analysis_backend_returns_llama_cpp_on_cpu(monkeypatch):
    monkeypatch.setattr(analysis_backend, "_detect_gpu", lambda: ("cpu", "CPU"))
    monkeypatch.setattr(analysis_backend, "_llama_cpp_is_available", lambda: True)
    monkeypatch.setattr(analysis_backend, "_llama_cpp_model_exists", lambda path: True)

    backend = analysis.detect_analysis_backend()

    assert backend.name == "cpu"
    assert backend.device_name == "CPU"
    assert "llama.cpp" in backend.notes[0]
    assert backend.max_new_tokens == 2048


def test_detect_analysis_backend_sizes_context_from_actual_transcript(tmp_path, monkeypatch):
    model_path = tmp_path / analysis.DEFAULT_LLAMA_CPP_MODEL_FILENAME
    model_path.write_bytes(b"0" * 2 * analysis._GIB)

    monkeypatch.setenv("TRANSCRIBER_MAX_TRANSCRIPT_CHARS", "250000")
    monkeypatch.delenv(analysis.LLAMA_CPP_CONTEXT_SIZE_ENV, raising=False)
    monkeypatch.setattr(analysis_backend, "_detect_gpu", lambda: ("cuda", "RTX 3060"))
    monkeypatch.setattr(analysis_backend, "_llama_cpp_is_available", lambda: True)
    monkeypatch.setattr(analysis_backend, "_llama_cpp_model_path", lambda: str(model_path))
    monkeypatch.setattr(analysis_backend, "_ensure_llama_cpp_model", lambda path: path)
    monkeypatch.setattr(analysis_backend, "_nvidia_free_vram_bytes", lambda: 12 * analysis._GIB)
    monkeypatch.setattr(analysis_backend, "_llama_cpp_gpu_offload_supported", lambda: True)

    backend = analysis.detect_analysis_backend(transcript_chars=12_000)

    assert backend.model_kwargs["n_ctx"] < 10_000
    assert backend.model_kwargs["n_gpu_layers"] > 0


def test_llama_cpp_gpu_layers_respects_explicit_override(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"0" * analysis._GIB)
    monkeypatch.setenv(analysis.LLAMA_CPP_GPU_LAYERS_ENV, "9")

    gpu_layers, notes = analysis._llama_cpp_gpu_layers(str(model_path))

    assert gpu_layers == 9
    assert analysis.LLAMA_CPP_GPU_LAYERS_ENV in notes[0]


def test_llama_cpp_gpu_layers_uses_safe_vram_budget(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"0" * 8 * analysis._GIB)
    monkeypatch.delenv(analysis.LLAMA_CPP_GPU_LAYERS_ENV, raising=False)
    monkeypatch.setenv(analysis.LLAMA_CPP_LAYER_COUNT_ENV, "32")
    monkeypatch.setenv(analysis.LLAMA_CPP_CONTEXT_SIZE_ENV, "4096")
    monkeypatch.setenv(analysis.LLAMA_CPP_GPU_HEADROOM_ENV, "2")
    # Pin to f16 so the layer count reflects the full (unquantized) KV cache.
    monkeypatch.setenv(analysis.LLAMA_CPP_KV_CACHE_TYPE_ENV, "f16")

    monkeypatch.setattr(analysis_backend, "_detect_gpu", lambda: ("cuda", "RTX 3060"))
    monkeypatch.setattr(analysis_backend, "_nvidia_free_vram_bytes", lambda: 6 * analysis._GIB)

    gpu_layers, notes = analysis._llama_cpp_gpu_layers(str(model_path))

    assert gpu_layers == 11
    assert "11/32" in notes[0]



def test_llama_cpp_gpu_layers_reports_rocm_offload_like_cuda(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"0" * 8 * analysis._GIB)
    monkeypatch.delenv(analysis.LLAMA_CPP_GPU_LAYERS_ENV, raising=False)
    monkeypatch.setenv(analysis.LLAMA_CPP_LAYER_COUNT_ENV, "32")
    monkeypatch.setenv(analysis.LLAMA_CPP_CONTEXT_SIZE_ENV, "4096")
    monkeypatch.setenv(analysis.LLAMA_CPP_GPU_HEADROOM_ENV, "2")
    # Pin to f16 so the layer count reflects the full (unquantized) KV cache.
    monkeypatch.setenv(analysis.LLAMA_CPP_KV_CACHE_TYPE_ENV, "f16")

    monkeypatch.setattr(analysis_backend, "_detect_gpu", lambda: ("rocm", "AMD Radeon RX 6800"))
    monkeypatch.setattr(analysis_backend, "_amd_free_vram_bytes", lambda: 6 * analysis._GIB)

    gpu_layers, notes = analysis._llama_cpp_gpu_layers(str(model_path))

    assert gpu_layers == 11
    assert "11/32" in notes[0]


def test_kv_cache_config_defaults_to_q8_0(monkeypatch):
    monkeypatch.delenv(analysis.LLAMA_CPP_KV_CACHE_TYPE_ENV, raising=False)

    cfg = analysis_backend._kv_cache_config()

    assert cfg["name"] == "q8_0"
    assert cfg["type_k"] == 8
    assert cfg["type_v"] == 8
    assert cfg["flash_attn"] is True
    assert cfg["kv_factor"] < 1.0


def test_kv_cache_config_q8_0_enables_flash_attn(monkeypatch):
    monkeypatch.setenv(analysis.LLAMA_CPP_KV_CACHE_TYPE_ENV, "q8_0")

    cfg = analysis_backend._kv_cache_config()

    assert cfg["name"] == "q8_0"
    assert cfg["type_k"] == 8
    assert cfg["type_v"] == 8
    assert cfg["flash_attn"] is True
    assert cfg["kv_factor"] < 1.0


def test_kv_cache_config_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(analysis.LLAMA_CPP_KV_CACHE_TYPE_ENV, "fp8")

    cfg = analysis_backend._kv_cache_config()

    assert cfg["name"] == "q8_0"
    assert cfg["flash_attn"] is True


def test_llama_cpp_model_kwargs_includes_kv_cache_type(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"0" * analysis._GIB)
    monkeypatch.setenv(analysis.LLAMA_CPP_KV_CACHE_TYPE_ENV, "q8_0")
    monkeypatch.setattr(analysis_backend, "_llama_cpp_gpu_layers", lambda *a, **k: (0, ()))

    kwargs, notes = analysis_backend._llama_cpp_model_kwargs(str(model_path))

    assert kwargs["flash_attn"] is True
    assert kwargs["type_k"] == 8
    assert kwargs["type_v"] == 8
    assert any("q8_0" in note for note in notes)


def test_llama_cpp_model_kwargs_defaults_to_q8_0_kv_cache(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"0" * analysis._GIB)
    monkeypatch.delenv(analysis.LLAMA_CPP_KV_CACHE_TYPE_ENV, raising=False)
    monkeypatch.setattr(analysis_backend, "_llama_cpp_gpu_layers", lambda *a, **k: (0, ()))

    kwargs, _notes = analysis_backend._llama_cpp_model_kwargs(str(model_path))

    assert kwargs["flash_attn"] is True
    assert kwargs["type_k"] == 8
    assert kwargs["type_v"] == 8


def test_llama_cpp_gpu_layers_applies_kv_cache_quant_factor(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"0" * 8 * analysis._GIB)
    monkeypatch.delenv(analysis.LLAMA_CPP_GPU_LAYERS_ENV, raising=False)
    monkeypatch.setenv(analysis.LLAMA_CPP_LAYER_COUNT_ENV, "32")
    monkeypatch.setenv(analysis.LLAMA_CPP_CONTEXT_SIZE_ENV, "4096")
    monkeypatch.setenv(analysis.LLAMA_CPP_GPU_HEADROOM_ENV, "2")
    monkeypatch.setattr(analysis_backend, "_detect_gpu", lambda: ("cuda", "RTX 3060"))
    monkeypatch.setattr(analysis_backend, "_nvidia_free_vram_bytes", lambda: 6 * analysis._GIB)

    # f16 reserves the full KV cache; q8_0 reserves less, so more layers fit.
    monkeypatch.setenv(analysis.LLAMA_CPP_KV_CACHE_TYPE_ENV, "f16")
    f16_layers, _ = analysis._llama_cpp_gpu_layers(str(model_path))

    monkeypatch.setenv(analysis.LLAMA_CPP_KV_CACHE_TYPE_ENV, "q8_0")
    q8_layers, _ = analysis._llama_cpp_gpu_layers(str(model_path))

    assert q8_layers > f16_layers


def test_default_llama_cpp_model_points_at_existing_repo_file():
    # The ggml-org repo ships Q4_0 (there is no Q4_K_M build), and the mtp-*
    # files are auxiliary multi-token-prediction modules, not the main model.
    assert analysis.DEFAULT_LLAMA_CPP_MODEL_REPO_ID == "ggml-org/gemma-4-E4B-it-GGUF"
    assert analysis.DEFAULT_LLAMA_CPP_MODEL_FILENAME == "gemma-4-E4B-it-Q4_0.gguf"


def test_ensure_llama_cpp_model_downloads_from_huggingface(tmp_path, monkeypatch):
    model_path = tmp_path / analysis.DEFAULT_LLAMA_CPP_MODEL_FILENAME
    downloaded_path = tmp_path / "downloaded.gguf"
    downloaded_path.write_bytes(b"model")
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )

    result = analysis._ensure_llama_cpp_model(str(model_path))

    assert result == str(model_path)
    assert model_path.read_bytes() == b"model"
    assert calls == [
        {
            "repo_id": analysis.DEFAULT_LLAMA_CPP_MODEL_REPO_ID,
            "filename": analysis.DEFAULT_LLAMA_CPP_MODEL_FILENAME,
            "local_dir": str(tmp_path),
        }
    ]


def test_build_user_message_includes_sections_metadata_and_transcript():
    message = analysis._build_user_message("Apollo roadmap and budget discussion", BASE_META)

    for heading, instruction in analysis.SUMMARY_TASKS:
        assert heading in message
        assert instruction in message
    assert "Title: Meeting Recording" in message
    assert "Date: May 28, 2026" in message
    assert "Requested transcription language: English (en)" in message
    assert "Apollo roadmap and budget discussion" in message


def test_build_compact_user_message_includes_sections_metadata_and_transcript():
    message = analysis._build_compact_user_message(
        "Apollo roadmap and budget discussion",
        BASE_META,
    )

    for heading, _instruction in analysis.SUMMARY_TASKS:
        assert heading in message
    assert "Apollo roadmap and budget discussion" in message


def test_query_llama_cpp_returns_completion_text():
    model = FakeLlamaCppModel({"choices": [{"text": "  ## Executive Summary\nApollo update  "}]})

    response = analysis._query_llama_cpp(
        model,
        "Prompt",
        max_new_tokens=128,
    )

    assert response == "## Executive Summary\nApollo update"
    assert model.calls == [
        {
            "prompt": "Prompt",
            "max_tokens": 128,
            "temperature": 0,
        }
    ]


def test_required_llama_cpp_context_size_scales_with_transcript_budget(monkeypatch):
    monkeypatch.delenv(analysis.LLAMA_CPP_CONTEXT_SIZE_ENV, raising=False)

    monkeypatch.setenv("TRANSCRIBER_MAX_TRANSCRIPT_CHARS", "28000")
    small = analysis._required_llama_cpp_context_size()

    monkeypatch.setenv("TRANSCRIBER_MAX_TRANSCRIPT_CHARS", "120000")
    large = analysis._required_llama_cpp_context_size()

    # Even the smallest transcript needs more than the legacy 4096 window.
    assert small > analysis.DEFAULT_LLAMA_CPP_CONTEXT_SIZE
    assert large > small
    assert large <= analysis.DEFAULT_LLAMA_CPP_MAX_CONTEXT_SIZE
    # Window must hold the whole transcript prompt plus the generation budget.
    expected_min = 28000 / analysis.LLAMA_CPP_CHARS_PER_TOKEN + 2048
    assert small >= expected_min


def test_required_llama_cpp_context_size_uses_actual_transcript_when_known():
    context_size = analysis._required_llama_cpp_context_size(transcript_chars=12_000)

    assert context_size < 10_000
    assert context_size >= 12_000 / analysis.LLAMA_CPP_CHARS_PER_TOKEN + 2048


def test_required_llama_cpp_context_size_respects_env_override(monkeypatch):
    monkeypatch.setenv(analysis.LLAMA_CPP_CONTEXT_SIZE_ENV, "8192")
    monkeypatch.setenv("TRANSCRIBER_MAX_TRANSCRIPT_CHARS", "250000")

    assert analysis._llama_cpp_context_size() == 8192


class FakeTokenizingLlamaCppModel(FakeLlamaCppModel):
    def __init__(self, response, context_size):
        super().__init__(response)
        self._context_size = context_size

    def n_ctx(self):
        return self._context_size

    def tokenize(self, text):
        # One byte per token keeps the arithmetic easy to assert.
        return list(text)

    def detokenize(self, tokens):
        return bytes(tokens)


def test_generate_summaries_passes_transcript_length_to_backend(monkeypatch):
    seen = {}

    def fake_detect_analysis_backend(transcript_chars=None):
        seen["transcript_chars"] = transcript_chars
        return analysis.AnalysisBackend(
            name="cpu",
            device_name="CPU",
            model_id="fake.gguf",
            model_kwargs={"model_path": "fake.gguf", "n_ctx": 4096, "n_gpu_layers": 0},
        )

    monkeypatch.setattr(analysis, "detect_analysis_backend", fake_detect_analysis_backend)
    monkeypatch.setattr(
        analysis,
        "_generate_report_with_llama_cpp",
        lambda backend, transcript_body, meta: "## Executive Summary\nOK\n\n## Detailed Summary\nOK\n\n## Action Items\nNone explicitly stated.\n\n## Key Decisions\nNone explicitly stated.\n\n## Topics Discussed\nTesting.",
    )
    monkeypatch.setattr(analysis, "_validate_grounding", lambda generated_report, transcript_body: None)

    sections = analysis.generate_summaries("short transcript", BASE_META)

    assert seen["transcript_chars"] == len("short transcript")
    assert sections[0][0] == "## Executive Summary"


def test_fit_prompt_to_context_trims_oversized_prompt():
    model = FakeTokenizingLlamaCppModel({"choices": [{"text": "ok"}]}, context_size=200)
    prompt = "x" * 500

    trimmed, max_new_tokens = analysis._fit_prompt_to_context(
        model, prompt, max_new_tokens=64
    )

    allowed = 200 - 64 - analysis.model.LLAMA_CPP_CONTEXT_GUARD_MARGIN
    assert len(trimmed) == allowed
    assert max_new_tokens == 64


def test_fit_prompt_to_context_keeps_small_prompt_untouched():
    model = FakeTokenizingLlamaCppModel({"choices": [{"text": "ok"}]}, context_size=4096)
    prompt = "short prompt"

    trimmed, max_new_tokens = analysis._fit_prompt_to_context(
        model, prompt, max_new_tokens=128
    )

    assert trimmed == prompt
    assert max_new_tokens == 128


def test_fit_prompt_to_context_noops_without_tokenizer():
    model = FakeLlamaCppModel({"choices": [{"text": "ok"}]})
    prompt = "x" * 10_000

    trimmed, max_new_tokens = analysis._fit_prompt_to_context(
        model, prompt, max_new_tokens=128
    )

    assert trimmed == prompt
    assert max_new_tokens == 128


def test_content_words_filters_stop_words_and_short_tokens():
    words = analysis._content_words("This is a short demo about Apollo budgets, UX, and releases.")

    assert "this" not in words
    assert "about" not in words
    assert "demo" in words
    assert "apollo" in words
    assert "budgets" in words
    assert "releases" in words
    assert "ux" not in words


def test_validate_grounding_accepts_grounded_report():
    transcript = "Apollo budget roadmap release owner timeline " * 8
    generated = "Apollo budget roadmap release owner timeline summary"

    analysis._validate_grounding(generated, transcript)


def test_validate_grounding_rejects_unrelated_report():
    transcript = "Apollo budget roadmap release owner timeline " * 8
    generated = "Volcano astronomy glacier museum orchestra festival unrelated claims"

    with pytest.raises(analysis.AnalysisGroundingError):
        analysis._validate_grounding(generated, transcript)


def test_parse_report_sections_returns_expected_order_and_missing_fallback():
    generated = """
 ## Executive Summary
The team reviewed the Apollo budget.

## Action Items
Dana will update the roadmap.
"""

    sections = analysis._parse_report_sections(generated)

    assert [heading for heading, _text in sections] == [heading for heading, _task in analysis.SUMMARY_TASKS]
    # Executive Summary is present in generated text
    assert sections[0] == ("## Executive Summary", "The team reviewed the Apollo budget.")
    # Action Items is present in generated text
    assert sections[1] == ("## Action Items", "Dana will update the roadmap.")
    # Key Decisions is missing — should get fallback
    assert sections[2][0] == "## Key Decisions"
    assert "No information was generated for this section." in sections[2][1]
    # Risks & Open Questions is missing — should get fallback
    assert sections[3][0] == "## Risks & Open Questions"
    assert "No information was generated for this section." in sections[3][1]
    # Detailed Summary is missing — should get fallback
    assert sections[4][0] == "## Detailed Summary"
    assert "No information was generated for this section." in sections[4][1]


def test_detect_analysis_backend_uses_vulkan_profile(tmp_path, monkeypatch):
    from lib.vulkan import VulkanDevice

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"0" * 1024)
    device = VulkanDevice(
        index=0,
        name="Intel Arc",
        device_type="discrete",
        vendor_id=0x8086,
        heap_size_bytes=8 * analysis._GIB,
        heap_budget_bytes=6 * analysis._GIB,
    )
    monkeypatch.setenv("TRANSCRIBER_RUNTIME_PROFILE", "vulkan")
    monkeypatch.setattr(analysis_backend, "_vulkan_device", lambda: device)
    monkeypatch.setattr(analysis_backend, "_llama_cpp_is_available", lambda: True)
    monkeypatch.setattr(analysis_backend, "_ensure_llama_cpp_model", lambda path: str(model_path))
    monkeypatch.setattr(analysis_backend, "_llama_cpp_model_path", lambda: str(model_path))
    monkeypatch.setattr(analysis_backend, "_llama_cpp_gpu_offload_supported", lambda: True)
    monkeypatch.setenv(analysis.LLAMA_CPP_GPU_HEADROOM_ENV, "1")
    monkeypatch.setenv(analysis.LLAMA_CPP_CONTEXT_SIZE_ENV, "4096")

    backend = analysis.detect_analysis_backend(transcript_chars=1000)

    assert backend.name == "vulkan"
    assert backend.device_name == "Intel Arc"
    assert backend.model_kwargs["n_gpu_layers"] > 0


def test_cpu_profile_forces_zero_gpu_layers(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    monkeypatch.setenv("TRANSCRIBER_RUNTIME_PROFILE", "cpu")
    monkeypatch.setenv(analysis.LLAMA_CPP_GPU_LAYERS_ENV, "42")

    layers, notes = analysis._llama_cpp_gpu_layers(str(model_path))

    assert layers == 0
    assert "system RAM" in notes[0]


def test_llama_cpp_gpu_backend_detects_vulkan_build(monkeypatch):
    monkeypatch.setattr(
        analysis_backend,
        "_llama_cpp_compiled_libs",
        lambda: ("libggml-base.so", "libggml-cpu.so", "libggml-vulkan.so"),
    )

    assert analysis_backend._llama_cpp_gpu_backend() == "vulkan"


def test_llama_cpp_gpu_backend_detects_cuda_build(monkeypatch):
    monkeypatch.setattr(
        analysis_backend,
        "_llama_cpp_compiled_libs",
        lambda: ("libggml-base.so", "libggml-cpu.so", "libggml-cuda.so"),
    )

    assert analysis_backend._llama_cpp_gpu_backend() == "cuda"


def test_llama_cpp_gpu_backend_none_for_cpu_only_build():
    # The autouse fixture stubs _llama_cpp_compiled_libs to () (no GPU backend).
    assert analysis_backend._llama_cpp_gpu_backend() is None


def test_detect_analysis_hardware_routes_vulkan_build_to_vulkan_device(monkeypatch):
    from lib.vulkan import VulkanDevice

    device = VulkanDevice(
        index=1, name="NVIDIA GeForce RTX 3060", device_type="discrete",
        vendor_id=0x10DE, heap_size_bytes=12 * analysis._GIB,
        heap_budget_bytes=8 * analysis._GIB,
    )
    monkeypatch.setattr(analysis_backend, "_llama_cpp_gpu_backend", lambda: "vulkan")
    monkeypatch.setattr(analysis_backend, "_vulkan_device", lambda: device)

    kind, name = analysis_backend._detect_analysis_hardware()

    assert kind == "vulkan"
    assert name == "NVIDIA GeForce RTX 3060"


def test_detect_analysis_hardware_falls_back_to_detect_gpu_for_cuda_build(monkeypatch):
    # A CUDA build drives the CUDA device that _detect_gpu reports.
    monkeypatch.setattr(analysis_backend, "_llama_cpp_gpu_backend", lambda: "cuda")
    monkeypatch.setattr(analysis_backend, "_detect_gpu", lambda: ("cuda", "RTX 3060"))

    kind, name = analysis_backend._detect_analysis_hardware()

    assert kind == "cuda"
    assert name == "RTX 3060"


def test_pin_vulkan_visible_device_sets_ggml_env(monkeypatch):
    from lib.vulkan import VulkanDevice

    device = VulkanDevice(
        index=1, name="NVIDIA", device_type="discrete", vendor_id=0x10DE,
        heap_size_bytes=12 * analysis._GIB, heap_budget_bytes=8 * analysis._GIB,
    )
    monkeypatch.delenv("GGML_VK_VISIBLE_DEVICES", raising=False)

    analysis_backend._pin_vulkan_visible_device(device)

    assert os.environ["GGML_VK_VISIBLE_DEVICES"] == "1"


def test_pin_vulkan_visible_device_respects_user_override(monkeypatch):
    from lib.vulkan import VulkanDevice

    device = VulkanDevice(
        index=1, name="NVIDIA", device_type="discrete", vendor_id=0x10DE,
        heap_size_bytes=12 * analysis._GIB, heap_budget_bytes=8 * analysis._GIB,
    )
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "0")

    analysis_backend._pin_vulkan_visible_device(device)

    assert os.environ["GGML_VK_VISIBLE_DEVICES"] == "0"


def test_vulkan_device_pins_ggml_visible_devices(monkeypatch):
    import lib.vulkan as vulkan_mod
    from lib.vulkan import VulkanDevice, VulkanProbeResult

    device = VulkanDevice(
        index=1, name="NVIDIA", device_type="discrete", vendor_id=0x10DE,
        heap_size_bytes=12 * analysis._GIB, heap_budget_bytes=8 * analysis._GIB,
    )
    monkeypatch.delenv("GGML_VK_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        vulkan_mod, "probe_vulkan",
        lambda: VulkanProbeResult(True, device, devices=(device,)),
    )

    selected = analysis_backend._vulkan_device()

    assert selected is device
    assert os.environ["GGML_VK_VISIBLE_DEVICES"] == "1"


def test_load_llama_cpp_model_retries_cpu_after_gpu_failure(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    calls = []

    class RetryLlama:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if kwargs["n_gpu_layers"] > 0:
                raise RuntimeError("out of device memory")

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=RetryLlama))
    backend = analysis.AnalysisBackend(
        name="vulkan",
        device_name="GPU",
        model_id=str(model_path),
        model_kwargs={"model_path": str(model_path), "n_gpu_layers": 10},
    )

    analysis._load_llama_cpp_model(backend)

    assert [call["n_gpu_layers"] for call in calls] == [10, 0]
    assert backend.model_kwargs["n_gpu_layers"] == 10
