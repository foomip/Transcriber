import sys
import types

import pytest

from lib import analysis


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


def test_detect_analysis_backend_reports_cuda(monkeypatch):
    monkeypatch.delenv(analysis.ANALYSIS_MODEL_ENV, raising=False)
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(analysis.torch.cuda, "get_device_name", lambda _index: "CUDA GPU")
    monkeypatch.setattr(
        analysis.torch.cuda,
        "mem_get_info",
        lambda: (_ for _ in ()).throw(RuntimeError("no device")),
    )
    monkeypatch.setattr(analysis.torch.version, "hip", None, raising=False)

    backend = analysis.detect_analysis_backend()

    assert backend.name == "cuda"
    assert backend.device_name == "CUDA GPU"
    assert backend.model_id == analysis.DEFAULT_ANALYSIS_MODEL_ID
    assert backend.model_kwargs == {"device_map": "auto", "torch_dtype": "auto"}


def test_detect_analysis_backend_reports_rocm(monkeypatch):
    monkeypatch.delenv(analysis.ANALYSIS_MODEL_ENV, raising=False)
    monkeypatch.setenv(analysis.ANALYSIS_BACKEND_ENV, analysis.TRANSFORMERS_BACKEND_NAME)
    monkeypatch.delenv(analysis.GPU_HEADROOM_ENV, raising=False)
    monkeypatch.delenv(analysis.GPU_MAX_MEMORY_ENV, raising=False)
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(analysis.torch.cuda, "get_device_name", lambda _index: "ROCm GPU")
    monkeypatch.setattr(
        analysis.torch.cuda,
        "mem_get_info",
        lambda: (16 * analysis._GIB, 16 * analysis._GIB),
    )
    monkeypatch.setattr(analysis.torch.version, "hip", "6.0", raising=False)
    monkeypatch.setattr(analysis, "_available_ram_bytes", lambda: 64 * analysis._GIB)

    backend = analysis.detect_analysis_backend()

    assert backend.name == "rocm"
    assert backend.device_name == "ROCm GPU"
    assert backend.model_id == analysis.DEFAULT_ROCM_ANALYSIS_MODEL_ID
    assert backend.engine == analysis.TRANSFORMERS_BACKEND_NAME
    assert backend.use_plain_prompt is True
    assert backend.max_new_tokens == analysis.ROCM_ANALYSIS_MAX_NEW_TOKENS
    assert backend.model_kwargs == {
        "device_map": {"": "cuda"},
        "torch_dtype": analysis.torch.float16,
        "attn_implementation": analysis.ROCM_ATTENTION_IMPLEMENTATION,
    }
    assert "ROCm analysis model" in backend.notes[0]
    assert "fully on GPU" in backend.notes[1]
    assert "float16 with eager attention" in backend.notes[2]


def test_detect_analysis_backend_respects_model_override_on_rocm(monkeypatch):
    monkeypatch.setenv(analysis.ANALYSIS_MODEL_ENV, "custom/model")
    monkeypatch.delenv(analysis.ANALYSIS_BACKEND_ENV, raising=False)
    monkeypatch.setattr(analysis, "_llama_cpp_is_available", lambda: True)
    monkeypatch.setattr(analysis, "_llama_cpp_model_exists", lambda: True)
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(analysis.torch.cuda, "get_device_name", lambda _index: "ROCm GPU")
    monkeypatch.setattr(analysis.torch.version, "hip", "6.0", raising=False)

    backend = analysis.detect_analysis_backend()

    assert backend.engine == analysis.TRANSFORMERS_BACKEND_NAME
    assert backend.model_id == "custom/model"
    assert backend.model_kwargs == {
        "device_map": {"": "cuda"},
        "torch_dtype": analysis.torch.float16,
        "attn_implementation": analysis.ROCM_ATTENTION_IMPLEMENTATION,
    }
    assert "Using TRANSCRIBER_ANALYSIS_MODEL=custom/model" in backend.notes[0]


def test_detect_analysis_backend_prefers_llama_cpp_on_rocm_when_available(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"0" * analysis._GIB)
    monkeypatch.delenv(analysis.ANALYSIS_BACKEND_ENV, raising=False)
    monkeypatch.setenv(analysis.LLAMA_CPP_MODEL_PATH_ENV, str(model_path))
    monkeypatch.setenv(analysis.LLAMA_CPP_LAYER_COUNT_ENV, "32")
    monkeypatch.setattr(analysis, "_llama_cpp_is_available", lambda: True)
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(analysis.torch.cuda, "get_device_name", lambda _index: "ROCm GPU")
    monkeypatch.setattr(
        analysis.torch.cuda,
        "mem_get_info",
        lambda: (8 * analysis._GIB, 8 * analysis._GIB),
    )
    monkeypatch.setattr(analysis.torch.version, "hip", "6.0", raising=False)

    backend = analysis.detect_analysis_backend()

    assert backend.name == "rocm"
    assert backend.device_name == "ROCm GPU"
    assert backend.engine == analysis.LLAMA_CPP_BACKEND_NAME
    assert backend.model_id == str(model_path)
    assert backend.model_kwargs["model_path"] == str(model_path)
    assert backend.model_kwargs["n_ctx"] == analysis.DEFAULT_ROCM_LLAMA_CPP_CONTEXT_SIZE
    assert backend.model_kwargs["n_batch"] == analysis.DEFAULT_ROCM_LLAMA_CPP_BATCH_SIZE
    assert 0 < backend.model_kwargs["n_gpu_layers"] <= 32
    assert "llama.cpp" in backend.notes[0]
    assert "splits layers" in backend.notes[1]


def test_detect_analysis_backend_forces_llama_cpp_on_rocm_without_import(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"0" * analysis._GIB)
    monkeypatch.setenv(analysis.ANALYSIS_BACKEND_ENV, analysis.LLAMA_CPP_BACKEND_NAME)
    monkeypatch.setenv(analysis.LLAMA_CPP_MODEL_PATH_ENV, str(model_path))
    monkeypatch.setattr(analysis, "_llama_cpp_is_available", lambda: False)
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(analysis.torch.cuda, "get_device_name", lambda _index: "ROCm GPU")
    monkeypatch.setattr(
        analysis.torch.cuda,
        "mem_get_info",
        lambda: (8 * analysis._GIB, 8 * analysis._GIB),
    )
    monkeypatch.setattr(analysis.torch.version, "hip", "6.0", raising=False)

    backend = analysis.detect_analysis_backend()

    assert backend.engine == analysis.LLAMA_CPP_BACKEND_NAME
    assert backend.model_id == str(model_path)


def test_detect_analysis_backend_downloads_missing_llama_cpp_model(tmp_path, monkeypatch):
    cache_dir = tmp_path / "gguf"
    model_path = cache_dir / analysis.DEFAULT_ROCM_LLAMA_CPP_MODEL_FILENAME

    monkeypatch.delenv(analysis.ANALYSIS_BACKEND_ENV, raising=False)
    monkeypatch.delenv(analysis.ANALYSIS_MODEL_ENV, raising=False)
    monkeypatch.delenv(analysis.LLAMA_CPP_MODEL_PATH_ENV, raising=False)
    monkeypatch.setenv(analysis.LLAMA_CPP_CACHE_DIR_ENV, str(cache_dir))
    monkeypatch.setattr(analysis, "_llama_cpp_is_available", lambda: True)
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(analysis.torch.cuda, "get_device_name", lambda _index: "ROCm GPU")
    monkeypatch.setattr(
        analysis.torch.cuda,
        "mem_get_info",
        lambda: (8 * analysis._GIB, 8 * analysis._GIB),
    )
    monkeypatch.setattr(analysis.torch.version, "hip", "6.0", raising=False)

    def fake_download(path):
        assert path == str(model_path)
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"0" * analysis._GIB)

    monkeypatch.setattr(analysis, "_download_llama_cpp_model", fake_download)

    backend = analysis.detect_analysis_backend()

    assert backend.engine == analysis.LLAMA_CPP_BACKEND_NAME
    assert backend.model_id == str(model_path)
    assert backend.model_kwargs["model_path"] == str(model_path)
    assert model_path.exists()


def test_ensure_llama_cpp_model_downloads_from_huggingface(tmp_path, monkeypatch):
    model_path = tmp_path / analysis.DEFAULT_ROCM_LLAMA_CPP_MODEL_FILENAME
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
            "repo_id": analysis.DEFAULT_ROCM_LLAMA_CPP_MODEL_REPO_ID,
            "filename": analysis.DEFAULT_ROCM_LLAMA_CPP_MODEL_FILENAME,
            "local_dir": str(tmp_path),
        }
    ]


def test_rocm_llama_cpp_gpu_layers_respects_explicit_override(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"0" * analysis._GIB)
    monkeypatch.setenv(analysis.LLAMA_CPP_GPU_LAYERS_ENV, "9")

    gpu_layers, notes = analysis._rocm_llama_cpp_gpu_layers(str(model_path))

    assert gpu_layers == 9
    assert analysis.LLAMA_CPP_GPU_LAYERS_ENV in notes[0]


def test_rocm_llama_cpp_gpu_layers_uses_safe_vram_budget(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"0" * 8 * analysis._GIB)
    monkeypatch.delenv(analysis.LLAMA_CPP_GPU_LAYERS_ENV, raising=False)
    monkeypatch.setenv(analysis.LLAMA_CPP_LAYER_COUNT_ENV, "32")
    monkeypatch.setenv(analysis.LLAMA_CPP_CONTEXT_SIZE_ENV, "4096")
    monkeypatch.setenv(analysis.LLAMA_CPP_GPU_HEADROOM_ENV, "2")
    monkeypatch.setattr(
        analysis.torch.cuda,
        "mem_get_info",
        lambda: (6 * analysis._GIB, 8 * analysis._GIB),
    )

    gpu_layers, notes = analysis._rocm_llama_cpp_gpu_layers(str(model_path))

    assert gpu_layers == 12
    assert "12/32" in notes[0]


def test_detect_analysis_backend_cpu_with_avx512_bf16_uses_auto_dtype(monkeypatch):
    monkeypatch.delenv(analysis.ANALYSIS_MODEL_ENV, raising=False)
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(analysis, "_cpu_supports_avx512_bf16", lambda: True)

    backend = analysis.detect_analysis_backend()

    assert backend.name == "cpu"
    assert backend.device_name == "CPU"
    assert backend.model_id == analysis.DEFAULT_ANALYSIS_MODEL_ID
    assert backend.model_kwargs == {"device_map": "auto", "torch_dtype": "auto"}
    assert "AVX-512 BF16" in backend.notes[0]


def test_detect_analysis_backend_cpu_without_avx512_bf16_uses_float32_when_ram_sufficient(monkeypatch):
    monkeypatch.delenv(analysis.ANALYSIS_MODEL_ENV, raising=False)
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(analysis, "_cpu_supports_avx512_bf16", lambda: False)
    monkeypatch.setattr(analysis, "_available_ram_bytes", lambda: 64 * analysis._GIB)

    backend = analysis.detect_analysis_backend()

    assert backend.name == "cpu"
    assert backend.device_name == "CPU"
    assert backend.model_id == analysis.DEFAULT_ANALYSIS_MODEL_ID
    assert backend.model_kwargs == {
        "device_map": "auto",
        "torch_dtype": analysis.torch.float32,
    }
    assert "float32" in backend.notes[0]
    assert "lacks" in backend.notes[0]


def test_detect_analysis_backend_cpu_without_avx512_bf16_falls_back_to_bf16_when_insufficient_ram(monkeypatch):
    monkeypatch.delenv(analysis.ANALYSIS_MODEL_ENV, raising=False)
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(analysis, "_cpu_supports_avx512_bf16", lambda: False)
    monkeypatch.setattr(analysis, "_available_ram_bytes", lambda: 16 * analysis._GIB)

    backend = analysis.detect_analysis_backend()

    assert backend.name == "cpu"
    assert backend.device_name == "CPU"
    assert backend.model_kwargs == {"device_map": "auto", "torch_dtype": "auto"}
    assert "falling back" in backend.notes[0]


def test_cpu_supports_avx512_bf16_returns_true_when_flag_present(tmp_path, monkeypatch):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nflags\t\t: fpu avx2 avx512f avx512_bf16 sse4_2\n")
    monkeypatch.setattr(analysis, "_CPUINFO_PATH", str(cpuinfo))

    assert analysis._cpu_supports_avx512_bf16() is True


def test_cpu_supports_avx512_bf16_returns_false_when_flag_absent(tmp_path, monkeypatch):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nflags\t\t: fpu avx2 avx512f sse4_2\n")
    monkeypatch.setattr(analysis, "_CPUINFO_PATH", str(cpuinfo))

    assert analysis._cpu_supports_avx512_bf16() is False


def test_cpu_supports_avx512_bf16_returns_false_when_file_missing(monkeypatch):
    monkeypatch.setattr(analysis, "_CPUINFO_PATH", "/nonexistent/cpuinfo")

    assert analysis._cpu_supports_avx512_bf16() is False


def test_available_ram_bytes_returns_value_from_meminfo(tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       131815760 kB\nMemAvailable:   65536000 kB\n")
    monkeypatch.setattr(analysis, "_MEMINFO_PATH", str(meminfo))

    assert analysis._available_ram_bytes() == 65536000 * 1024


def test_available_ram_bytes_returns_none_when_file_missing(monkeypatch):
    monkeypatch.setattr(analysis, "_MEMINFO_PATH", "/nonexistent/meminfo")

    assert analysis._available_ram_bytes() is None


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
    assert sections[0] == ("## Executive Summary", "The team reviewed the Apollo budget.")
    assert sections[1] == (
        "## Detailed Summary",
        "No information was generated for this section.",
    )
    assert sections[2] == ("## Action Items", "Dana will update the roadmap.")
