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
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(analysis.torch.cuda, "get_device_name", lambda _index: "ROCm GPU")
    monkeypatch.setattr(analysis.torch.version, "hip", "6.0", raising=False)

    backend = analysis.detect_analysis_backend()

    assert backend.model_id == "custom/model"
    assert backend.model_kwargs == {
        "device_map": {"": "cuda"},
        "torch_dtype": analysis.torch.float16,
        "attn_implementation": analysis.ROCM_ATTENTION_IMPLEMENTATION,
    }
    assert "Using TRANSCRIBER_ANALYSIS_MODEL=custom/model" in backend.notes[0]


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
