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
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(analysis.torch.cuda, "get_device_name", lambda _index: "CUDA GPU")
    monkeypatch.setattr(analysis.torch.version, "hip", None, raising=False)

    backend = analysis.detect_analysis_backend()

    assert backend.name == "cuda"
    assert backend.device_name == "CUDA GPU"
    assert backend.model_kwargs == {"device_map": "auto", "torch_dtype": "auto"}


def test_detect_analysis_backend_reports_rocm(monkeypatch):
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(analysis.torch.cuda, "get_device_name", lambda _index: "ROCm GPU")
    monkeypatch.setattr(analysis.torch.version, "hip", "6.0", raising=False)

    backend = analysis.detect_analysis_backend()

    assert backend.name == "rocm"
    assert backend.device_name == "ROCm GPU"


def test_detect_analysis_backend_cpu_with_avx512_bf16_uses_auto_dtype(monkeypatch):
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(analysis, "_cpu_supports_avx512_bf16", lambda: True)

    backend = analysis.detect_analysis_backend()

    assert backend.name == "cpu"
    assert backend.device_name == "CPU"
    assert backend.model_kwargs == {"device_map": "auto", "torch_dtype": "auto"}


def test_detect_analysis_backend_cpu_without_avx512_bf16_uses_float32(monkeypatch):
    monkeypatch.setattr(analysis.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(analysis, "_cpu_supports_avx512_bf16", lambda: False)

    backend = analysis.detect_analysis_backend()

    assert backend.name == "cpu"
    assert backend.device_name == "CPU"
    assert backend.model_kwargs == {
        "device_map": "auto",
        "torch_dtype": analysis.torch.float32,
    }


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


def test_build_user_message_includes_sections_metadata_and_transcript():
    message = analysis._build_user_message("Apollo roadmap and budget discussion", BASE_META)

    for heading, instruction in analysis.SUMMARY_TASKS:
        assert heading in message
        assert instruction in message
    assert "Title: Meeting Recording" in message
    assert "Date: May 28, 2026" in message
    assert "Requested transcription language: English (en)" in message
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
