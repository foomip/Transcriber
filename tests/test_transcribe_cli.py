from types import SimpleNamespace

import pytest

import transcribe
from lib import transcription


class NonInteractiveInput:
    def isatty(self):
        return False


class InteractiveInput:
    def isatty(self):
        return True


def make_result(lines=None):
    return transcription.TranscriptionResult(
        lines=lines if lines is not None else ["[00:00:00.00 -> 00:01:05.00]  Hello team"],
        requested_language="en",
        requested_language_description="English",
        detected_language="en",
        detected_language_description="English",
        language_probability=0.94,
        model_size="small",
    )


def test_parse_args_reads_audio_path_and_language():
    args = transcribe.parse_args(["-l", "EN", "meeting.wav"])

    assert args.language == "EN"
    assert args.audio_path == "meeting.wav"


def test_validate_language_normalizes_supported_language():
    assert transcribe.validate_language(" EN ") == "en"
    assert transcribe.validate_language(None) is None


def test_validate_language_exits_for_unsupported_language():
    with pytest.raises(SystemExit) as exc_info:
        transcribe.validate_language("zz")

    assert exc_info.value.code == 1


def test_transcript_file_lines_include_language_metadata():
    lines = transcribe.transcript_file_lines("/tmp/meeting.wav", make_result())

    assert "Source file: meeting.wav" in lines
    assert "Whisper model: small" in lines
    assert "Requested language: English (en)" in lines
    assert "Detected language: English (en)" in lines
    assert "Detection confidence: 94%" in lines
    assert "# Transcript" in lines
    assert "[00:00:00.00 -> 00:01:05.00]  Hello team" in lines


def test_should_continue_with_analysis_continues_for_non_interactive_input(monkeypatch):
    monkeypatch.setattr(transcribe.sys, "stdin", NonInteractiveInput())

    assert transcribe.should_continue_with_analysis() is True


def test_should_continue_with_analysis_accepts_no_after_invalid_answer(monkeypatch):
    answers = iter(["maybe", "n"])
    monkeypatch.setattr(transcribe.sys, "stdin", InteractiveInput())
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert transcribe.should_continue_with_analysis() is False


def test_run_writes_transcript_only_when_analysis_is_declined(tmp_path, monkeypatch):
    audio_path = tmp_path / "meeting_20260528_0930.wav"
    audio_path.write_bytes(b"")

    monkeypatch.setattr(transcribe.transcription, "detect_device", lambda: ("cpu", "int8"))
    monkeypatch.setattr(
        transcribe.transcription,
        "transcribe_audio",
        lambda audio, device, compute_type, language=None: make_result(),
    )
    monkeypatch.setattr(transcribe, "should_continue_with_analysis", lambda: False)

    transcribe.run(str(audio_path), language="en")

    transcript_path = tmp_path / "meeting_20260528_0930_transcript.txt"
    assert transcript_path.exists()
    assert "# Transcript" in transcript_path.read_text(encoding="utf-8")
    assert not (tmp_path / "meeting_20260528_0930_report.md").exists()


def test_run_writes_report_for_full_mocked_pipeline(tmp_path, monkeypatch):
    audio_path = tmp_path / "meeting_20260528_0930.wav"
    audio_path.write_bytes(b"")
    calls = SimpleNamespace(transcript_body=None, meta=None)

    def fake_generate_summaries(transcript_body, meta):
        calls.transcript_body = transcript_body
        calls.meta = meta
        return [("## Executive Summary", "The team said hello.")]

    monkeypatch.setattr(transcribe.transcription, "detect_device", lambda: ("cpu", "int8"))
    monkeypatch.setattr(
        transcribe.transcription,
        "transcribe_audio",
        lambda audio, device, compute_type, language=None: make_result(),
    )
    monkeypatch.setattr(transcribe, "should_continue_with_analysis", lambda: True)
    monkeypatch.setattr(transcribe.analysis, "generate_summaries", fake_generate_summaries)

    transcribe.run(str(audio_path), language="en")

    assert calls.transcript_body == "Hello team"
    assert calls.meta["duration"] == "1 minute"
    report_path = tmp_path / "meeting_20260528_0930_report.md"
    assert report_path.exists()
    assert "The team said hello." in report_path.read_text(encoding="utf-8")


def test_run_exits_when_audio_file_is_missing(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        transcribe.run(str(tmp_path / "missing.wav"))

    assert exc_info.value.code == 1


def test_run_exits_without_report_when_no_speech_is_detected(tmp_path, monkeypatch):
    audio_path = tmp_path / "meeting_20260528_0930.wav"
    audio_path.write_bytes(b"")

    monkeypatch.setattr(transcribe.transcription, "detect_device", lambda: ("cpu", "int8"))
    monkeypatch.setattr(
        transcribe.transcription,
        "transcribe_audio",
        lambda audio, device, compute_type, language=None: make_result(lines=[]),
    )

    with pytest.raises(SystemExit) as exc_info:
        transcribe.run(str(audio_path), language="en")

    assert exc_info.value.code == 0
    assert not (tmp_path / "meeting_20260528_0930_transcript.txt").exists()
