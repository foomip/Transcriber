from lib import report


def test_parse_recording_meta_extracts_date_and_time_from_standard_filename():
    meta = report.parse_recording_meta("/tmp/meeting_20260528_093015.wav")

    assert meta["title"] == "Meeting Recording"
    assert meta["date"] == "May 28, 2026"
    assert meta["time"] == "09:30 AM"
    assert meta["duration"] == "Unknown"


def test_parse_recording_meta_falls_back_for_unrecognized_filename():
    meta = report.parse_recording_meta("/tmp/random_audio.wav")

    assert meta["date"] == "Unknown"
    assert meta["time"] == "Unknown"


def test_estimate_duration_handles_empty_malformed_and_valid_lines():
    assert report.estimate_duration([]) == "Unknown"
    assert report.estimate_duration(["No timestamp here"]) == "Unknown"
    assert report.estimate_duration(["[00:00:00.00 -> 00:00:42.10]  Hi"]) == "< 1 minute"
    assert report.estimate_duration(["[00:00:00.00 -> 00:12:05.00]  Hi"]) == "12 minutes"
    assert report.estimate_duration(["[00:00:00.00 -> 01:02:05.00]  Hi"]) == "1 hour, 2 minutes"


def test_transcript_char_budget_uses_environment_override(monkeypatch):
    monkeypatch.setenv(report.TRANSCRIPT_BUDGET_ENV, "42_000")

    assert report.transcript_char_budget(available_memory_bytes=report.GIB) == 42_000


def test_transcript_char_budget_scales_with_available_memory(monkeypatch):
    monkeypatch.delenv(report.TRANSCRIPT_BUDGET_ENV, raising=False)

    assert report.transcript_char_budget(available_memory_bytes=8 * report.GIB) == report.MIN_TRANSCRIPT_CHARS
    assert report.transcript_char_budget(available_memory_bytes=10 * report.GIB) == 52_000
    assert report.transcript_char_budget(available_memory_bytes=100 * report.GIB) == report.MAX_TRANSCRIPT_CHARS


def test_build_transcript_body_strips_timestamps_and_truncates(monkeypatch):
    monkeypatch.setattr(report, "transcript_char_budget", lambda: 12)
    lines = [
        "[00:00:00.00 -> 00:00:02.00]  First sentence",
        "[00:00:02.00 -> 00:00:04.00]  Second sentence",
    ]

    body = report.build_transcript_body(lines)

    assert body == "First senten\n[... transcript truncated ...]"


def test_compile_includes_metadata_and_sections():
    meta = {
        "date": "May 28, 2026",
        "duration": "12 minutes",
        "requested_language": "English (en)",
        "detected_language": "English (en)",
        "language_probability": "94%",
    }
    sections = [("## Executive Summary", "A useful summary.")]

    markdown = report.compile(meta, sections, "/tmp/meeting_20260528_0930.wav")

    assert "# Meeting Report" in markdown
    assert "**Source file:** `meeting_20260528_0930.wav`" in markdown
    assert "**Date:** May 28, 2026" in markdown
    assert "**Requested language:** English (en)" in markdown
    assert "**Detected language:** English (en) (94% confidence)" in markdown
    assert "## Executive Summary" in markdown
    assert "A useful summary." in markdown


def test_compile_uses_custom_report_title_and_source_label():
    meta = {
        "date": "Unknown",
        "duration": "8 minutes",
        "requested_language": "English (en)",
        "detected_language": "English (en)",
        "language_probability": "",
    }
    sections = [("## Executive Summary", "Great talk.")]

    markdown = report.compile(
        meta,
        sections,
        report_title="Video Summary Report",
        source_label="My Video — youtube.com/watch?v=abc12345678",
    )

    assert "# Video Summary Report" in markdown
    assert "# Meeting Report" not in markdown
    assert "**Source:** `My Video — youtube.com/watch?v=abc12345678`" in markdown
    assert "**Source file:**" not in markdown


def test_compile_omits_confidence_suffix_when_language_probability_is_empty():
    meta = {
        "date": "Unknown",
        "duration": "5 minutes",
        "requested_language": "Auto-select",
        "detected_language": "German (de)",
        "language_probability": "",
    }
    sections = []

    markdown = report.compile(meta, sections)

    assert "**Detected language:** German (de)" in markdown
    assert "confidence" not in markdown
