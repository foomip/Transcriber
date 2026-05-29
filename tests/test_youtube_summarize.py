"""
test_youtube_summarize.py — Tests for the YouTube summarization pipeline.

External boundaries mocked throughout:
  - YouTubeTranscriptApi (transcript fetching)
  - urllib.request.urlopen (metadata fetching)
  - analysis.generate_summaries (local LLM)
  - should_continue_with_analysis (interactive prompt)
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import youtube_summarize
from lib import report
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled


# ── Fake transcript objects ────────────────────────────────────────────────


class FakeSnippet:
    """Mimics a FetchedTranscriptSnippet from youtube-transcript-api."""

    def __init__(self, text: str, start: float, duration: float) -> None:
        self.text = text
        self.start = start
        self.duration = duration


class FakeTranscript:
    """Mimics a Transcript object returned by TranscriptList."""

    def __init__(
        self,
        language: str = "English",
        language_code: str = "en",
        is_generated: bool = True,
        snippets: list | None = None,
    ) -> None:
        self.language = language
        self.language_code = language_code
        self.is_generated = is_generated
        self._snippets = snippets or [
            FakeSnippet("Hello world", 0.0, 5.0),
            FakeSnippet("Second segment", 5.0, 4.0),
        ]

    def fetch(self) -> list:
        return self._snippets


class FakeTranscriptList:
    """Mimics a TranscriptList returned by YouTubeTranscriptApi.list()."""

    def __init__(
        self,
        transcripts: list | None = None,
        raise_on_find: bool = False,
    ) -> None:
        self._transcripts = transcripts or [FakeTranscript()]
        self._raise_on_find = raise_on_find

    def __iter__(self):
        return iter(self._transcripts)

    def find_transcript(self, language_codes: list[str]):
        if self._raise_on_find:
            raise NoTranscriptFound("test_id", language_codes, [])
        for lang in language_codes:
            for t in self._transcripts:
                if t.language_code == lang:
                    return t
        raise NoTranscriptFound("test_id", language_codes, [])


# Module-level default snippets reused across tests.
_FAKE_SNIPPETS = [
    FakeSnippet("Hello world", 0.0, 5.0),
    FakeSnippet("Second segment", 5.0, 4.0),
]


def _make_fake_api_class(
    transcript_list: FakeTranscriptList | None = None,
    raise_list_error: Exception | None = None,
) -> MagicMock:
    """Return a mock class that behaves like YouTubeTranscriptApi."""
    fake_instance = MagicMock()
    if raise_list_error is not None:
        fake_instance.list.side_effect = raise_list_error
    else:
        fake_instance.list.return_value = transcript_list or FakeTranscriptList()
    return MagicMock(return_value=fake_instance)


def _patch_run_deps(
    monkeypatch,
    tmp_path,
    *,
    continue_analysis: bool = True,
) -> None:
    """Apply standard monkeypatches for run() integration tests."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        youtube_summarize,
        "fetch_video_metadata",
        lambda _id: {"title": "Test Video Title", "author": "Test Channel"},
    )
    monkeypatch.setattr(
        youtube_summarize,
        "fetch_transcript",
        lambda _vid, _lang: (_FAKE_SNIPPETS, "en", "English", True),
    )
    monkeypatch.setattr(
        youtube_summarize,
        "should_continue_with_analysis",
        lambda: continue_analysis,
    )
    if continue_analysis:
        monkeypatch.setattr(
            youtube_summarize.analysis,
            "generate_summaries",
            lambda body, meta: [("## Executive Summary", "This is a test summary.")],
        )


# ── extract_video_id ───────────────────────────────────────────────────────


def test_extract_video_id_from_full_watch_url():
    assert (
        youtube_summarize.extract_video_id(
            "https://www.youtube.com/watch?v=XmpKPs9Emx0"
        )
        == "XmpKPs9Emx0"
    )


def test_extract_video_id_from_full_watch_url_with_extra_params():
    assert (
        youtube_summarize.extract_video_id(
            "https://www.youtube.com/watch?v=XmpKPs9Emx0&t=60s&list=PLxxx"
        )
        == "XmpKPs9Emx0"
    )


def test_extract_video_id_from_short_url():
    assert (
        youtube_summarize.extract_video_id("https://youtu.be/XmpKPs9Emx0")
        == "XmpKPs9Emx0"
    )


def test_extract_video_id_from_embed_url():
    assert (
        youtube_summarize.extract_video_id(
            "https://www.youtube.com/embed/XmpKPs9Emx0"
        )
        == "XmpKPs9Emx0"
    )


def test_extract_video_id_from_bare_id():
    assert youtube_summarize.extract_video_id("XmpKPs9Emx0") == "XmpKPs9Emx0"


def test_extract_video_id_returns_none_for_non_youtube_url():
    assert youtube_summarize.extract_video_id("https://example.com/watch?v=abc") is None


def test_extract_video_id_returns_none_for_short_bare_string():
    assert youtube_summarize.extract_video_id("short") is None


def test_extract_video_id_returns_none_for_plain_text():
    assert youtube_summarize.extract_video_id("not-a-url-at-all") is None


# ── format_transcript_lines ────────────────────────────────────────────────


def test_format_transcript_lines_produces_correct_timestamps():
    segments = [FakeSnippet("Hello world", 4.5, 7.8)]
    lines = youtube_summarize.format_transcript_lines(segments)
    assert lines == ["[00:00:04.50 -> 00:00:12.30]  Hello world"]


def test_format_transcript_lines_skips_empty_segments():
    segments = [
        FakeSnippet("", 0.0, 5.0),
        FakeSnippet("  ", 5.0, 3.0),
        FakeSnippet("Real text", 8.0, 2.0),
    ]
    lines = youtube_summarize.format_transcript_lines(segments)
    assert len(lines) == 1
    assert "Real text" in lines[0]


def test_format_transcript_lines_replaces_embedded_newlines():
    segments = [FakeSnippet("Line one\nLine two", 0.0, 5.0)]
    lines = youtube_summarize.format_transcript_lines(segments)
    assert "Line one Line two" in lines[0]
    assert "\n" not in lines[0]


def test_format_transcript_lines_multiple_segments():
    segments = [
        FakeSnippet("Hello world", 0.0, 5.0),
        FakeSnippet("Second segment", 5.0, 4.0),
    ]
    lines = youtube_summarize.format_transcript_lines(segments)
    assert len(lines) == 2
    assert lines[0] == "[00:00:00.00 -> 00:00:05.00]  Hello world"
    assert lines[1] == "[00:00:05.00 -> 00:00:09.00]  Second segment"


# ── fetch_video_metadata ───────────────────────────────────────────────────


def test_fetch_video_metadata_returns_title_on_success(monkeypatch):
    payload = json.dumps(
        {"title": "Test Video Title", "author_name": "Test Channel"}
    ).encode()

    class FakeResponse:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        youtube_summarize.urllib.request,
        "urlopen",
        lambda *a, **kw: FakeResponse(),
    )

    meta = youtube_summarize.fetch_video_metadata("XmpKPs9Emx0")
    assert meta["title"] == "Test Video Title"
    assert meta["author"] == "Test Channel"


def test_fetch_video_metadata_falls_back_to_video_id_on_network_error(monkeypatch):
    def raise_error(*a, **kw):
        raise OSError("Network error")

    monkeypatch.setattr(
        youtube_summarize.urllib.request, "urlopen", raise_error
    )

    meta = youtube_summarize.fetch_video_metadata("XmpKPs9Emx0")
    assert meta["title"] == "XmpKPs9Emx0"
    assert meta["author"] == "Unknown"


# ── fetch_transcript ───────────────────────────────────────────────────────


def test_fetch_transcript_returns_segments_and_metadata(monkeypatch):
    monkeypatch.setattr(
        youtube_summarize, "YouTubeTranscriptApi", _make_fake_api_class()
    )

    segments, lang_code, lang_name, is_generated = youtube_summarize.fetch_transcript(
        "XmpKPs9Emx0", None
    )

    assert len(segments) == 2
    assert lang_code == "en"
    assert lang_name == "English"
    assert is_generated is True


def test_fetch_transcript_selects_requested_language(monkeypatch):
    fr_transcript = FakeTranscript(
        language="French", language_code="fr", is_generated=False
    )
    en_transcript = FakeTranscript(language="English", language_code="en")
    transcript_list = FakeTranscriptList(transcripts=[fr_transcript, en_transcript])
    monkeypatch.setattr(
        youtube_summarize,
        "YouTubeTranscriptApi",
        _make_fake_api_class(transcript_list=transcript_list),
    )

    _, lang_code, _, _ = youtube_summarize.fetch_transcript("XmpKPs9Emx0", "fr")
    assert lang_code == "fr"


def test_fetch_transcript_falls_back_when_language_not_available(monkeypatch, capsys):
    de_transcript = FakeTranscript(language="German", language_code="de")
    transcript_list = FakeTranscriptList(
        transcripts=[de_transcript], raise_on_find=True
    )
    monkeypatch.setattr(
        youtube_summarize,
        "YouTubeTranscriptApi",
        _make_fake_api_class(transcript_list=transcript_list),
    )

    _, lang_code, _, _ = youtube_summarize.fetch_transcript("XmpKPs9Emx0", "en")

    # Falls back to the only available transcript (German)
    assert lang_code == "de"
    captured = capsys.readouterr()
    assert "'en'" in captured.out


def test_fetch_transcript_raises_on_transcripts_disabled(monkeypatch):
    monkeypatch.setattr(
        youtube_summarize,
        "YouTubeTranscriptApi",
        _make_fake_api_class(raise_list_error=TranscriptsDisabled("test_id")),
    )

    with pytest.raises(RuntimeError, match="[Dd]isabled"):
        youtube_summarize.fetch_transcript("test_id", None)


def test_fetch_transcript_raises_on_video_unavailable(monkeypatch):
    from youtube_transcript_api._errors import VideoUnavailable

    monkeypatch.setattr(
        youtube_summarize,
        "YouTubeTranscriptApi",
        _make_fake_api_class(raise_list_error=VideoUnavailable("test_id")),
    )

    with pytest.raises(RuntimeError, match="unavailable"):
        youtube_summarize.fetch_transcript("test_id", None)


# ── parse_args / validate_language ────────────────────────────────────────


def test_parse_args_flag_before_url():
    args = youtube_summarize.parse_args(
        ["-l", "en", "https://youtu.be/XmpKPs9Emx0"]
    )
    assert args.language == "en"
    assert args.youtube_url == "https://youtu.be/XmpKPs9Emx0"


def test_parse_args_url_before_flag():
    # argparse accepts positional + optional in any order
    args = youtube_summarize.parse_args(
        ["https://youtu.be/XmpKPs9Emx0", "-l", "en"]
    )
    assert args.language == "en"
    assert args.youtube_url == "https://youtu.be/XmpKPs9Emx0"


def test_parse_args_long_form_language():
    args = youtube_summarize.parse_args(
        ["--language=pt", "https://youtu.be/XmpKPs9Emx0"]
    )
    assert args.language == "pt"


def test_parse_args_no_language():
    args = youtube_summarize.parse_args(["https://youtu.be/XmpKPs9Emx0"])
    assert args.language is None


def test_validate_language_normalizes_supported_code():
    assert youtube_summarize.validate_language("EN") == "en"
    assert youtube_summarize.validate_language("  pt  ") == "pt"
    assert youtube_summarize.validate_language(None) is None


def test_validate_language_exits_for_unsupported_code():
    with pytest.raises(SystemExit) as exc_info:
        youtube_summarize.validate_language("zz")
    assert exc_info.value.code == 1


# ── run() integration tests ────────────────────────────────────────────────


def test_run_exits_on_invalid_url():
    with pytest.raises(SystemExit) as exc_info:
        youtube_summarize.run("not-a-valid-url")
    assert exc_info.value.code == 1


def test_run_creates_output_directory(tmp_path, monkeypatch):
    _patch_run_deps(monkeypatch, tmp_path)
    youtube_summarize.run("https://www.youtube.com/watch?v=XmpKPs9Emx0", "en")
    assert (tmp_path / "output").is_dir()


def test_run_writes_transcript_file(tmp_path, monkeypatch):
    _patch_run_deps(monkeypatch, tmp_path)
    youtube_summarize.run("https://www.youtube.com/watch?v=XmpKPs9Emx0", "en")

    transcript_path = tmp_path / "output" / "XmpKPs9Emx0_transcript.txt"
    assert transcript_path.exists()
    content = transcript_path.read_text(encoding="utf-8")
    assert "Source: YouTube" in content
    assert "Video ID: XmpKPs9Emx0" in content
    assert "Title: Test Video Title" in content
    assert "# Transcript" in content
    assert "[00:00:00.00 -> 00:00:05.00]  Hello world" in content
    assert "[00:00:05.00 -> 00:00:09.00]  Second segment" in content


def test_run_writes_report_file(tmp_path, monkeypatch):
    _patch_run_deps(monkeypatch, tmp_path)
    youtube_summarize.run("https://www.youtube.com/watch?v=XmpKPs9Emx0", "en")

    report_path = tmp_path / "output" / "XmpKPs9Emx0_report.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "# Video Summary Report" in content
    assert "Test Video Title" in content
    assert "This is a test summary." in content


def test_run_report_does_not_contain_meeting_report_title(tmp_path, monkeypatch):
    _patch_run_deps(monkeypatch, tmp_path)
    youtube_summarize.run("https://www.youtube.com/watch?v=XmpKPs9Emx0")

    content = (tmp_path / "output" / "XmpKPs9Emx0_report.md").read_text(encoding="utf-8")
    assert "# Meeting Report" not in content


def test_run_report_omits_confidence_suffix(tmp_path, monkeypatch):
    """YouTube reports must not show '( confidence)' when probability is empty."""
    _patch_run_deps(monkeypatch, tmp_path)
    youtube_summarize.run("https://www.youtube.com/watch?v=XmpKPs9Emx0")

    content = (tmp_path / "output" / "XmpKPs9Emx0_report.md").read_text(encoding="utf-8")
    assert "confidence" not in content


def test_run_writes_transcript_only_when_analysis_declined(tmp_path, monkeypatch):
    _patch_run_deps(monkeypatch, tmp_path, continue_analysis=False)
    youtube_summarize.run("https://www.youtube.com/watch?v=XmpKPs9Emx0")

    assert (tmp_path / "output" / "XmpKPs9Emx0_transcript.txt").exists()
    assert not (tmp_path / "output" / "XmpKPs9Emx0_report.md").exists()


def test_run_exits_when_transcript_fetch_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        youtube_summarize,
        "fetch_video_metadata",
        lambda _id: {"title": "Test Video", "author": "Channel"},
    )

    def raise_transcript_error(_vid, _lang):
        raise RuntimeError("Transcripts are disabled for this video.")

    monkeypatch.setattr(youtube_summarize, "fetch_transcript", raise_transcript_error)

    with pytest.raises(SystemExit) as exc_info:
        youtube_summarize.run("https://www.youtube.com/watch?v=XmpKPs9Emx0")
    assert exc_info.value.code == 1
    # No output directory should have been created (makedirs comes after fetch)
    assert not (tmp_path / "output").exists()


def test_run_uses_video_id_as_title_when_metadata_unavailable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Simulate metadata failure: fetch_video_metadata returns video_id as title
    monkeypatch.setattr(
        youtube_summarize,
        "fetch_video_metadata",
        lambda vid: {"title": vid, "author": "Unknown"},
    )
    monkeypatch.setattr(
        youtube_summarize,
        "fetch_transcript",
        lambda _vid, _lang: (_FAKE_SNIPPETS, "en", "English", True),
    )
    monkeypatch.setattr(
        youtube_summarize, "should_continue_with_analysis", lambda: False
    )

    youtube_summarize.run("https://www.youtube.com/watch?v=XmpKPs9Emx0")

    content = (
        (tmp_path / "output" / "XmpKPs9Emx0_transcript.txt")
        .read_text(encoding="utf-8")
    )
    # Title falls back to video ID
    assert "Title: XmpKPs9Emx0" in content


def test_run_transcript_language_info_recorded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        youtube_summarize,
        "fetch_video_metadata",
        lambda _id: {"title": "My Video", "author": "Channel"},
    )
    monkeypatch.setattr(
        youtube_summarize,
        "fetch_transcript",
        lambda _vid, _lang: (_FAKE_SNIPPETS, "de", "German", False),
    )
    monkeypatch.setattr(
        youtube_summarize, "should_continue_with_analysis", lambda: False
    )

    youtube_summarize.run("https://www.youtube.com/watch?v=XmpKPs9Emx0", "de")

    content = (
        (tmp_path / "output" / "XmpKPs9Emx0_transcript.txt")
        .read_text(encoding="utf-8")
    )
    assert "Transcript language: German (de)" in content
    assert "Transcript type: manual" in content
