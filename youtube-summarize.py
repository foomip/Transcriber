#!/usr/bin/env python3
"""
youtube-summarize.py — YouTube video transcript fetch and summarization.

Usage:
    python youtube-summarize.py [-l LANGUAGE] <youtube_url>

Pipeline:
  1. Parse the YouTube URL and extract the video ID
  2. Fetch video metadata (title) from the YouTube oEmbed API
  3. Fetch the YouTube transcript (captions / auto-generated subtitles)
  4. Save timestamped transcript  → output/<video_id>_transcript.txt
  5. Load the local analysis model and generate a summary report
  6. Save Markdown report          → output/<video_id>_report.md

No audio is downloaded or processed locally.  The transcript text is
fetched from YouTube's public subtitle endpoint and everything else
(analysis and summarization) runs entirely on the local machine.
"""

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from argparse import ArgumentParser, Namespace

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from lib import analysis, report
from lib.transcription import (
    fmt_ts,
    format_supported_languages,
    language_description,
    normalize_language_code,
)

OUTPUT_DIR = "output"
_YOUTUBE_OEMBED = (
    "https://www.youtube.com/oembed"
    "?url=https://www.youtube.com/watch?v={video_id}&format=json"
)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str]) -> Namespace:
    parser = ArgumentParser(
        description=(
            "Fetch a YouTube video transcript and generate a local summary report."
        )
    )
    parser.add_argument(
        "youtube_url",
        help="YouTube video URL (watch, short, or embed) or bare 11-character video ID.",
    )
    parser.add_argument(
        "-l",
        "--language",
        help=(
            "Optional language code to prefer when selecting the YouTube transcript "
            "and to guide the summary output language (e.g. en, de, pt). "
            "Omit to auto-select the first available transcript."
        ),
    )
    return parser.parse_args(argv)


def validate_language(language_code: str | None) -> str | None:
    if language_code is None:
        return None

    normalized = normalize_language_code(language_code)
    if language_description(normalized) is not None:
        return normalized

    print(f"❌  Unsupported language code: '{language_code}'")
    print("\nSupported language codes:")
    print(format_supported_languages())
    sys.exit(1)


# ── YouTube URL / ID parsing ───────────────────────────────────────────────

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str | None:
    """
    Return the 11-character video ID from any common YouTube URL format,
    or from a bare video ID string.  Returns None if no ID can be found.

    Supported formats
    -----------------
    https://www.youtube.com/watch?v=VIDEO_ID
    https://www.youtube.com/watch?v=VIDEO_ID&t=60s
    https://youtu.be/VIDEO_ID
    https://www.youtube.com/embed/VIDEO_ID
    https://www.youtube.com/v/VIDEO_ID
    VIDEO_ID   (bare 11-character ID)
    """
    url_or_id = url_or_id.strip()
    parsed = urllib.parse.urlparse(url_or_id)

    if parsed.scheme in ("http", "https"):
        host = (parsed.hostname or "").lower()

        # youtu.be/VIDEO_ID
        if host == "youtu.be":
            vid = parsed.path.lstrip("/").split("/")[0].split("?")[0]
            if _VIDEO_ID_RE.match(vid):
                return vid
            return None

        if host in ("www.youtube.com", "youtube.com", "m.youtube.com"):
            # /watch?v=VIDEO_ID
            if "/watch" in parsed.path:
                params = urllib.parse.parse_qs(parsed.query)
                v_list = params.get("v", [])
                if v_list and _VIDEO_ID_RE.match(v_list[0]):
                    return v_list[0]

            # /embed/VIDEO_ID  or  /v/VIDEO_ID
            m = re.match(r"^/(?:embed|v)/([A-Za-z0-9_-]{11})", parsed.path)
            if m:
                return m.group(1)

        return None

    # Bare video ID (no URL scheme)
    if _VIDEO_ID_RE.match(url_or_id):
        return url_or_id

    return None


# ── Metadata ──────────────────────────────────────────────────────────────


def fetch_video_metadata(video_id: str) -> dict[str, str]:
    """
    Fetch video title and author from the YouTube oEmbed endpoint.

    The oEmbed API is a documented, key-free public endpoint.  Falls back
    to the video ID as the title if the request fails for any reason.
    """
    url = _YOUTUBE_OEMBED.format(video_id=video_id)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            return {
                "title": data.get("title") or video_id,
                "author": data.get("author_name") or "Unknown",
            }
    except Exception:
        return {"title": video_id, "author": "Unknown"}


# ── Transcript fetching ───────────────────────────────────────────────────


def fetch_transcript(
    video_id: str,
    language: str | None,
) -> tuple[list, str, str, bool]:
    """
    Fetch transcript segments for a YouTube video.

    If *language* is given, the matching transcript is preferred; if none
    exists in that language a warning is printed and the first available
    transcript is used instead.

    Returns
    -------
    (segments, language_code, language_name, is_generated)
        segments       — list of FetchedTranscriptSnippet objects
        language_code  — e.g. "en"
        language_name  — e.g. "English"
        is_generated   — True when the transcript is auto-generated

    Raises RuntimeError on any unrecoverable failure.
    """
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
    except VideoUnavailable:
        raise RuntimeError(
            f"Video '{video_id}' is unavailable or does not exist."
        )
    except TranscriptsDisabled:
        raise RuntimeError(
            f"Transcripts are disabled for video '{video_id}'."
        )
    except CouldNotRetrieveTranscript as exc:
        raise RuntimeError(
            f"Could not retrieve transcripts for '{video_id}': {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error listing transcripts for '{video_id}': {exc}"
        ) from exc

    transcript = None

    if language:
        try:
            transcript = transcript_list.find_transcript([language])
        except NoTranscriptFound:
            available = list(transcript_list)
            if available:
                transcript = available[0]
                print(
                    f"\n  ⚠️  No '{language}' transcript found — "
                    f"falling back to: {transcript.language} ({transcript.language_code})"
                )

    if transcript is None:
        available = list(transcript_list)
        if not available:
            raise RuntimeError(
                f"No transcripts are available for video '{video_id}'."
            )
        transcript = available[0]

    try:
        fetched = transcript.fetch()
    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch transcript content: {exc}"
        ) from exc

    return (
        list(fetched),
        transcript.language_code,
        transcript.language,
        transcript.is_generated,
    )


# ── Transcript formatting ─────────────────────────────────────────────────


def format_transcript_lines(segments: list) -> list[str]:
    """
    Convert YouTube transcript segments into the standard timestamped line
    format used throughout this project:

        [HH:MM:SS.xx -> HH:MM:SS.xx]  segment text

    Each segment supplies .start and .duration (in seconds) and .text.
    """
    lines: list[str] = []
    for seg in segments:
        text = seg.text.strip().replace("\n", " ")
        if not text:
            continue
        start: float = seg.start
        end: float = seg.start + seg.duration
        lines.append(f"[{fmt_ts(start)} -> {fmt_ts(end)}]  {text}")
    return lines


# ── Shared helpers ────────────────────────────────────────────────────────


def _language_display(language_code: str | None, description: str | None) -> str:
    if language_code is None or description is None:
        return "Auto-select"
    return f"{description} ({language_code})"


def should_continue_with_analysis() -> bool:
    if not sys.stdin.isatty():
        print("\nNon-interactive input detected; continuing with analysis.")
        return True

    while True:
        answer = input("\nContinue with analysis and summarising? [Y/n]: ").strip().lower()
        if answer in {"", "y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


# ── Main pipeline ─────────────────────────────────────────────────────────


def run(youtube_url: str, language: str | None = None) -> None:
    video_id = extract_video_id(youtube_url)
    if not video_id:
        print(f"❌  Could not extract a YouTube video ID from: {youtube_url!r}")
        print("    Accepted formats:")
        print("      https://www.youtube.com/watch?v=VIDEO_ID")
        print("      https://youtu.be/VIDEO_ID")
        print("      VIDEO_ID  (11-character bare ID)")
        sys.exit(1)

    print("─" * 56)

    # ── Step 1: Fetch metadata and transcript ──────────────────────────────
    print("\n▶ Step 1/3: Fetching YouTube transcript")

    print(f"  🌐 Fetching video metadata for {video_id}...")
    meta_raw = fetch_video_metadata(video_id)
    title = meta_raw["title"]
    if title == video_id:
        print(f"  ⚠️  Metadata unavailable — using video ID as title: {video_id}")
    else:
        print(f"  📹 Title: {title}")

    print("  📥 Fetching transcript...")
    try:
        segments, lang_code, lang_name, is_generated = fetch_transcript(
            video_id, language
        )
    except RuntimeError as exc:
        print(f"\n❌  {exc}")
        sys.exit(1)

    transcript_type = "auto-generated" if is_generated else "manual"
    print(f"  🌐 Transcript: {lang_name} ({lang_code}) — {transcript_type}")

    lines = format_transcript_lines(segments)
    if not lines:
        print("⚠️  The fetched transcript is empty. Exiting.")
        sys.exit(0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    requested_lang_display = _language_display(
        language, language_description(language) if language else None
    )
    detected_lang_display = f"{lang_name} ({lang_code})"

    transcript_path = os.path.join(OUTPUT_DIR, f"{video_id}_transcript.txt")
    transcript_file_lines = [
        "# Transcription metadata",
        "Source: YouTube",
        f"Video ID: {video_id}",
        f"Title: {title}",
        f"URL: https://www.youtube.com/watch?v={video_id}",
        f"Requested language: {requested_lang_display}",
        f"Transcript language: {detected_lang_display}",
        f"Transcript type: {transcript_type}",
        "",
        "# Transcript",
        "",
        *lines,
    ]
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("\n".join(transcript_file_lines) + "\n")
    print(f"\n✅  Transcript saved → {transcript_path}")

    if not should_continue_with_analysis():
        print("\nTranscript-only run complete.")
        print("─" * 56)
        return

    # ── Step 2: Generate summaries ─────────────────────────────────────────
    print("\n▶ Step 2/3: Generating video summary report sections")

    meta: dict[str, str] = {
        "title": title,
        "date": "Unknown",
        "time": "Unknown",
        "duration": report.estimate_duration(lines),
        "requested_language": requested_lang_display,
        "detected_language": detected_lang_display,
        # language_probability is not available for YouTube transcripts; an
        # empty string signals report.compile() to omit the confidence suffix.
        "language_probability": "",
        "transcription_model": f"YouTube Transcript ({transcript_type})",
    }
    transcript_body = report.build_transcript_body(lines)

    try:
        sections = analysis.generate_summaries(transcript_body, meta)
    except (analysis.AnalysisGroundingError, analysis.AnalysisModelError) as exc:
        print(f"\n❌  Analysis failed: {exc}")
        print("    The transcript was saved, but no Markdown report was written.")
        sys.exit(1)

    # ── Step 3: Compile and save report ───────────────────────────────────
    print("\n▶ Step 3/3: Saving Markdown report")

    source_label = f"{title} — youtube.com/watch?v={video_id}"
    report_md = report.compile(
        meta,
        sections,
        report_title="Video Summary Report",
        source_label=source_label,
    )
    report_path = os.path.join(OUTPUT_DIR, f"{video_id}_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\n📋  Video summary report saved → {report_path}")
    print("─" * 56)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run(args.youtube_url, validate_language(args.language))
