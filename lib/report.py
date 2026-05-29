"""
report.py — Transcript preparation and Markdown report compilation.

Responsibilities:
  - Parse recording metadata (date, time) from the WAV filename
  - Estimate meeting duration from the final transcript timestamp
  - Strip timestamps from raw transcript lines into a clean body of text
  - Assemble the final Markdown report from metadata and generated sections
"""

import os
import re
from datetime import datetime, timedelta

GIB = 1024**3

# Keep the transcript body bounded so local analysis models have prompt headroom.
MIN_TRANSCRIPT_CHARS = 28_000
MAX_TRANSCRIPT_CHARS = 120_000
RESERVED_ANALYSIS_MEMORY_GIB = 8
TRANSCRIPT_CHARS_PER_AVAILABLE_GIB = 12_000
TRANSCRIPT_BUDGET_ENV = "TRANSCRIBER_MAX_TRANSCRIPT_CHARS"


def _available_memory_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None

    if pages <= 0 or page_size <= 0:
        return None
    return int(pages) * int(page_size)


def transcript_char_budget(available_memory_bytes: int | None = None) -> int:
    override = os.environ.get(TRANSCRIPT_BUDGET_ENV)
    if override:
        try:
            budget = int(override.replace("_", ""))
        except ValueError:
            print(f"  ⚠️  Ignoring invalid {TRANSCRIPT_BUDGET_ENV}={override!r}")
        else:
            if budget > 0:
                return budget
            print(f"  ⚠️  Ignoring non-positive {TRANSCRIPT_BUDGET_ENV}={override!r}")

    if available_memory_bytes is None:
        available_memory_bytes = _available_memory_bytes()
    if available_memory_bytes is None:
        return MIN_TRANSCRIPT_CHARS

    reserved_bytes = RESERVED_ANALYSIS_MEMORY_GIB * GIB
    extra_gib = max(0.0, (available_memory_bytes - reserved_bytes) / GIB)
    dynamic_budget = MIN_TRANSCRIPT_CHARS + int(
        extra_gib * TRANSCRIPT_CHARS_PER_AVAILABLE_GIB
    )
    return min(MAX_TRANSCRIPT_CHARS, max(MIN_TRANSCRIPT_CHARS, dynamic_budget))


def parse_recording_meta(audio_path: str) -> dict[str, str]:
    """
    Extract date and time from filenames produced by record_meeting.sh,
    which follow the pattern  meeting_YYYYMMDD_HHMM[SS].wav.
    Falls back to 'Unknown' for any field that cannot be parsed.
    """
    meta: dict[str, str] = {
        "title":    "Meeting Recording",
        "date":     "Unknown",
        "time":     "Unknown",
        "duration": "Unknown",
    }
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    m = re.search(r"(\d{8})_(\d{4,6})", stem)
    if m:
        try:
            dt = datetime.strptime(m.group(1) + m.group(2)[:4], "%Y%m%d%H%M")
            meta["date"] = dt.strftime("%B %d, %Y")
            meta["time"] = dt.strftime("%I:%M %p")
        except ValueError:
            pass
    return meta


def estimate_duration(lines: list[str]) -> str:
    """
    Read the end-timestamp of the last transcript line and return a
    human-friendly duration string, e.g. '1 hour, 2 minutes'.
    """
    if not lines:
        return "Unknown"
    m = re.search(r"-> (\d{2}):(\d{2}):(\d{2}\.\d+)", lines[-1])
    if not m:
        return "Unknown"
    total_secs = (
        int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    )
    td = timedelta(seconds=int(total_secs))
    h    = td.seconds // 3600
    mins = (td.seconds % 3600) // 60
    parts: list[str] = []
    if h:
        parts.append(f"{h} hour{'s' if h > 1 else ''}")
    if mins:
        parts.append(f"{mins} minute{'s' if mins > 1 else ''}")
    return ", ".join(parts) or "< 1 minute"


def build_transcript_body(lines: list[str]) -> str:
    """
    Strip the  [HH:MM:SS.xx -> HH:MM:SS.xx]  prefix from every line and
    join them into a single block of plain text, ready for analysis.
    Truncates with a warning if the result exceeds the configured prompt budget.
    """
    plain = [
        re.sub(r"^\[.*?\]\s*", "", line).strip()
        for line in lines
        if line.strip()
    ]
    body = "\n".join(plain)
    max_transcript_chars = transcript_char_budget()

    if len(body) > max_transcript_chars:
        print(
            f"  ⚠️  Transcript is long — truncating to {max_transcript_chars:,} chars "
            f"to stay within the analysis prompt budget."
        )
        body = body[:max_transcript_chars] + "\n[... transcript truncated ...]"

    return body


def compile(
    meta: dict[str, str],
    sections: list[tuple[str, str]],
    audio_path: str = "",
    *,
    report_title: str = "Meeting Report",
    source_label: str | None = None,
) -> str:
    """
    Assemble the final Markdown report from metadata and the generated
    (heading, text) section pairs returned by analysis.generate_summaries().

    audio_path is used to derive the source label when source_label is not
    provided (the existing meeting-recording path).  Pass source_label
    explicitly from non-audio callers such as youtube-summarize.py.

    report_title overrides the top-level Markdown heading so YouTube reports
    can use "Video Summary Report" while meeting reports keep "Meeting Report".
    """
    effective_source = (
        source_label if source_label is not None else os.path.basename(audio_path)
    )
    source_key = "Source" if source_label is not None else "Source file"

    lang_prob = meta.get("language_probability", "")
    detected_lang_line = (
        f"**Detected language:** {meta.get('detected_language', 'Unknown')} "
        f"({lang_prob} confidence)  "
        if lang_prob
        else f"**Detected language:** {meta.get('detected_language', 'Unknown')}  "
    )

    lines: list[str] = [
        f"# {report_title}",
        "",
        f"**{source_key}:** `{effective_source}`  ",
        f"**Date:** {meta['date']}  ",
        f"**Duration:** {meta['duration']}  ",
        f"**Requested language:** {meta.get('requested_language', 'Auto-detect')}  ",
        detected_lang_line,
        "",
        "---",
        "",
    ]

    for heading, text in sections:
        lines.extend([heading, "", text, ""])

    return "\n".join(lines)
