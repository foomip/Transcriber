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

# Keep the transcript body bounded so local analysis models have prompt headroom.
MAX_TRANSCRIPT_CHARS = 28_000


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

    if len(body) > MAX_TRANSCRIPT_CHARS:
        print(
            f"  ⚠️  Transcript is long — truncating to {MAX_TRANSCRIPT_CHARS:,} chars "
            f"to stay within the analysis prompt budget."
        )
        body = body[:MAX_TRANSCRIPT_CHARS] + "\n[... transcript truncated ...]"

    return body


def compile(
    meta: dict[str, str],
    sections: list[tuple[str, str]],
    audio_path: str,
) -> str:
    """
    Assemble the final Markdown report from metadata and the generated
    (heading, text) section pairs returned by analysis.generate_summaries().
    """
    lines: list[str] = [
        "# Meeting Report",
        "",
        f"**Source file:** `{os.path.basename(audio_path)}`  ",
        f"**Date:** {meta['date']}  ",
        f"**Duration:** {meta['duration']}  ",
        "",
        "---",
        "",
    ]

    for heading, text in sections:
        lines.extend([heading, "", text, ""])

    return "\n".join(lines)
