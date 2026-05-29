# YouTube Summarization Plan

## Goal

Add a separate CLI script, `youtube-summarize.py`, that accepts a YouTube URL and generates a local Markdown summary report using the video's YouTube transcript instead of running Faster-Whisper transcription.

This feature should:

- reuse the existing local analysis pipeline where possible
- support an optional language flag with the same CLI style as `transcribe.py`
- save all generated files into a dedicated `output/` directory
- fetch YouTube video metadata such as title and description when available
- fall back gracefully when metadata cannot be fetched

---

## Agreed Decisions

### Output location

All generated output should go into a dedicated `output/` directory.

This applies to:

- the new `youtube-summarize.py` script
- the existing `transcribe.py` script

### Report title

Use:

`Video Summary Report`

for YouTube-generated reports.

### Language flag behavior

The language flag should behave like option **C**:

1. guide transcript selection from YouTube
2. guide the summarization/output language passed into the analysis metadata

### Video metadata

Fetch the YouTube video title and description.

If metadata fetch fails, fall back to using the video ID.

### CLI consistency

The language option must follow the same formatting style as `transcribe.py`.

Primary documented usage:

```bash
python youtube-summarize.py -l en <youtube_url>
```

Also supported via normal `argparse` behavior:

```bash
python youtube-summarize.py --language=en <youtube_url>
python youtube-summarize.py <youtube_url> -l en
```

---

## High-Level Design

### Existing pipeline today

```text
record_meeting.sh -> transcribe.py -> lib/transcription.py -> lib/analysis.py -> lib/report.py
```

### New YouTube pipeline

```text
youtube-summarize.py -> YouTube transcript fetch -> lib/analysis.py -> lib/report.py
```

The YouTube path will skip local speech-to-text entirely and instead use YouTube's existing transcript/caption data.

---

## Proposed Implementation

## 1. New script: `youtube-summarize.py`

### Responsibilities

- parse CLI args
- validate and normalize the optional language flag using the same helpers as `transcribe.py`
- accept a YouTube URL or bare video ID
- extract the video ID
- fetch video metadata
- fetch transcript/captions from YouTube
- format transcript segments into the same line format already used elsewhere:

```text
[HH:MM:SS.xx -> HH:MM:SS.xx]  segment text
```

- write transcript output into `output/`
- build metadata for analysis
- call `lib.analysis.generate_summaries(...)`
- write the final report into `output/`

### Planned CLI

```bash
python youtube-summarize.py -l en https://www.youtube.com/watch?v=XmpKPs9Emx0
```

### Pipeline steps

The script should mirror the style of `transcribe.py`:

1. fetch YouTube metadata and transcript
2. save transcript text file
3. optionally continue to analysis
4. generate summary sections
5. save Markdown report

---

## 2. Transcript source strategy

### Use YouTube transcript data instead of local transcription

Planned dependency:

- `youtube-transcript-api`

Reasoning:

- lightweight
- no API key required
- supports manually created and auto-generated transcripts
- supports language selection
- avoids running Faster-Whisper when a transcript already exists

### Transcript selection behavior

If `-l/--language` is provided:

1. try to fetch transcript in that language
2. prefer manually created transcript if available
3. fall back to auto-generated transcript in that language
4. if nothing exists in that language, optionally fall back to any available transcript with a clear message

If no language is provided:

- allow normal auto-selection behavior

### Transcript formatting

YouTube transcript segments usually provide:

- `text`
- `start`
- `duration`

These will be converted into the existing transcript line format using start and end timestamps.

This allows `lib.report.build_transcript_body(...)` and the rest of the downstream logic to work with minimal or no changes.

---

## 3. Metadata fetching

Fetch the following when possible:

- video title
- video description
- publish date if practical

### Fallback behavior

If metadata cannot be retrieved:

- use the video ID as the title fallback
- omit or mark unknown fields where necessary

### Intended report metadata

The generated metadata passed to analysis/reporting should include values such as:

- title
- date
- duration (estimated from transcript timestamps)
- requested language
- detected/selected transcript language
- transcript source

---

## 4. `output/` directory changes

### For `youtube-summarize.py`

Write outputs to:

- `output/<video_id>_transcript.txt`
- `output/<video_id>_report.md`

### For `transcribe.py`

Update existing behavior so outputs are also written to `output/`:

- `output/<audio_stem>_transcript.txt`
- `output/<audio_stem>_report.md`

### Behavior

- create `output/` automatically if missing
- keep source WAV files untouched in their original location

---

## 5. Planned reuse of existing modules

## `lib.analysis.py`

Planned approach:

- reuse as-is if possible
- pass transcript body and metadata dict just like `transcribe.py` does today

No major logic changes are expected here.

## `lib.report.py`

Likely small changes only.

### Possible updates

- allow report title override so YouTube reports can use `Video Summary Report`
- allow source label override so the report header can show a YouTube source instead of an audio filename
- keep existing meeting-report flow backward compatible

### WAV-specific parsing

`parse_recording_meta(...)` is specific to meeting audio filenames. The YouTube path should likely build its own metadata dict directly rather than trying to force YouTube data through the WAV filename parser.

---

## 6. Output file contents

## Transcript file

Planned format similar to existing transcript output, but adapted for YouTube:

```text
# Transcription metadata
Source: YouTube
Video ID: XmpKPs9Emx0
Title: Example Video Title
Requested language: English (en)
Transcript language: English (en)
Description: ...

# Transcript

[00:00:04.50 -> 00:00:12.30]  Example segment
```

## Report file

Use the existing Markdown section structure, but with title:

```markdown
# Video Summary Report
```

Sections should continue to use the current headings unless later changed separately:

- Executive Summary
- Detailed Summary
- Action Items
- Key Decisions
- Topics Discussed

---

## 7. Error handling plan

The new script should handle the following cases clearly:

- invalid YouTube URL
- unsupported or malformed video ID
- no transcript available for the video
- requested language transcript not available
- metadata fetch failure
- analysis model failure

### Failure behavior

- if transcript fetch fails entirely, exit with a clear error and do not write a report
- if transcript is saved but analysis fails, keep the transcript file and do not write the report
- if metadata fetch fails but transcript succeeds, continue using fallback metadata

---

## 8. Testing plan

Create a new test file:

- `tests/test_youtube_summarize.py`

### Planned coverage

- parse args and language flag handling
- URL parsing for:
  - full YouTube URL
  - short `youtu.be` URL
  - bare video ID
- transcript segment formatting into timestamped lines
- output path generation under `output/`
- `output/` directory creation
- metadata fallback to video ID
- analysis pipeline invocation with expected transcript body and metadata
- failure when no transcript is available
- language selection behavior

### Existing test updates

Update `tests/test_transcribe_cli.py` to expect output files under `output/` instead of beside the audio file.

All tests should continue mocking external boundaries:

- YouTube transcript API
- metadata HTTP fetches
- analysis model loading
- filesystem where needed

---

## 9. README update plan

Update `README.md` to document the new feature and the new output behavior.

### Planned README changes

#### How It Works

Show both input paths:

- local meeting recording path
- YouTube URL summarization path

#### Quick Start

Add a YouTube example:

```bash
python youtube-summarize.py -l en https://www.youtube.com/watch?v=XmpKPs9Emx0
```

#### Output Files

Document that all generated files now go into `output/`.

#### Requirements

Add `youtube-transcript-api` to Python dependencies.

#### Privacy

Clarify:

- meeting audio and summarization remain local
- YouTube summarization requires fetching transcript/metadata from YouTube
- no local audio/video upload occurs as part of this feature

---

## 10. File-by-file change summary

| File | Planned change |
| --- | --- |
| `youtube-summarize.py` | New CLI entry point for YouTube transcript fetch + summarization |
| `transcribe.py` | Write transcript/report outputs into `output/` |
| `lib/report.py` | Likely add optional report title/source override support |
| `lib/analysis.py` | Reuse as-is if possible |
| `requirements.txt` | Add `youtube-transcript-api` |
| `tests/test_youtube_summarize.py` | New tests |
| `tests/test_transcribe_cli.py` | Update output path expectations |
| `README.md` | Document YouTube support and `output/` directory behavior |

---

## Implementation Notes

- No implementation has been done yet.
- This document is the agreed plan only.
- The next step after approval is implementation plus tests and README updates.
