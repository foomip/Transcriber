#!/usr/bin/env python3
"""
transcribe.py — Meeting transcription and analysis pipeline.

Usage:
    python transcribe.py [-l LANGUAGE] <path_to_audio.wav>

Pipeline:
  1. transcription.py  — detect GPU, run Faster-Whisper, produce timestamped lines
    2. analysis.py       — load the local analysis model, generate report sections
  3. report.py         — compile everything into a structured Markdown report

Output (written alongside the source WAV):
  <name>_transcript.txt  — raw timestamped transcript
  <name>_report.md       — structured meeting report
"""

import os
import sys
from argparse import ArgumentParser, Namespace

from lib import analysis, report, transcription


def parse_args(argv: list[str]) -> Namespace:
    parser = ArgumentParser(
        description="Transcribe a meeting recording and generate a local report."
    )
    parser.add_argument(
        "audio_path",
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument(
        "-l",
        "--language",
        help=(
            "Optional Whisper language code to force during transcription "
            "(for example: en, af, pt). Omit this to auto-detect."
        ),
    )
    return parser.parse_args(argv)


def validate_language(language_code: str | None) -> str | None:
    if language_code is None:
        return None

    normalized = transcription.normalize_language_code(language_code)
    if transcription.language_description(normalized) is not None:
        return normalized

    print(f"❌  Unsupported language code: '{language_code}'")
    print("\nSupported language codes:")
    print(transcription.format_supported_languages())
    sys.exit(1)


def _language_display(language_code: str | None, description: str | None) -> str:
    if language_code is None or description is None:
        return "Auto-detect"
    return f"{description} ({language_code})"


def transcript_file_lines(
    audio_path: str,
    result: transcription.TranscriptionResult,
) -> list[str]:
    return [
        "# Transcription metadata",
        f"Source file: {os.path.basename(audio_path)}",
        f"Whisper model: {result.model_size}",
        "Requested language: "
        + _language_display(
            result.requested_language,
            result.requested_language_description,
        ),
        "Detected language: "
        + _language_display(
            result.detected_language,
            result.detected_language_description,
        ),
        f"Detection confidence: {result.language_probability:.0%}",
        "",
        "# Transcript",
        "",
        *result.lines,
    ]


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


def run(audio_path: str, language: str | None = None) -> None:
    if not os.path.exists(audio_path):
        print(f"❌  Error: File '{audio_path}' not found.")
        sys.exit(1)

    base = os.path.splitext(audio_path)[0]

    print("─" * 56)
    print("  🔎  Detecting compute device...")
    device, compute_type = transcription.detect_device()

    # ── Step 1: Transcribe ─────────────────────────────────────────────────
    print("\n▶ Step 1/3: Transcribing audio")
    result = transcription.transcribe_audio(
        audio_path,
        device,
        compute_type,
        language=language,
    )

    if not result.lines:
        print("⚠️  No speech detected in the recording. Exiting.")
        sys.exit(0)

    transcript_path = f"{base}_transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("\n".join(transcript_file_lines(audio_path, result)) + "\n")
    print(f"\n✅  Transcript saved → {transcript_path}")

    if not should_continue_with_analysis():
        print("\nTranscript-only run complete.")
        print("─" * 56)
        return

    # ── Step 2: Generate summaries ─────────────────────────────────────────
    print("\n▶ Step 2/3: Generating meeting report sections")
    meta = report.parse_recording_meta(audio_path)
    meta["duration"] = report.estimate_duration(result.lines)
    meta["requested_language"] = _language_display(
        result.requested_language,
        result.requested_language_description,
    )
    meta["detected_language"] = _language_display(
        result.detected_language,
        result.detected_language_description,
    )
    meta["language_probability"] = f"{result.language_probability:.0%}"
    meta["transcription_model"] = f"Faster-Whisper {result.model_size}"
    transcript_body  = report.build_transcript_body(result.lines)

    try:
        sections = analysis.generate_summaries(transcript_body, meta)
    except (analysis.AnalysisGroundingError, analysis.AnalysisModelError) as exc:
        print(f"\n❌  Analysis failed: {exc}")
        print("    The transcript was saved, but no Markdown report was written.")
        sys.exit(1)

    # ── Step 3: Compile and save Markdown report ───────────────────────────
    print("\n▶ Step 3/3: Saving Markdown report")
    report_md   = report.compile(meta, sections, audio_path)
    report_path = f"{base}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\n📋  Meeting report saved → {report_path}")
    print("─" * 56)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run(args.audio_path, validate_language(args.language))
