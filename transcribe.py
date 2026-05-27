#!/usr/bin/env python3
"""
transcribe.py — Meeting transcription and analysis pipeline.

Usage:
  python transcribe.py <path_to_audio.wav>

Pipeline:
  1. transcription.py  — detect GPU, run Faster-Whisper, produce timestamped lines
  2. analysis.py       — load LFM2-2.6B-Transcript, run five summary passes
  3. report.py         — compile everything into a structured Markdown report

Output (written alongside the source WAV):
  <name>_transcript.txt  — raw timestamped transcript
  <name>_report.md       — structured meeting report
"""

import os
import sys

from lib import analysis, report, transcription


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


def run(audio_path: str) -> None:
    if not os.path.exists(audio_path):
        print(f"❌  Error: File '{audio_path}' not found.")
        sys.exit(1)

    base = os.path.splitext(audio_path)[0]

    print("─" * 56)
    print("  🔎  Detecting compute device...")
    device, compute_type = transcription.detect_device()

    # ── Step 1: Transcribe ─────────────────────────────────────────────────
    print("\n▶ Step 1/3: Transcribing audio")
    lines = transcription.transcribe_audio(audio_path, device, compute_type)

    if not lines:
        print("⚠️  No speech detected in the recording. Exiting.")
        sys.exit(0)

    transcript_path = f"{base}_transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✅  Raw transcript saved → {transcript_path}")

    if not should_continue_with_analysis():
        print("\nTranscript-only run complete.")
        print("─" * 56)
        return

    # ── Step 2: Generate summaries ─────────────────────────────────────────
    print("\n▶ Step 2/3: Generating meeting report sections")
    meta = report.parse_recording_meta(audio_path)
    meta["duration"] = report.estimate_duration(lines)
    transcript_body  = report.build_transcript_body(lines)

    sections = analysis.generate_summaries(transcript_body, meta)

    # ── Step 3: Compile and save Markdown report ───────────────────────────
    print("\n▶ Step 3/3: Saving Markdown report")
    report_md   = report.compile(meta, sections, audio_path)
    report_path = f"{base}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\n📋  Meeting report saved → {report_path}")
    print("─" * 56)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("💡  Usage: python transcribe.py <path_to_audio.wav>")
        sys.exit(1)

    run(sys.argv[1])
