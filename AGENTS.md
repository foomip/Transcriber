# Project Guidelines

## Overview

- This is a local-only Linux meeting recorder, transcription, and summarization pipeline. Keep the privacy boundary intact: audio, transcripts, reports, and model inference stay on the user's machine.
- Use [README.md](README.md) for full setup, platform notes, and troubleshooting instead of duplicating those details here.

## Architecture

- [record_meeting.sh](record_meeting.sh) records desktop audio plus microphone through PipeWire/PulseAudio and FFmpeg into a 16 kHz mono WAV file.
- [transcribe.py](transcribe.py) is the CLI entry point. It checks the audio path, runs transcription, writes `<base>_transcript.txt`, runs analysis, then writes `<base>_report.md`.
- [lib/transcription.py](lib/transcription.py) owns CUDA detection, Faster-Whisper model selection, timestamp formatting, and raw transcript line generation.
- [lib/analysis.py](lib/analysis.py) owns llama.cpp/GGUF loading, prompting, and the five report-generation passes.
- [lib/report.py](lib/report.py) owns filename metadata parsing, transcript body preparation, duration estimation, and Markdown report assembly.

## Environment And Commands

- Python is pinned by [.tool-versions](.tool-versions) and the workspace uses the local `whisper_env` virtual environment, auto-activated by [.envrc](.envrc) when direnv is enabled.
- Setup: `python3 -m venv whisper_env && source whisper_env/bin/activate && pip install -r requirements.txt`.
- Record audio: `./record_meeting.sh [name.wav]`.
- Transcribe and summarize: `python transcribe.py <audio.wav>`.
- Run tests: `whisper_env/bin/python -m pytest`.
- Quick syntax check: `python -m py_compile transcribe.py lib/*.py`.
- After changing Python files, verify the touched files with Pylance and ensure the change adds no errors or warnings.
- Double-check Python compilation after Python file changes, preferably with `whisper_env/bin/python -m py_compile` on the touched files or the full project command above.
- CUDA smoke check: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"`.

## Code Change Verification

- Any Python code change must include accompanying relevant tests or updates to existing tests.
- Before completion, run the relevant tests and ensure they pass; for broad changes, run the full suite with `whisper_env/bin/python -m pytest`.
- Verify touched Python files with Pylance and ensure there are zero errors or warnings introduced by the change.
- Compile touched Python files with `whisper_env/bin/python -m py_compile <changed_files>`; for broad changes, use `whisper_env/bin/python -m py_compile transcribe.py lib/*.py tests/*.py`.
- Tests should mock Whisper, HuggingFace, GPU, filesystem, and audio boundaries unless the user explicitly asks for a real integration run.

## Conventions And Pitfalls

- Do not edit or search inside `whisper_env/` except when explicitly troubleshooting the environment; it is a local dependency tree.
- Treat `*.wav`, `*_transcript.txt`, and `*_report.md` as generated personal data. Do not inspect, commit, or summarize them unless the user explicitly asks.
- Keep recorder changes compatible with PipeWire through the PulseAudio compatibility layer; `pactl` and FFmpeg `-f pulse` are intentional.
- Preserve the `meeting_YYYYMMDD_HHMM[SS].wav` naming convention when changing recording/output behavior; [lib/report.py](lib/report.py) extracts metadata from that pattern.
- The Gemma 4 model has a finite context window; keep or adjust `MAX_TRANSCRIPT_CHARS` in [lib/report.py](lib/report.py) deliberately when changing summarization behavior.
- First runs may download large HuggingFace models into the user's cache. Avoid adding commands or tests that unexpectedly re-download models or process real recordings.
