# Initial Implementation — Session Notes

**Date:** May 27, 2026  
**Scope:** Full project build from scratch, based on a prior Gemini design conversation.

---

## Table of Contents

1. [Project Origin](#1-project-origin)
2. [Initial File Creation](#2-initial-file-creation)
3. [Improvements Over the Gemini Design](#3-improvements-over-the-gemini-design)
4. [direnv Auto-Activation](#4-direnv-auto-activation)
5. [Pop!_OS 24.04 / COSMIC Compatibility Research](#5-popos-2404--cosmic-compatibility-research)
6. [README and .gitignore](#6-readme-and-gitignore)
7. [Modular Refactor](#7-modular-refactor)
8. [lib/ Directory Reorganisation](#8-lib-directory-reorganisation)
9. [Final Project Structure](#9-final-project-structure)

---

## 1. Project Origin

The starting point was a conversation with Gemini that produced a design for a DIY, fully local meeting transcription rig on Linux. The Gemini design consisted of two files:

- `record_meeting.sh` — capture desktop audio + microphone via PipeWire/PulseAudio and FFmpeg into a WAV file
- `transcribe.py` — transcribe the WAV with Faster-Whisper and write a timestamped `.txt` file

The design was sound but had several issues that were addressed before any file was written, and others that emerged through research during the session. None of the Gemini code was used as-is.

---

## 2. Initial File Creation

### `record_meeting.sh`

Captures both audio streams simultaneously using FFmpeg's PulseAudio input driver (`-f pulse`):

- **Desktop sink monitor** — the `.monitor` of the default PulseAudio sink, which captures all system/application audio (the other meeting participants)
- **Microphone source** — the default PulseAudio input (your own voice)

Both streams are mixed with FFmpeg's `amix` filter and written to a 16 kHz mono WAV. 16 kHz mono is Whisper's native input format, so no resampling is needed at transcription time.

Auto-names the output file with a `meeting_YYYYMMDD_HHMMSS.wav` timestamp if no filename argument is given.

### `transcribe.py` (original monolithic version)

The original single-file implementation covered:

1. Device detection for Faster-Whisper
2. Timestamp formatting
3. Whisper transcription
4. Gemma 4 model loading and querying
5. Five summary generation passes
6. Markdown report assembly

This file was later refactored — see [Section 7](#7-modular-refactor).

### `requirements.txt`

```
faster-whisper>=1.0.1
transformers>=5.2.0
torch
accelerate
```

`transformers>=5.2.0` is a hard requirement imposed by the analysis model. `accelerate` is required for `device_map="auto"` to work correctly.

---

## 3. Improvements Over the Gemini Design

### 3a. GPU auto-detection

**What changed:** Added automatic CUDA detection using `ctranslate2.get_cuda_device_count()` (already available as a transitive dependency of `faster-whisper`), selecting `device="cuda"` + `compute_type="float16"` when a GPU is found and falling back to `device="cpu"` + `compute_type="int8"` otherwise.

**Why:** The Gemini design hardcoded `device="cpu"`. This meant users with NVIDIA GPUs (which can run Whisper 4–10× faster) got no benefit without manually editing the file.

**Why `ctranslate2` rather than `torch.cuda.is_available()`:** CTranslate2 is already loaded at this point (Faster-Whisper depends on it), so we avoid an extra PyTorch import just for device detection. PyTorch is still used to print the GPU name, but wrapped in a `try/except` so it's non-fatal if not available.

For the Gemma 4 model, `device_map="auto"` with `torch_dtype="auto"` (HuggingFace Accelerate) handles GPU placement transparently with no additional detection code needed.

### 3b. Gemma 4 instead of Ollama

**What changed:** Replaced the Gemini suggestion of piping the transcript into a local Ollama instance with `google/gemma-4-E4B-it` loaded directly via the `transformers` library.

**Why:** Ollama is a general-purpose model server and requires a separate running daemon. Gemma 4 is the default model purpose-built specifically for meeting summarisation from transcripts — trained on meeting data, not general conversation. Key properties:

- 2.6B parameters, under 3 GB RAM for long meetings
- 32K token context window
- Recommended `temperature=0.3` (low, for factual accuracy)
- Specific input schema (title, date, time, duration, participant block, separator, transcript body)
- Generates five structured output types: executive summary, detailed summary, action items, key decisions, topics discussed

The model is downloaded once to the HuggingFace cache (`~/.cache/huggingface`) and runs fully offline thereafter. No API key, no daemon, no cloud.

**Five summary passes:** Rather than a single prompt, `analysis.py` runs five separate forward passes — one per section type — using the exact instruction strings from the Gemma 4 prompting recipe. This produces higher-quality, more focused output than a single combined prompt.

### 3c. Timestamp format

**What changed:** Replaced the Gemini format `[00.00s -> 04.50s]` with `[HH:MM:SS.xx -> HH:MM:SS.xx]`.

**Why:** The Gemini format broke for meetings longer than ~16 minutes. A one-hour meeting would produce timestamps like `[3723.10s -> 3728.40s]`, which is unreadable. The new format produces `[01:02:03.10 -> 01:02:08.40]` regardless of duration.

**Implementation:**

```python
def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"
```

### 3d. FFmpeg `amix` duration fix

**What changed:** Changed `duration=first` to `duration=longest` in the FFmpeg `amix` filter.

**Why:** The Gemini design used `duration=first`, which stops the recording as soon as the _first_ input stream ends. In the context of a live recording via PulseAudio, the desktop sink monitor stream could end (e.g. if the application briefly goes silent) before the microphone stream, silently cutting off the rest of the recording. `duration=longest` keeps recording until both streams have ended, which in practice means it runs until `CTRL+C`.

---

## 4. direnv Auto-Activation

**Files created/modified:** `.envrc`

**What:** Created a `.envrc` file that automatically activates the `whisper_env` virtual environment when entering the project directory and deactivates it on exit.

```bash
export VIRTUAL_ENV="$PWD/whisper_env"
PATH_add "$VIRTUAL_ENV/bin"
unset PYTHON_HOME
```

Ran `direnv allow .` to authorise the file. The user's shell (zsh) already had `eval "$(direnv hook zsh)"` in `.zshrc`.

**Why `PATH_add` instead of `source whisper_env/bin/activate`:** direnv works by diffing environment variable state before and after evaluating `.envrc` — it captures variable changes, not shell function calls. `PATH_add` is the direnv stdlib function for prepending to `PATH`; it registers the change in a way direnv can cleanly reverse when leaving the directory. Using `source activate` would set `VIRTUAL_ENV` and modify `PATH` but also attempt to redefine shell functions (like `deactivate`) which direnv cannot reverse. `unset PYTHON_HOME` prevents a stale system-level `PYTHON_HOME` variable from overriding the venv's interpreter.

---

## 5. Pop!_OS 24.04 / COSMIC Compatibility Research

**Files modified:** `record_meeting.sh`

Researched compatibility of the full stack against Pop!_OS 24.04 running the COSMIC desktop environment. Findings:

### What works without changes

| Component | Status | Notes |
|---|---|---|
| `ffmpeg -f pulse` | ✅ Works | pipewire-pulse compatibility layer is transparent to FFmpeg |
| `pactl info` (reading defaults) | ✅ Works | pipewire-pulse exposes the standard PulseAudio query API |
| Python venv / pip | ✅ Works | PEP 668 (externally managed env) is bypassed by the venv |
| `transformers>=5.2.0` | ✅ Works | Installed via pip inside venv, not from apt |
| `pulseaudio-utils` package | ✅ Works | Package name unchanged on Ubuntu 24.04 base |

### Known issue — Dummy Output bug (kernel 6.16.x)

An active regression in some Pop!_OS 24.04 installs on kernels 6.16.x causes the HDA audio driver to intermittently lose the hardware device. When this happens, PipeWire falls back to a null sink named `auto_null` or `dummy`. 

The original script would have silently accepted this device, run FFmpeg against it, and produced a completely silent WAV file with no warning or error.

**Fix added to `record_meeting.sh`:**

```bash
if echo "$DESKTOP_SINK $MIC_SOURCE" | grep -qiE "dummy|auto_null"; then
    echo "❌ Error: A dummy/null audio device was detected — your PipeWire session has"
    echo "   lost track of the hardware. Reset it with:"
    echo ""
    echo "   systemctl --user restart wireplumber pipewire pipewire-pulse"
    echo ""
    echo "   Then re-run this script. If the problem persists, check kernel version with"
    echo "   'uname -r' — kernel 6.16.x has a known HDA audio regression on Pop!_OS 24.04."
    exit 1
fi
```

The one-line fix (`systemctl --user restart wireplumber pipewire pipewire-pulse`) resets the entire PipeWire session stack and resolves the issue in most cases. If it recurs persistently, the stable kernel to fall back to is `6.12.x`.

### Awareness item — suspend/resume audio failure

Separately documented (but not patched, as it does not affect a workstation that stays on during meetings): Pop!_OS 24.04 has a known four-layer audio failure after suspend/resume involving ALSA hardware muting, PipeWire software muting, and the COSMIC panel applet losing its PipeWire connection. The same `systemctl --user restart wireplumber pipewire pipewire-pulse` command, followed by `killall cosmic-panel`, resolves it.

---

## 6. README and .gitignore

**Files created:** `README.md`, `.gitignore`

### README.md

Written specifically for a GitHub repository audience. Sections:

- ASCII pipeline diagram showing the data flow from audio capture through to output files
- System and Python requirements, split into separate tables (different install paths)
- Step-by-step installation (clone → venv → pip → chmod → optional direnv)
- Two-terminal quick start mirroring the actual workflow
- Output files table with real example content for both the transcript and report
- Whisper model size comparison table with the config constant to change
- Pop!_OS 24.04 / COSMIC platform notes covering the Dummy Output issue
- Troubleshooting section for the four most likely failure modes
- Privacy statement confirming fully local operation

### .gitignore

Prevents committing:

- `whisper_env/` — the Python virtual environment (large, machine-specific)
- `*.wav` — meeting recordings (personal data, large binary files)
- `*.txt` — transcript output files (personal data)
- `*_report.md` — generated meeting reports (personal data)
- `__pycache__/` and `lib/__pycache__/` — Python bytecode
- `.cache/` — HuggingFace model cache if ever redirected inside the project
- `.direnv/` — direnv's internal state directory

---

## 7. Modular Refactor

**Files created:** `lib/transcription.py`, `lib/analysis.py`, `lib/report.py`  
**Files modified:** `transcribe.py` (rewritten as thin orchestrator)

The original `transcribe.py` was ~290 lines covering five conceptually distinct concerns in a single file. It was split into three modules matching the three natural pipeline stages, with `transcribe.py` reduced to a ~70-line orchestrator.

### Module breakdown

#### `lib/transcription.py` — Stage 1: Audio → timestamped lines

| Symbol | Description |
|---|---|
| `WHISPER_MODEL_SIZE` | Configurable constant (`"base"` default) |
| `detect_device()` | Returns `(device, compute_type)` tuple for Faster-Whisper |
| `fmt_ts(seconds)` | Converts float seconds to `HH:MM:SS.xx` string |
| `transcribe_audio(path, device, compute_type)` | Runs Whisper, returns `list[str]` of timestamped lines |

#### `lib/analysis.py` — Stage 2: Transcript → generated section text

| Symbol | Description |
|---|---|
| `ANALYSIS_MODEL_ID` / `ANALYSIS_*` constants | Model ID, system prompt, temperature, token limit |
| `SUMMARY_TASKS` | `list[tuple[str, str]]` of `(heading, instruction)` pairs |
| `_build_user_message(...)` | Formats transcript + metadata into the Gemma 4 prompting schema |
| `_query(model, tokenizer, message)` | Runs one Gemma 4 forward pass, returns generated text only |
| `generate_summaries(transcript_body, meta)` | Loads model, runs all five passes, returns `list[tuple[str, str]]` |

#### `lib/report.py` — Stage 3: Sections → Markdown file

| Symbol | Description |
|---|---|
| `MAX_TRANSCRIPT_CHARS` | Context window safety limit (28,000 chars) |
| `parse_recording_meta(audio_path)` | Extracts date/time from `meeting_YYYYMMDD_HHMM.wav` filename |
| `estimate_duration(lines)` | Parses final timestamp to produce `'1 hour, 2 minutes'` string |
| `build_transcript_body(lines)` | Strips timestamps, joins plain text, enforces truncation limit |
| `compile(meta, sections, audio_path)` | Assembles final Markdown string from metadata and section pairs |

#### `transcribe.py` — Orchestrator

Imports from all three modules and wires them together in sequence. Handles all file I/O (reading the WAV path from argv, writing `_transcript.txt` and `_report.md`). Contains no business logic of its own.

### Key interface design decisions

**`generate_summaries()` returns `list[tuple[str, str]]`** (heading + text pairs) rather than a dict. This means `report.py` knows nothing about `SUMMARY_TASKS`, the analysis model, or the number/order of sections — it simply iterates over whatever pairs it receives. The analysis and report modules are fully decoupled.

**`build_transcript_body()` lives in `report.py`**, not `analysis.py`, even though its output is fed into the analysis step. Stripping timestamps and enforcing the context-window truncation limit are data-preparation concerns, not analysis model concerns. `transcribe.py` calls it first and passes the result to `generate_summaries()`.

---

## 8. lib/ Directory Reorganisation

**Files moved:** `transcription.py`, `analysis.py`, `report.py` → `lib/`  
**Files created:** `lib/__init__.py`  
**Files modified:** `transcribe.py` (import line), `.gitignore`

The three library modules were moved into a `lib/` subdirectory to keep the repository root clean — only the directly-executable entry point (`transcribe.py`) and the shell script (`record_meeting.sh`) remain at the top level alongside configuration and documentation files.

`lib/__init__.py` is empty; it exists solely to mark the directory as a Python package so that the import resolves correctly:

```python
# transcribe.py
from lib import analysis, report, transcription
```

`.gitignore` was updated to explicitly cover `lib/__pycache__/` in addition to the root `__pycache__/`.

---

## 9. Final Project Structure

```
transcriber/
├── lib/
│   ├── __init__.py
│   ├── transcription.py    # Stage 1: GPU detection, Whisper, timestamps
│   ├── analysis.py         # Stage 2: Gemma 4 model, five summary passes
│   └── report.py           # Stage 3: metadata parsing, Markdown assembly
├── docs/
│   └── initial-implementation.md   # this file
├── transcribe.py           # entry point — orchestrates the three stages
├── record_meeting.sh       # capture desktop audio + mic to WAV
├── requirements.txt        # pip dependencies
├── README.md               # installation and usage guide
├── .envrc                  # direnv venv auto-activation
└── .gitignore
```

### Pipeline data flow

```
record_meeting.sh
  └─ FFmpeg (-f pulse) ──────────────────────────► meeting_YYYYMMDD_HHMMSS.wav
                                                            │
transcribe.py                                               │
  ├─ transcription.detect_device()                         │
  ├─ transcription.transcribe_audio() ◄────────────────────┘
  │    └─ Faster-Whisper (base / CUDA or CPU)
  │    └─ returns list[str] of timestamped lines
  │         └───────────────────────────────────────► _transcript.txt
  │
  ├─ report.parse_recording_meta()
  ├─ report.estimate_duration()
  ├─ report.build_transcript_body()  ← strips timestamps, enforces 28K char limit
  │
  ├─ analysis.generate_summaries()
  │    └─ Gemma 4 (device_map="auto")
  │    └─ 5 × forward pass  (exec summary / detailed / actions / decisions / topics)
  │    └─ returns list[tuple[str, str]]  (heading, generated text)
  │
  └─ report.compile() ──────────────────────────────────► _report.md
```
