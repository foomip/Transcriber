# 🎙️ transcriber

A fully **local**, **private** meeting recorder, transcription, and summarization pipeline for Linux.
Record any online meeting **or** point it at a YouTube video — transcribe with [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) or fetch YouTube's built-in captions, then generate a structured Markdown report using a local Hugging Face analysis model. No cloud services. No API keys. Nothing leaves your machine.

## Table of Contents

- [How It Works](#how-it-works)
- [Requirements](#requirements)
  - [System packages](#system-packages)
  - [Python packages](#python-packages)
- [Installation](#installation)
- [Docker Workflow (Optional)](#docker-workflow-optional)
  - [Docker prerequisites](#docker-prerequisites)
  - [Building the images](#building-the-images)
  - [Running the Docker wrapper](#running-the-docker-wrapper)
  - [Testing in Docker](#testing-in-docker)
- [Quick Start — Meeting Recording](#quick-start--meeting-recording)
- [Quick Start — YouTube Summarization](#quick-start--youtube-summarization)
- [Output Files](#output-files)
  - [Transcript format](#transcript-format)
  - [Report format](#report-format)
- [Configuration](#configuration)
  - [Whisper model size](#whisper-model-size)
  - [Analysis model](#analysis-model)
  - [Transcript prompt budget](#transcript-prompt-budget)
- [Platform Notes — Pop!\_OS 24.04 with COSMIC Desktop](#platform-notes--pop_os-2404-with-cosmic-desktop)
  - [Dummy Output bug (kernel 6.16.x)](#dummy-output-bug-kernel-616x)
- [Troubleshooting](#troubleshooting)
- [Privacy](#privacy)

---

## How It Works

**Meeting recording path**

```
┌─────────────────────┐     WAV      ┌──────────────────────┐
│  record_meeting.sh  │ ──────────▶  │    transcribe.py     │
│                     │              │                      │
│  PipeWire / FFmpeg  │              │  Faster-Whisper      │
│  Desktop audio +    │              │  → timestamped       │
│  Microphone mixed   │              │    transcript (.txt) │
│  → 16 kHz mono WAV  │              │                      │  TXT + MD
└─────────────────────┘              │  Analysis model      │ ──────────▶  output/
                                     │  → meeting report    │
                                     │    (.md)             │
                                     └──────────────────────┘
```

**YouTube summarization path**

```
┌─────────────────────┐  transcript  ┌──────────────────────┐
│ youtube-summarize   │ ──────────▶  │  lib/analysis.py     │
│       .py           │              │                      │
│  YouTube oEmbed API │              │  Analysis model      │  TXT + MD
│  → title / metadata │              │  → video summary     │ ──────────▶  output/
│                     │              │    report (.md)      │
│  youtube-transcript │              └──────────────────────┘
│  -api → captions    │
└─────────────────────┘
```

Both paths share the same local analysis pipeline (`lib/analysis.py`, `lib/report.py`). No audio is downloaded or uploaded for the YouTube path — only the text transcript is fetched from YouTube's public subtitle endpoint.

`transcribe.py` auto-detects the available local accelerators at startup. Faster-Whisper transcription uses NVIDIA CUDA when CTranslate2 can see it and falls back cleanly to CPU otherwise. Analysis summarisation uses PyTorch device placement, so it can use NVIDIA CUDA or AMD ROCm when the matching PyTorch build is installed. ROCm summarisation uses Qwen2.5-3B-Instruct fully on the GPU by default because consumer AMD cards are most reliable when the model fits entirely in VRAM. If you want the higher-quality Gemma 4 analysis path, or your AMD GPU has too little VRAM, use the CPU analysis path instead.

---

## Requirements

### System packages

```bash
sudo apt update && sudo apt install ffmpeg pulseaudio-utils python3-venv -y
```

| Package            | Purpose                                                   |
| ------------------ | --------------------------------------------------------- |
| `ffmpeg`           | Captures and mixes the two audio streams                  |
| `pulseaudio-utils` | Provides `pactl` for PipeWire/PulseAudio device detection |
| `python3-venv`     | Creates the isolated Python environment                   |

> **PipeWire / PulseAudio** — any modern Linux distribution running PipeWire with the `pipewire-pulse` compatibility layer works out of the box. Ubuntu 22.04+, Fedora, Arch, and Pop!\_OS 22.04+ all qualify.

### Python packages

Installed automatically into the virtual environment during setup (see below).

| Package                   | Purpose                                                                        |
| ------------------------- | ------------------------------------------------------------------------------ |
| `faster-whisper >= 1.0.1` | CTranslate2-based Whisper inference                                            |
| `transformers >= 5.2.0`   | Loads the local Hugging Face analysis model selected for the active backend    |
| `torch`                   | PyTorch acceleration for analysis summarisation, including CUDA or ROCm builds |
| `accelerate`              | Enables `device_map="auto"` for automatic GPU placement                        |
| `youtube-transcript-api`  | Fetches YouTube captions/subtitles without an API key or headless browser      |

---

## Installation

Choose one of two workflows:

- **Native Python workflow** — create `whisper_env` locally and run the scripts directly.
- **Docker workflow** — keep recording on the host, but run transcription and analysis inside a container. This avoids host-side Python package management for the heavy-lifting steps.

The native Python workflow remains the default for local development and for `record_meeting.sh`.

### 1. Clone the repository

```bash
git clone https://github.com/your-username/transcriber.git
cd transcriber
```

### 2. Create the virtual environment and install dependencies

```bash
python3 -m venv whisper_env
source whisper_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For AMD GPUs, install a ROCm-enabled PyTorch build after the base requirements. Use the [PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose the wheel index that matches your ROCm version. ROCm-enabled PyTorch exposes AMD GPUs through the `torch.cuda` API, which is what the summarisation step uses for automatic placement.

ROCm analysis is designed to keep the selected model fully on the AMD GPU. This is more reliable than CPU/GPU offload on consumer ROCm systems, but it means low-VRAM AMD cards may not be suitable for the ROCm analysis path. Use CPU analysis if the AMD GPU cannot fit the ROCm model comfortably or if you prefer Gemma 4's higher-quality analysis over GPU speed.

### 3. Make the recording script executable

```bash
chmod +x record_meeting.sh
```

### 4. Copy the direnv example file

If you want to use the provided direnv setup, copy the example file into place:

```bash
cp .envrc.example .envrc
```

### 5. (Optional) Auto-activate the venv with direnv

If you have [direnv](https://direnv.net/) installed and hooked into your shell, the virtual environment activates and deactivates automatically whenever you `cd` into or out of the project directory:

```bash
# Install direnv (if not already installed)
sudo apt install direnv

# Add the shell hook — add this line to your ~/.bashrc or ~/.zshrc
eval "$(direnv hook bash)"   # or zsh / fish

# Allow the .envrc already present in the repo
direnv allow .
```

After copying `.envrc.example` to `.envrc`, no further configuration is needed.

---

## Docker Workflow (Optional)

The repository includes a Docker-based path for the transcription and report-generation steps. Audio capture still happens on the host with `record_meeting.sh`, so the privacy boundary stays local while Python dependencies, PyTorch wheels, and model runtime libraries live inside container images.

### Docker prerequisites

- Docker Engine 20.10+
- **NVIDIA GPU**: NVIDIA Container Toolkit plus a compatible NVIDIA driver
- **AMD / ROCm GPU**: ROCm-compatible host with `/dev/kfd` access
- **Intel GPU**: Intel graphics stack with `/dev/dri` access

> The host does **not** need a Python virtual environment for the Docker workflow. The host still needs `ffmpeg` and `pactl` if you want to record meetings locally with `record_meeting.sh`.

### Building the images

You can pre-build the images or let `docker-run-transcribe.sh` build the selected image on first use.

Build the shared base image first:

```bash
docker build -f Dockerfile.base -t transcriber:base .
```

Then build any variant you want to use:

```bash
docker build -f Dockerfile.cpu -t transcriber:cpu .
docker build -f Dockerfile.nvidia -t transcriber:nvidia .
docker build -f Dockerfile.rocm -t transcriber:rocm .
docker build -f Dockerfile.intel -t transcriber:intel .
```

If you want a safe default tag for manual Docker runs, point `latest` at the CPU image:

```bash
docker tag transcriber:cpu transcriber:latest
```

### Running the Docker wrapper

Make the wrapper executable once:

```bash
chmod +x docker-run-transcribe.sh
```

Then run it against a recorded WAV file:

```bash
./docker-run-transcribe.sh meeting_20260527_114300.wav
```

The wrapper automatically prefers backends in this order: **NVIDIA → ROCm → Intel → CPU**.

For ROCm runs, the wrapper passes `/dev/kfd`, `/dev/dri`, the required device owner groups, and `HSA_ENABLE_SDMA=0` so AMD GPU analysis runs by default on RDNA cards that otherwise fault during model transfers or generation. ROCm uses Qwen2.5-3B-Instruct because it fits fully on typical AMD GPU VRAM; it does not offload Gemma 4 into system RAM.

If your AMD GPU has limited VRAM, or you want the higher-quality Gemma 4 analysis path, force the CPU Docker image instead. The first two overrides below do that.

Useful overrides:

```bash
# Force CPU even if a GPU is available
FORCE_CPU=1 ./docker-run-transcribe.sh meeting_20260527_114300.wav
./docker-run-transcribe.sh --force-cpu meeting_20260527_114300.wav

# Force a specific image
./docker-run-transcribe.sh --image transcriber:rocm meeting_20260527_114300.wav

# Forward normal transcribe.py flags unchanged
./docker-run-transcribe.sh meeting_20260527_114300.wav -l en
```

The wrapper mounts:

- the selected audio file read-only under `/input/...`
- `output/` from the repository root to `/app/output`
- your HuggingFace cache (default: `~/.cache/huggingface`) to `/cache/huggingface`

Generated files still land in the same project `output/` directory:

- `output/<name>_transcript.txt`
- `output/<name>_report.md`

### Testing in Docker

Run the test suite inside any image by overriding the image entrypoint:

```bash
docker run --rm --entrypoint pytest transcriber:cpu
```

To test the current checkout instead of the code baked into the image, mount the repository into `/app`:

```bash
docker run --rm --entrypoint pytest -v "$(pwd)":/app transcriber:cpu
```

Quick accelerator smoke tests:

```bash
docker run --rm --entrypoint python transcriber:cpu -c "import torch; print(torch.cuda.is_available())"
docker run --rm --gpus all --entrypoint python transcriber:nvidia -c "import torch; print(torch.cuda.is_available())"
```

> `ENTRYPOINT` in the Docker images is `python transcribe.py`, so use `--entrypoint` whenever you want to run something else such as `pytest` or `python -c ...`.

---

## Quick Start — Meeting Recording

You will need **two terminal windows** open side by side.

### Step 1 — Start recording before your meeting begins

```bash
./record_meeting.sh
```

The script auto-detects your default audio output (desktop/system audio) and microphone, mixes them into a single stream, and begins writing a 16 kHz mono WAV file. The filename is stamped with the current date and time automatically:

```
🔍 Searching for PipeWire/PulseAudio devices...
🎙️  Microphone   : alsa_input.pci-0000_00_1f.3.analog-stereo
🔊  System audio : alsa_output.pci-0000_00_1f.3.analog-stereo.monitor
💾  Output file  : meeting_20260527_114300.wav
----------------------------------------------------
🛑  Press [CTRL+C] to stop recording.
----------------------------------------------------
```

You can also provide a custom output filename:

```bash
./record_meeting.sh q2_planning.wav
```

When the meeting ends, press `Ctrl+C`. The WAV file is finalised immediately.

### Step 2 — Transcribe and analyse

Activate the virtual environment if it is not already active, then pass the recording to `transcribe.py`:

```bash
source whisper_env/bin/activate   # skip if using direnv
python transcribe.py meeting_20260527_114300.wav
```

If you know the recording is in a specific language, pass a Whisper language code with `-l` or `--language`:

```bash
python transcribe.py -l en meeting_20260527_114300.wav
python transcribe.py --language=en meeting_20260527_114300.wav
```

The script will:

1. Detect the available transcription and summarisation backends
2. Transcribe the audio with Faster-Whisper
3. Save a transcript with language metadata and timestamped segments
4. Load the configured analysis model and generate a full meeting report; ROCm uses Qwen2.5-3B-Instruct on the AMD GPU by default

Long-running phases print elapsed-time progress messages so model downloads, model loading, transcription, and summary generation do not look stalled.

> **First run only:** the selected analysis model is downloaded to the HuggingFace model cache (`~/.cache/huggingface`). ROCm defaults to Qwen2.5-3B-Instruct. All subsequent runs load from disk.

---

## Quick Start — YouTube Summarization

Pass any YouTube URL to `youtube-summarize.py`. No recording or audio download is needed — the script fetches YouTube's existing captions.

```bash
source whisper_env/bin/activate   # skip if using direnv
python youtube-summarize.py https://www.youtube.com/watch?v=XmpKPs9Emx0
```

To specify a preferred transcript language and guide the summary output language, use `-l` or `--language` with the same language codes accepted by `transcribe.py`:

```bash
python youtube-summarize.py -l en https://www.youtube.com/watch?v=XmpKPs9Emx0
python youtube-summarize.py --language=de https://youtu.be/XmpKPs9Emx0
python youtube-summarize.py https://youtu.be/XmpKPs9Emx0 -l en
```

All three URL formats are accepted:

```bash
# Full watch URL
python youtube-summarize.py https://www.youtube.com/watch?v=XmpKPs9Emx0

# Short URL
python youtube-summarize.py https://youtu.be/XmpKPs9Emx0

# Bare 11-character video ID
python youtube-summarize.py XmpKPs9Emx0
```

The script will:

1. Extract the video ID and fetch the video title from YouTube's oEmbed API
2. Fetch the YouTube transcript (manual captions preferred; auto-generated as fallback)
3. Save a timestamped transcript file
4. Load the configured analysis model locally and generate a Video Summary Report

If the requested language transcript is not available, the script falls back to the first available transcript with a clear warning rather than failing.

> **First run only:** the analysis model download applies here too — see the note above.

---

## Output Files

All generated files are written into an `output/` subdirectory created automatically in the current working directory.

### Meeting recording outputs

For a recording named `meeting_20260527_114300.wav`:

| File                                            | Contents                                                   |
| ----------------------------------------------- | ---------------------------------------------------------- |
| `output/meeting_20260527_114300_transcript.txt` | Language metadata plus one timestamped line per segment    |
| `output/meeting_20260527_114300_report.md`      | Structured Markdown report generated by the analysis model |

### YouTube outputs

For a video with ID `XmpKPs9Emx0`:

| File                                  | Contents                                                        |
| ------------------------------------- | --------------------------------------------------------------- |
| `output/XmpKPs9Emx0_transcript.txt`   | YouTube metadata header plus one timestamped line per caption   |
| `output/XmpKPs9Emx0_report.md`        | Video Summary Report generated by the local analysis model      |

### Transcript format — meeting recording

```
# Transcription metadata
Source file: meeting_20260527_114300.wav
Whisper model: small
Requested language: English (en)
Detected language: English (en)
Detection confidence: 97%

# Transcript

[00:00:04.50 -> 00:00:12.30]  Hey everyone, let's look at the database schema update.
[00:00:12.80 -> 00:00:21.10]  We need to make sure the user_id column is indexed before deploying to staging.
[01:02:03.45 -> 01:02:08.90]  Agreed — let's get that merged by end of week.
```

### Transcript format — YouTube

```
# Transcription metadata
Source: YouTube
Video ID: XmpKPs9Emx0
Title: Example Video Title
URL: https://www.youtube.com/watch?v=XmpKPs9Emx0
Requested language: English (en)
Transcript language: English (en)
Transcript type: auto-generated

# Transcript

[00:00:04.50 -> 00:00:12.30]  Example caption segment.
[00:00:12.80 -> 00:00:21.10]  Another caption line here.
```

### Report format

Both pipelines produce the same five-section Markdown report. The analysis step uses deterministic decoding and refuses to write a report if the generated text appears unrelated to the transcript.

**Meeting report header:**

```markdown
# Meeting Report

**Source file:** `meeting_20260527_114300.wav`
**Date:** May 27, 2026
**Duration:** 1 hour, 2 minutes
**Requested language:** English (en)
**Detected language:** English (en) (97% confidence)
```

**YouTube report header:**

```markdown
# Video Summary Report

**Source:** `Example Video Title — youtube.com/watch?v=XmpKPs9Emx0`
**Date:** Unknown
**Duration:** 12 minutes
**Requested language:** English (en)
**Detected language:** English (en)
```

Both report types then continue with the same five sections:

```markdown
---

## Executive Summary

...

## Detailed Summary

...

## Action Items

...

## Key Decisions

...

## Topics Discussed

...
```

---

## Configuration

### Transcription language

Both scripts share the same `-l` / `--language` flag and the same set of supported language codes.

**`transcribe.py`** — forces Faster-Whisper to transcribe in the given language:

```bash
python transcribe.py -l en meeting_20260527_114300.wav
python transcribe.py --language=en meeting_20260527_114300.wav
python transcribe.py --language en meeting_20260527_114300.wav
```

**`youtube-summarize.py`** — prefers the matching YouTube transcript language and guides the summary output language:

```bash
python youtube-summarize.py -l en https://www.youtube.com/watch?v=XmpKPs9Emx0
python youtube-summarize.py --language=de https://youtu.be/XmpKPs9Emx0
```

If the requested language transcript is not available on YouTube, the script falls back to the first available transcript with a warning and continues. Invalid codes exit before any network requests and print the full supported language-code list sorted alphabetically by language name.

Common examples:

| Code | Language   |
| ---- | ---------- |
| `af` | Afrikaans  |
| `en` | English    |
| `pt` | Portuguese |

### Whisper model size

The default model is `small`, which gives better accuracy than `base` while remaining practical for local use. Change the `WHISPER_MODEL_SIZE` constant in `lib/transcription.py` to trade speed for accuracy:

| Model      | Size    | Relative speed | Notes                      |
| ---------- | ------- | -------------- | -------------------------- |
| `tiny`     | ~75 MB  | Fastest        | Suitable for quick drafts  |
| `base`     | ~145 MB | Fast           | Good speed/accuracy balance |
| `small`    | ~483 MB | Moderate       | **Default** — better accuracy |
| `medium`   | ~1.5 GB | Slow on CPU    | Recommended with a GPU     |
| `large-v3` | ~3 GB   | Slowest        | Best accuracy available    |

```python
# lib/transcription.py
WHISPER_MODEL_SIZE = "small"   # change here
```

### Analysis model

The default analysis model is backend-specific: CPU/CUDA use `google/gemma-4-E4B-it`, while ROCm uses `Qwen/Qwen2.5-3B-Instruct` fully on the AMD GPU.

Gemma 4 is the higher-quality default analysis model, but it is too large and transfer-heavy for reliable ROCm offload on many consumer AMD cards. The ROCm default therefore prioritises a model that fits fully in VRAM and completes reliably. If you want Gemma 4 on an AMD system, use the CPU analysis path rather than trying to offload Gemma 4 between AMD VRAM and system RAM.

To compare another local Hugging Face chat/instruct model without editing source, set `TRANSCRIBER_ANALYSIS_MODEL` for a single run:

```bash
TRANSCRIBER_ANALYSIS_MODEL="mistralai/Mistral-7B-Instruct-v0.3" \
   python transcribe.py meeting_20260527_114300.wav
```

The replacement model must load with `AutoModelForCausalLM.from_pretrained(...)` and provide a tokenizer chat template via `apply_chat_template(...)`.

### Transcript prompt budget

The transcript text sent to the analysis model is capped dynamically based on currently available RAM. The app reserves memory for the model/runtime, then raises the transcript budget on machines with more headroom while keeping a conservative ceiling for CPU inference.

To force a specific cap for one run, set `TRANSCRIBER_MAX_TRANSCRIPT_CHARS`:

```bash
TRANSCRIBER_MAX_TRANSCRIPT_CHARS=80000 python transcribe.py meeting_20260527_114300.wav
```

---

## Platform Notes — Pop!\_OS 24.04 with COSMIC Desktop

Pop!\_OS 24.04 runs PipeWire with the `pipewire-pulse` compatibility layer, so all `pactl` commands and FFmpeg's `-f pulse` flag work transparently. There is one known platform-specific issue to be aware of.

### Dummy Output bug (kernel 6.16.x)

Some machines running kernel 6.16.x experience an intermittent regression where the HDA audio driver loses the hardware device and PipeWire falls back to a null sink. `record_meeting.sh` detects this condition at startup and exits with a clear error rather than silently recording silence:

```
❌ Error: A dummy/null audio device was detected — your PipeWire session has
   lost track of the hardware. Reset it with:

   systemctl --user restart wireplumber pipewire pipewire-pulse
```

Run that command, then re-run the recording script. If the problem returns persistently, downgrading to kernel `6.12.x` resolves it until an upstream fix lands.

---

## Troubleshooting

**Audio devices not detected**

```bash
pactl list short sources   # lists all available sources
pactl info                 # shows current defaults
```

Set your desired device as the system default in your audio settings, then re-run the script.

**`No speech detected in the recording`**

The WAV file was silent or contained only noise below Whisper's detection threshold. Verify the recording has audible speech by opening it in any audio player before re-running `transcribe.py`.

**GPU not being used**

For Faster-Whisper transcription, confirm CTranslate2 can see a CUDA device. AMD ROCm is not exposed through this Faster-Whisper path, so AMD-only systems currently transcribe on CPU and can still accelerate the analysis summarisation step.

```bash
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

For analysis summarisation, confirm PyTorch can see your CUDA or ROCm device:

```bash
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('hip=', getattr(torch.version, 'hip', None)); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

If `cuda_available` is `False`, your PyTorch installation may not match your CUDA or ROCm driver stack. Refer to the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for the correct install command.

For the Docker workflow, verify that the matching image is being used and that Docker received the correct accelerator flags:

```bash
./docker-run-transcribe.sh --help-docker

docker run --rm --gpus all --entrypoint python transcriber:nvidia -c "import torch; print(torch.cuda.is_available())"
docker run --rm --entrypoint python transcriber:rocm -c "import torch; print(torch.cuda.is_available(), getattr(torch.version, 'hip', None))"
```

**AMD ROCm VRAM limits or Gemma 4 quality preference**

ROCm analysis keeps the model fully on the AMD GPU. This avoids the CPU/GPU offload path that can trigger ROCm memory access faults on consumer AMD cards, but it also means the ROCm model must fit in VRAM.

As a rough guide:

- 12-16 GB AMD GPUs should usually handle the ROCm default model comfortably.
- 8 GB AMD GPUs may be marginal, especially with long transcripts.
- 4-6 GB AMD GPUs are likely to fail or run out of memory on the ROCm analysis path.

If ROCm analysis fails with out-of-memory errors, GPU memory access faults, or repeated container crashes, use CPU analysis instead. This is also the recommended path when you want the higher-quality Gemma 4 analysis model:

```bash
./docker-run-transcribe.sh --force-cpu meeting_20260527_114300.wav
FORCE_CPU=1 ./docker-run-transcribe.sh meeting_20260527_114300.wav
```

Do not set `TRANSCRIBER_ANALYSIS_MODEL=google/gemma-4-E4B-it` while forcing the ROCm image unless you know the model fits and your ROCm stack is stable. On AMD systems, Gemma 4 is expected to be used through the CPU analysis path.

**Analysis model download fails or is slow**

The model downloads from HuggingFace Hub. If you are behind a proxy or have an unstable connection, you can pre-download it separately and it will be found in the cache automatically on the next run. For the ROCm default model:

```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
           AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); \
           AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"
```

**`torchvision::nms` error while loading the analysis model**

This app does not use torchvision. If Transformers tries to import a mismatched torchvision build, model loading can fail with `RuntimeError: operator torchvision::nms does not exist`. Remove the stray package from the virtual environment:

```bash
whisper_env/bin/python -m pip uninstall -y torchvision
```

---

## Privacy

**Meeting recording path** — everything runs entirely on your local machine:

- **Faster-Whisper** runs the Whisper model locally via CTranslate2
- The selected analysis model is downloaded once and runs fully offline thereafter
- No audio, transcript, or report data is ever transmitted anywhere

**YouTube summarization path** — two outbound requests are made:

- The video title is fetched from YouTube's public [oEmbed endpoint](https://www.youtube.com/oembed) (a single lightweight JSON request, no authentication)
- The transcript text is fetched from YouTube's public subtitle endpoint via [`youtube-transcript-api`](https://github.com/jdepoix/youtube-transcript-api)

No audio or video is downloaded. No local transcript or report data is uploaded. All analysis and summarization runs locally on your machine exactly as it does for meeting recordings.
