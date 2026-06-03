# 🎙️ transcriber

A fully **local**, **private** meeting recorder, transcription, and summarization pipeline for Linux.
Record any online meeting **or** point it at a YouTube video - transcribe with [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) or fetch YouTube's built-in captions, then generate a structured Markdown report using a local Hugging Face analysis model. No cloud services. No API keys. Nothing leaves your machine.

## Table of Contents

- [How It Works](#how-it-works)
- [Requirements](#requirements)
  - [System packages](#system-packages)
  - [Python packages](#python-packages)
  - [Hardware recommendations](#hardware-recommendations)
- [Installation](#installation)
- [Docker Workflow (Optional)](#docker-workflow-optional)
  - [Docker prerequisites](#docker-prerequisites)
  - [Building the images](#building-the-images)
  - [Running the Docker wrapper](#running-the-docker-wrapper)
  - [Testing in Docker](#testing-in-docker)
- [Quick Start - Meeting Recording](#quick-start--meeting-recording)
- [Quick Start - YouTube Summarization](#quick-start--youtube-summarization)
- [Output Files](#output-files)
  - [Transcript format](#transcript-format)
  - [Report format](#report-format)
- [Configuration](#configuration)
  - [Whisper model size](#whisper-model-size)
  - [Analysis model](#analysis-model)
  - [Transcript prompt budget](#transcript-prompt-budget)
- [Platform Notes - Pop!\_OS 24.04 with COSMIC Desktop](#platform-notes--pop_os-2404-with-cosmic-desktop)
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

Both paths share the same local analysis pipeline (`lib/analysis.py`, `lib/report.py`). No audio is downloaded or uploaded for the YouTube path - only the text transcript is fetched from YouTube's public subtitle endpoint.

`transcribe.py` auto-detects the available local accelerators at startup. Faster-Whisper transcription uses NVIDIA CUDA or AMD ROCm when CTranslate2 can see a GPU, and falls back cleanly to CPU otherwise. Both GPU families use CTranslate2 `device="cuda"` internally — the distinction is handled automatically. Analysis summarisation uses llama.cpp/GGUF for all backends (CPU, NVIDIA, AMD ROCm, and Intel), providing a unified, resource-efficient inference engine that can dynamically split model layers between GPU VRAM and system RAM.

---

## Why a Local Pipeline? (vs. Cloud Services)

Most transcription and summarization tools rely on cloud APIs (like OpenAI, Google, or AssemblyAI). This app is designed as a **local-first alternative**.

| Feature | Local-First (This App) | Cloud-Based Solutions |
| :--- | :--- | :--- |
| **Privacy** | **Maximum**. Audio and transcripts never leave your machine. | **Limited**. Data is transmitted and stored on third-party servers. |
| **Cost** | **Free**. No subscriptions or per-token/per-minute fees. | **Recurring**. Monthly costs or "pay-as-you-go" API charges. |
| **Connectivity** | **Offline**. Works without internet once models are downloaded. | **Online**. Requires a stable connection to function. |
| **Control** | **Total**. You choose the specific Whisper and LLM models. | **Fixed**. You are limited to the models provided by the vendor. |
| **Hardware** | **Demanding**. Requires a decent CPU/GPU for speed. | **Zero**. Compute is handled by the cloud provider. |
| **Setup** | **Manual**. Requires local installation and config. | **Instant**. Sign up and start using via a web interface. |

**In short:** Use this app if you prioritize **privacy, data sovereignty, and cost-efficiency** over the convenience of a managed cloud service.

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

> **PipeWire / PulseAudio** - any modern Linux distribution running PipeWire with the `pipewire-pulse` compatibility layer works out of the box. Ubuntu 22.04+, Fedora, Arch, and Pop!\_OS 22.04+ all qualify.

### Python packages

Installed automatically into the virtual environment during setup (see below).

| Package                   | Purpose                                                                        |
| ------------------------- | ------------------------------------------------------------------------------ |
| `faster-whisper >= 1.0.1` | CTranslate2-based Whisper inference                                            |
| `llama-cpp-python`        | GGUF-based analysis model inference for all hardware targets                    |
| `huggingface-hub`         | Downloads the default GGUF analysis model                                       |
| `nvidia-ml-py`            | Provides `pynvml` for NVIDIA GPU VRAM probing                                   |
| `youtube-transcript-api`  | Fetches YouTube captions/subtitles without an API key or headless browser      |

### Hardware Recommendations

The pipeline has two main stages: transcription (Faster-Whisper) and summarization (llama.cpp with Gemma-4).
Below are GPU recommendations for each stage.

**Transcription (NVIDIA CUDA and AMD ROCm)**
- Minimum: Any NVIDIA or AMD GPU with **2 GB VRAM** – will run the Whisper small model in FP16.
- Recommended: **4 GB VRAM or more** (e.g. RTX 3060, RX 7600) for comfortable headroom and faster throughput.
- AMD ROCm: Requires a ROCm-enabled CTranslate2 build (see [AMD ROCm setup](#amd-rocm-transcription-setup) below). 
  - **Supported GPUs**: Generally RDNA 2 (RX 6000 series) and RDNA 3 (RX 7000 series) or newer.
  - **Unsupported GPUs**: Older architectures (e.g. RX 500 series / Vega) are not supported by the current ROCm toolchain and will fall back to CPU transcription.
- CPU fallback: used automatically when no GPU is detected or supported.

**Summarization (llama.cpp, works with CUDA and ROCm)**
- Minimum: **8 GB VRAM** (NVIDIA or AMD) - loads the Gemma-4 E4B Q4_K_M model with limited GPU offloading.
- Recommended: **12 GB VRAM or more** (e.g. RTX 3060-12GB, RTX 4070-12GB, RX 7900 XT) for smoother layer offloading and better throughput, especially with longer meetings.
- VRAM scales with context length; for very long transcripts (>1 h) consider 16 GB+.

**Overall system**
- RAM: 8 GB+ system memory is adequate; 16 GB+ recommended for multitasking.
- Storage: A few GB for models and temporary files; SSD preferred for faster model loading.

---

## Installation

Choose one of two workflows:

- **Native Python workflow** - create `whisper_env` locally and run the scripts directly.
- **Docker workflow** - keep recording on the host, but run transcription and analysis inside a container. This avoids host-side Python package management for the heavy-lifting steps.

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

For a native GPU build of `llama-cpp-python`, you must set `CMAKE_ARGS` during installation to enable CUDA or HIP kernels.

**NVIDIA CUDA:**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**AMD ROCm:**
```bash
CMAKE_ARGS="-DGGML_HIP=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

If no `CMAKE_ARGS` are provided, `pip` installs the CPU-only version.

#### AMD ROCm transcription setup

By default the `ctranslate2` wheel from PyPI does not include ROCm/HIP support. To enable AMD GPU transcription, run the provided helper after the standard install:

```bash
bash scripts/install_ctranslate2_rocm.sh
```

This force-reinstalls `ctranslate2` from the ROCm wheel index and falls back gracefully to the standard PyPI wheel when no ROCm-specific wheel is available. It also runs a quick smoke test to confirm GPU visibility:

```bash
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count(), ctranslate2.get_supported_compute_types('cuda'))"
```

On a working ROCm setup this prints a device count ≥ 1 and a list of supported compute types such as `['float16', 'int8_float16', 'int8']`.

> **Note:** CTranslate2 ROCm builds still use `device="cuda"` in the Python API. This is expected — Faster-Whisper and the transcription pipeline handle this transparently.

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

# Add the shell hook - add this line to your ~/.bashrc or ~/.zshrc
eval "$(direnv hook bash)"   # or zsh / fish

# Allow the .envrc already present in the repo
direnv allow .
```

After copying `.envrc.example` to `.envrc`, no further configuration is needed.

---

## Docker Workflow (Optional)

The repository includes a Docker-based path for the transcription and report-generation steps. Audio capture still happens on the host with `record_meeting.sh`, so the privacy boundary stays local while Python dependencies, model runtime libraries, and CUDA/ROCm toolkits live inside container images.

### Docker prerequisites

- Docker Engine 20.10+
- **NVIDIA GPU**: NVIDIA Container Toolkit plus a compatible NVIDIA driver
  - **Pop!_OS / Ubuntu users:** If you hit `failed to discover GPU vendor from CDI: no known GPU vendor found`, you must [install and configure the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  ```bash
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo mkdir -p /etc/cdi && sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  sudo systemctl restart docker
  ```
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

The ROCm image builds `llama-cpp-python` from source with HIP support. For manual AMD builds, you can reduce compile time by targeting your GPU architecture, for example RX 6000-series cards use `gfx1030`:

```bash
docker build --build-arg AMDGPU_TARGETS=gfx1030 -f Dockerfile.rocm -t transcriber:rocm .
```

When the wrapper auto-builds `transcriber:rocm` on a ROCm host, it reads `rocminfo` and passes the detected `gfx...` targets automatically. Set `AMDGPU_TARGETS` before running the wrapper if you want to override that detection.

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

For ROCm runs, the wrapper selects `transcriber:rocm`, passes `/dev/kfd`, `/dev/dri`, the required device owner groups, sets `HSA_ENABLE_SDMA=0`, and defaults `CT2_CUDA_ALLOCATOR=cub_caching` to avoid known RDNA2 CTranslate2 memory-fault crashes during Whisper model loading. It then mounts a GGUF cache at `/cache/transcriber/gguf`. The llama.cpp backend estimates a safe `n_gpu_layers` value from available ROCm VRAM and leaves the rest of the model in system RAM.

If your AMD GPU has limited VRAM, force the CPU Docker image instead. The first two overrides below do that.

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
- your GGUF cache (default: `~/.cache/transcriber/gguf`) to `/cache/transcriber/gguf`

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

## Quick Start - Meeting Recording

You will need **two terminal windows** open side by side.

### Step 1 - Start recording before your meeting begins

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

### Step 2 - Transcribe and analyse

Activate the virtual environment if it is not already active, then pass the recording to `transcribe.py`:

```bash
source whisper_env/bin/activate   # skip if using direnv
python transcribe.py meeting_20260527_114300.wav
```

If you know the recording is in a specific language, pass a Whisper language code with `-l` or `--language`:

```bash
python transcribe.py -l en meeting_20260527_114300.wav
python transcribe.py --language=en meeting_20260527_114300.wav
python transcribe.py --language en meeting_20260527_114300.wav
```

The script will:

1. Detect the available transcription and summarisation backends
2. Transcribe the audio with Faster-Whisper
3. Save a transcript with language metadata and timestamped segments
4. Load the configured analysis model and generate a full meeting report

Long-running phases print elapsed-time progress messages so model downloads, model loading, transcription, and summary generation do not look stalled.

> **First run only:** the default GGUF analysis model is downloaded (via Hugging Face Hub) into the Transcriber GGUF cache (`~/.cache/transcriber/gguf` by default). All subsequent runs load from disk.

---

## Quick Start - YouTube Summarization

Pass any YouTube URL to `youtube-summarize.py`. No recording or audio download is needed - the script fetches YouTube's existing captions.

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

If the requested language transcript is not available, the script falls back to the first available transcript with a clear warning rather than failing. Invalid codes exit before any network requests and print the full supported language-code list sorted alphabetically by language name.

> **First run only:** the analysis model download applies here too - see the note above.

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

### Transcript format - meeting recording

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
[01:02:03.45 -> 01:02:08.90]  Agreed - let's get that merged by end of week.
```

### Transcript format - YouTube

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

**Source:** `Example Video Title - youtube.com/watch?v=XmpKPs9Emx0`
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

**`transcribe.py`** - forces Faster-Whisper to transcribe in the given language:

```bash
python transcribe.py -l en meeting_20260527_114300.wav
python transcribe.py --language=en meeting_20260527_114300.wav
python transcribe.py --language en meeting_20260527_114300.wav
```

**`youtube-summarize.py`** - prefers the matching YouTube transcript language and guides the summary output language:

```bash
python youtube-summarize.py -l en https://www.youtube.com/watch?v=XmpKPs9Emx0
python youtube-summarize.py --language=de https://youtu.be/XmpKPs9Emx0
python youtube-summarize.py https://youtu.be/XmpKPs9Emx0 -l en
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
| `small`    | ~483 MB | Moderate       | **Default** - better accuracy |
| `medium`   | ~1.5 GB | Slow on CPU    | Recommended with a GPU     |
| `large-v3` | ~3 GB   | Slowest        | Best accuracy available    |

```python
# lib/transcription.py
WHISPER_MODEL_SIZE = "small"   # change here
```

### Analysis model

The default analysis model is `google/gemma-4-E4B-it`. For all backends (CPU, NVIDIA, Intel, and AMD ROCm), this model is run through the llama.cpp engine using GGUF quantizations, which allow the model to be split across GPU VRAM and system RAM.

For a native Python installation, a GGUF version of the model is downloaded automatically. For Docker runs, the images come with the necessary build tools, and the wrapper handles the GGUF cache.

To compare another local GGUF model without editing source, point to it explicitly:

```bash
TRANSCRIBER_LLAMA_CPP_MODEL_PATH="/path/to/model.gguf" \
  python transcribe.py meeting_20260527_114300.wav
```

The default download source is `ggml-org/gemma-4-E4B-it-GGUF`. To use a different Hugging Face GGUF repository that contains the same filename, set `TRANSCRIBER_LLAMA_CPP_MODEL_REPO`.

The automatic layer split defaults to 42 model layers, matching the Gemma 4 E4B text configuration. Override `TRANSCRIBER_LLAMA_CPP_LAYER_COUNT` only when using a different GGUF architecture.

The llama.cpp context window is sized automatically to hold the current transcript prompt plus the generated report. When the transcript length is known, the window is derived from the actual transcript size; otherwise it falls back to the configured `TRANSCRIBER_MAX_TRANSCRIPT_CHARS` budget. The result is capped at the model's trained 131072-token window. Set `TRANSCRIBER_LLAMA_CPP_CONTEXT_SIZE` only to pin a fixed window.

Advanced llama.cpp tuning is available through `TRANSCRIBER_LLAMA_CPP_MODEL_REPO`, `TRANSCRIBER_LLAMA_CPP_CONTEXT_SIZE`, `TRANSCRIBER_LLAMA_CPP_BATCH_SIZE`, `TRANSCRIBER_LLAMA_CPP_GPU_LAYERS`, `TRANSCRIBER_LLAMA_CPP_GPU_HEADROOM_GIB`, and `TRANSCRIBER_LLAMA_CPP_LAYER_COUNT`. The defaults are intended to be conservative.

To compare another local Hugging Face model, first convert it to GGUF format using the tools provided by the llama.cpp project.

### Transcript prompt budget

The transcript text sent to the analysis model is capped dynamically based on currently available RAM. The app reserves memory for the model/runtime, then raises the transcript budget on machines with more headroom while keeping a conservative ceiling for CPU inference.

To force a specific cap for one run, set `TRANSCRIBER_MAX_TRANSCRIPT_CHARS`:

```bash
TRANSCRIBER_MAX_TRANSCRIPT_CHARS=80000 python transcribe.py meeting_20260527_114300.wav
```

---

## Platform Notes - Pop!\_OS 24.04 with COSMIC Desktop

Pop!\_OS 24.04 runs PipeWire with the `pipewire-pulse` compatibility layer, so all `pactl` commands and FFmpeg's `-f pulse` flag work transparently. There is one known platform-specific issue to be aware of.

### Dummy Output bug (kernel 6.16.x)

Some machines running kernel 6.16.x experience an intermittent regression where the HDA audio driver loses the hardware device and PipeWire falls back to a null sink. `record_meeting.sh` detects this condition at startup and exits with a clear error rather than silently recording silence:

```
❌ Error: A dummy/null audio device was detected - your PipeWire session has
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

For Faster-Whisper transcription, confirm CTranslate2 can see a CUDA or ROCm device:

```bash
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count(), ctranslate2.get_supported_compute_types('cuda'))"
```

On NVIDIA this requires the standard PyPI `ctranslate2` wheel. On AMD ROCm, the standard PyPI wheel reports 0 devices — run `bash scripts/install_ctranslate2_rocm.sh` to install a ROCm-enabled build (see [AMD ROCm transcription setup](#amd-rocm-transcription-setup) above).

**AMD ROCm transcription: RDNA2 allocator quirk**

Some RDNA2 cards (RX 6000-series) report illegal memory access errors when CTranslate2 loads a Whisper model. If you hit this:

```bash
export CT2_CUDA_ALLOCATOR=cub_caching
python transcribe.py <audio.wav>
```

Add the export to your `.envrc` to make it permanent for native runs. The Docker wrapper (`docker-run-transcribe.sh`) now defaults this variable to `cub_caching` on ROCm runs and still lets you override it explicitly from the host environment when needed.

For analysis summarisation, confirm your hardware is detected by the llama.cpp backend:

```bash
python transcribe.py --help
```

If you are using Docker, verify that the matching image is being used and that Docker received the correct accelerator flags:

```bash
./docker-run-transcribe.sh --help-docker

docker run --rm --gpus all --entrypoint python transcriber:nvidia -c "import llama_cpp; print('llama_cpp ok')"
docker run --rm --entrypoint python transcriber:rocm -c "import llama_cpp; print('llama_cpp ok')"
```

**AMD ROCm VRAM limits**

ROCm Docker analysis uses llama.cpp and dynamically chooses how many Gemma 4 E4B GGUF layers to offload to the AMD GPU. This avoids the PyTorch/Transformers CPU/GPU offload path that can trigger ROCm memory access faults on consumer AMD cards.

As a rough guide:

- 12-16 GB AMD GPUs should usually offload most or all layers of the default Gemma 4 E4B GGUF model.
- 8 GB AMD GPUs should offload fewer layers and use more system RAM.
- 4-6 GB AMD GPUs may still work with a smaller or more heavily quantized GGUF model, but will be slower.

If ROCm llama.cpp analysis fails with out-of-memory errors, GPU memory access faults, or repeated container crashes, use a smaller local GGUF model or CPU analysis instead:

```bash
./docker-run-transcribe.sh --force-cpu meeting_20260527_114300.wav
FORCE_CPU=1 ./docker-run-transcribe.sh meeting_20260527_114300.wav
```

**`torchvision::nms` error while loading the analysis model**

This app does not use torchvision. If Transformers tries to import a mismatched torchvision build, model loading can fail with `RuntimeError: operator torchvision::nms does not exist`. Remove the stray package from the virtual environment:

```bash
whisper_env/bin/python -m pip uninstall -y torchvision
```

---

## Privacy

**Meeting recording path** - everything runs entirely on your local machine:

- **Faster-Whisper** runs the Whisper model locally via CTranslate2
- The analysis model is downloaded once and runs fully offline thereafter
- No audio, transcript, or report data is ever transmitted anywhere

**YouTube summarization path** - two outbound requests are made:

- The video title is fetched from YouTube's public [oEmbed endpoint](https://www.youtube.com/oembed) (a single lightweight JSON request, no authentication)
- The transcript text is fetched from YouTube's public subtitle endpoint via [`youtube-transcript-api`](https://github.com/jdepoix/youtube-transcript-api)

No audio or video is downloaded. No local transcript or report data is uploaded. All analysis and summarisation runs locally on your machine exactly as it does for meeting recordings.
