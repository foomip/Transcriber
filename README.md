# 🎙️ transcriber

A fully **local**, **private** meeting recorder and transcription pipeline for Linux.
Record any online meeting, transcribe it with [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper), and generate a structured Markdown report using [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct). No cloud services. No API keys. Nothing leaves your machine.

---

## How It Works

```
┌─────────────────────┐     WAV      ┌──────────────────────┐     TXT + MD
│  record_meeting.sh  │ ──────────▶  │    transcribe.py     │ ─────────────▶  output files
│                     │              │                      │
│  PipeWire / FFmpeg  │              │  Faster-Whisper      │
│  Desktop audio +    │              │  → timestamped       │
│  Microphone mixed   │              │    transcript (.txt) │
│  → 16 kHz mono WAV  │              │                      │
└─────────────────────┘              │  Qwen2.5-3B-Instruct│
                                     │  → meeting report    │
                                     │    (.md)             │
                                     └──────────────────────┘
```

`transcribe.py` auto-detects the available local accelerators at startup. Faster-Whisper transcription uses NVIDIA CUDA when CTranslate2 can see it and falls back cleanly to CPU otherwise. Analysis summarisation uses PyTorch device placement, so it can use NVIDIA CUDA or AMD ROCm when the matching PyTorch build is installed.

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
| `transformers >= 5.2.0`   | Loads the local Hugging Face analysis model, defaulting to Qwen2.5-3B-Instruct |
| `torch`                   | PyTorch acceleration for analysis summarisation, including CUDA or ROCm builds |
| `accelerate`              | Enables `device_map="auto"` for automatic GPU placement                        |

---

## Installation

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

## Quick Start

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

The script will:

1. Detect the available transcription and summarisation backends
2. Transcribe the audio with Faster-Whisper
3. Save a timestamped raw transcript
4. Load Qwen2.5-3B-Instruct on CUDA, ROCm, or CPU and generate a full meeting report

Long-running phases print elapsed-time progress messages so model downloads, model loading, transcription, and summary generation do not look stalled.

> **First run only:** Qwen2.5-3B-Instruct is downloaded to the HuggingFace model cache (`~/.cache/huggingface`). All subsequent runs load from disk.

---

## Output Files

For a recording named `meeting_20260527_114300.wav`, two files are produced alongside it:

| File                                     | Contents                                                   |
| ---------------------------------------- | ---------------------------------------------------------- |
| `meeting_20260527_114300_transcript.txt` | Raw timestamped transcript, one line per Whisper segment   |
| `meeting_20260527_114300_report.md`      | Structured Markdown report generated by the analysis model |

### Transcript format

```
[00:00:04.50 -> 00:00:12.30]  Hey everyone, let's look at the database schema update.
[00:00:12.80 -> 00:00:21.10]  We need to make sure the user_id column is indexed before deploying to staging.
[01:02:03.45 -> 01:02:08.90]  Agreed — let's get that merged by end of week.
```

### Report format

The Markdown report contains five sections generated in one grounded analysis-model inference pass. The analysis step uses deterministic decoding and refuses to write a report if the generated text appears unrelated to the transcript:

```markdown
# Meeting Report

**Source file:** `meeting_20260527_114300.wav`
**Date:** May 27, 2026
**Duration:** 1 hour, 2 minutes

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

### Whisper model size

The default model is `base`, which is fast on CPU and accurate enough for clear speech. Change the `WHISPER_MODEL_SIZE` constant at the top of `transcribe.py` to trade speed for accuracy:

| Model      | Size    | Relative speed | Notes                      |
| ---------- | ------- | -------------- | -------------------------- |
| `tiny`     | ~75 MB  | Fastest        | Suitable for quick drafts  |
| `base`     | ~145 MB | Fast           | **Default** — good balance |
| `small`    | ~483 MB | Moderate       | Noticeably better accuracy |
| `medium`   | ~1.5 GB | Slow on CPU    | Recommended with a GPU     |
| `large-v3` | ~3 GB   | Slowest        | Best accuracy available    |

```python
# transcribe.py — line 27
WHISPER_MODEL_SIZE = "small"   # change here
```

### Analysis model

The default analysis model is `Qwen/Qwen2.5-3B-Instruct`. To compare another local Hugging Face chat/instruct model without editing source, set `TRANSCRIBER_ANALYSIS_MODEL` for a single run:

```bash
TRANSCRIBER_ANALYSIS_MODEL="mistralai/Mistral-7B-Instruct-v0.3" \
   python transcribe.py meeting_20260527_114300.wav
```

The replacement model must load with `AutoModelForCausalLM.from_pretrained(...)` and provide a tokenizer chat template via `apply_chat_template(...)`.

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

**Analysis model download fails or is slow**

The model downloads from HuggingFace Hub. If you are behind a proxy or have an unstable connection, you can pre-download it separately and it will be found in the cache automatically on the next run:

```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
           AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); \
           AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"
```

**`torchvision::nms` error while loading Qwen**

This app does not use torchvision. If Transformers tries to import a mismatched torchvision build, model loading can fail with `RuntimeError: operator torchvision::nms does not exist`. Remove the stray package from the virtual environment:

```bash
whisper_env/bin/python -m pip uninstall -y torchvision
```

---

## Privacy

Everything runs entirely on your local machine:

- **Faster-Whisper** runs the Whisper model locally via CTranslate2
- **Qwen2.5-3B-Instruct** is downloaded once and runs fully offline thereafter
- No audio, transcript, or report data is ever transmitted anywhere
