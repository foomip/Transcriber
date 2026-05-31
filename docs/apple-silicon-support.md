# Apple Silicon Support — Implementation and Platform Guide

**Status:** In Progress — Platform support is evolving. Local-only transcription and analysis are available; Apple Silicon GPU acceleration is planned.

---

## New Feature: YouTube Summarization

The project now includes a standalone tool, [youtube-summarize.py](youtube-summarize.py), which allows users to generate summary reports for YouTube videos without needing to record or locally transcribe audio.

### How it Works
1. **Transcript Fetching:** Uses `youtube-transcript-api` to fetch existing manual or auto-generated subtitles directly from YouTube.
2. **Local Analysis:** Passes the fetched text to the existing `lib/analysis.py` pipeline to generate the same five-section Markdown report used for recorded meetings.
3. **Output:** Saves results to `output/<video_id>_transcript.txt` and `output/<video_id>_report.md`.

### Platform Performance
- **CPU/GPU:** Because it bypasses the computationally expensive Whisper transcription step, this feature is fast on all platforms.
- **Apple Silicon Acceleration:** The summarization phase uses the Gemma 4 model. Currently, this runs on the CPU on macOS. Once the **Phase 2** acceleration (detailed below) is implemented, `youtube-summarize.py` will automatically use the Apple Silicon GPU (via MPS) for significantly faster report generation.

---

## Background and Constraints

Two facts established by research before any implementation decisions were made:

1. **CTranslate2 (the engine under `faster-whisper`) has no MPS or Metal support.** An open GitHub issue (CTranslate2 #1607) has been tracking this since January 2024 with no landing date. A Metal PR (#1819) was closed without merging. `faster-whisper` will always run on CPU on macOS, regardless of chip generation.

2. **`device_map="auto"` in HuggingFace Accelerate does not automatically use MPS.** It is CUDA-first, then CPU. Placing Gemma 4 on an Apple Silicon GPU requires an explicit `device_map={"": "mps"}` combined with `torch_dtype=torch.float16`.

These two constraints determine the architecture for both phases.

---

## Phase 1 — Everything Runs on macOS (CPU only)

**Goal:** A user on an Intel or Apple Silicon Mac can install the project, record a meeting, and get a full transcript and report. No GPU acceleration in this phase.

### What already works without any changes

| Component | Status | Reason |
|---|---|---|
| `lib/transcription.py` — `detect_device()` | ✅ No change | `ctranslate2.get_cuda_device_count()` returns `0` on macOS; CPU fallback triggers automatically |
| `lib/analysis.py` — `detect_analysis_backend()` | ✅ No change | `torch.cuda.is_available()` returns `False` on macOS (no CUDA); CPU path triggers automatically |
| `lib/report.py` | ✅ No change | Pure Python, no platform-specific code |
| `transcribe.py` | ✅ No change | Pure Python orchestrator |
| `requirements.txt` | ✅ No change | `faster-whisper`, `transformers`, `torch`, `accelerate` all publish macOS wheels on PyPI |

### What needs to change

#### 1. `record_meeting.sh`

This is the only Phase 1 blocker. The entire script is Linux-specific:

- `pactl info` — PulseAudio/PipeWire CLI; not available on macOS
- `ffmpeg -f pulse` — PulseAudio FFmpeg input driver; not available on macOS
- The Dummy Output guard greps for PipeWire-specific device names

**macOS audio architecture:**
- macOS uses CoreAudio; FFmpeg accesses it through the `avfoundation` input driver (`-f avfoundation`)
- Microphone capture works out of the box via AVFoundation
- **System audio capture requires a third-party virtual audio device.** macOS provides no built-in loopback. The standard free option is [BlackHole](https://github.com/ExistentialAudio/BlackHole) (`brew install blackhole-2ch`)

**macOS device detection approach:**

```bash
# List all AVFoundation audio devices
DEVICE_LIST=$(ffmpeg -f avfoundation -list_devices true -i "" 2>&1)

# Audio section starts after "AVFoundation audio devices:"
# Each device line looks like:
#   [AVFoundation indev @ 0x...] [0] BlackHole 2ch
#   [AVFoundation indev @ 0x...] [1] MacBook Pro Microphone
#   [AVFoundation indev @ 0x...] [2] ZoomAudioDevice

# Extract the index of BlackHole (system audio capture)
BLACKHOLE_IDX=$(echo "$DEVICE_LIST" \
    | awk '/AVFoundation audio devices/,0' \
    | grep -i "blackhole" \
    | grep -oE '\[([0-9]+)\]' | head -1 \
    | tr -d '[]')

# Extract the index of the microphone
MIC_IDX=$(echo "$DEVICE_LIST" \
    | awk '/AVFoundation audio devices/,0' \
    | grep -iE "microphone|mic\b" \
    | grep -oE '\[([0-9]+)\]' | head -1 \
    | tr -d '[]')
```

**Recording command:**

```bash
# Both system audio (BlackHole) and mic present — mix them
ffmpeg -loglevel warning \
    -f avfoundation -i "none:$BLACKHOLE_IDX" \
    -f avfoundation -i "none:$MIC_IDX" \
    -filter_complex "amix=inputs=2:duration=longest:dropout_transition=2" \
    -ac 1 -ar 16000 "$OUTPUT_FILE"

# BlackHole not found — mic only, with a clear warning
ffmpeg -loglevel warning \
    -f avfoundation -i "none:$MIC_IDX" \
    -ac 1 -ar 16000 "$OUTPUT_FILE"
```

**Guard conditions to add for macOS:**
- BlackHole not installed → warn that system audio will not be captured, continue with mic-only (do not exit)
- Neither BlackHole nor a microphone found → error and exit with instructions
- No AVFoundation audio devices at all → error and exit

**Implementation approach:**

Wrap the existing Linux code in an OS check and add the macOS path alongside it:

```bash
OS=$(uname -s)

if [[ "$OS" == "Darwin" ]]; then
    # macOS path — AVFoundation
    ...
else
    # Linux path — PulseAudio/PipeWire (existing code, unchanged)
    ...
fi
```

This keeps the Linux path completely untouched and avoids any risk of regression.

#### 2. `README.md`

Add a **macOS** prerequisites block alongside the existing Linux block:

```markdown
### macOS

brew install ffmpeg
brew install blackhole-2ch   # for system audio capture (recommended)
```

Note that `python3-venv` and `pulseaudio-utils` are Linux-only. On macOS, Python ships with `venv` built in and `pulseaudio-utils` is not needed.

Add a **BlackHole setup note** explaining:
- Without BlackHole, only microphone audio is captured — remote participants will not appear in the transcript
- BlackHole appears in Audio MIDI Setup as a regular audio device
- The recording script detects it automatically by name; no configuration needed after install

---

## Phase 2 — Apple Silicon GPU Acceleration

**Goal:** On an Apple Silicon Mac (M1/M2/M3/M4), both the transcription step and the Gemma 4 analysis step use the GPU via Apple's native ML frameworks.

### Transcription — `lib/transcription.py`

Because `faster-whisper` / CTranslate2 will not run on the Apple Silicon GPU, a different backend is needed: **`mlx-whisper`**.

`mlx-whisper` is a Whisper implementation built on Apple's [MLX](https://github.com/ml-explore/mlx) framework. It runs on the Metal GPU and Neural Engine and does not use PyTorch at all. Recent versions include Flash Attention and batched decoding, achieving benchmarked 9.5× speedup over CPU on M-series hardware.

**Detection logic to add to `detect_device()`:**

```python
import platform

# Apple Silicon MPS path — checked before the CUDA probe
if platform.system() == "Darwin" and platform.machine() == "arm64":
    try:
        import mlx_whisper  # noqa: F401 — presence check only
        print("  ✅ Apple Silicon detected — transcription will use mlx-whisper (Metal)")
        return "mlx", "mlx"
    except ImportError:
        print("  ℹ️  Apple Silicon detected but mlx-whisper not installed — using CPU")
        print("       Install with: pip install mlx-whisper")
```

`"mlx"` is not a CTranslate2 device string — it is a sentinel value that `transcribe_audio()` branches on:

**New branch inside `transcribe_audio()`:**

```python
if device == "mlx":
    return _transcribe_mlx(audio_path)
else:
    return _transcribe_ctranslate2(audio_path, device, compute_type)
```

`_transcribe_mlx()` calls `mlx_whisper.transcribe()` and normalises the output into the same `list[str]` of `[HH:MM:SS.xx -> HH:MM:SS.xx]  text` lines that the rest of the pipeline already expects. The rest of `transcribe.py`, `report.py`, and `analysis.py` remain completely unaware of which backend was used.

**mlx-whisper model mapping:**

`faster-whisper` downloads CTranslate2-converted models. `mlx-whisper` loads from `mlx-community/whisper-*` repos on HuggingFace. A mapping is needed so `WHISPER_MODEL_SIZE = "base"` still works:

```python
_MLX_MODEL_MAP = {
    "tiny":     "mlx-community/whisper-tiny-mlx",
    "base":     "mlx-community/whisper-base-mlx",
    "small":    "mlx-community/whisper-small-mlx",
    "medium":   "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}
```

### Analysis — `lib/analysis.py`

**New branch in `detect_analysis_backend()`:**

```python
import platform
import torch

# Check MPS before CPU fallback
if platform.system() == "Darwin" and platform.machine() == "arm64":
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            device_name = "Apple Silicon GPU (MPS)"
            return AnalysisBackend(
                name="mps",
                device_name=device_name,
                model_kwargs={
                    "device_map": {"": "mps"},
                    "torch_dtype": torch.float16,
                },
            )
        except Exception:
            pass  # fall through to CPU
```

**Why `device_map={"": "mps"}` rather than `device_map="auto"`:**
`device_map="auto"` in HuggingFace Accelerate is CUDA-first; it does not enumerate MPS as a candidate. The dict form `{"": "mps"}` explicitly routes all model layers to the MPS device, which is the pattern recommended in the HuggingFace Accelerate MPS guide.

**`torch_dtype`:** `torch.float16` is used rather than `"auto"` because `"auto"` on MPS resolves to `float32` on some Accelerate versions, doubling memory usage with no accuracy benefit.

### `requirements.txt`

Add `mlx-whisper` with a platform marker so it is only installed on Apple Silicon Macs:

```
mlx-whisper ; sys_platform == "darwin" and platform_machine == "arm64"
```

`mlx` (the underlying framework) is a dependency of `mlx-whisper` and will be pulled in automatically. It should not be listed separately.

**Note on Intel Macs:** The `platform_machine == "arm64"` guard means `mlx-whisper` is not installed on Intel Macs (`x86_64`). Intel Macs remain on the CPU path for both transcription and analysis, which is correct — MLX requires Apple Silicon.

### `README.md`

Add an **Apple Silicon** subsection under the platform notes, covering:
- Transcription uses `mlx-whisper` on the Metal GPU / Neural Engine instead of `faster-whisper` / CTranslate2
- Gemma 4 analysis uses PyTorch MPS
- Both are installed automatically from `requirements.txt` on `arm64` macOS
- The `WHISPER_MODEL_SIZE` constant works the same way; larger models are available as `mlx-community/whisper-large-v3-mlx`
- First run downloads both the mlx-whisper model weights (~same sizes as faster-whisper) and the Gemma 4 model to `~/.cache/huggingface`

---

## Dockerization on Apple Silicon

While a Docker-based workflow is planned for Linux/Windows environments (via `Dockerfile.cpu`, `Dockerfile.nvidia`, etc.), **running an Apple Silicon-centered Docker image is natively constrained by Docker for Mac's architecture**:

1. **No GPU Passthrough:** Docker Desktop on macOS runs containers inside a Linux Virtual Machine. Apple's hardware acceleration frameworks—specifically Metal, MPS (`torch.mps`), and MLX—are strictly native to macOS and **cannot be passed through to Linux containers**.
2. **CPU-Only Execution:** Building an Apple Silicon Docker image (`linux/arm64`) means the container will run efficiently on the M-series CPU, but it will have absolutely **no access to the GPU or Neural Engine**. 
3. **Implications for New Features:** 
   - Operations like the newly added `youtube-summarize.py` will function correctly inside an ARM64 Docker container, but Gemma 4 analysis will be limited entirely to CPU-bound performance.
   - For users seeking full hardware acceleration (such as the 9.5× speedup provided by MLX in Phase 2), they **must** run the pipeline natively on macOS rather than through Docker.

If an Apple Silicon container image is provided for portability/convenience, it will effectively be a build of `Dockerfile.cpu` compiled for the `linux/arm64` architecture, accompanied by documentation explaining the fallback to CPU execution.

---

## Summary of All File Changes

### Phase 1

| File | Change |
|---|---|
| `record_meeting.sh` | Add `uname -s` OS check; add macOS AVFoundation recording path with BlackHole auto-detection and mic-only fallback |
| `README.md` | Add macOS prerequisites block; add BlackHole setup note |

### Phase 2

| File | Change |
|---|---|
| `lib/transcription.py` | Add Apple Silicon detection in `detect_device()`; add `_transcribe_mlx()` function; add model name mapping dict; branch in `transcribe_audio()` |
| `lib/analysis.py` | Add MPS branch in `detect_analysis_backend()` with `device_map={"": "mps"}` and `torch_dtype=torch.float16` |
| `requirements.txt` | Add `mlx-whisper ; sys_platform == "darwin" and platform_machine == "arm64"` |
| `README.md` | Add Apple Silicon GPU subsection under platform notes |

### No changes needed in either phase

| File | Reason |
|---|---|
| `transcribe.py` | Pure orchestrator, no platform-specific logic |
| `lib/report.py` | Pure Python data processing, fully cross-platform |
| `.envrc` | direnv is cross-platform; `PATH_add` works on macOS |
| `.gitignore` | No platform-specific entries needed |

---

## Open Questions Before Implementation

1. **BlackHole mic-only fallback UX** — When BlackHole is absent, should the script warn once and proceed automatically, or should it pause for confirmation? A recording that captures only the microphone is still useful, but the user should not be surprised to find remote participants missing from the transcript.

2. **Intel Mac support scope** — Intel Macs run on CPU for both phases. The Phase 2 `mlx-whisper` marker excludes them explicitly. Should they be mentioned in the README as a supported but unaccelerated platform, or left undocumented?

3. **mlx-whisper model download location** — `mlx-whisper` downloads from `mlx-community/*` on HuggingFace, landing in `~/.cache/huggingface`. This is separate from the `faster-whisper` model cache (which goes to `~/.cache/huggingface/hub/models--Systran--faster-whisper-*`). The README note about first-run downloads should be updated to reflect the correct size for the MLX model variant.
