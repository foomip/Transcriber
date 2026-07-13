# Dockerization Plan for the Transcriber Application

**Project:** Local‑only meeting recorder, transcription, and summarization pipeline
**Document:** `docs/dockerization.md`
**Status:** Planning (no code changes applied yet)
**Last Updated:** 2026‑05‑30

---

## 1. Introduction

The current workflow requires users to manage a Python virtual environment (`whisper_env`) and to have the appropriate GPU drivers (CUDA, ROCm, or Intel) installed on the host in order to obtain hardware‑accelerated transcription and summarisation.

Containerising the heavy‑lifting steps (transcription and analysis) decouples the application from host‑specific Python packages and driver installations, while still allowing the host to perform audio capture via `record_meeting.sh`.

This document outlines a **phased implementation** that introduces:

* A common base Docker image containing OS‑level dependencies and the pure‑Python requirements.
* Variant images for CPU, NVIDIA GPU, AMD/ROCm GPU, and Intel GPU.
* An enhanced wrapper script (`docker‑run‑transcribe.sh`) that automatically detects the host’s hardware and selects the most appropriate image.
* Corresponding updates to `README.md` to guide users through building, running, and maintaining the containerised workflow.

No modifications to the existing Python source code (`transcribe.py`, `lib/*.py`) are required in any phase.

---

## 2. Goals

| Goal | Description |
|------|-------------|
| **Portability** | Users need only Docker (and, optionally, vendor‑specific container toolkits) to run the pipeline. |
| **Reproducibility** | All runtime dependencies are locked inside the image; the same image yields identical behaviour across hosts. |
| **Zero host‑side Python setup** | No virtual environment, `pip install`, or manual dependency resolution on the host. |
| **Automatic hardware selection** | The wrapper chooses the best‑available backend (NVIDIA > ROCm > Intel > CPU) without user intervention. |
| **Maintainability** | Shared base image reduces duplication; adding a new variant only requires a small Dockerfile. |
| **Future‑proofing** | Keeping `ffmpeg` in the base enables optional audio format conversion or video‑to‑audio extraction later. |

---

## 3. Prerequisites (host side)

| Requirement | Minimum version / notes |
|-------------|------------------------|
| Docker Engine | ≥ 20.10 |
| **For NVIDIA GPU acceleration** | NVIDIA Container Toolkit (installs `nvidia-docker2` or equivalent) and a compatible NVIDIA driver. |
| **For AMD/ROCm GPU acceleration** | ROCm‑compatible kernel and drivers; the wrapper uses `--device /dev/kfd --group-add video`. |
| **For Intel GPU acceleration** | Intel graphics driver with Level Zero (`intel-level-zero-gpu`) and access to `/dev/dri`. |
| **Optional (for future features)** | `ffmpeg` on the host if you wish to pre‑process audio outside the container (not required for the container itself). |

> **Note:** The container itself always contains `ffmpeg`, `libsndfile`, and other system libraries needed to read PCM WAV files. The host does **not** need these for the transcription step.

---

## 4. Phased Implementation

### Phase 1 – Common Base Image (`Dockerfile.base`)

**Objective:** Create a reusable foundation that installs OS packages, creates a non‑root user, copies the source tree, and installs all pure‑Python dependencies (everything except the accelerator‑specific PyTorch wheel).

**Steps:**

1. Create `Dockerfile.base` in the repository root with the following content (see Appendix A).
2. Verify that the image builds successfully:
   ```bash
   docker build -f Dockerfile.base -t transcriber:base .
   ```
3. Run a quick smoke test to ensure `python transcribe.py --help` works inside the container:
   ```bash
   docker run --rm transcriber:base python transcribe.py --help
   ```

**Outcome:** A lightweight base image (~150 MB) that can be extended for any hardware target.

---

### Phase 2 – Variant Dockerfiles

**Objective:** Produce four concrete images that add the appropriate PyTorch/ROCm/Intel wheel to the base.

**Files to create:**

| Variant | Dockerfile | Base Image | PyTorch install command |
|---------|------------|------------|--------------------------|
| CPU | `Dockerfile.cpu` | `transcriber:base` | `pip install torch` |
| NVIDIA | `Dockerfile.nvidia` | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` (or latest LTS) | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| AMD/ROCm | `Dockerfile.rocm` | `rocm/pytorch:latest` (official ROCm‑PyTorch image) | *none* (image already contains ROCm‑enabled PyTorch) |
| Intel | `Dockerfile.intel` | `transcriber:base` | `pip install torch intel-extension-for-pytorch` (plus optional Level‑Zero packages) |

**Steps for each variant:**

1. Write the Dockerfile (see Appendix B‑E).
2. Build the image and tag it according to the convention:
   ```bash
   docker build -f Dockerfile.nvidia -t transcriber:nvidia .
   docker build -f Dockerfile.rocm   -t transcriber:rocm .
   docker build -f Dockerfile.intel  -t transcriber:intel .
   docker build -f Dockerfile.cpu    -t transcriber:cpu .
   ```
3. Optionally set `latest` to the CPU image:
   ```bash
   docker tag transcriber:cpu transcriber:latest
   ```
4. Test each variant with a small sample WAV (or a dummy file) to confirm that the container starts and can import `torch`:
   ```bash
   docker run --rm transcriber:nvidia python -c "import torch; print(torch.cuda.is_available())"
   ```

**Outcome:** Four ready‑to‑run images, each ~2‑4 GB (dominated by the PyTorch wheel).

---

### Phase 3 – Enhanced Wrapper Script (`docker‑run‑transcribe.sh`)

**Objective:** Provide a single command that:

* Detects host hardware (NVIDIA > ROCm > Intel > CPU).
* Selects the matching image (or respects user overrides).
* Applies the necessary Docker run flags (`--gpus all`, `--device /dev/kfd`, `--device /dev/dri`).
* Mounts the audio file, an `output/` directory, and the HuggingFace cache.
* Executes `transcribe.py` inside the container, forwarding any user‑supplied arguments.

**Steps:**

1. Create `docker‑run‑transcribe.sh` in the repository root (make it executable).
2. Implement detection functions (`_has_nvidia`, `_has_rocm`, `_has_intel`) as described in Appendix F.
3. Add logic to honor `FORCE_CPU=1` or a `--image <tag>` flag for manual overrides.
4. Ensure the script builds the selected image on‑first‑use if it is missing (optional convenience feature).
5. Test the wrapper with each hardware scenario (you can simulate by temporarily unsetting detection flags or using `--force-cpu`).

**Outcome:** Users invoke:
```bash
./docker-run-transcribe.sh meeting.wav   # auto‑selects backend
./docker-run-transcribe.sh --force-cpu meeting.wav
./docker-run-transcribe.sh --image transcriber:rocm meeting.wav
```
and obtain the same `*_transcript.txt` and `*_report.md` files in `./output/`.

---

### Phase 4 – Documentation Updates

**Objective:** Update `README.md` to reflect the new Docker‑based workflow, prerequisites, and maintenance instructions.

**Sections to add/modify (see Appendix G):**

* **Prerequisites** – add Docker and vendor‑specific container toolkit notes.
* **Installation / Build** – show how to build all variants (or rely on on‑demand building).
* **Usage – Docker‑based workflow** – basic invocation, forcing a backend, passing `transcribe.py` options.
* **Testing inside Docker** – how to run the test suite.
* **Maintenance notes** – image size, rebuilding when `requirements.txt` changes, updating base images.

**Steps:**

1. Draft the markdown changes.
2. Insert them into the existing `README.md` (preserve existing sections).
3. Verify that the rendered README is clear and correctly formatted.

---

### Phase 5 – Testing and Validation

**Objective:** Ensure that the containerised pipeline produces bit‑identical output to the native workflow for a variety of inputs and configurations.

**Steps:**

1. Select a few representative WAV files (different lengths, languages, presence/absence of speech).
2. Run the native workflow:
   ```bash
   source whisper_env/bin/activate
   python transcribe.py file.wav
   ```
3. Run the Dockerised workflow via the wrapper for each variant:
   ```bash
   ./docker-run-transcribe.sh file.wav          # auto
   FORCE_CPU=1 ./docker-run-transcribe.sh file.wav
   ./docker-run-transcribe.sh --image transcriber:nvidia file.wav
   ./docker-run-transcribe.sh --image transcriber:rocm file.wav
   ./docker-run-transcribe.sh --image transcriber:intel file.wav
   ```
4. Compare the generated `_transcript.txt` and `_report.md` files (e.g., using `diff -u`). They should be identical except for possible non‑functional metadata such as timestamps in the report (which are derived from the file name and thus identical).
5. Run the test suite inside each image:
   ```bash
   docker run --rm -v "$(pwd)":/app transcriber:latest pytest
   ```
   Repeat for each variant.

**Outcome:** Confirmation that the Dockerised workflow is functionally equivalent to the native one.

---

### Phase 6 – Rollout and Maintenance

**Objective:** Transition users to the Docker‑based workflow while keeping the native workflow available for those who prefer it.

**Steps:**

1. Announce the new method in the project’s communication channels (e.g., README, release notes).
2. Keep the existing instructions for the virtual‑env workflow as a “legacy” option, marked as optional.
3. Monitor for issues (e.g., missing devices, permission problems) and adjust the wrapper detection logic as needed.
4. Periodically (e.g., monthly) rebuild images from the latest base images to incorporate security patches.
5. When `requirements.txt` changes, increment a version tag (e.g., `transcriber:cpu-v2`) and update the wrapper’s image‑selection logic or ask users to rebuild.

---

## 5. Optional Optimisations (post‑implementation)

| Idea | Benefit |
|------|---------|
| **Multi‑stage build** – compile wheels in a separate stage to reduce final image size. |
| **Cache busting** – use build‑args for `USER_ID`/`GROUP_ID` to match the host user without needing `--user` at runtime. |
| **Signing & SBOM** – generate a Software Bill of Materials for each image to improve supply‑chain security. |
| **GitHub Actions** – automate building and pushing all variants to a container registry (e.g., GHCR) on every push to `main`. |
| **Help flag** – add `--help-docker` to the wrapper to print usage notes. |

These are **out of scope** for the initial phased implementation but can be considered later.

---

## 6. Appendices

### Appendix A – `Dockerfile.base`

```Dockerfile
# ---------- Dockerfile.base ----------
# Common base image: OS packages, non-root user, source tree, pure-Python deps
# Use Ubuntu 24.04 + the deadsnakes Python 3.14 PPA instead of the official
# python:3.14-slim images. The Debian bookworm/trixie variants are currently
# flagged by Docker DX for unpatched perl/tar/xz CVEs; Ubuntu's package set is
# not affected by the same advisories.
FROM ubuntu:24.04@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6

# Install Python 3.14 via deadsnakes, then install system build/runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.14 \
        python3.14-dev \
        python3.14-venv \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        libsndfile1 \
        ffmpeg \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.14 the default `python3` / `python`, then bootstrap pip.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.14 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.14 1 && \
    python3.14 -m ensurepip --upgrade && \
    python3.14 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Create a non-root user (UID/GID overridden at runtime via --user)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN existing_group="$(getent group "$GROUP_ID" | cut -d: -f1 || true)" && \
    if [ -n "$existing_group" ] && [ "$existing_group" != "appgroup" ]; then \
        groupmod -n appgroup "$existing_group"; \
    elif [ -z "$existing_group" ]; then \
        addgroup --gid "$GROUP_ID" appgroup; \
    fi && \
    existing_user="$(getent passwd "$USER_ID" | cut -d: -f1 || true)" && \
    if [ -n "$existing_user" ] && [ "$existing_user" != "appuser" ]; then \
        usermod -l appuser -d /home/appuser -m -g "$GROUP_ID" "$existing_user"; \
    elif [ -z "$existing_user" ]; then \
        adduser --uid "$USER_ID" --gid "$GROUP_ID" --disabled-password --gecos "" appuser; \
    else \
        usermod -g "$GROUP_ID" appuser; \
    fi

WORKDIR /app

# Install Python dependencies (pure deps first, then llama-cpp-python).
COPY requirements.txt .
RUN grep -v '^llama-cpp-python$' requirements.txt > requirements_no_llama.txt && \
    python -m pip install --no-cache-dir -r requirements_no_llama.txt && \
    rm requirements_no_llama.txt && \
    FORCE_CMAKE=1 \
    python -m pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python llama-cpp-python

COPY . .

USER appuser
ENTRYPOINT ["python", "transcribe.py"]
# --------------------------------------
```

### Appendix B – `Dockerfile.cpu`

```Dockerfile
# ---------- Dockerfile.cpu ----------
FROM transcriber:base

# Install CPU‑only PyTorch
RUN pip install --no-cache-dir torch

# Entrypoint inherited from base
# --------------------------------------
```

### Appendix C – `Dockerfile.nvidia`

```Dockerfile
# ---------- Dockerfile.nvidia ----------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
        ffmpeg \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN addgroup --gid $GROUP_ID appgroup && \
    adduser --uid $USER_ID --gid $GROUP_ID --disabled-password --gecos "" appuser

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# CUDA‑enabled PyTorch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

COPY . .
USER appuser
ENTRYPOINT ["python", "transcribe.py"]
# --------------------------------------
```

### Appendix D – `Dockerfile.rocm`

```Dockerfile
# ---------- Dockerfile.rocm ----------
FROM rocm/pytorch:latest

# System packages (the base image may already have them; we add for safety)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
        ffmpeg \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN addgroup --gid $GROUP_ID appgroup && \
    adduser --uid $USER_ID --gid $GROUP_ID --disabled-password --gecos "" appuser

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
USER appuser
ENTRYPOINT ["python", "transcribe.py"]
# --------------------------------------
```

### Appendix E – `Dockerfile.intel`

```Dockerfile
# ---------- Dockerfile.intel ----------
FROM transcriber:base

# Install Intel‑extension‑for‑PyTorch (requires Level Zero runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        intel-level-zero-gpu \
        level-zero \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch intel-extension-for-pytorch

# Entrypoint inherited from base
# --------------------------------------
```

### Appendix F – Detection Logic (bash functions) for `docker‑run‑transcribe.sh`

```bash
_has_nvidia() { command -v nvidia-smi >/dev/null 2>&1; }
_has_rocm()   { [ -c /dev/kfd ] && rocminfo >/dev/null 2>&1; }
_has_intel()  {
    # Check for any render node and Intel vendor ID
    [ -d /dev/dri ] && ls /dev/dri/render* >/dev/null 2>&1 && \
    ( command -v clinfo >/dev/null 2>&1 || \
      ls /sys/class/drm/*/device/vendor 2>/dev/null | grep -q 8086 )
}
```

### Appendix G – Sample README Updates (markdown snippet)

```markdown
## Prerequisites

- Docker Engine ≥ 20.10
- **NVIDIA GPU**: NVIDIA Container Toolkit + compatible driver
- **AMD/ROCm GPU**: ROCm‑compatible kernel & drivers
- **Intel GPU**: Intel graphics driver with Level Zero (`intel-level-zero-gpu`) and `/dev/dri` access
- (Optional) `ffmpeg` on the host if you wish to pre‑process audio outside the container

## Building the Docker images (optional but recommended)

You can pre‑build any or all variants to avoid the first‑run build delay, or let the wrapper build the needed image on‑demand.

To build a specific variant:
```bash
docker build -f Dockerfile.cpu      -t transcriber:cpu      .
docker build -f Dockerfile.nvidia   -t transcriber:nvidia   .
docker build -f Dockerfile.rocm     -t transcriber:rocm     .
docker build -f Dockerfile.intel    -t transcriber:intel    .
```

To build all at once, run the four commands above (or script them).
After building, you may set the `latest` alias to the CPU image (safe default):
```bash
docker tag transcriber:cpu transcriber:latest
```

If you prefer not to pre‑build, simply run the wrapper; it will automatically build the selected image the first time it is needed (provided the corresponding Dockerfile exists and you have network access to download base layers).

## Usage – Docker‑based workflow

```bash
# Record a meeting (unchanged)
./record_meeting.sh my_meeting.wav

# Transcribe & summarise – wrapper auto‑selects the best backend
./docker-run-transcribe.sh my_meeting.wav   # add -l en etc. if needed

# Force a specific backend
FORCE_CPU=1 ./docker-run-transcribe.sh my_meeting.wav
./docker-run-transcribe.sh --image transcriber:rocm my_meeting.wav

# All options supported by transcribe.py are forwarded:
./docker-run-transcribe.sh my_meeting.wav -l fr
```

After the run finishes, find the results in `./output/`:
- `my_meeting_transcript.txt`
- `my_meeting_report.md`

## Testing inside Docker

```bash
docker run --rm -v "$(pwd)":/app transcriber:latest pytest
# Replace latest with any variant (nvidia, rocm, intel, cpu) to test that build.
```

## Maintenance

- Image size: CPU ~1.2 GB, GPU variants add ~2‑3 GB each (mostly the PyTorch wheel).
- When `requirements.txt` changes, rebuild the images:
  ```bash
  docker build -f Dockerfile.nvidia -t transcriber:nvidia .
  # repeat for other variants
  ```
- Periodically pull newer base images (`docker pull ubuntu:24.04`, `docker pull nvidia/cuda:…`, etc.) and rebuild to incorporate security patches.
```

---

*End of document.*
