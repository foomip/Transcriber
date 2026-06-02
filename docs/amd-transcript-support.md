# AMD GPU Transcription Support — Implementation Plan

## Goal

Enable the transcription stage to use an AMD GPU through ROCm instead of always falling back to CPU on AMD systems.

This plan assumes we **keep Faster-Whisper** as the transcription frontend and add support by using a **ROCm-enabled CTranslate2 build**.

## Current State

Today the project supports:

- **NVIDIA GPU transcription** via Faster-Whisper + CTranslate2 CUDA
- **AMD GPU summarization** via llama.cpp + ROCm
- **AMD transcription fallback to CPU**

The current AMD limitation is enforced in `lib/transcription.py`, which detects `/dev/kfd`, prints that ROCm is available for summarization, and then explicitly falls back to CPU for transcription.

## Key Finding

The previous assumption that Faster-Whisper could not use ROCm is now outdated.

Recent CTranslate2 releases support AMD ROCm/HIP builds, and Faster-Whisper can use them because it already runs on top of CTranslate2.

Important detail:

- On ROCm, CTranslate2 still uses the device string **`"cuda"`** in its Python API.
- So the Faster-Whisper call pattern does not need a new AMD-specific device string.

## Recommended Approach

Use the existing Faster-Whisper pipeline for AMD too, but ensure AMD environments install a **ROCm-capable CTranslate2** rather than the standard PyPI wheel path.

This is lower-risk than introducing a second transcription backend such as `whisper.cpp`.

---

## Required Changes

## 1. Install a ROCm-capable CTranslate2 on AMD

### Why

`requirements.txt` currently installs `faster-whisper`, which in turn relies on `ctranslate2`, but the default installation path does not guarantee a ROCm-enabled CTranslate2 build.

### Required work

Add an AMD-specific install path for CTranslate2.

Possible options:

- add a helper script such as `scripts/install_ctranslate2_rocm.sh`
- add a dedicated ROCm setup section in `README.md`
- optionally add a ROCm-specific requirements or install workflow

### Native install expectation

The install flow should:

1. install normal project requirements
2. replace or force-install `ctranslate2` from the ROCm release wheel
3. keep ROCm runtime libraries available on the host

### Docker expectation

`Dockerfile.rocm` should:

1. install normal project requirements
2. explicitly install a ROCm-enabled `ctranslate2`
3. ensure required ROCm runtime libraries are visible at runtime

---

## 2. Update transcription device detection

### Current behavior

`lib/transcription.py:detect_device()` currently does:

- use GPU if `ctranslate2.get_cuda_device_count() > 0`
- otherwise, if ROCm is detected, log that transcription must run on CPU

### Required behavior

Change detection so AMD ROCm can use the CTranslate2 GPU path.

### Design intent

Separate:

- **accelerator kind**: `nvidia`, `rocm`, `cpu`
- **CTranslate2 runtime device string**: `cuda` or `cpu`
- **compute type**: `float16`, `int8_float16`, `float32`, etc.

### Practical result

If:

- CTranslate2 reports a visible GPU, and
- the machine is ROCm-based,

then transcription should run with:

- `device="cuda"`
- an appropriate GPU compute type

instead of falling back to CPU.

---

## 3. Make compute type selection dynamic

### Current behavior

Transcription currently hardcodes:

- GPU → `float16`
- CPU → `int8`

### Required behavior

Query CTranslate2 for supported GPU compute types and select the best available one.

Suggested preference order:

1. `float16`
2. `int8_float16`
3. `float32`
4. `int8`

This makes AMD support more robust across different ROCm cards and driver stacks.

---

## 4. Improve user-facing reporting

Even though ROCm uses `device="cuda"` internally, the CLI output should distinguish between:

- NVIDIA CUDA transcription
- AMD ROCm transcription
- CPU transcription

### Suggested improvement

Return richer device metadata from detection, for example:

- accelerator kind
- runtime device string
- compute type
- device name

Then log messages like:

- `✅ CUDA GPU detected for transcription: NVIDIA RTX 3060`
- `✅ ROCm GPU detected for transcription: Radeon RX 7800 XT`
- `ℹ️ No GPU detected — running on CPU`

---

## 5. Handle ROCm-specific runtime quirks

There are known reports that some RDNA2 cards need:

```bash
CT2_CUDA_ALLOCATOR=cub_caching
```

to avoid illegal memory access errors when loading models with CTranslate2 on ROCm.

### Initial recommendation

Start by **documenting** this in troubleshooting rather than auto-applying it.

### Optional later enhancement

If testing confirms the pattern is reliable, auto-enable it for known affected AMD architectures when the environment variable is not already set.

---

## 6. Update the ROCm Docker path

`docker-run-transcribe.sh` already passes through the key AMD device nodes:

- `/dev/kfd`
- `/dev/dri`

That means the main missing piece is the **software stack**, not the device plumbing.

### Required Docker work

Update `Dockerfile.rocm` so it supports both:

- AMD summarization through llama.cpp
- AMD transcription through ROCm-enabled CTranslate2

### Possible additions

- pass through `CT2_CUDA_ALLOCATOR` when set
- ensure ROCm library paths are available
- add a transcription smoke test to the Docker documentation

---

## 7. Add and update tests

## Test file

Primary changes will be in:

- `tests/test_transcription.py`

## New tests needed

Add tests covering:

- ROCm-capable CTranslate2 path selecting GPU instead of CPU
- dynamic compute-type selection from `ctranslate2.get_supported_compute_types("cuda")`
- CPU fallback only when CTranslate2 reports no visible GPU
- correct AMD/NVIDIA user-facing reporting if richer metadata is introduced

## Existing test to replace

The current test asserting ROCm must fall back to CPU should be updated or removed:

- `test_detect_device_uses_cpu_for_rocm_faster_whisper_fallback`

That behavior is what this implementation changes.

---

## 8. Update documentation

## README changes

Update `README.md` to reflect that transcription is no longer NVIDIA-only.

### Areas to revise

- hardware support summary
- installation instructions
- Docker notes
- troubleshooting
- transcription hardware recommendation wording

### Useful smoke test to document

```bash
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count(), ctranslate2.get_supported_compute_types('cuda'))"
```

On a working ROCm setup, this should report at least one visible GPU and one or more supported compute types.

---

## Files Expected To Change

Most likely implementation touch points:

- `lib/transcription.py`
- `tests/test_transcription.py`
- `Dockerfile.rocm`
- `README.md`
- optionally a new helper script such as:
  - `scripts/install_ctranslate2_rocm.sh`
- optionally Docker wrapper env passthrough in:
  - `docker-run-transcribe.sh`

---

## What Should Not Need Architectural Change

These parts likely do **not** need fundamental redesign:

- `transcribe.py`
- transcript output format
- report generation
- summarization pipeline
- Faster-Whisper model usage itself

This work is primarily:

- packaging / installation
- hardware detection
- runtime selection
- tests
- documentation

---

## Fallback Option If ROCm CTranslate2 Proves Unreliable

If ROCm support through CTranslate2 turns out too inconsistent across target AMD cards, the fallback architecture would be to add a separate AMD transcription backend based on `whisper.cpp`.

That would be a larger change because it would require:

- a second transcription engine
- separate model management
- output normalization
- broader test coverage

For now, that should be treated as a contingency plan, not the preferred implementation.

---

## Summary

The preferred implementation is:

1. keep Faster-Whisper
2. install a ROCm-enabled CTranslate2 on AMD
3. remove the explicit ROCm → CPU fallback in transcription detection
4. run AMD transcription through CTranslate2 with `device="cuda"`
5. select compute types dynamically
6. add ROCm-focused tests and docs
7. document known allocator/runtime quirks for affected AMD cards

This is the smallest-change path to getting AMD GPU transcription working while preserving the current pipeline design.

---

## Reference Notes

Relevant upstream references reviewed while preparing this plan:

- CTranslate2 PyPI project page
- CTranslate2 installation docs (`WITH_HIP=ON`)
- OpenNMT/CTranslate2 ROCm support PR #1989
- OpenNMT/CTranslate2 ROCm issue reports confirming Faster-Whisper works on AMD with ROCm-enabled builds
