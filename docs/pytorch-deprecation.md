# PyTorch Deprecation Plan — llama.cpp for All Analysis

**Project:** Local-only meeting recorder, transcription, and summarization pipeline
**Document:** `docs/pytorch-deprecation.md`
**Status:** Planning (no code changes applied yet)
**Last Updated:** 2026-05-31

---

## 1. Objective

Replace the PyTorch/Transformers analysis (summarization) backend with
**llama.cpp/GGUF for every hardware target** (CPU, NVIDIA, Intel, AMD ROCm), and
remove `torch`, `transformers`, and `accelerate` from the entire project.

The llama.cpp engine, GGUF download/caching, and dynamic layer-splitting logic
already exist and are used by the AMD ROCm path today. This migration mostly
**deletes** the PyTorch half of the analysis module and **generalizes** the
existing llama.cpp half so it serves all backends.

Transcription (Faster-Whisper via CTranslate2) is functionally unchanged. The
only transcription edit is removing torch's cosmetic GPU-name lookup, since
CTranslate2 — not torch — performs the real device detection.

---

## 2. Motivation

- The PyTorch/Transformers CUDA path is significantly slower than llama.cpp.
  The dominant cost is Accelerate's `max_memory` cap forcing partial CPU offload,
  which shuttles layers across the PCIe bus on every generated token.
- 16-bit Transformers weights are ~4x larger than 4-bit GGUF quantizations,
  making them a poor fit for consumer GPUs (e.g. 12 GB RTX 3060).
- The Python `model.generate()` decode loop is slower than llama.cpp's native
  C++/CUDA decode loop.
- Maintaining two analysis engines (Transformers + llama.cpp) doubles the
  surface area for configuration, tests, and Docker images. Consolidating on
  llama.cpp simplifies the whole pipeline.

---

## 3. Decisions

These decisions were confirmed before planning and govern the implementation:

| Topic | Decision |
| ----- | -------- |
| Intel GPU | Use **CPU llama.cpp only** (no SYCL/oneAPI or Vulkan build). The Intel image becomes equivalent to the CPU build. |
| Torch removal | Remove `torch` from the **entire** project, including the cosmetic GPU-name display in `lib/transcription.py`. |
| GPU layer split | Keep the dynamic `n_gpu_layers` calculation, but probe free VRAM via `pynvml` (NVIDIA) and `amdsmi`/`rocm-smi` (AMD) instead of `torch.cuda.mem_get_info()`. |
| ROCm images | Collapse the two ROCm images. Delete the deprecated `Dockerfile.rocm`; rename `Dockerfile.rocm-llama` → `Dockerfile.rocm` (image `transcriber:rocm`). |
| Prompt style | Always use the compact/plain prompt. Drop chat-template prompting. |
| `AnalysisBackend` shape | Remove the `engine` and `use_plain_prompt` fields (only one engine remains). |
| `max_new_tokens` | Unify on a single configurable default (proposed **2048**; see §10). |

---

## 4. Affected Files (Inventory)

**Python — analysis module**

- `lib/analysis/backend.py` — hardware detection, backend config, GGUF helpers
- `lib/analysis/model.py` — model loading and inference
- `lib/analysis/__init__.py` — public re-exports and `generate_summaries()`
- `lib/analysis/prompt.py` — prompt construction
- `lib/analysis/utils.py` — grounding/parsing (no torch; unchanged)

**Python — transcription**

- `lib/transcription.py` — remove torch GPU-name display only

**Dependencies**

- `requirements.txt`
- `pyproject.toml` (pytest config only; no dependency changes expected)

**Docker**

- `Dockerfile.base`
- `Dockerfile.cpu`
- `Dockerfile.intel`
- `Dockerfile.nvidia`
- `Dockerfile.rocm` (delete)
- `Dockerfile.rocm-llama` (rename → `Dockerfile.rocm`)
- `docker-run-transcribe.sh`

**Tests**

- `tests/test_analysis.py`
- `tests/test_docker_run_transcribe.py`
- `tests/test_transcription.py`
- `conftest.py` (no change expected)

**Docs**

- `README.md`
- `docs/model-recommendations.md`
- `AGENTS.md`

---

## 5. Phase 1 — Python Analysis Core (`lib/analysis/`) - IMPLEMENTED

### 5.1 `backend.py`

Remove:

- `import torch`
- Transformers-only constants and functions:
  - `_gpu_max_memory()`
  - `_cpu_analysis_backend()`
  - `_cpu_supports_avx512_bf16()`
  - `_rocm_transformers_analysis_backend()`
  - `FLOAT32_MIN_RAM_GIB`
  - `DEFAULT_GPU_HEADROOM_GIB`
  - `CPU_OFFLOAD_HEADROOM_GIB`
  - `GPU_HEADROOM_ENV` (`TRANSCRIBER_ANALYSIS_GPU_HEADROOM_GIB`)
  - `GPU_MAX_MEMORY_ENV` (`TRANSCRIBER_ANALYSIS_GPU_MAX_MEMORY_GIB`)
  - `TRANSFORMERS_BACKEND_NAME`
  - `DEFAULT_ANALYSIS_MODEL_ID` (HF id)
  - `DEFAULT_ROCM_ANALYSIS_MODEL_ID` (HF id)
  - `ROCM_ATTENTION_IMPLEMENTATION`
  - `ROCM_ANALYSIS_MAX_NEW_TOKENS` (folded into a single default)
  - `ANALYSIS_MODEL_ENV` (Hugging Face model override; no longer applicable)

Rename for clarity (single engine, all backends):

- `DEFAULT_ROCM_LLAMA_CPP_MODEL_REPO_ID` → `DEFAULT_LLAMA_CPP_MODEL_REPO_ID`
- `DEFAULT_ROCM_LLAMA_CPP_MODEL_FILENAME` → `DEFAULT_LLAMA_CPP_MODEL_FILENAME`
- `DEFAULT_ROCM_LLAMA_CPP_CONTEXT_SIZE` → `DEFAULT_LLAMA_CPP_CONTEXT_SIZE`
- `DEFAULT_ROCM_LLAMA_CPP_BATCH_SIZE` → `DEFAULT_LLAMA_CPP_BATCH_SIZE`
- `DEFAULT_ROCM_LLAMA_CPP_LAYER_COUNT` → `DEFAULT_LLAMA_CPP_LAYER_COUNT`
- `DEFAULT_ROCM_LLAMA_CPP_GPU_HEADROOM_GIB` → `DEFAULT_LLAMA_CPP_GPU_HEADROOM_GIB`
- `DEFAULT_ROCM_LLAMA_CPP_KV_CACHE_GIB` → `DEFAULT_LLAMA_CPP_KV_CACHE_GIB`

Keep the Gemma 4 E4B Q4_K_M GGUF default and 42-layer default.

Add torch-free helpers:

- `_detect_gpu() -> tuple[str, str]` — returns `(kind, device_name)` where
  `kind ∈ {"cuda", "rocm", "cpu"}`. Detection without torch:
  - NVIDIA: `pynvml` (preferred) or presence of `nvidia-smi`.
  - AMD: presence of `/dev/kfd` and/or `amdsmi`/`rocm-smi`.
  - Otherwise CPU.
- `_nvidia_free_vram_bytes() -> int | None` — via `pynvml`
  (`nvmlDeviceGetMemoryInfo`).
- `_amd_free_vram_bytes() -> int | None` — via `amdsmi`, falling back to parsing
  `rocm-smi --showmeminfo vram`.
- Generalize `_rocm_llama_cpp_gpu_layers()` → `_llama_cpp_gpu_layers()` to use the
  new VRAM probes (NVIDIA or AMD) and return `0` when no GPU is present.

`AnalysisBackend` dataclass:

- Remove the `engine` field (only llama.cpp remains).
- Remove the `use_plain_prompt` field (always plain prompt).
- Keep: `name`, `device_name`, `model_id`, `model_kwargs`, `notes`,
  `max_new_tokens`.

`detect_analysis_backend()`:

- Determine GPU kind via `_detect_gpu()`.
- Resolve/ensure the GGUF model path (existing download helpers).
- Compute `n_gpu_layers` via `_llama_cpp_gpu_layers()` (0 on CPU).
- Build llama.cpp `model_kwargs` (`model_path`, `n_ctx`, `n_batch`,
  `n_gpu_layers`, `verbose=False`) and informative `notes`.

### 5.2 `model.py`

Remove:

- `import torch`
- `from transformers import AutoModelForCausalLM, AutoTokenizer`
- `ChatTokenizer` and `GenerativeModel` Protocols
- `_query()` (Transformers forward pass)
- `_generate_report_with_transformers()`
- `torch.OutOfMemoryError` handling

Keep the llama.cpp path only:

- `LlamaCppModel` Protocol
- `_query_llama_cpp()`
- `_load_llama_cpp_model()`
- `_generate_report_with_llama_cpp()`

### 5.3 `__init__.py`

- Remove `import torch` and every torch/transformers re-export
  (`ChatTokenizer`, `GenerativeModel`, `_query`,
  `_generate_report_with_transformers`, `_cpu_analysis_backend`,
  `_cpu_supports_avx512_bf16`, `_gpu_max_memory`, deprecated constants, etc.).
- Update the re-export list to match the slimmed `backend.py`/`model.py`.
- Simplify `generate_summaries()`:
  - Remove the `engine` branch and the Transformers print branch.
  - Always call `_generate_report_with_llama_cpp()`.
  - Keep grounding validation and section parsing.

### 5.4 `prompt.py`

- Always build the compact/plain user message.
- Remove the chat-template `_build_user_message()` and the
  `use_plain_prompt` switch in `_build_prompt_for_backend()` (or hardcode the
  compact path). `ANALYSIS_SYSTEM_PROMPT` and `SUMMARY_TASKS` are retained.

### 5.5 `lib/transcription.py`

- Remove `import torch`.
- Replace the torch-based GPU-name print with a `pynvml` name lookup when
  available, otherwise a generic label (`"CUDA GPU"` / `"GPU"`).
- CTranslate2 detection (`ctranslate2.get_cuda_device_count()`) is unchanged —
  it still drives the actual device/compute-type selection.

---

## 6. Phase 2 — Dependencies

`requirements.txt`:

- Remove: `torch`, `transformers`, `accelerate`.
- Add: `llama-cpp-python`, `nvidia-ml-py` (provides `pynvml`; imported
  defensively so CPU/AMD hosts without it still work).
- Keep: `faster-whisper`, `huggingface-hub`, `youtube-transcript-api`,
  `pytest`, `pytest-cov`.

`pyproject.toml`: no dependency changes (pytest configuration only).

---

## 7. Phase 3 — Dockerfiles

- `Dockerfile.base`: add `cmake`, `ninja-build`, `pkg-config`; build a CPU
  `llama-cpp-python`. (Already installs OS deps and the non-torch requirements.)
- `Dockerfile.cpu`: inherit `transcriber:base`; drop the `torch` install. The
  CPU llama.cpp build comes from base.
- `Dockerfile.intel`: inherit `transcriber:base` as a CPU-only image; drop
  `torch`, `intel-extension-for-pytorch`, and the Level Zero packages.
- `Dockerfile.nvidia`: switch the base image from
  `nvidia/cuda:12.4.1-runtime-ubuntu22.04` to the matching `-devel` image (needs
  `nvcc` to compile CUDA kernels); build `llama-cpp-python` with
  `CMAKE_ARGS="-DGGML_CUDA=on"`; add `nvidia-ml-py`; drop the torch install.
- `Dockerfile.rocm`: delete the deprecated PyTorch/Transformers ROCm image.
- `Dockerfile.rocm-llama` → rename to `Dockerfile.rocm`: keep the HIP
  `llama-cpp-python` build and `AMDGPU_TARGETS` arg; drop `transformers` from the
  requirements install step.

---

## 8. Phase 4 — `docker-run-transcribe.sh`

- Collapse the `rocm` and `rocm-llama` backends into a single `rocm` backend
  across:
  - `backend_for_known_image()`
  - `backend_for_image_hint()`
  - `image_for_backend()`
  - `dockerfile_for_backend()`
  - `select_backend()`
  - the run-flags `case` block
  - `ensure_image()`
- Remove the `transcriber:rocm` deprecation warning block.
- Keep the device flags (`--gpus all`, `--device /dev/kfd`, `--device /dev/dri`)
  and the `AMDGPU_TARGETS` build detection.
- Stop forwarding transformers-only env vars:
  - `TRANSCRIBER_ANALYSIS_GPU_HEADROOM_GIB`
  - `TRANSCRIBER_ANALYSIS_GPU_MAX_MEMORY_GIB`
  - `TRANSCRIBER_ANALYSIS_MODEL` (Hugging Face id)
- Keep forwarding the llama.cpp env vars (`TRANSCRIBER_LLAMA_CPP_*`,
  `TRANSCRIBER_GGUF_CACHE_DIR`, `TRANSCRIBER_MAX_TRANSCRIPT_CHARS`).

---

## 9. Phase 5 — Tests

`tests/test_analysis.py`:

- Delete Transformers-path tests:
  - `test_detect_analysis_backend_reports_cuda`
  - `test_detect_analysis_backend_reports_rocm`
  - `test_detect_analysis_backend_respects_model_override_on_rocm`
  - the three CPU dtype tests (AVX-512 BF16 / float32 / bf16 fallback)
  - `test_cpu_supports_avx512_bf16_*`
- Rewrite detection tests so CUDA, ROCm, and CPU all resolve to the llama.cpp
  engine with appropriate `n_gpu_layers`.
- Adapt the layer-split tests to mock the new VRAM probes (`_nvidia_free_vram_bytes`,
  `_amd_free_vram_bytes`) instead of `torch.cuda.mem_get_info()`.
- Add tests for `_detect_gpu()` and the VRAM probe helpers.

`tests/test_docker_run_transcribe.py`:

- Replace `rocm-llama` references with `rocm`.
- Delete the deprecated `transcriber:rocm` device-group test (or repoint it to
  the new unified image).
- Drop assertions about GPU-headroom env forwarding.
- Assert the NVIDIA path builds the llama.cpp-enabled image.

`tests/test_transcription.py`:

- Adjust GPU-name display expectations now that torch is removed (the device
  selection assertions via CTranslate2 stay the same).

`conftest.py`: no change expected (tests inject a `FakeLlamaCppModel`).

---

## 10. Phase 6 — Documentation

- `README.md`:
  - Update the Python packages table: drop `torch`, `transformers`,
    `accelerate`; add `llama-cpp-python`.
  - Rewrite the analysis description to "llama.cpp/GGUF for all backends".
  - Collapse the Docker build list to a single `transcriber:rocm`.
  - Add a native (non-Docker) GPU build note documenting
    `CMAKE_ARGS="-DGGML_CUDA=on"` / `-DGGML_HIP=on` when installing
    `llama-cpp-python`.
- `docs/model-recommendations.md`: present GGUF-only analysis models for every
  backend; remove the Hugging Face/PyTorch per-backend split and the
  GPU-headroom env variables tied to Accelerate.
- `AGENTS.md`: update the `lib/analysis` description to reflect the llama.cpp-only
  design.

---

## 11. Open Considerations

1. **NVIDIA image size** — Building CUDA kernels requires the larger `-devel`
   base (ships `nvcc`). Options: (A) single-stage `-devel` now (simpler); (B)
   multi-stage build that copies the compiled wheel into a slim `-runtime` image
   later. Recommended: (A) now, (B) as a follow-up optimization.
2. **Native (non-Docker) GPU builds** — `pip install llama-cpp-python` produces a
   CPU-only build unless `CMAKE_ARGS` is set at install time. Options: (A)
   document the manual CUDA/HIP rebuild; (B) add a helper script. Recommended:
   (A) document only.
3. **Unified `max_new_tokens`** — Transformers used 4096; the llama.cpp ROCm path
   used 1024 for the full 5-section report. Options: (A) 2048 (proposed default);
   (B) keep 1024; (C) 4096. Recommended: (A) 2048.

---

## 12. Verification

1. `whisper_env/bin/python -m py_compile transcribe.py lib/*.py lib/analysis/*.py tests/*.py`
2. `whisper_env/bin/python -m pytest` — full suite green.
3. Pylance reports zero errors/warnings on touched files.
4. Build each image and smoke test:
   - `docker build -f Dockerfile.cpu -t transcriber:cpu .`
   - `docker build -f Dockerfile.nvidia -t transcriber:nvidia .`
   - `docker build -f Dockerfile.rocm -t transcriber:rocm .`
   - In each container: `import llama_cpp` succeeds, `detect_analysis_backend()`
     returns a llama.cpp config, and `import torch` fails (proving removal).
5. End-to-end run:
   `./docker-run-transcribe.sh --image transcriber:nvidia samples/example_recording.wav -l en`
   produces a transcript and report, using GPU offload via llama.cpp.

---

## 13. Rollback

All changes are tracked in git. If a regression appears:

- Revert the analysis-module and Dockerfile commits to restore the
  PyTorch/Transformers path.
- The GGUF cache and HuggingFace cache are unaffected by code reverts.
- No generated user data (`*.wav`, `*_transcript.txt`, `*_report.md`) is touched
  by this migration.
