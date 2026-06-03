# YouTube Transcription in Docker — Shared Execution Framework

## 1. Problem Statement

`youtube-summarize.py` fetches YouTube video transcripts (no audio processing)
and runs the same local analysis + report pipeline as `transcribe.py`. Both scripts
share the same Python dependencies, model caches, and GPU backends — so both should
run inside the same Docker containers (`transcriber:cpu`, `transcriber:nvidia`,
`transcriber:rocm`, `transcriber:intel`).

**Goals:**

- Create `docker-run-youtube.sh` — a Docker wrapper for `youtube-summarize.py`
- Extract common execution logic into `lib/docker-common.sh` so both wrappers share
  GPU detection, image selection, build logic, volume mounts, and device flags
- Leave `docker-run-transcribe.sh` with identical behavior (drop-in compatible)

## 2. Architecture Overview

```
                    ┌───────────────────────────────────┐
                    │       lib/docker-common.sh        │
                    │  (Shared shell library — sourced) │
                    │                                   │
                    │  • GPU detection (nvidia/rocm/    │
                    │    intel/cpu)                     │
                    │  • Image selection & building     │
                    │  • Common volume mounts & env     │
                    │  • Device flag assembly           │
                    │  • Group ID management            │
                    └──────────┬────────────────────▲───┘
                               │ sourced by         │ sourced by
                    ┌──────────▼──────────┐  ┌──────┴──────────────┐
                    │docker-run-          │  │docker-run-youtube.sh│
                    │transcribe.sh        │  │  (NEW)              │
                    │                     │  │                     │
                    │ • Audio-file mount  │  │ • Entrypoint →      │
                    │ • Forwards remaining│  │   youtube-          │
                    │   args to           │  │   summarize.py      │
                    │   transcribe.py     │  │ • Passes URL & lang │
                    └──────────┬──────────┘  └──────────┬──────────┘
                               │                        │
                               │  docker run            │  docker run --entrypoint python
                               │                        │
                    ┌──────────▼────────────────────────▼───────────┐
                    │            target Docker image                │
                    │  transcriber:cpu / :nvidia / :rocm / :intel   │
                    │                                               │
                    │  Volume mounts: output/, HF cache, GGUF cache │
                    │  GPU device passthrough (if applicable)       │
                    └───────────────────────────────────────────────┘
```

### 2.1 Key Design Decisions

| Decision | Rationale |
|---|---|
| **Shell library, not Python** | The shared logic is shell-level (GPU detection, image tags, docker flags). Python abstractions would require Docker SDK or convoluted subprocess wrappers. Keeping it in bash preserves the existing pattern and keeps complexity low. |
| **Array-based flag accumulation** | `COMMON_RUN_FLAGS` and `RUNTIME_GROUP_IDS` are mutable global arrays appended by setup functions. Each script then appends its own script-specific flags before the final `docker run` call. This avoids fragile string concatenation. |
| **`--entrypoint python` override** for YouTube | The default `ENTRYPOINT` in all Dockerfiles is `["python", "transcribe.py"]`. To run `youtube-summarize.py`, we override the entrypoint to `python` and pass the script path + arguments as CMD. The script lives at `/app/youtube-summarize.py` (already `COPY`'d by all Dockerfiles). |
| **No changes to transcribe wrapper args** | `docker-run-transcribe.sh` argument parsing, usage text, and audio handling remain completely untouched — only the function bodies move to the common lib. |

## 3. Shared Library API — `lib/docker-common.sh`

### 3.1 Exported Functions

| Function | Purpose | Sets / Returns |
|---|---|---|
| `die <msg>` | Print error to stderr and exit 1 | Exits |
| `is_truthy <value>` | Check if a value is truthy (1/true/yes/on) | Return code |
| `has_nvidia` | Detect NVIDIA GPU via `nvidia-smi` | Return code |
| `has_rocm` | Detect AMD GPU via `/dev/kfd` + `rocminfo` | Return code |
| `has_intel` | Detect Intel GPU via `/dev/dri/render*` | Return code |
| `group_id_for_path <path>` | Get numeric GID of a device path | stdout |
| `add_runtime_group <gid>` | Append GID to `RUNTIME_GROUP_IDS` (deduplicated) | Appends array |
| `add_device_group <path>` | Resolve path GID and call `add_runtime_group` | Appends array |
| `backend_for_known_image <tag>` | Known image → canoncial backend name | stdout |
| `backend_for_image_hint <tag>` | Heuristic: substring match on image tag | stdout |
| `image_for_backend <backend>` | Backend name → image tag | stdout |
| `dockerfile_for_backend <backend>` | Backend name → Dockerfile path | stdout |
| `image_exists <tag>` | Check if Docker image is present locally | Return code |
| `build_image <backend> <tag>` | Build (and possibly build base first) | `docker build` |
| `ensure_image <tag>` | Build if missing; fail for unknown tags | `docker build` / exit |
| `detect_rocm_amdgpu_targets` | Detect AMDGPU targets via `rocminfo` | stdout |
| `select_backend` | Auto-detect: nvidia > rocm > intel > cpu | stdout |
| `setup_common_run_flags <backend>` | Populate `COMMON_RUN_FLAGS` and `RUNTIME_GROUP_IDS` | Appends arrays |
| `apply_backend_device_flags <backend>` | Add GPU device mounts + group-ids to run flags | Appends arrays |
| `pass_env_if_set <name>` | Forward host env var if set | Appends to `COMMON_RUN_FLAGS` |
| `print_run_header <image> <backend> <script_name>` | Echo the "▶ Running ..." block | stdout |
| `forwarded_has_help <args...>` | Check if `--help` or `-h` appears in forwarded args | Return code |

### 3.2 Global Variables (Set by library, consumed by caller)

| Variable | Type | Set By | Description |
|---|---|---|---|
| `COMMON_RUN_FLAGS` | array | `setup_common_run_flags` | `docker run` flags: `--rm`, `--user`, `-e`, `-v` |
| `RUNTIME_GROUP_IDS` | array | `add_runtime_group` / `add_device_group` | Deduplicated GIDs for `--group-add` |
| `BASE_IMAGE` | string | declaration | `transcriber:base` |

### 3.3 Environment Variables Passed Through (`pass_env_if_set`)

- `DEBUG`
- `TRANSCRIBER_ANALYSIS_BACKEND`
- `TRANSCRIBER_LLAMA_CPP_MODEL_PATH`
- `TRANSCRIBER_LLAMA_CPP_MODEL_REPO`
- `TRANSCRIBER_LLAMA_CPP_CONTEXT_SIZE`
- `TRANSCRIBER_LLAMA_CPP_BATCH_SIZE`
- `TRANSCRIBER_LLAMA_CPP_GPU_LAYERS`
- `TRANSCRIBER_LLAMA_CPP_GPU_HEADROOM_GIB`
- `TRANSCRIBER_LLAMA_CPP_LAYER_COUNT`
- `TRANSCRIBER_MAX_TRANSCRIPT_CHARS`

## 4. Script: `docker-run-transcribe.sh` (Modified)

### 4.1 What Stays

- Script header / `set -euo pipefail`
- `SCRIPT_DIR`, `REPO_ROOT`, configurable paths (`OUTPUT_DIR`, `HF_CACHE_DIR`, etc.)
- `usage()` function (script-specific help text)
- Wrapper argument parsing (`--force-cpu`, `--image`, `--help-docker`)
- `find_audio_index()` — finds the audio path among forwarded args
- Audio validation (`[ -f "$audio_path" ] || die ...`)
- Audio mount assembly (`-v "$audio_abs:$container_audio:ro"`)
- The final `docker run` invocation with script-specific flags appended to `COMMON_RUN_FLAGS`

### 4.2 What Moves to `lib/docker-common.sh`

- `die()`, `is_truthy()`
- All GPU detection functions
- All image / backend / Dockerfile mapping functions
- `build_image()`, `ensure_image()`, `detect_rocm_amdgpu_targets()`
- `select_backend()`, `pass_env_if_set()`
- `setup_common_run_flags()` and `apply_backend_device_flags()`
- `add_runtime_group()`, `add_device_group()`, `group_id_for_path()`
- `print_run_header()` and `forwarded_has_help()`

### 4.3 Structure After Refactor

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/lib/docker-common.sh"

REPO_ROOT="${TRANSCRIBER_REPO_ROOT:-$SCRIPT_DIR}"
OUTPUT_DIR="${TRANSCRIBER_OUTPUT_DIR:-$REPO_ROOT/output}"
HF_CACHE_DIR=...
GGUF_CACHE_DIR=...
DEV_KFD_PATH=...
DRI_DIR=...
DRM_VENDOR_GLOB=...
NVIDIA_DEVICE_GLOB=...

usage() { ... }   # script-specific help text

# === Wrapper arg parsing (unchanged) ===
force_cpu=0; custom_image=""; forwarded_args=()
while [ "$#" -gt 0 ]; do ...; shift; done

# === Backend / image selection (now from common lib) ===
if [ "$force_cpu" -eq 1 ]; then FORCE_CPU=1; fi
if [ -n "$custom_image" ]; then
  selected_image="$custom_image"
  selected_backend="$(backend_for_known_image "$selected_image" 2>/dev/null || \
                       backend_for_image_hint "$selected_image" 2>/dev/null || echo "cpu")"
else
  selected_backend="$(select_backend)"
  selected_image="$(image_for_backend "$selected_backend")"
fi

mkdir -p "$OUTPUT_DIR" "$HF_CACHE_DIR" "$GGUF_CACHE_DIR"

# === Common run flags ===
COMMON_RUN_FLAGS=(); RUNTIME_GROUP_IDS=()
setup_common_run_flags "$selected_backend"
apply_backend_device_flags "$selected_backend"

# === Audio mount (script-specific) ===
audio_mount=()
if audio_index="$(find_audio_index)"; then
  audio_path="${forwarded_args[$audio_index]}"
  [ -f "$audio_path" ] || die "Audio file not found: $audio_path"
  audio_abs="$(readlink -f -- "$audio_path")"
  container_audio="/input/$(basename -- "$audio_abs")"
  container_args[$audio_index]="$container_audio"
  audio_mount=( -v "$audio_abs:$container_audio:ro" )
else
  if ! forwarded_has_help "${forwarded_args[@]}"; then
    usage >&2; die "Provide an audio file or pass --help."
  fi
fi

ensure_image "$selected_image"
print_run_header "$selected_image" "$selected_backend" "transcribe.py"

"$DOCKER_BIN" run "${COMMON_RUN_FLAGS[@]}" "${audio_mount[@]}" "$selected_image" "${container_args[@]}"
```

## 5. Script: `docker-run-youtube.sh` (New)

### 5.1 CLI Interface

```
Usage:
  ./docker-run-youtube.sh [wrapper options] <youtube_url> [youtube-summarize.py options]
  ./docker-run-youtube.sh [wrapper options] --help

Wrapper options:
  --force-cpu          Force the CPU image even if a GPU is available.
  --image <tag>        Use a specific Docker image tag.
  --help-docker        Show this wrapper help.

Examples:
  ./docker-run-youtube.sh https://www.youtube.com/watch?v=dQw4w9WgXcQ
  ./docker-run-youtube.sh --force-cpu https://youtu.be/dQw4w9WgXcQ -l en
  ./docker-run-youtube.sh --image transcriber:nvidia dQw4w9WgXcQ

Notes:
  - Results are written to ./output in the repository root.
  - HuggingFace cache is mounted from ~/.cache/huggingface.
  - GGUF cache is mounted from ~/.cache/transcriber/gguf.
  - All other arguments are forwarded to youtube-summarize.py unchanged.
```

### 5.2 Execution Flow

1. Source `lib/docker-common.sh`
2. Parse wrapper-specific flags (`--force-cpu`, `--image`, `--help-docker`)
3. Detect backend / select image (reuses `select_backend`, `image_for_backend`)
4. Setup common run flags via `setup_common_run_flags` + `apply_backend_device_flags`
5. Ensure image is built via `ensure_image`
6. Validate that the first positional argument looks like a YouTube URL
7. Run:
   ```bash
   "$DOCKER_BIN" run "${COMMON_RUN_FLAGS[@]}" \
       --entrypoint python \
       "$selected_image" \
       /app/youtube-summarize.py "${forwarded_args[@]}"
   ```

### 5.3 Key Differences from Transcribe Wrapper

| Aspect | `docker-run-transcribe.sh` | `docker-run-youtube.sh` |
|---|---|---|
| **Entrypoint** | Inherited (default `transcribe.py`) | Overridden: `--entrypoint python` |
| **CMD** | `transcribe.py <audio.wav> [opts]` | `youtube-summarize.py <url> [opts]` |
| **Audio mount** | `-v audio:container:ro` | None — no audio needed |
| **Positional arg validation** | Must be an existing `.wav` file | Must be a non-flag string (URL or video ID) |
| **`find_audio_index`** | Needed to substitute path | Not needed |
| **`container_args` substitution** | Replaces audio path with container path | No substitution needed |

## 6. Backend Detection & Device Flag Reference

### 6.1 Backend Selection Priority

```
select_backend():
  1. FORCE_CPU=1                         → "cpu"
  2. nvidia-smi works                    → "nvidia"
  3. /dev/kfd exists + rocminfo works   → "rocm"
  4. /dev/dri/render* exists             → "intel"
  5. fallback                            → "cpu"
```

### 6.2 Image Tags and Dockerfiles

| Backend | Image Tag | Dockerfile | Depends On |
|---|---|---|---|
| cpu | `transcriber:cpu` | `Dockerfile.cpu` | `transcriber:base` |
| nvidia | `transcriber:nvidia` | `Dockerfile.nvidia` | (none, standalone) |
| rocm | `transcriber:rocm` | `Dockerfile.rocm` | (none, standalone) |
| intel | `transcriber:intel` | `Dockerfile.intel` | `transcriber:base` |

### 6.3 Device Flags per Backend

| Backend | `--gpus` | `--device` | `--group-add` |
|---|---|---|---|
| nvidia | `all` | — | GID of each `/dev/nvidia*` |
| rocm | — | `/dev/kfd`, `/dev/dri` | GID of `/dev/kfd` + each `/dev/dri/*` |
| intel | — | `/dev/dri` (if exists) | — |
| cpu | — | — | — |

### 6.4 Environment Variables Set per Backend

| Backend | Extra `-e` Flags |
|---|---|
| nvidia | `NVIDIA_DRIVER_CAPABILITIES=compute,utility` |
| rocm | `HSA_ENABLE_SDMA=0`, `CT2_CUDA_ALLOCATOR=cub_caching` (overridable) |
| intel | (none) |
| cpu | (none) |

## 7. Volume Mounts (Same for Both Scripts)

| Host Path | Container Path | Purpose |
|---|---|---|
| `$OUTPUT_DIR` | `/app/output` | Transcripts and reports |
| `$HF_CACHE_DIR` | `/cache/huggingface` | HuggingFace model cache (Whisper) |
| `$GGUF_CACHE_DIR` | `/cache/transcriber/gguf` | GGUF model cache (llama.cpp) |

## 8. Incremental Implementation Steps

To minimize risk, the refactor is split into small, testable increments. Every step must preserve existing functionality.

### Step 1: Create `lib/docker-common.sh` (Stateless Utilities)
- Create `lib/docker-common.sh` with all pure, stateless utility functions (die, is_truthy, has_nvidia, etc.).
- Include `BASE_IMAGE` and default path variables (`DEV_KFD_PATH`, etc.).
- **No array mutations** in this step.
- **Verification**: `shellcheck lib/docker-common.sh` and basic smoke test via `source`.

### Step 2: Source common lib in transcribe wrapper, remove duplicates
- Edit `docker-run-transcribe.sh` to source `lib/docker-common.sh`.
- Remove the local copies of the functions now provided by the lib.
- Keep all local array logic and the backend case statement.
- **Verification**: `whisper_env/bin/python -m pytest tests/test_docker_run_transcribe.py -v`.

### Step 3: Move array-mutating functions into common lib
- Move `add_runtime_group`, `add_device_group`, and `pass_env_if_set` to `lib/docker-common.sh`.
- These functions still reference the original array names (`run_flags`, `runtime_group_ids`).
- Remove local definitions from `docker-run-transcribe.sh`.
- **Verification**: `whisper_env/bin/python -m pytest tests/test_docker_run_transcribe.py -v`.

### Step 4: Rename arrays + introduce setup/print helpers
- In `lib/docker-common.sh`, rename arrays to `COMMON_RUN_FLAGS` and `RUNTIME_GROUP_IDS`.
- Add `setup_common_run_flags()` (populates common flags and env vars) and `print_run_header()`.
- Update `docker-run-transcribe.sh` to use the new array names and the setup helper.
- **Verification**: Update tests if necessary; `whisper_env/bin/python -m pytest tests/test_docker_run_transcribe.py -v`.

### Step 5: Consolidate `apply_backend_device_flags` into common lib
- Move the backend case statement and group-add loop into `lib/docker-common.sh` as `apply_backend_device_flags()`.
- Update `docker-run-transcribe.sh` to call this function.
- **Verification**: `whisper_env/bin/python -m pytest tests/test_docker_run_transcribe.py -v`.

### Step 6: Create `docker-run-youtube.sh`
- Create the new wrapper script sourcing `lib/docker-common.sh`.
- Implement YouTube-specific URL validation and the `--entrypoint python` override.
- **Verification**: Manual smoke test with `TRANSCRIBER_DOCKER_BIN=echo`.

### Step 7: Create `tests/test_docker_run_youtube.py`
- Implement a full test suite mirroring the transcribe wrapper tests.
- Verify GPU detection, image building, and correct `docker run` arguments.
- **Verification**: `whisper_env/bin/python -m pytest tests/ -v`.

## 9. Tests — `tests/test_docker_run_youtube.py`

### 9.1 Test Strategy

Follow the exact same pattern as `tests/test_docker_run_transcribe.py`:

- Mock `docker` via a fake Python script on `PATH`
- Mock `nvidia-smi` / `rocminfo` as trivial exit-zero scripts
- Override `TRANSCRIBER_DOCKER_BIN`, `TRANSCRIBER_OUTPUT_DIR`, cache dirs
- Capture `FAKE_DOCKER_LOG` to assert on `docker build` and `docker run` args

### 9.2 Test Cases

| Test | What It Verifies |
|---|---|
| `test_default_backend_builds_image_and_forwards_url` | No GPU tools → CPU backend, image built, URL forwarded |
| `test_nvidia_backend_sets_gpus_all` | `nvidia-smi` present → NVIDIA, `--gpus all` set |
| `test_rocm_backend_device_mounts` | ROCm detected → device mounts + group-adds |
| `test_custom_image_skips_build` | Image already present → no `build` call, `--gpus` if hint |
| `test_force_cpu_overrides_gpu_detection` | `--force-cpu` used even with GPU tools → CPU image |
| `test_entrypoint_python_and_script_path` | `--entrypoint python` and `/app/youtube-summarize.py` are in run call |
| `test_custom_env_vars_forwarded` | `TRANSCRIBER_LLAMA_CPP_GPU_LAYERS=99` appears in run call |

### 9.3 Fake Docker Script (Same as Transcribe Tests)

```python
#!/usr/bin/env python3
import json, os, pathlib, sys

log_path = pathlib.Path(os.environ["FAKE_DOCKER_LOG"])
present_images = set(json.loads(os.environ.get("FAKE_DOCKER_PRESENT_IMAGES", "[]")))
args = sys.argv[1:]

with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": args}) + "\n")

if args[:2] == ["image", "inspect"]:
    image = args[2] if len(args) > 2 else ""
    raise SystemExit(0 if image in present_images else 1)

raise SystemExit(0)
```

## 10. Rollback Strategy

If a regression is discovered after deployment:

1. **YouTube wrapper only**: Delete `docker-run-youtube.sh` — transcribe wrapper is unaffected.
2. **Shared library bug**: Revert `docker-run-transcribe.sh` and `lib/docker-common.sh` to the pre-refactor commit. The transcribe wrapper before the refactor was a single monolithic file with identical behavior.
3. **To restore the old transcribe wrapper**:
   ```bash
   git checkout HEAD~1 -- docker-run-transcribe.sh
   git rm lib/docker-common.sh      # if it was introduced in the same commit
   git rm docker-run-youtube.sh     # if it was introduced in the same commit
   ```
