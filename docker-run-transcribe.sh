#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="${TRANSCRIBER_REPO_ROOT:-$SCRIPT_DIR}"
DOCKER_BIN="${TRANSCRIBER_DOCKER_BIN:-docker}"
OUTPUT_DIR="${TRANSCRIBER_OUTPUT_DIR:-$REPO_ROOT/output}"
HF_CACHE_DIR="${TRANSCRIBER_HF_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
GGUF_CACHE_DIR="${TRANSCRIBER_GGUF_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/transcriber/gguf}"
WHISPER_CACHE_DIR="${TRANSCRIBER_WHISPER_CPP_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/transcriber/whisper}"
DRI_DIR="${TRANSCRIBER_DRI_DIR:-/dev/dri}"
NVIDIA_DEVICE_GLOB="${TRANSCRIBER_NVIDIA_DEVICE_GLOB:-/dev/nvidia*}"

usage() {
    cat <<'EOF'
Usage:
  ./docker-run-transcribe.sh [wrapper options] <audio.wav> [transcribe.py options]
  ./docker-run-transcribe.sh [wrapper options] --help

Wrapper options:
  --force-cpu          Force the CPU image even if a GPU is available.
  --image <tag>        Use a specific Docker image tag.
  --help-docker        Show this wrapper help.

Images:
  transcriber:cpu      Faster-Whisper and llama.cpp on CPU.
  transcriber:vulkan   Vulkan acceleration with transparent CPU fallback.

Examples:
  ./docker-run-transcribe.sh meeting.wav
  ./docker-run-transcribe.sh --force-cpu meeting.wav
  ./docker-run-transcribe.sh --image transcriber:vulkan meeting.wav -l en
  FORCE_CPU=1 ./docker-run-transcribe.sh meeting.wav
EOF
}

die() {
    echo "❌  $*" >&2
    exit 1
}

is_truthy() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

has_nvidia() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    compgen -G "$NVIDIA_DEVICE_GLOB" >/dev/null
}

has_dri() {
    compgen -G "$DRI_DIR/render*" >/dev/null
}

image_exists() {
    "$DOCKER_BIN" image inspect "$1" >/dev/null 2>&1
}

build_image() {
    local target="$1" image="$2"
    [ -f "$REPO_ROOT/Dockerfile" ] || die "Missing Dockerfile: $REPO_ROOT/Dockerfile"
    echo "🐳  Building $image (target: $target)"
    "$DOCKER_BIN" build --target "$target" -f "$REPO_ROOT/Dockerfile" -t "$image" "$REPO_ROOT"
}

ensure_image() {
    local image="$1"
    image_exists "$image" && return
    case "$image" in
        transcriber:cpu) build_image cpu "$image" ;;
        transcriber:vulkan) build_image vulkan "$image" ;;
        transcriber:latest) build_image cpu "$image" ;;
        *) die "Docker image '$image' was not found and cannot be auto-built. Use transcriber:cpu or transcriber:vulkan." ;;
    esac
}

group_id_for_path() {
    [ -e "$1" ] || return 1
    stat -c '%g' -- "$1" 2>/dev/null
}

add_runtime_group() {
    local group_id="$1" existing
    [ -n "$group_id" ] || return
    for existing in "${runtime_group_ids[@]:-}"; do
        [ "$existing" = "$group_id" ] && return
    done
    runtime_group_ids+=("$group_id")
    accelerator_flags+=(--group-add "$group_id")
}

add_device_group() {
    local group_id
    if group_id="$(group_id_for_path "$1")"; then
        add_runtime_group "$group_id"
    fi
}

configure_accelerator_flags() {
    accelerator_flags=()
    runtime_group_ids=()
    if has_nvidia; then
        accelerator_flags+=(--gpus all)
        accelerator_flags+=(-e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-graphics,compute,utility}")
        return
    fi
    if [ -e "$DRI_DIR" ]; then
        accelerator_flags+=(--device "$DRI_DIR")
        local dri_device
        for dri_device in "$DRI_DIR"/*; do
            [ -e "$dri_device" ] || continue
            add_device_group "$dri_device"
        done
    fi
}

vulkan_preflight() {
    local image="$1"
    "$DOCKER_BIN" run --rm "${accelerator_flags[@]}" \
        --entrypoint transcriber-vulkan-probe "$image" >/dev/null 2>&1
}

pass_env_if_set() {
    local name="$1"
    if [ "${!name+x}" ]; then
        run_flags+=(-e "$name=${!name}")
    fi
}

forwarded_has_help() {
    local token
    for token in "$@"; do
        [ "$token" = "--help" ] || [ "$token" = "-h" ] && return 0
    done
    return 1
}

find_audio_index() {
    local idx token skip_next=0
    for idx in "${!forwarded_args[@]}"; do
        token="${forwarded_args[$idx]}"
        if [ "$skip_next" -eq 1 ]; then skip_next=0; continue; fi
        case "$token" in
            -l|--language) skip_next=1 ;;
            --language=*|--|-*) ;;
            *) echo "$idx"; return 0 ;;
        esac
    done
    return 1
}

force_cpu=0
custom_image=""
forwarded_args=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --help-docker) usage; exit 0 ;;
        --force-cpu) force_cpu=1 ;;
        --image)
            shift
            [ "$#" -gt 0 ] || die "--image requires a Docker image tag."
            custom_image="$1"
            ;;
        --image=*) custom_image="${1#*=}" ;;
        --) shift; forwarded_args+=("$@"); break ;;
        *) forwarded_args+=("$1") ;;
    esac
    shift
done

if [ "$force_cpu" -eq 1 ] || is_truthy "${FORCE_CPU:-0}"; then
    selected_backend=cpu
    selected_image=transcriber:cpu
elif [ -n "$custom_image" ]; then
    selected_image="$custom_image"
    case "${custom_image,,}" in
        *vulkan*|*nvidia*|*cuda*|*rocm*|*amd*|*intel*) selected_backend=vulkan ;;
        *) selected_backend=cpu ;;
    esac
elif has_nvidia || has_dri; then
    selected_backend=vulkan
    selected_image=transcriber:vulkan
else
    selected_backend=cpu
    selected_image=transcriber:cpu
fi

ensure_image "$selected_image"
configure_accelerator_flags
if [ "$selected_backend" = vulkan ] && [ "$selected_image" = transcriber:vulkan ]; then
    echo "🔎  Checking Vulkan access..."
    if ! vulkan_preflight "$selected_image"; then
        echo "⚠️  Vulkan preflight failed; using the CPU image."
        selected_backend=cpu
        selected_image=transcriber:cpu
        accelerator_flags=()
        runtime_group_ids=()
        ensure_image "$selected_image"
    fi
fi

mkdir -p "$OUTPUT_DIR" "$HF_CACHE_DIR" "$GGUF_CACHE_DIR" "$WHISPER_CACHE_DIR"
audio_mount=()
container_args=("${forwarded_args[@]}")
if audio_index="$(find_audio_index)"; then
    audio_path="${forwarded_args[$audio_index]}"
    [ -f "$audio_path" ] || die "Audio file not found: $audio_path"
    audio_abs="$(readlink -f -- "$audio_path")"
    container_audio="/input/$(basename -- "$audio_abs")"
    container_args[$audio_index]="$container_audio"
    audio_mount=(-v "$audio_abs:$container_audio:ro")
elif ! forwarded_has_help "${forwarded_args[@]}"; then
    usage >&2
    die "Please provide an audio file path or pass --help to transcribe.py."
fi

run_flags=(
    --rm
    --user "$(id -u):$(id -g)"
    -e HF_HOME=/cache/huggingface
    -e XDG_CACHE_HOME=/cache
    -e TRANSCRIBER_GGUF_CACHE_DIR=/cache/transcriber/gguf
    -e TRANSCRIBER_WHISPER_CPP_CACHE_DIR=/cache/transcriber/whisper
    -v "$OUTPUT_DIR:/app/output"
    -v "$HF_CACHE_DIR:/cache/huggingface"
    -v "$GGUF_CACHE_DIR:/cache/transcriber/gguf"
    -v "$WHISPER_CACHE_DIR:/cache/transcriber/whisper"
)
if [ "$selected_backend" = vulkan ]; then
    run_flags+=("${accelerator_flags[@]}")
fi

for env_name in \
    DEBUG TRANSCRIBER_ANALYSIS_BACKEND TRANSCRIBER_TRANSCRIPTION_BACKEND \
    TRANSCRIBER_WHISPER_CPP_MODEL_PATH TRANSCRIBER_VULKAN_DEVICE \
    TRANSCRIBER_LLAMA_CPP_MODEL_PATH TRANSCRIBER_LLAMA_CPP_MODEL_REPO \
    TRANSCRIBER_LLAMA_CPP_CONTEXT_SIZE TRANSCRIBER_LLAMA_CPP_BATCH_SIZE \
    TRANSCRIBER_LLAMA_CPP_GPU_LAYERS TRANSCRIBER_LLAMA_CPP_GPU_HEADROOM_GIB \
    TRANSCRIBER_LLAMA_CPP_LAYER_COUNT TRANSCRIBER_MAX_TRANSCRIPT_CHARS; do
    pass_env_if_set "$env_name"
done

echo "▶ Running transcribe.py in Docker"
echo "   Image   : $selected_image"
echo "   Backend : $selected_backend"
[ -n "${audio_path:-}" ] && echo "   Audio   : $audio_path"
echo "   Output  : $OUTPUT_DIR"

"$DOCKER_BIN" run "${run_flags[@]}" "${audio_mount[@]}" \
    "$selected_image" "${container_args[@]}"
