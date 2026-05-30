#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="${TRANSCRIBER_REPO_ROOT:-$SCRIPT_DIR}"
DOCKER_BIN="${TRANSCRIBER_DOCKER_BIN:-docker}"
OUTPUT_DIR="${TRANSCRIBER_OUTPUT_DIR:-$REPO_ROOT/output}"
HF_CACHE_DIR="${TRANSCRIBER_HF_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
DEV_KFD_PATH="${TRANSCRIBER_DEV_KFD_PATH:-/dev/kfd}"
DRI_DIR="${TRANSCRIBER_DRI_DIR:-/dev/dri}"
DRM_VENDOR_GLOB="${TRANSCRIBER_DRM_VENDOR_GLOB:-/sys/class/drm/*/device/vendor}"

BASE_IMAGE="transcriber:base"

usage() {
    cat <<'EOF'
Usage:
  ./docker-run-transcribe.sh [wrapper options] <audio.wav> [transcribe.py options]
  ./docker-run-transcribe.sh [wrapper options] --help

Wrapper options:
  --force-cpu          Force the CPU image even if a GPU is available.
  --image <tag>        Use a specific Docker image tag.
  --help-docker        Show this wrapper help.

Examples:
  ./docker-run-transcribe.sh meeting.wav
  ./docker-run-transcribe.sh --force-cpu meeting.wav
  ./docker-run-transcribe.sh --image transcriber:rocm meeting.wav -l en
  FORCE_CPU=1 ./docker-run-transcribe.sh meeting.wav

Notes:
  - Audio capture still happens on the host via record_meeting.sh.
  - Results are written to ./output in the repository root.
  - HuggingFace cache is mounted from ~/.cache/huggingface by default.
  - All other arguments are forwarded to transcribe.py unchanged.
EOF
}

die() {
    echo "❌  $*" >&2
    exit 1
}

is_truthy() {
    local value
    value="${1:-}"
    value="${value,,}"
    case "$value" in
        1|true|yes|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

has_nvidia() {
    command -v nvidia-smi >/dev/null 2>&1
}

has_rocm() {
    [ -e "$DEV_KFD_PATH" ] && command -v rocminfo >/dev/null 2>&1 && rocminfo >/dev/null 2>&1
}

has_intel() {
    local render_glob vendor_file
    render_glob="$DRI_DIR"/render*

    if ! compgen -G "$render_glob" >/dev/null; then
        return 1
    fi

    if command -v clinfo >/dev/null 2>&1; then
        return 0
    fi

    for vendor_file in $DRM_VENDOR_GLOB; do
        if [ -f "$vendor_file" ] && grep -qiE '0x?8086' "$vendor_file"; then
            return 0
        fi
    done

    return 1
}

group_id_for_path() {
    local path
    path="$1"

    [ -e "$path" ] || return 1
    stat -c '%g' -- "$path" 2>/dev/null
}

add_runtime_group() {
    local group_id existing_group_id
    group_id="$1"

    [ -n "$group_id" ] || return

    for existing_group_id in "${runtime_group_ids[@]}"; do
        if [ "$existing_group_id" = "$group_id" ]; then
            return
        fi
    done

    runtime_group_ids+=("$group_id")
    run_flags+=(--group-add "$group_id")
}

add_device_group() {
    local path group_id
    path="$1"

    if group_id="$(group_id_for_path "$path")"; then
        add_runtime_group "$group_id"
    fi
}

pass_env_if_set() {
    local name
    name="$1"

    if [ "${!name+x}" ]; then
        run_flags+=(-e "$name=${!name}")
    fi
}

backend_for_known_image() {
    case "$1" in
        transcriber:cpu|transcriber:latest)
            echo "cpu"
            ;;
        transcriber:nvidia)
            echo "nvidia"
            ;;
        transcriber:rocm)
            echo "rocm"
            ;;
        transcriber:intel)
            echo "intel"
            ;;
        *)
            return 1
            ;;
    esac
}

backend_for_image_hint() {
    local image
    image="${1,,}"
    case "$image" in
        *nvidia*|*cuda*)
            echo "nvidia"
            ;;
        *rocm*|*amd*)
            echo "rocm"
            ;;
        *intel*|*xpu*)
            echo "intel"
            ;;
        *cpu*|*latest)
            echo "cpu"
            ;;
        *)
            return 1
            ;;
    esac
}

image_for_backend() {
    case "$1" in
        cpu)
            echo "transcriber:cpu"
            ;;
        nvidia)
            echo "transcriber:nvidia"
            ;;
        rocm)
            echo "transcriber:rocm"
            ;;
        intel)
            echo "transcriber:intel"
            ;;
        *)
            return 1
            ;;
    esac
}

dockerfile_for_backend() {
    case "$1" in
        base)
            echo "$REPO_ROOT/Dockerfile.base"
            ;;
        cpu)
            echo "$REPO_ROOT/Dockerfile.cpu"
            ;;
        nvidia)
            echo "$REPO_ROOT/Dockerfile.nvidia"
            ;;
        rocm)
            echo "$REPO_ROOT/Dockerfile.rocm"
            ;;
        intel)
            echo "$REPO_ROOT/Dockerfile.intel"
            ;;
        *)
            return 1
            ;;
    esac
}

image_exists() {
    "$DOCKER_BIN" image inspect "$1" >/dev/null 2>&1
}

build_image() {
    local backend image dockerfile
    backend="$1"
    image="$2"
    dockerfile="$(dockerfile_for_backend "$backend")"

    if [ ! -f "$dockerfile" ]; then
        die "Missing Dockerfile for backend '$backend': $dockerfile"
    fi

    echo "🐳  Building $image from $(basename "$dockerfile")"
    "$DOCKER_BIN" build -f "$dockerfile" -t "$image" "$REPO_ROOT"
}

ensure_image() {
    local image backend
    image="$1"

    if image_exists "$image"; then
        return
    fi

    if backend="$(backend_for_known_image "$image")"; then
        if [ "$backend" = "cpu" ] || [ "$backend" = "intel" ]; then
            if ! image_exists "$BASE_IMAGE"; then
                build_image base "$BASE_IMAGE"
            fi
        fi
        build_image "$backend" "$image"
        return
    fi

    die "Docker image '$image' was not found and cannot be auto-built. Build it manually or use one of: transcriber:cpu, transcriber:nvidia, transcriber:rocm, transcriber:intel."
}

select_backend() {
    if is_truthy "${FORCE_CPU:-0}"; then
        echo "cpu"
        return
    fi

    if has_nvidia; then
        echo "nvidia"
    elif has_rocm; then
        echo "rocm"
    elif has_intel; then
        echo "intel"
    else
        echo "cpu"
    fi
}

forwarded_has_help() {
    local token
    for token in "$@"; do
        if [ "$token" = "--help" ] || [ "$token" = "-h" ]; then
            return 0
        fi
    done
    return 1
}

find_audio_index() {
    local idx token skip_next
    skip_next=0

    for idx in "${!forwarded_args[@]}"; do
        token="${forwarded_args[$idx]}"

        if [ "$skip_next" -eq 1 ]; then
            skip_next=0
            continue
        fi

        case "$token" in
            -l|--language)
                skip_next=1
                continue
                ;;
            --language=*)
                continue
                ;;
            --)
                continue
                ;;
            -*)
                continue
                ;;
            *)
                echo "$idx"
                return 0
                ;;
        esac
    done

    return 1
}

force_cpu=0
custom_image=""
forwarded_args=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --help-docker)
            usage
            exit 0
            ;;
        --force-cpu)
            force_cpu=1
            ;;
        --image)
            shift
            [ "$#" -gt 0 ] || die "--image requires a Docker image tag."
            custom_image="$1"
            ;;
        --image=*)
            custom_image="${1#*=}"
            ;;
        --)
            shift
            forwarded_args+=("$@")
            break
            ;;
        *)
            forwarded_args+=("$1")
            ;;
    esac
    shift
done

if [ "$force_cpu" -eq 1 ]; then
    FORCE_CPU=1
fi

if [ -n "$custom_image" ]; then
    selected_image="$custom_image"
    if ! selected_backend="$(backend_for_known_image "$selected_image" 2>/dev/null)"; then
        selected_backend="$(backend_for_image_hint "$selected_image" 2>/dev/null || echo "cpu")"
    fi
else
    selected_backend="$(select_backend)"
    selected_image="$(image_for_backend "$selected_backend")"
fi

mkdir -p "$OUTPUT_DIR" "$HF_CACHE_DIR"

audio_mount=()
container_args=("${forwarded_args[@]}")

if audio_index="$(find_audio_index)"; then
    audio_path="${forwarded_args[$audio_index]}"
    [ -f "$audio_path" ] || die "Audio file not found: $audio_path"
    audio_abs="$(readlink -f -- "$audio_path")"
    container_audio="/input/$(basename -- "$audio_abs")"
    container_args[$audio_index]="$container_audio"
    audio_mount=( -v "$audio_abs:$container_audio:ro" )
else
    if ! forwarded_has_help "${forwarded_args[@]}"; then
        usage >&2
        die "Please provide an audio file path or pass --help to transcribe.py."
    fi
fi

run_flags=(
    --rm
    --user "$(id -u):$(id -g)"
    -e HF_HOME=/cache/huggingface
    -e XDG_CACHE_HOME=/cache
    -v "$OUTPUT_DIR:/app/output"
    -v "$HF_CACHE_DIR:/cache/huggingface"
)
runtime_group_ids=()

pass_env_if_set TRANSCRIBER_ANALYSIS_GPU_HEADROOM_GIB
pass_env_if_set TRANSCRIBER_ANALYSIS_GPU_MAX_MEMORY_GIB
pass_env_if_set TRANSCRIBER_ANALYSIS_MODEL
pass_env_if_set TRANSCRIBER_MAX_TRANSCRIPT_CHARS

case "$selected_backend" in
    nvidia)
        run_flags+=(--gpus all)
        ;;
    rocm)
        run_flags+=(-e "HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-0}")
        run_flags+=(--device "$DEV_KFD_PATH")
        add_device_group "$DEV_KFD_PATH"
        if [ -e "$DRI_DIR" ]; then
            run_flags+=(--device "$DRI_DIR")
            for dri_device in "$DRI_DIR"/*; do
                add_device_group "$dri_device"
            done
        fi
        ;;
    intel)
        if [ -e "$DRI_DIR" ]; then
            run_flags+=(--device "$DRI_DIR")
        fi
        ;;
    cpu)
        ;;
    *)
        die "Unsupported backend '$selected_backend'."
        ;;
esac

ensure_image "$selected_image"

echo "▶ Running transcribe.py in Docker"
echo "   Image   : $selected_image"
echo "   Backend : $selected_backend"
if [ -n "${audio_path:-}" ]; then
    echo "   Audio   : $audio_path"
fi
echo "   Output  : $OUTPUT_DIR"

audio_mount+=("$selected_image")
audio_mount+=("${container_args[@]}")

"$DOCKER_BIN" run "${run_flags[@]}" "${audio_mount[@]}"
