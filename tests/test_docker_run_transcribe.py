import os
import stat
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "docker-run-transcribe.sh"


MOCK_DOCKER = """#!/usr/bin/env bash
set -euo pipefail

log_path="${MOCK_DOCKER_LOG:?}"
built_path="${MOCK_DOCKER_BUILT:?}"
touch "$built_path"

record_call() {
    printf '%s' "$1" >> "$log_path"
    shift
    for arg in "$@"; do
        printf '|%s' "$arg" >> "$log_path"
    done
    printf '\n' >> "$log_path"
}

image_known() {
    local image
    image="$1"

    if grep -qxF "$image" "$built_path" 2>/dev/null; then
        return 0
    fi

    case " ${MOCK_DOCKER_IMAGES:-} " in
        *" $image "*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

command_name="$1"
shift

case "$command_name" in
    image)
        subcommand="$1"
        shift
        if [ "$subcommand" = "inspect" ]; then
            record_call image_inspect "$1"
            if image_known "$1"; then
                exit 0
            fi
            exit 1
        fi
        ;;
    build)
        record_call build "$@"
        image_tag=""
        while [ "$#" -gt 0 ]; do
            if [ "$1" = "-t" ]; then
                image_tag="$2"
                break
            fi
            shift
        done
        if [ -n "$image_tag" ]; then
            printf '%s\n' "$image_tag" >> "$built_path"
        fi
        exit 0
        ;;
    run)
        record_call run "$@"
        exit 0
        ;;
    *)
        record_call "$command_name" "$@"
        exit 0
        ;;
esac
"""


SIMPLE_SUCCESS_COMMAND = "#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _make_env(tmp_path: Path, *, existing_images: str = "") -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    docker_path = bin_dir / "docker"
    _write_executable(docker_path, MOCK_DOCKER)

    log_path = tmp_path / "docker.log"
    built_path = tmp_path / "built-images.log"

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "MOCK_DOCKER_LOG": str(log_path),
            "MOCK_DOCKER_BUILT": str(built_path),
            "MOCK_DOCKER_IMAGES": existing_images,
            "TRANSCRIBER_OUTPUT_DIR": str(tmp_path / "output"),
            "TRANSCRIBER_HF_CACHE_DIR": str(tmp_path / "hf-cache"),
        }
    )
    return env, log_path


def _read_calls(log_path: Path) -> list[list[str]]:
    return [line.split("|") for line in log_path.read_text(encoding="utf-8").splitlines()]


def _find_first_call(calls: list[list[str]], name: str) -> list[str]:
    return next(call for call in calls if call[0] == name)


def _find_all_calls(calls: list[list[str]], name: str) -> list[list[str]]:
    return [call for call in calls if call[0] == name]


def test_wrapper_auto_selects_nvidia_builds_image_and_forwards_transcribe_args(tmp_path):
    env, log_path = _make_env(tmp_path)
    _write_executable(tmp_path / "bin" / "nvidia-smi", SIMPLE_SUCCESS_COMMAND)

    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"RIFF")

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), str(audio_path), "-l", "en"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout

    calls = _read_calls(log_path)
    build_calls = _find_all_calls(calls, "build")
    run_call = _find_first_call(calls, "run")

    assert any(call[:5] == ["build", "-f", str(SCRIPT_PATH.parent / "Dockerfile.nvidia"), "-t", "transcriber:nvidia"] for call in build_calls)
    assert "--gpus" in run_call
    assert "all" in run_call
    assert "transcriber:nvidia" in run_call
    assert f"{audio_path.resolve()}:/input/meeting.wav:ro" in run_call
    assert f"{(tmp_path / 'output')}:/app/output" in run_call
    assert f"{(tmp_path / 'hf-cache')}:/cache/huggingface" in run_call
    assert "/input/meeting.wav" in run_call
    assert "-l" in run_call
    assert "en" in run_call


def test_wrapper_force_cpu_builds_base_then_cpu_and_omits_gpu_flags(tmp_path):
    env, log_path = _make_env(tmp_path)

    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"RIFF")

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--force-cpu", str(audio_path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout

    calls = _read_calls(log_path)
    build_calls = _find_all_calls(calls, "build")
    run_call = _find_first_call(calls, "run")

    built_tags = [call[call.index("-t") + 1] for call in build_calls]

    assert built_tags == ["transcriber:base", "transcriber:cpu"]
    assert "transcriber:cpu" in run_call
    assert "--gpus" not in run_call
    assert "--device" not in run_call


def test_wrapper_manual_rocm_image_uses_rocm_device_groups_without_rebuilding(tmp_path):
    env, log_path = _make_env(tmp_path, existing_images="transcriber:rocm")

    kfd_path = tmp_path / "kfd"
    kfd_path.write_text("", encoding="utf-8")
    dri_dir = tmp_path / "dri"
    dri_dir.mkdir()
    (dri_dir / "renderD128").write_text("", encoding="utf-8")

    env.update(
        {
            "TRANSCRIBER_DEV_KFD_PATH": str(kfd_path),
            "TRANSCRIBER_DRI_DIR": str(dri_dir),
        }
    )

    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"RIFF")

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--image", "transcriber:rocm", str(audio_path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout

    calls = _read_calls(log_path)
    build_calls = _find_all_calls(calls, "build")
    run_call = _find_first_call(calls, "run")

    assert build_calls == []
    assert "transcriber:rocm" in run_call
    assert "--device" in run_call
    assert str(kfd_path) in run_call
    assert str(dri_dir) in run_call
    assert "--group-add" in run_call
    assert str(kfd_path.stat().st_gid) in run_call
    assert str((dri_dir / "renderD128").stat().st_gid) in run_call
