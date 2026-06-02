import json
import os
import pathlib
import stat
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "docker-run-transcribe.sh"


FAKE_DOCKER_SCRIPT = """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

log_path = pathlib.Path(os.environ[\"FAKE_DOCKER_LOG\"])
present_images = set(json.loads(os.environ.get(\"FAKE_DOCKER_PRESENT_IMAGES\", \"[]\")))
args = sys.argv[1:]

with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": args}) + "\\n")

if args[:2] == ["image", "inspect"]:
    image = args[2] if len(args) > 2 else ""
    raise SystemExit(0 if image in present_images else 1)

raise SystemExit(0)
"""


SMOKE_SCRIPT = """#!/usr/bin/env bash
exit 0
"""


def _make_executable(path: pathlib.Path, content: str) -> pathlib.Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _read_docker_calls(log_path: pathlib.Path) -> list[list[str]]:
    return [json.loads(line)["argv"] for line in log_path.read_text(encoding="utf-8").splitlines()]


def _run_wrapper(
    tmp_path: pathlib.Path,
    args: list[str],
    *,
    present_images: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    add_nvidia_smi: bool = False,
    add_rocminfo: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "fake-docker.log"

    _make_executable(fake_bin / "docker", FAKE_DOCKER_SCRIPT)
    if add_nvidia_smi:
        _make_executable(fake_bin / "nvidia-smi", SMOKE_SCRIPT)
    if add_rocminfo:
        _make_executable(fake_bin / "rocminfo", SMOKE_SCRIPT)

    output_dir = tmp_path / "output"
    hf_cache_dir = tmp_path / "hf-cache"
    gguf_cache_dir = tmp_path / "gguf-cache"

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_DOCKER_LOG": str(docker_log),
            "FAKE_DOCKER_PRESENT_IMAGES": json.dumps(present_images or []),
            "TRANSCRIBER_DOCKER_BIN": str(fake_bin / "docker"),
            "TRANSCRIBER_OUTPUT_DIR": str(output_dir),
            "TRANSCRIBER_HF_CACHE_DIR": str(hf_cache_dir),
            "TRANSCRIBER_GGUF_CACHE_DIR": str(gguf_cache_dir),
        }
    )
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [str(SCRIPT_PATH), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    calls = _read_docker_calls(docker_log)
    return result, calls


def _call_for_subcommand(calls: list[list[str]], subcommand: str) -> list[str]:
    return next(call for call in calls if call and call[0] == subcommand)


def _values_after_flag(argv: list[str], flag: str) -> list[str]:
    values = []
    for index, token in enumerate(argv[:-1]):
        if token == flag:
            values.append(argv[index + 1])
    return values


def test_auto_nvidia_backend_builds_nvidia_image_and_forwards_flags(tmp_path):
    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"RIFF")
    same_gid_device_a = tmp_path / "nvidia-uvm"
    same_gid_device_b = tmp_path / "nvidiactl"
    same_gid_device_a.write_text("", encoding="utf-8")
    same_gid_device_b.write_text("", encoding="utf-8")

    nvidia_glob = f"/dev/null {same_gid_device_a} {same_gid_device_b}"
    result, calls = _run_wrapper(
        tmp_path,
        [str(audio_path), "-l", "en", "--dry-run"],
        add_nvidia_smi=True,
        extra_env={"TRANSCRIBER_NVIDIA_DEVICE_GLOB": nvidia_glob},
    )

    assert result.returncode == 0, result.stderr

    build_call = _call_for_subcommand(calls, "build")
    assert build_call == [
        "build",
        "-f",
        str(REPO_ROOT / "Dockerfile.nvidia"),
        "-t",
        "transcriber:nvidia",
        str(REPO_ROOT),
    ]

    run_call = _call_for_subcommand(calls, "run")
    assert "--gpus" in run_call
    assert run_call[run_call.index("--gpus") + 1] == "all"
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in run_call

    expected_group_ids = {
        str(os.stat("/dev/null").st_gid),
        str(os.stat(same_gid_device_a).st_gid),
    }
    group_ids = _values_after_flag(run_call, "--group-add")
    assert set(group_ids) == expected_group_ids
    assert len(group_ids) == len(expected_group_ids)

    audio_mount = f"{audio_path.resolve()}:/input/{audio_path.name}:ro"
    assert audio_mount in run_call

    image_index = run_call.index("transcriber:nvidia")
    assert run_call[image_index + 1 :] == [f"/input/{audio_path.name}", "-l", "en", "--dry-run"]


def test_custom_cuda_image_hint_uses_nvidia_runtime_flags_without_rebuild(tmp_path):
    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"RIFF")
    custom_image = "ghcr.io/example/transcriber-cuda:latest"

    result, calls = _run_wrapper(
        tmp_path,
        ["--image", custom_image, str(audio_path), "-l", "fr"],
        present_images=[custom_image],
        extra_env={"NVIDIA_DRIVER_CAPABILITIES": "compute,video"},
    )

    assert result.returncode == 0, result.stderr
    assert not any(call[0] == "build" for call in calls)

    run_call = _call_for_subcommand(calls, "run")
    assert "--gpus" in run_call
    assert run_call[run_call.index("--gpus") + 1] == "all"
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,video" in run_call

    image_index = run_call.index(custom_image)
    assert run_call[image_index + 1 :] == [f"/input/{audio_path.name}", "-l", "fr"]


def test_rocm_backend_adds_device_mounts_and_deduplicated_groups(tmp_path):
    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"RIFF")
    dev_kfd = tmp_path / "kfd"
    dev_kfd.write_text("", encoding="utf-8")
    dri_dir = tmp_path / "dri"
    dri_dir.mkdir()
    (dri_dir / "card0").write_text("", encoding="utf-8")
    (dri_dir / "renderD128").write_text("", encoding="utf-8")

    result, calls = _run_wrapper(
        tmp_path,
        ["--image", "transcriber:rocm", str(audio_path)],
        extra_env={
            "TRANSCRIBER_DEV_KFD_PATH": str(dev_kfd),
            "TRANSCRIBER_DRI_DIR": str(dri_dir),
        },
    )

    assert result.returncode == 0, result.stderr

    build_call = _call_for_subcommand(calls, "build")
    assert build_call[0] == "build"
    assert "-f" in build_call
    assert build_call[build_call.index("-f") + 1] == str(REPO_ROOT / "Dockerfile.rocm")
    assert "-t" in build_call
    assert build_call[build_call.index("-t") + 1] == "transcriber:rocm"
    assert build_call[-1] == str(REPO_ROOT)

    run_call = _call_for_subcommand(calls, "run")
    assert f"HSA_ENABLE_SDMA=0" in run_call
    device_values = _values_after_flag(run_call, "--device")
    assert str(dev_kfd) in device_values
    assert str(dri_dir) in device_values

    expected_group_ids = {str(os.stat(dev_kfd).st_gid)}
    group_ids = _values_after_flag(run_call, "--group-add")
    assert set(group_ids) == expected_group_ids
    assert len(group_ids) == len(expected_group_ids)

    image_index = run_call.index("transcriber:rocm")
    assert run_call[image_index + 1 :] == [f"/input/{audio_path.name}"]
