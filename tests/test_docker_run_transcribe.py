import json
import os
import pathlib
import stat
import subprocess


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "docker-run-transcribe.sh"

FAKE_DOCKER_SCRIPT = """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

log_path = pathlib.Path(os.environ["FAKE_DOCKER_LOG"])
present_images = set(json.loads(os.environ.get("FAKE_DOCKER_PRESENT_IMAGES", "[]")))
args = sys.argv[1:]
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": args}) + "\\n")
if args[:2] == ["image", "inspect"]:
    image = args[2] if len(args) > 2 else ""
    raise SystemExit(0 if image in present_images else 1)
if args and args[0] == "run" and "transcriber-vulkan-probe" in args:
    raise SystemExit(int(os.environ.get("FAKE_VULKAN_PREFLIGHT_EXIT", "0")))
raise SystemExit(0)
"""

SMOKE_SCRIPT = """#!/usr/bin/env bash
exit 0
"""

FAIL_SCRIPT = """#!/usr/bin/env bash
exit 1
"""


def _make_executable(path: pathlib.Path, content: str) -> pathlib.Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _run_wrapper(
    tmp_path: pathlib.Path,
    args: list[str],
    *,
    present_images: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    add_nvidia_smi: bool = False,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    _make_executable(fake_bin / "docker", FAKE_DOCKER_SCRIPT)
    _make_executable(
        fake_bin / "nvidia-smi",
        SMOKE_SCRIPT if add_nvidia_smi else FAIL_SCRIPT,
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_DOCKER_LOG": str(docker_log),
            "FAKE_DOCKER_PRESENT_IMAGES": json.dumps(present_images or []),
            "TRANSCRIBER_DOCKER_BIN": str(fake_bin / "docker"),
            "TRANSCRIBER_OUTPUT_DIR": str(tmp_path / "output"),
            "TRANSCRIBER_HF_CACHE_DIR": str(tmp_path / "hf"),
            "TRANSCRIBER_GGUF_CACHE_DIR": str(tmp_path / "gguf"),
            "TRANSCRIBER_WHISPER_CPP_CACHE_DIR": str(tmp_path / "whisper"),
            "TRANSCRIBER_DRI_DIR": str(tmp_path / "missing-dri"),
            "TRANSCRIBER_NVIDIA_DEVICE_GLOB": str(tmp_path / "missing-nvidia*"),
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
    calls = [
        json.loads(line)["argv"]
        for line in docker_log.read_text(encoding="utf-8").splitlines()
    ]
    return result, calls


def _calls(calls, subcommand):
    return [call for call in calls if call and call[0] == subcommand]


def _real_run(calls):
    return next(call for call in reversed(_calls(calls, "run")) if "--user" in call)


def _values_after_flag(argv, flag):
    return [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == flag]


def test_no_gpu_builds_and_runs_cpu_target(tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"RIFF")

    result, calls = _run_wrapper(tmp_path, [str(audio), "-l", "en"])

    assert result.returncode == 0, result.stderr
    assert _calls(calls, "build") == [
        [
            "build",
            "--target",
            "cpu",
            "-f",
            str(REPO_ROOT / "Dockerfile"),
            "-t",
            "transcriber:cpu",
            str(REPO_ROOT),
        ]
    ]
    run = _real_run(calls)
    assert "transcriber:cpu" in run
    assert "--gpus" not in run
    assert f"{audio.resolve()}:/input/{audio.name}:ro" in run
    image_index = run.index("transcriber:cpu")
    assert run[image_index + 1 :] == [f"/input/{audio.name}", "-l", "en"]


def test_nvidia_uses_vulkan_preflight_and_graphics_capability(tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"RIFF")

    result, calls = _run_wrapper(
        tmp_path,
        [str(audio)],
        present_images=["transcriber:vulkan"],
        add_nvidia_smi=True,
    )

    assert result.returncode == 0, result.stderr
    run_calls = _calls(calls, "run")
    assert len(run_calls) == 2
    preflight, run = run_calls
    assert "transcriber-vulkan-probe" in preflight
    assert not any(str(audio) in token for token in preflight)
    assert "--gpus" in run
    assert run[run.index("--gpus") + 1] == "all"
    assert "NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility" in run
    assert "transcriber:vulkan" in run


def test_dri_uses_vulkan_and_deduplicates_device_groups(tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"RIFF")
    dri = tmp_path / "dri"
    dri.mkdir()
    (dri / "card0").write_text("", encoding="utf-8")
    (dri / "renderD128").write_text("", encoding="utf-8")

    result, calls = _run_wrapper(
        tmp_path,
        [str(audio)],
        present_images=["transcriber:vulkan"],
        extra_env={"TRANSCRIBER_DRI_DIR": str(dri)},
    )

    assert result.returncode == 0, result.stderr
    run = _real_run(calls)
    assert str(dri) in _values_after_flag(run, "--device")
    group_ids = _values_after_flag(run, "--group-add")
    assert group_ids == [str(os.stat(dri / "card0").st_gid)]
    assert "transcriber:vulkan" in run


def test_failed_vulkan_preflight_falls_back_to_cpu_before_audio_mount(tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"RIFF")
    dri = tmp_path / "dri"
    dri.mkdir()
    (dri / "renderD128").write_text("", encoding="utf-8")

    result, calls = _run_wrapper(
        tmp_path,
        [str(audio)],
        present_images=["transcriber:vulkan", "transcriber:cpu"],
        extra_env={
            "TRANSCRIBER_DRI_DIR": str(dri),
            "FAKE_VULKAN_PREFLIGHT_EXIT": "2",
        },
    )

    assert result.returncode == 0, result.stderr
    preflight = _calls(calls, "run")[0]
    assert not any(str(audio) in token for token in preflight)
    run = _real_run(calls)
    assert "transcriber:cpu" in run
    assert "--device" not in run
    assert "Vulkan preflight failed" in result.stdout


def test_force_cpu_wins_over_detected_nvidia_and_custom_image(tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"RIFF")

    result, calls = _run_wrapper(
        tmp_path,
        ["--force-cpu", "--image", "transcriber:vulkan", str(audio)],
        present_images=["transcriber:cpu", "transcriber:vulkan"],
        add_nvidia_smi=True,
    )

    assert result.returncode == 0, result.stderr
    run = _real_run(calls)
    assert "transcriber:cpu" in run
    assert "--gpus" not in run
    assert len(_calls(calls, "run")) == 1


def test_vulkan_run_mounts_new_whisper_cache(tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"RIFF")
    dri = tmp_path / "dri"
    dri.mkdir()
    (dri / "renderD128").write_text("", encoding="utf-8")
    whisper_cache = tmp_path / "custom-whisper-cache"

    result, calls = _run_wrapper(
        tmp_path,
        [str(audio)],
        present_images=["transcriber:vulkan"],
        extra_env={
            "TRANSCRIBER_DRI_DIR": str(dri),
            "TRANSCRIBER_WHISPER_CPP_CACHE_DIR": str(whisper_cache),
        },
    )

    assert result.returncode == 0, result.stderr
    run = _real_run(calls)
    assert f"{whisper_cache}:/cache/transcriber/whisper" in run
    assert "TRANSCRIBER_WHISPER_CPP_CACHE_DIR=/cache/transcriber/whisper" in run


def test_unknown_missing_custom_image_fails_without_auto_build(tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"RIFF")

    result, calls = _run_wrapper(
        tmp_path,
        ["--image", "example/custom:vulkan", str(audio)],
    )

    assert result.returncode == 1
    assert not _calls(calls, "build")
    assert "cannot be auto-built" in result.stderr
