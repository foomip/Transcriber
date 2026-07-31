import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lib import whisper_cpp
from lib.vulkan import VulkanDevice


def _valid_payload():
    return {
        "result": {"language": "en", "language_probability": 0.94},
        "transcription": [
            {"offsets": {"from": 0, "to": 4500}, "text": " Hello "},
            {"offsets": {"from": 4500, "to": 65250}, "text": "Next topic"},
        ],
    }


def test_parse_result_returns_segments_and_language(tmp_path):
    output = tmp_path / "result.json"
    output.write_text(json.dumps(_valid_payload()), encoding="utf-8")

    result = whisper_cpp.parse_result(output)

    assert result.detected_language == "en"
    assert result.language_probability == 0.94
    assert result.segments == (
        (0.0, 4.5, "Hello"),
        (4.5, 65.25, "Next topic"),
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda data: data["result"].update(language_probability=1.5), "between 0 and 1"),
        (lambda data: data["transcription"][0]["offsets"].update(to=-1), "invalid timestamps"),
        (lambda data: data.update(transcription="bad"), "unsupported schema"),
    ],
)
def test_parse_result_rejects_invalid_output(tmp_path, mutation, message):
    payload = _valid_payload()
    mutation(payload)
    output = tmp_path / "result.json"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(whisper_cpp.WhisperCppError, match=message):
        whisper_cpp.parse_result(output)


def test_ensure_model_uses_explicit_path_without_download(tmp_path, monkeypatch):
    model = tmp_path / "custom.bin"
    model.write_bytes(b"model")
    monkeypatch.setenv(whisper_cpp.WHISPER_CPP_MODEL_PATH_ENV, str(model))

    assert whisper_cpp.ensure_model() == model


def test_verify_default_model_writes_marker(tmp_path, monkeypatch):
    model = tmp_path / whisper_cpp.DEFAULT_MODEL_FILENAME
    model.write_bytes(b"model")
    expected = whisper_cpp.hashlib.sha256(b"model").hexdigest()
    monkeypatch.setattr(whisper_cpp, "DEFAULT_MODEL_SHA256", expected)

    whisper_cpp._verify_default_model(model)

    assert model.with_suffix(".bin.sha256").read_text(encoding="utf-8").strip() == expected


def test_verify_default_model_rejects_bad_checksum(tmp_path, monkeypatch):
    model = tmp_path / whisper_cpp.DEFAULT_MODEL_FILENAME
    model.write_bytes(b"corrupt")
    monkeypatch.setattr(whisper_cpp, "DEFAULT_MODEL_SHA256", "0" * 64)

    with pytest.raises(whisper_cpp.WhisperCppError, match="checksum mismatch"):
        whisper_cpp._verify_default_model(model)


def test_transcribe_normalises_audio_and_parses_json(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"RIFF")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        output_prefix = Path(command[command.index("--output-file") + 1])
        output_prefix.with_suffix(".json").write_text(
            json.dumps(_valid_payload()), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(whisper_cpp.subprocess, "run", fake_run)
    device = VulkanDevice(2, "GPU", "discrete", 0x1002, 8 * 1024**3, 7 * 1024**3)

    result = whisper_cpp.transcribe(
        "meeting with spaces.wav",
        model_path=tmp_path / "model.bin",
        device=device,
        language="en",
    )

    assert result.detected_language == "en"
    assert calls[0][0] == "ffmpeg"
    assert "meeting with spaces.wav" in calls[0]
    assert calls[1][calls[1].index("--device") + 1] == "2"
    assert calls[1][calls[1].index("--language") + 1] == "en"


def test_transcribe_reports_whisper_process_failure(tmp_path, monkeypatch):
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=3, stdout="", stderr="GPU init failed"),
        ]
    )
    monkeypatch.setattr(whisper_cpp.subprocess, "run", lambda *args, **kwargs: next(responses))
    device = VulkanDevice(0, "GPU", "discrete", 0, None, None)

    with pytest.raises(whisper_cpp.WhisperCppError, match="GPU init failed"):
        whisper_cpp.transcribe(
            "meeting.wav",
            model_path=tmp_path / "model.bin",
            device=device,
            language=None,
        )
