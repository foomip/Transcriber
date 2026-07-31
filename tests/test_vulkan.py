import json
import subprocess
from types import SimpleNamespace

from lib import vulkan


def _payload(devices):
    return json.dumps({"schema_version": 1, "devices": devices})


def _device(index, name, device_type, budget=None):
    return {
        "index": index,
        "name": name,
        "type": device_type,
        "vendor_id": 0x1002,
        "heap_size_bytes": 8 * 1024**3,
        "heap_budget_bytes": budget,
    }


def test_parse_probe_ignores_cpu_vulkan_and_prefers_discrete(monkeypatch):
    output = _payload(
        [
            _device(0, "llvmpipe", "cpu"),
            _device(1, "Intel Arc", "integrated"),
            _device(2, "Radeon RX", "discrete", 6 * 1024**3),
        ]
    )
    monkeypatch.setattr(
        vulkan.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )

    result = vulkan.probe_vulkan()

    assert result.available is True
    assert result.selected_device is not None
    assert result.selected_device.index == 2
    assert [device.index for device in result.devices] == [1, 2]


def test_probe_respects_explicit_device(monkeypatch):
    output = _payload(
        [
            _device(1, "Integrated", "integrated"),
            _device(2, "Discrete", "discrete"),
        ]
    )
    monkeypatch.setenv(vulkan.VULKAN_DEVICE_ENV, "1")
    monkeypatch.setattr(
        vulkan.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )

    result = vulkan.probe_vulkan()

    assert result.selected_device is not None
    assert result.selected_device.name == "Integrated"


def test_probe_reports_invalid_override(monkeypatch):
    monkeypatch.setenv(vulkan.VULKAN_DEVICE_ENV, "not-a-number")
    monkeypatch.setattr(
        vulkan.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=_payload([_device(0, "GPU", "discrete")]),
            stderr="",
        ),
    )

    result = vulkan.probe_vulkan()

    assert result.available is False
    assert "Requested Vulkan device" in (result.reason or "")


def test_probe_handles_missing_binary(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(vulkan.subprocess, "run", missing)

    result = vulkan.probe_vulkan()

    assert result.available is False
    assert "not found" in (result.reason or "")


def test_probe_handles_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("probe", 10)

    monkeypatch.setattr(vulkan.subprocess, "run", timeout)

    result = vulkan.probe_vulkan()

    assert result.available is False
    assert "timed out" in (result.reason or "")


def test_probe_rejects_malformed_and_unknown_schema(monkeypatch):
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="not json", stderr=""),
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"schema_version": 2, "devices": []}),
                stderr="",
            ),
        ]
    )
    monkeypatch.setattr(vulkan.subprocess, "run", lambda *args, **kwargs: next(responses))

    assert vulkan.probe_vulkan().available is False
    assert vulkan.probe_vulkan().available is False


def test_probe_accepts_no_hardware_exit_status(monkeypatch):
    monkeypatch.setattr(
        vulkan.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2,
            stdout=_payload([_device(0, "lavapipe", "cpu")]),
            stderr="",
        ),
    )

    result = vulkan.probe_vulkan()

    assert result.available is False
    assert result.devices == ()
