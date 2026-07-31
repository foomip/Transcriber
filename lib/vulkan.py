"""Vendor-neutral Vulkan capability probing.

The Docker Vulkan image includes a tiny probe binary that emits a stable JSON
schema.  Keeping parsing here lets the Python pipeline make accelerator choices
without depending on CUDA, ROCm, or vendor-specific command-line tools.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Literal

VULKAN_PROBE_BIN_ENV = "TRANSCRIBER_VULKAN_PROBE_BIN"
VULKAN_DEVICE_ENV = "TRANSCRIBER_VULKAN_DEVICE"
DEFAULT_VULKAN_PROBE_BIN = "transcriber-vulkan-probe"
_SUPPORTED_SCHEMA_VERSION = 1
_MAX_PROBE_OUTPUT_BYTES = 1024 * 1024

VulkanDeviceType = Literal["discrete", "integrated", "virtual"]


@dataclass(frozen=True)
class VulkanDevice:
    index: int
    name: str
    device_type: VulkanDeviceType
    vendor_id: int
    heap_size_bytes: int | None
    heap_budget_bytes: int | None


@dataclass(frozen=True)
class VulkanProbeResult:
    available: bool
    selected_device: VulkanDevice | None
    devices: tuple[VulkanDevice, ...] = ()
    reason: str | None = None


def _optional_non_negative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("expected a non-negative integer or null")
    return value


def _parse_device(raw: Any) -> VulkanDevice | None:
    if not isinstance(raw, dict):
        raise ValueError("device entry must be an object")

    device_type = raw.get("type")
    # CPU Vulkan implementations (Lavapipe/LLVMpipe/SwiftShader) are valid
    # Vulkan devices but are not accelerators. Ignore them deliberately.
    if device_type == "cpu":
        return None
    if device_type not in {"discrete", "integrated", "virtual"}:
        raise ValueError(f"unsupported Vulkan device type: {device_type!r}")

    index = raw.get("index")
    vendor_id = raw.get("vendor_id")
    name = raw.get("name")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("device index must be a non-negative integer")
    if isinstance(vendor_id, bool) or not isinstance(vendor_id, int) or vendor_id < 0:
        raise ValueError("vendor_id must be a non-negative integer")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("device name must be a non-empty string")

    return VulkanDevice(
        index=index,
        name=name.strip(),
        device_type=device_type,
        vendor_id=vendor_id,
        heap_size_bytes=_optional_non_negative_int(raw.get("heap_size_bytes")),
        heap_budget_bytes=_optional_non_negative_int(raw.get("heap_budget_bytes")),
    )


def parse_vulkan_probe_output(output: str) -> tuple[VulkanDevice, ...]:
    if len(output.encode("utf-8")) > _MAX_PROBE_OUTPUT_BYTES:
        raise ValueError("Vulkan probe output exceeded the size limit")

    payload = json.loads(output)
    if not isinstance(payload, dict):
        raise ValueError("Vulkan probe output must be an object")
    if payload.get("schema_version") != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError("unsupported Vulkan probe schema version")

    raw_devices = payload.get("devices")
    if not isinstance(raw_devices, list):
        raise ValueError("Vulkan probe devices must be an array")

    devices: list[VulkanDevice] = []
    seen_indexes: set[int] = set()
    for raw_device in raw_devices:
        device = _parse_device(raw_device)
        if device is None:
            continue
        if device.index in seen_indexes:
            raise ValueError(f"duplicate Vulkan device index {device.index}")
        seen_indexes.add(device.index)
        devices.append(device)
    return tuple(devices)


def _select_device(devices: tuple[VulkanDevice, ...]) -> VulkanDevice | None:
    configured_index = os.environ.get(VULKAN_DEVICE_ENV)
    if configured_index is not None:
        try:
            requested_index = int(configured_index)
        except ValueError:
            return None
        return next((device for device in devices if device.index == requested_index), None)

    priority = {"discrete": 0, "integrated": 1, "virtual": 2}
    return min(
        devices,
        key=lambda device: (priority[device.device_type], device.index),
        default=None,
    )


def probe_vulkan(timeout_seconds: float = 10.0) -> VulkanProbeResult:
    probe_bin = os.environ.get(VULKAN_PROBE_BIN_ENV, DEFAULT_VULKAN_PROBE_BIN)
    try:
        completed = subprocess.run(
            [probe_bin],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return VulkanProbeResult(False, None, reason=f"Vulkan probe not found: {probe_bin}")
    except subprocess.TimeoutExpired:
        return VulkanProbeResult(False, None, reason="Vulkan probe timed out")
    except OSError as exc:
        return VulkanProbeResult(False, None, reason=f"Vulkan probe failed: {exc}")

    # The probe returns 2 when Vulkan works but only CPU/no devices exist.
    if completed.returncode not in {0, 2}:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        return VulkanProbeResult(False, None, reason=f"Vulkan probe failed: {detail}")

    try:
        devices = parse_vulkan_probe_output(completed.stdout)
    except (json.JSONDecodeError, ValueError) as exc:
        return VulkanProbeResult(False, None, reason=f"Invalid Vulkan probe output: {exc}")

    selected = _select_device(devices)
    if selected is None:
        if os.environ.get(VULKAN_DEVICE_ENV) is not None and devices:
            reason = f"Requested Vulkan device {os.environ[VULKAN_DEVICE_ENV]!r} is unavailable"
        else:
            reason = "No hardware Vulkan device was found"
        return VulkanProbeResult(False, None, devices=devices, reason=reason)

    return VulkanProbeResult(True, selected, devices=devices)
