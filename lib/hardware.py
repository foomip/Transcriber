"""
hardware — shared hardware-detection helpers.

Responsibilities:
    - Parse accelerator CLI output into stable device names
    - Probe for an available GPU and return a (kind, device_name) tuple
      that callers can use to configure runtimes without duplicating
      detection logic across the project.
"""

import os
import re
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def detect_gpu() -> tuple[str, str]:
    """Return ``(kind, device_name)`` for the best available accelerator.

    ``kind`` is one of:

    * ``"cuda"``  — NVIDIA CUDA GPU detected via pynvml or ``/dev/nvidia0``
    * ``"rocm"``  — AMD ROCm GPU detected via ``/dev/kfd`` / ``rocm-smi``
    * ``"cpu"``   — no GPU found

    ``device_name`` is a human-readable label such as ``"NVIDIA RTX 3060"`` or
    ``"Radeon RX 7800 XT"``.

    The function is intentionally side-effect-free: it never prints, never
    modifies environment variables, and never imports optional GPU libraries
    at module-load time.  Callers are responsible for any user-facing output.
    """
    # --- NVIDIA ---
    try:
        import pynvml  # type: ignore[import]

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            return "cuda", name
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
    except (ImportError, Exception) as exc:  # noqa: BLE001
        if os.environ.get("DEBUG") == "1":
            print(f"  DEBUG: NVIDIA detection via pynvml failed: {exc}")

        if os.path.exists("/dev/nvidia0"):
            return "cuda", "NVIDIA GPU (detected via /dev/nvidia0)"

    # --- AMD / ROCm ---
    if os.path.exists("/dev/kfd"):
        try:
            res = subprocess.check_output(
                ["rocm-smi", "--showproductname"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if name := parse_rocm_product_name(res):
                return "rocm", name
            if os.environ.get("DEBUG") == "1":
                print("  DEBUG: Could not parse a ROCm GPU name from rocm-smi output")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            if os.environ.get("DEBUG") == "1":
                print(f"  DEBUG: ROCm SMI failed: {exc}")
        return "rocm", "AMD GPU"

    return "cpu", "CPU"


def parse_rocm_product_name(output: str) -> str | None:
    """Extract a human-friendly AMD GPU name from rocm-smi output.

    Supports both the older tabular format::

        GPU  Product Name
        0    Navi 21

    and newer verbose blocks such as::

        ======================================== Product Info ========================================
        GPU[0]          : Card series: Radeon RX 7800 XT
        GPU[0]          : Card model: 0x747e
    """
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower_line = line.casefold()
        if (
            "rocm system management interface" in lower_line
            or "product info" in lower_line
            or set(line) <= {"="}
            or lower_line.startswith("gpu  product name")
        ):
            continue

        verbose_match = re.search(
            r"GPU\[\d+\]\s*:\s*(?:card\s+series|product\s+name)\s*:\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        if verbose_match:
            return verbose_match.group(1).strip()

        tabular_match = re.match(r"^\d+\s+(.+)$", line)
        if tabular_match:
            return tabular_match.group(1).strip()

    return None
