"""
hardware — shared hardware-detection helpers.

Responsibilities:
    - Parse accelerator CLI output into stable device names
"""

import re


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
