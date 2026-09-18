"""Resolves where benchmark logs are read from / written to.

Precedence: $PCIE_VIZ_LOGS_DIR env var, then the first non-comment line of
.storage-config (gitignored, machine-local override), then ./logs so a fresh
clone stays fully self-contained with no extra setup.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def resolve_logs_root() -> Path:
    env_override = os.environ.get("PCIE_VIZ_LOGS_DIR")
    if env_override:
        return Path(env_override).expanduser()
    config_file = REPO_ROOT / ".storage-config"
    if config_file.exists():
        for line in config_file.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                return Path(line).expanduser()
    return REPO_ROOT / "logs"
