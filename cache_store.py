"""Disk-backed cache for the LOGS_ROOT scan.

Lives in .cache/ next to dashboard.py so only the very first-ever scan (or an
explicit Refresh click) pays the network-share cost -- a restart of the
Streamlit process reads the same cached JSON instantly instead of re-walking
LOGS_ROOT. Mirrors the pattern used by the WCL Sweep dashboard's cache_store.py.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"


def _cache_path(logs_root: Path) -> Path:
    key = hashlib.sha1(str(logs_root).encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"configs_{key}.json"


def _paths_to_str(configs: dict) -> dict:
    out = {}
    for key, cfg in configs.items():
        runs = [{k: (str(v) if isinstance(v, Path) else v) for k, v in run.items()} for run in cfg["runs"]]
        out[key] = {**cfg, "runs": runs}
    return out


def _str_to_paths(configs: dict) -> dict:
    out = {}
    for key, cfg in configs.items():
        runs = [
            {k: (Path(v) if k in ("json", "csv", "engine", "meta") else v) for k, v in run.items()}
            for run in cfg["runs"]
        ]
        out[key] = {**cfg, "runs": runs}
    return out


def load_configs(logs_root: Path) -> tuple[dict | None, dict | None]:
    path = _cache_path(logs_root)
    if not path.is_file():
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None
    return _str_to_paths(payload["configs"]), payload["meta"]


def save_configs(logs_root: Path, configs: dict, meta: dict) -> None:
    """Best-effort write: a failure here should never crash the caller -- the
    freshly-scanned data is still shown this run, it just won't be cached
    until a later attempt succeeds."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(logs_root)
    tmp_path = path.with_suffix(f".tmp{time.time_ns()}")
    payload = {"configs": _paths_to_str(configs), "meta": meta}
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp_path.replace(path)  # atomic-ish swap so a reader never sees a half-written file
    except OSError:
        tmp_path.unlink(missing_ok=True)


def make_meta(started_at: float) -> dict:
    now = time.time()
    return {"generated_at": now, "duration_seconds": round(now - started_at, 1)}
