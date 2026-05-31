"""Path and runtime-mode helpers for notebooks and CLI wrappers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime configuration for reproducibility entrypoints."""

    repo_root: Path
    data_dir: Path
    results_dir: Path
    figures_dir: Path
    run_mode: str


def _resolve_dir(env_name: str, default: str | Path) -> Path:
    value = os.environ.get(env_name)
    path = Path(value) if value else Path(default)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def get_run_mode(default: str = "smoke") -> str:
    """Return FD_REPRO_MODE normalized to either smoke or full."""
    mode = os.environ.get("FD_REPRO_MODE", default).strip().lower()
    if mode not in {"smoke", "full"}:
        raise ValueError("FD_REPRO_MODE must be 'smoke' or 'full'")
    return mode


def get_runtime_config(
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    run_mode: str | None = None,
) -> RuntimeConfig:
    """Resolve repo, data, results, figure paths and run mode."""
    data = _resolve_dir("FD_DATA_DIR", data_dir or "data")
    results = _resolve_dir("FD_RESULTS_DIR", results_dir or "results")
    figures = _resolve_dir("FD_FIGURES_DIR", figures_dir or "figures")
    mode = (run_mode or get_run_mode()).strip().lower()
    if mode not in {"smoke", "full"}:
        raise ValueError("run_mode must be 'smoke' or 'full'")
    return RuntimeConfig(REPO_ROOT, data, results, figures, mode)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def missing_inputs_message(paths: list[Path], command_hint: str) -> str:
    """Create a concise message for missing full-run inputs."""
    missing = [str(p) for p in paths if not p.exists()]
    if not missing:
        return ""
    joined = "\n  - ".join(missing)
    return f"Missing required input files:\n  - {joined}\nRun:\n  {command_hint}"

