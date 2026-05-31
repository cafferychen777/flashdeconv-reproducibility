#!/usr/bin/env python3
"""Execute reproducibility notebooks in smoke mode on temporary copies."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute notebook smoke runs without modifying tracked notebooks."
    )
    parser.add_argument(
        "--notebook-dir",
        type=Path,
        default=Path("notebooks"),
        help="Directory containing notebooks to execute.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root used as notebook working directory.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-notebook timeout in seconds.",
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Jupyter kernel name.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    notebook_dir = (repo_root / args.notebook_dir).resolve()
    notebooks = sorted(notebook_dir.glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError(f"No notebooks found in {notebook_dir}")

    os.environ.setdefault("FD_REPRO_MODE", "smoke")

    tmp = Path(tempfile.mkdtemp(prefix="fd-notebook-smoke-"))
    print(f"Executing {len(notebooks)} notebooks from temporary copies in {tmp}")
    for src in notebooks:
        dst = tmp / src.name
        shutil.copy2(src, dst)
        nb = nbformat.read(dst, as_version=4)
        NotebookClient(
            nb,
            timeout=args.timeout,
            kernel_name=args.kernel,
            resources={"metadata": {"path": str(repo_root)}},
        ).execute()
        nbformat.write(nb, dst)
        print(f"OK {src.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
