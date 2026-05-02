"""Nightly runner for photo ingest (intended for launchd schedule)."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run one photo ingest pass for nightly schedule.")
    p.add_argument("--db", default=str(Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"))
    p.add_argument("--vlm-model", default=os.environ.get("PHOTO_INDEX_VLM_MODEL", "gemma4:26b"))
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--skip-vlm", action="store_true")
    p.add_argument("--no-keep-awake", action="store_true")
    args = p.parse_args(argv)

    cmd = [
        os.environ.get("PYTHON_EXECUTABLE", "python"),
        "-m",
        "photo_index.ingest",
        "--db",
        args.db,
        "--vlm-model",
        args.vlm_model,
        "--progress-every",
        str(args.progress_every),
    ]
    if args.skip_vlm:
        cmd.append("--skip-vlm")
    if args.no_keep_awake:
        cmd.append("--no-keep-awake")

    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
