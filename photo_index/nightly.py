"""Nightly runner for photo + messages ingest (intended for launchd schedule)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _REPO_ROOT / "data" / "photo_index.sqlite"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _run_module(module: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *extra_args]
    _log(f"[nightly] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    code = int(result.returncode or 0)
    if code != 0:
        _log(f"[nightly warn] {module} exited with code {code}")
    return code


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Run incremental photo + messages ingest for nightly schedule."
    )
    p.add_argument("--db", default=str(_DEFAULT_DB), help="SQLite index DB path.")
    p.add_argument(
        "--vlm-model",
        default=os.environ.get("PHOTO_INDEX_VLM_MODEL", "gemma4:26b"),
        help="Ollama vision model for new photos (photo_index.ingest).",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Photo ingest progress log interval (0=off).",
    )
    p.add_argument(
        "--messages-progress-every",
        type=int,
        default=500,
        help="Messages ingest progress log interval (0=off).",
    )
    p.add_argument("--skip-vlm", action="store_true", help="Photos: OCR only, no VLM captions.")
    p.add_argument("--skip-photos", action="store_true", help="Skip photo_index.ingest.")
    p.add_argument("--skip-messages", action="store_true", help="Skip photo_index.messages_ingest.")
    p.add_argument(
        "--no-keep-awake",
        action="store_true",
        help="Do not prevent idle sleep during photo ingest.",
    )
    args = p.parse_args(argv)

    db = str(args.db)
    rc = 0

    if not args.skip_photos:
        photo_args = [
            "--db",
            db,
            "--vlm-model",
            args.vlm_model,
            "--progress-every",
            str(args.progress_every),
        ]
        if args.skip_vlm:
            photo_args.append("--skip-vlm")
        if args.no_keep_awake:
            photo_args.append("--no-keep-awake")
        rc = max(rc, _run_module("photo_index.ingest", photo_args))
    else:
        _log("[nightly] skipping photo_index.ingest")

    if not args.skip_messages:
        msg_args = [
            "--db",
            db,
            "--progress-every",
            str(args.messages_progress_every),
        ]
        rc = max(rc, _run_module("photo_index.messages_ingest", msg_args))
    else:
        _log("[nightly] skipping photo_index.messages_ingest")

    _log(f"[nightly] finished with exit code {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
