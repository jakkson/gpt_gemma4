"""Nightly runner for photo + messages + documents ingest (intended for launchd schedule).

Documents: text extract (documents_ingest) then Apple Vision OCR + Ollama VLM
on PDF/PNG/JPG (documents_vlm_ingest). Both skip unchanged files unless --force
on the child modules. Requires Ollama for photo and document VLM steps.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _REPO_ROOT / "data" / "photo_index.sqlite"
_DEFAULT_DOCUMENTS_ROOT = Path.home() / "Dropbox" / "Documents"


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
        description=(
            "Run incremental photo, messages, and documents ingest for nightly schedule. "
            "Documents: text extract then OCR+VLM on PDF/PNG/JPG. Each child skips "
            "unchanged rows (no --force)."
        )
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
    p.add_argument(
        "--documents-root",
        type=str,
        default=os.environ.get("PHOTO_INDEX_DOCUMENTS_ROOT", str(_DEFAULT_DOCUMENTS_ROOT)),
        help="Folder to walk for documents ingest (default: ~/Dropbox/Documents).",
    )
    p.add_argument(
        "--documents-progress-every",
        type=int,
        default=500,
        help="Documents ingest progress log interval (0=off).",
    )
    p.add_argument("--skip-vlm", action="store_true", help="Photos: OCR only, no VLM captions.")
    p.add_argument("--skip-photos", action="store_true", help="Skip photo_index.ingest.")
    p.add_argument("--skip-messages", action="store_true", help="Skip photo_index.messages_ingest.")
    p.add_argument("--skip-mail", action="store_true", help="Skip photo_index.mail_ingest.")
    p.add_argument("--skip-documents", action="store_true", help="Skip photo_index.documents_ingest.")
    p.add_argument(
        "--skip-documents-vlm",
        action="store_true",
        help="Skip Apple Vision OCR + Ollama VLM on PDF/PNG/JPG in documents folder.",
    )
    p.add_argument(
        "--documents-vlm-model",
        default=os.environ.get("PHOTO_INDEX_DOC_VLM_MODEL", "gemma4:latest"),
        help="Ollama vision model for documents_vlm_ingest (PDF/image OCR+VLM).",
    )
    p.add_argument(
        "--documents-vlm-progress-every",
        type=int,
        default=10,
        help="documents_vlm_ingest progress log interval (0=off).",
    )
    p.add_argument(
        "--skip-documents-vlm-ocr",
        action="store_true",
        help="documents_vlm: VLM only (no Apple Vision OCR). Use if Vision fails on host.",
    )
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

    if not args.skip_mail:
        mail_args = ["--db", db, "--progress-every", "1000"]
        rc = max(rc, _run_module("photo_index.mail_ingest", mail_args))
    else:
        _log("[nightly] skipping photo_index.mail_ingest")

    if not args.skip_documents:
        doc_root = str(Path(args.documents_root).expanduser().resolve())
        doc_args = [
            "--db",
            db,
            "--root",
            doc_root,
            "--progress-every",
            str(args.documents_progress_every),
        ]
        rc = max(rc, _run_module("photo_index.documents_ingest", doc_args))
    else:
        _log("[nightly] skipping photo_index.documents_ingest")

    if not args.skip_documents_vlm:
        doc_root = str(Path(args.documents_root).expanduser().resolve())
        vlm_args = [
            "--db",
            db,
            "--root",
            doc_root,
            "--vlm-model",
            args.documents_vlm_model,
            "--progress-every",
            str(args.documents_vlm_progress_every),
        ]
        if args.skip_documents_vlm_ocr:
            vlm_args.append("--skip-ocr")
        rc = max(rc, _run_module("photo_index.documents_vlm_ingest", vlm_args))
    else:
        _log("[nightly] skipping photo_index.documents_vlm_ingest")

    _log(f"[nightly] finished with exit code {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
