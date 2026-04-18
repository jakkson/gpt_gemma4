"""JSON checkpoint files written during long ingest runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def checkpoint_path_for_db(db_path: Path) -> Path:
    """``photo_index.sqlite`` → ``photo_index.checkpoint.json`` (same directory)."""
    return db_path.with_name(db_path.stem + ".checkpoint.json")


def write_checkpoint(
    path: Path,
    *,
    db_path: Path,
    prefer: str,
    total_candidates: int,
    processed_new_this_run: int,
    last_uuid: str,
    last_filename: str,
    started_at_unix: float,
    elapsed_s: float | None = None,
    finished: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "db_path": str(db_path.resolve()),
        "prefer": prefer,
        "total_candidates": total_candidates,
        "processed_new_this_run": processed_new_this_run,
        "last_uuid": last_uuid,
        "last_filename": last_filename,
        "started_at_unix": started_at_unix,
        "updated_at_iso": datetime.now(timezone.utc).isoformat(),
        "ingest_finished": finished,
    }
    if elapsed_s is not None:
        payload["elapsed_s"] = round(elapsed_s, 3)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
