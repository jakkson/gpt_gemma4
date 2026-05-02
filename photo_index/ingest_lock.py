"""Cross-process lock to prevent concurrent content ingests."""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path


_DEFAULT_LOCK_PATH = Path(__file__).resolve().parent.parent / "data" / "content_ingest.lock"


@contextmanager
def global_ingest_lock(lock_path: Path | None = None):
    """
    Acquire a non-blocking global lock for content ingestion.

    Any ingest job (photos/docs/email) can share this lock path to avoid
    competing for CPU, I/O, OCR/VLM runtime, and SQLite writes.
    """
    path = lock_path or _DEFAULT_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            acquired = False
        yield acquired
    finally:
        if acquired:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()
