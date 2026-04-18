"""Retry when Photos.sqlite or our index DB is temporarily locked (e.g. Photos.app open)."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def is_transient_lock_error(exc: BaseException) -> bool:
    """True if the error may clear after waiting (Photos or SQLite busy/locked)."""
    msg = str(exc).lower()
    markers = (
        "locked",
        "busy",
        "lock",
        "could not obtain",
        "database is locked",
        "unable to open",
        "temporary failure",
        "resource busy",
        "error copying",  # osxphotos temp copy while DB held
    )
    if any(m in msg for m in markers):
        return True
    if isinstance(exc, sqlite3.OperationalError) and (
        "locked" in msg or "busy" in msg
    ):
        return True
    # macOS file busy / resource temporarily unavailable
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (11, 16, 35, 89):
        return True
    return False


def retry_on_transient_lock(
    fn: Callable[[], T],
    *,
    log,
    wait_seconds: float = 30.0,
    max_attempts: int = 120,
    what: str = "operation",
) -> T:
    """
    Run ``fn``; on transient lock/busy errors, log and sleep ``wait_seconds``, then retry.

    ``max_attempts`` is the number of *retries* after the first failure (total tries = 1 + retries
    until success or cap). Set high so a long Photos session does not kill an overnight ingest.
    """
    last: BaseException | None = None
    for attempt in range(max_attempts + 1):
        try:
            return fn()
        except BaseException as e:
            last = e
            if not is_transient_lock_error(e):
                raise
            if attempt >= max_attempts:
                break
            log(
                f"[wait] {what} blocked ({e!r}); "
                f"try closing Photos.app. Sleeping {wait_seconds:.0f}s "
                f"(attempt {attempt + 1}/{max_attempts})…"
            )
            time.sleep(wait_seconds)
    assert last is not None
    raise last
