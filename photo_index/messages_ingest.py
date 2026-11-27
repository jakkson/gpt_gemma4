"""Ingest Apple Messages text into the shared SQLite/FTS index."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from photo_index.ingest_lock import global_ingest_lock
from photo_index.store import already_indexed, commit_ingest, connect, init_schema, upsert_photo

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"
_DEFAULT_CHAT_DB = Path.home() / "Library" / "Messages" / "chat.db"
_APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)

_MESSAGES_ACCESS_HELP = """
macOS blocked read access to Messages (chat.db).

Fix: System Settings -> Privacy & Security -> Full Disk Access -> enable for
Cursor or Terminal, then restart the app and try again.
""".strip()

def _extract_attributed_body(blob: bytes | None) -> str:
    """Extract plain text from an NSAttributedString typedstream blob.

    Apple Messages sometimes stores text only in ``attributedBody``
    (an NSMutableAttributedString archive) and leaves ``text`` NULL.
    The plain text follows the ``\\x2b`` marker after the NSString class
    header and ends at the ``\\x86`` attribute-dictionary byte.
    """
    if not blob:
        return ""
    try:
        idx = blob.find(b"NSString")
        if idx < 0:
            return ""
        # Locate the \x2b ('+') type marker that precedes the length + text.
        plus = blob.find(b"\x2b", idx + 8)
        if plus < 0 or plus > idx + 30:
            return ""
        pos = plus + 1
        if pos >= len(blob):
            return ""
        # BER-style length: if high bit set, next byte(s) hold the length.
        length_byte = blob[pos]
        pos += 1
        if length_byte & 0x80:
            n_len_bytes = length_byte & 0x7F
            if n_len_bytes < 1 or pos + n_len_bytes > len(blob):
                return ""
            length = int.from_bytes(blob[pos:pos + n_len_bytes], "big")
            pos += n_len_bytes
        else:
            length = length_byte
        # Skip any NUL padding between length and text.
        while pos < len(blob) and blob[pos] == 0x00:
            pos += 1
        # Read until \x86 (attribute dictionary) or exhausted length.
        end = blob.find(b"\x86", pos)
        if end < 0:
            end = pos + length
        else:
            end = min(end, pos + length)
        text_bytes = blob[pos:end]
        return text_bytes.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _log(msg: str) -> None:
    print(msg, flush=True)


def _to_iso_from_apple_time(raw: int | float | None) -> str | None:
    if raw is None:
        return None
    try:
        v = float(raw)
    except Exception:
        return None
    if v == 0:
        return None
    # Messages often stores nanoseconds since 2001-01-01, but some systems
    # expose seconds. Handle both.
    sec = v / 1_000_000_000.0 if abs(v) > 1_000_000_000_000 else v
    dt = _APPLE_EPOCH + timedelta(seconds=sec)
    return dt.isoformat()


def run_messages_ingest(
    *,
    index_db_path: Path,
    chat_db_path: Path,
    limit: int | None,
    force: bool,
    commit_every: int,
    progress_every: int,
) -> dict[str, int | float]:
    conn = connect(index_db_path)
    init_schema(conn)

    # Handle->name map (built from a Contacts CSV, see photo_index.contacts).
    # Lets new messages carry the contact NAME so name searches find them; empty
    # {} if no map has been built yet.
    from photo_index.contacts import load_map, resolve as _resolve_contact
    _contacts_map = load_map()

    if not chat_db_path.exists():
        raise FileNotFoundError(f"Messages DB not found: {chat_db_path}")

    try:
        chat_conn = sqlite3.connect(str(chat_db_path))
    except OSError as e:
        err = str(e).lower()
        if "operation not permitted" in err or "permission denied" in err:
            print(_MESSAGES_ACCESS_HELP, file=sys.stderr)
        raise

    chat_conn.row_factory = sqlite3.Row
    q = """
    SELECT
      m.ROWID AS rowid,
      m.guid AS guid,
      m.text AS text,
      m.attributedBody AS attributed_body,
      m.date AS date_raw,
      m.is_from_me AS is_from_me,
      h.id AS handle_id
    FROM message m
    LEFT JOIN handle h ON h.ROWID = m.handle_id
    WHERE m.text IS NOT NULL AND trim(m.text) != ''
       OR m.attributedBody IS NOT NULL
    ORDER BY m.ROWID
    """
    if limit is not None:
        q += " LIMIT ?"
        rows = chat_conn.execute(q, (limit,)).fetchall()
    else:
        rows = chat_conn.execute(q).fetchall()

    total = len(rows)
    _log(f"[messages] Found {total} text message(s) to consider.")
    t0 = time.perf_counter()

    indexed = skipped_dup = skipped_empty = errors = 0
    for i, r in enumerate(rows, start=1):
        if progress_every and i % progress_every == 1:
            _log(f"[messages progress] {i}/{total} ...")

        body = (r["text"] or "").strip()
        if not body:
            body = _extract_attributed_body(r["attributed_body"])
        if not body:
            skipped_empty += 1
            continue

        base_id = r["guid"] or f"rowid:{r['rowid']}"
        uuid = f"imsg:{base_id}"
        if not force and already_indexed(conn, uuid):
            skipped_dup += 1
            continue

        handle = (r["handle_id"] or "").strip() or "unknown"
        direction = "from_me" if int(r["is_from_me"] or 0) == 1 else "from_them"
        # Resolve the handle to a contact name so searches by a person's name
        # match their texts (the message text only holds the phone/email).
        name = _resolve_contact(handle, _contacts_map) if _contacts_map else None
        filename = f"message:{handle} | {name}" if name else f"message:{handle}"
        date_iso = _to_iso_from_apple_time(r["date_raw"])
        meta = f"source=messages direction={direction} handle={handle}"
        if name:
            meta += f" contact={name}"

        try:
            upsert_photo(
                conn,
                uuid=uuid,
                filename=filename,
                date_iso=date_iso,
                ocr_text=body,
                vlm_text=meta,
                image_path_used="",
                commit=False,
            )
            indexed += 1
            if commit_every <= 1 or indexed % commit_every == 0:
                commit_ingest(conn)
        except Exception as e:
            errors += 1
            _log(f"[messages warn] Failed indexing {uuid}: {e}")

    commit_ingest(conn)
    elapsed = time.perf_counter() - t0
    _log(
        f"[messages done] indexed={indexed} skipped_already={skipped_dup} "
        f"skipped_empty={skipped_empty} errors={errors} time={elapsed:.1f}s db={index_db_path}"
    )
    return {
        "indexed": indexed,
        "skipped_dup": skipped_dup,
        "skipped_empty": skipped_empty,
        "errors": errors,
        "elapsed": elapsed,
        "total": total,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Index Messages.app text into photo_index SQLite/FTS.")
    p.add_argument("--db", type=str, default=str(_DEFAULT_DB), help="Target SQLite index DB path.")
    p.add_argument("--chat-db", type=str, default=str(_DEFAULT_CHAT_DB), help="Messages chat.db path.")
    p.add_argument("--limit", type=int, default=None, help="Only process first N messages.")
    p.add_argument("--force", action="store_true", help="Re-index rows even if already indexed.")
    p.add_argument("--commit-every", type=int, default=200, help="Commit every N newly indexed messages.")
    p.add_argument("--progress-every", type=int, default=500, help="Log progress every N rows (0=off).")
    p.add_argument(
        "--no-global-ingest-lock",
        action="store_true",
        help="Disable shared content-ingest lock (not recommended).",
    )
    args = p.parse_args(argv)

    if args.commit_every < 1:
        p.error("--commit-every must be >= 1")

    index_db_path = Path(os.path.abspath(args.db))
    chat_db_path = Path(os.path.abspath(args.chat_db))

    if args.no_global_ingest_lock:
        run_messages_ingest(
            index_db_path=index_db_path,
            chat_db_path=chat_db_path,
            limit=args.limit,
            force=args.force,
            commit_every=args.commit_every,
            progress_every=args.progress_every,
        )
        return

    with global_ingest_lock() as have_lock:
        if not have_lock:
            _log("[lock] Another content ingest is already running; skipping this run.")
            return
        run_messages_ingest(
            index_db_path=index_db_path,
            chat_db_path=chat_db_path,
            limit=args.limit,
            force=args.force,
            commit_every=args.commit_every,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()
