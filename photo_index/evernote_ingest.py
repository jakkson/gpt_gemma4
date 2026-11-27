"""Ingest Evernote notes into the shared SQLite/FTS index.

Reads notes directly from an ``evernote-backup`` database (``en_backup.db``),
which stores each note as ``lzma.compress(pickle.dumps(note))`` keyed by the
stable Evernote note GUID. That GUID makes re-runs idempotent, and the note's
``updated`` timestamp lets us re-ingest edited notes (not just brand-new ones).

Pipeline (all local, zero API credits at ingest time):
    evernote-backup sync    # incremental download of new/changed notes
    evernote-backup export  # (optional) portable .enex files
    python -m photo_index.evernote_ingest   # index into search DB

Each note becomes one photo_meta row:
    uuid       = "evernote:<note-guid>"
    filename   = "<created> | <title> | [<notebook>] <tags>"
    date_iso   = note created date (ISO)
    ocr_text   = "Title / Notebook / Tags\\n\\n<plain-text body>"

Change detection: a note is (re)ingested when it is new, or when its
``updated`` time is newer than the row's ``ingested_at`` (i.e. edited since we
last indexed it). Re-ingest nulls the embedding, so embed_index re-embeds it.

Usage:
    python -m photo_index.evernote_ingest
    python -m photo_index.evernote_ingest --backup-db /path/to/en_backup.db
    python -m photo_index.evernote_ingest --force   # re-index everything
"""
from __future__ import annotations

import argparse
import lzma
import pickle
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from photo_index.documents_ingest import _chunk_text
from photo_index.ingest_lock import global_ingest_lock
from photo_index.mail_ingest import _clean_text, _strip_html
from photo_index.store import commit_ingest, connect, init_schema, optimize, upsert_photo

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"
_DEFAULT_BACKUP_DB = (
    Path(__file__).resolve().parent.parent / "data" / "evernote" / "en_backup.db"
)

# Long notes are CHUNKED into passages (see below), so this cap only guards
# against pathological multi-megabyte notes. The old 12_000 cap silently cut
# note tails at ingest — embedding and prompts then never saw them.
_BODY_MAX_CHARS = 200_000
_COMMIT_EVERY = 200


def _log(msg: str) -> None:
    print(msg, flush=True)


class _StubNote:
    """Placeholder for pickled evernote SDK classes (Note, NoteAttributes, …).

    The backup DB stores notes as pickled thrift objects from the `evernote`
    package. We only read plain attributes (title/created/updated/tagNames/
    content), so instead of requiring that package to be importable (it is NOT
    installed in .venv — its click pin conflicts with gradio), unpickle every
    evernote.* class into this attribute bag. Without this, nightly ingest ran
    under .venv and errored on 100% of notes."""


class _NoteUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: D102
        if module.startswith("evernote."):
            return _StubNote
        return super().find_class(module, name)


def _load_note(blob: bytes):
    import io

    return _NoteUnpickler(io.BytesIO(lzma.decompress(blob))).load()


def _ms_to_iso(ms: int | None) -> str | None:
    """Evernote timestamps are epoch milliseconds."""
    if not ms:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _ms_to_unix(ms: int | None) -> float:
    if not ms:
        return 0.0
    try:
        return float(ms) / 1000.0
    except Exception:
        return 0.0


def _notebook_names(conn: sqlite3.Connection) -> dict[str, str]:
    """guid -> 'Stack / Name' (or just Name)."""
    out: dict[str, str] = {}
    for guid, name, stack in conn.execute("SELECT guid, name, stack FROM notebooks"):
        label = f"{stack} / {name}" if stack else (name or "")
        out[guid] = label
    return out


def _enml_to_text(content: str | None) -> str:
    """Strip ENML (Evernote's XHTML markup) down to plain text."""
    if not content:
        return ""
    # ENML is XHTML wrapped in <en-note>; the same HTML stripper handles it.
    text = _strip_html(content)
    return _clean_text(text)[:_BODY_MAX_CHARS]


def _existing_ingested_map(conn: sqlite3.Connection) -> dict[str, float]:
    """uuid -> ingested_at for all Evernote rows, loaded in one query."""
    return {
        r[0]: float(r[1])
        for r in conn.execute(
            "SELECT uuid, ingested_at FROM photo_meta WHERE uuid LIKE 'evernote:%'"
        )
    }


def ingest(
    conn: sqlite3.Connection,
    backup_db: Path,
    *,
    progress_every: int = 500,
    force: bool = False,
    include_inactive: bool = False,
) -> tuple[int, int, int, int]:
    """Return (indexed_new, reindexed_edited, skipped, errors)."""
    bconn = sqlite3.connect(f"file:{backup_db}?mode=ro", uri=True)
    bconn.row_factory = sqlite3.Row
    nb_names = _notebook_names(bconn)
    ingested_map = _existing_ingested_map(conn)

    new = edited = skipped = errors = 0
    batch = 0

    where = "raw_note IS NOT NULL" + ("" if include_inactive else " AND is_active = 1")
    cur = bconn.execute(
        f"SELECT guid, title, notebook_guid, raw_note FROM notes WHERE {where}"
    )

    seen = 0
    for row in cur:
        seen += 1
        try:
            note = _load_note(row["raw_note"])
        except Exception:
            errors += 1
            continue

        guid = row["guid"]
        uuid = f"evernote:{guid}"

        created_iso = _ms_to_iso(getattr(note, "created", None))
        updated_unix = _ms_to_unix(getattr(note, "updated", None))

        # A note is stored either as one row (uuid) or as chunk rows (uuid#0…);
        # check both forms so chunked notes aren't mistaken for new ones.
        prev_ingested = ingested_map.get(uuid)
        if prev_ingested is None:
            prev_ingested = ingested_map.get(f"{uuid}#0")
        if prev_ingested is not None and not force:
            # Skip unless the note was edited after we last indexed it.
            if updated_unix <= prev_ingested:
                skipped += 1
                continue
            is_edit = True
        else:
            is_edit = prev_ingested is not None  # force re-index of existing

        title = (getattr(note, "title", None) or "(untitled)").strip()
        notebook = nb_names.get(row["notebook_guid"], "")
        tag_names = getattr(note, "tagNames", None) or []
        tags = ", ".join(t for t in tag_names if t)

        body = _enml_to_text(getattr(note, "content", None))

        if not body and not title:
            skipped += 1
            continue

        header_bits = [f"Title: {title}"]
        if notebook:
            header_bits.append(f"Notebook: {notebook}")
        if tags:
            header_bits.append(f"Tags: {tags}")
        header = "\n".join(header_bits)

        filename_bits = [created_iso or "unknown-date", title]
        loc = f"[{notebook}]" if notebook else ""
        if loc or tags:
            filename_bits.append(f"{loc}{(' ' + tags) if tags else ''}".strip())
        filename = " | ".join(b for b in filename_bits if b)

        # Clean slate: a note may switch between single-row and chunked forms
        # across edits — remove any prior representation before writing.
        conn.execute(
            "DELETE FROM photo_meta WHERE uuid = ? OR uuid LIKE ?",
            (uuid, f"{uuid}#%"),
        )

        # Long notes become passage rows (like book chunking): embedding and
        # the answer prompt only ever see ~2K chars per row, so a 12K note
        # stored whole hides most of itself. Each chunk repeats the header so
        # every row stays attributable to its note title/notebook.
        body_chunks = _chunk_text(body)
        n_chunks = len(body_chunks)
        for ci, body_chunk in enumerate(body_chunks):
            cu = uuid if n_chunks == 1 else f"{uuid}#{ci}"
            fn = (
                filename
                if n_chunks == 1
                else f"{filename} [part {ci + 1}/{n_chunks}]"
            )
            upsert_photo(
                conn,
                uuid=cu,
                filename=fn,
                date_iso=created_iso,
                ocr_text=header + "\n\n" + body_chunk,
                vlm_text="",
                image_path_used="",  # no local file to open for a note
                open_url="",
                commit=False,
            )
        if is_edit:
            edited += 1
        else:
            new += 1
        batch += 1
        if batch >= _COMMIT_EVERY:
            commit_ingest(conn)
            batch = 0

        if progress_every and seen % progress_every == 0:
            _log(f"  {seen:,} notes read | {new:,} new | {edited:,} edited | "
                 f"{skipped:,} unchanged | {errors:,} err")

    commit_ingest(conn)
    bconn.close()
    return new, edited, skipped, errors


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Ingest Evernote notes (via evernote-backup DB).")
    p.add_argument("--db", default=str(_DEFAULT_DB), help="Search index SQLite DB.")
    p.add_argument("--backup-db", default=str(_DEFAULT_BACKUP_DB),
                   help="evernote-backup en_backup.db path.")
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--force", action="store_true", help="Re-index all notes.")
    p.add_argument("--include-inactive", action="store_true",
                   help="Also index trashed/inactive notes.")
    args = p.parse_args(argv)

    backup_db = Path(args.backup_db)
    if not backup_db.exists():
        _log(f"[evernote_ingest] backup DB not found: {backup_db}")
        _log("[evernote_ingest] run 'evernote-backup sync' first.")
        return

    conn = connect(Path(args.db))
    init_schema(conn)

    t0 = time.time()
    with global_ingest_lock():
        new, edited, skipped, errors = ingest(
            conn,
            backup_db,
            progress_every=args.progress_every,
            force=args.force,
            include_inactive=args.include_inactive,
        )
    if new or edited:
        optimize(conn)
    dt = time.time() - t0
    _log(f"[evernote_ingest] done in {dt:.0f}s — "
         f"{new:,} new | {edited:,} edited | {skipped:,} unchanged | {errors:,} errors")


if __name__ == "__main__":
    main()
