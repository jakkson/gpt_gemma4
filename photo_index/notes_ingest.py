"""Ingest Apple Notes into the shared SQLite/FTS index (replaces Evernote).

Reads notes straight from the local Apple Notes store,
``~/Library/Group Containers/group.com.apple.notes/NoteStore.sqlite`` — a live
Core Data + CloudKit database. We COPY it read-only first (like history_ingest)
and never write to it. Each note body is stored as gzip-compressed protobuf in
``ZICNOTEDATA.ZDATA``; we gunzip it and walk the protobuf to pull the plain
text (no external deps, no `protobuf` package needed).

Each note becomes one photo_meta row (long notes are chunked, like books):
    uuid       = "notes:<ZIDENTIFIER>"          # stable note GUID -> idempotent
    filename   = "<created> | <title> | [<folder>]"
    date_iso   = best recency date (see resolution below)
    ocr_text   = "Title / Folder\\n\\n<plain-text body>"

Date resolution — two opposite Apple-Notes traps to dodge:
    * Importing a batch bumped the *modification* date of imported notes to
      import-time (falsely recent).
    * Native notes are often duplicated from an old template, which freezes the
      *creation* date years in the past even though the note is current.
    So date_iso comes from, in order:
      1. the note's original Evernote ``created`` (matched by title, tie-broken
         by closeness to the clean creation date) — authoritative for imported
         notes, read from the retained ``en_backup.db``. We use Evernote
         *created* (not *updated*): the migration bumped the updated date on
         BOTH sides, but created survived clean;
      2. ZMODIFICATIONDATE1 — clean for native notes (import never touched them)
         and correctly recent when a template is edited;
      3. ZCREATIONDATE3 (last resort, only if modification is missing).

Change detection: a note is (re)ingested when new, or when its modification
time is newer than the row's ``ingested_at`` (edited since last indexed).

Usage:
    python -m photo_index.notes_ingest
    python -m photo_index.notes_ingest --dry-run
    python -m photo_index.notes_ingest --force            # re-index everything
    python -m photo_index.notes_ingest --no-evernote-dates
"""
from __future__ import annotations

import argparse
import gzip
import io
import lzma
import pickle
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from photo_index.documents_ingest import _chunk_text
from photo_index.ingest_lock import global_ingest_lock
from photo_index.mail_ingest import _clean_text
from photo_index.store import commit_ingest, connect, init_schema, optimize, upsert_photo

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"
_DEFAULT_NOTE_STORE = (
    Path.home() / "Library" / "Group Containers" / "group.com.apple.notes"
    / "NoteStore.sqlite"
)
_DEFAULT_EVERNOTE_DB = (
    Path(__file__).resolve().parent.parent / "data" / "evernote" / "en_backup.db"
)

# Core Data / CFAbsoluteTime epoch is 2001-01-01; add this to reach Unix epoch.
_CF_EPOCH = 978_307_200

# Long notes are chunked into passages; this cap only guards pathological notes.
_BODY_MAX_CHARS = 200_000
_COMMIT_EVERY = 200
# U+FFFC OBJECT REPLACEMENT CHARACTER — Apple Notes uses it for every inline
# attachment/image; a note can be hundreds of them, so collapse to nothing.
_ATTACHMENT_MARK = "￼"


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Apple Notes protobuf body decode
# ---------------------------------------------------------------------------
def _read_varint(b: bytes, i: int) -> tuple[int, int]:
    shift = val = 0
    while True:
        byte = b[i]
        i += 1
        val |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return val, i
        shift += 7


def _walk(b: bytes) -> dict[int, list]:
    """One level of a protobuf message -> {field_no: [values]} (bytes for len-
    delimited, int for varint). Best-effort; stops on anything malformed."""
    out: dict[int, list] = {}
    i, n = 0, len(b)
    try:
        while i < n:
            key, i = _read_varint(b, i)
            field, wire = key >> 3, key & 7
            if wire == 0:
                v, i = _read_varint(b, i)
            elif wire == 2:
                ln, i = _read_varint(b, i)
                v, i = b[i:i + ln], i + ln
            elif wire == 5:
                v, i = b[i:i + 4], i + 4
            elif wire == 1:
                v, i = b[i:i + 8], i + 8
            else:
                break
            out.setdefault(field, []).append(v)
    except Exception:  # noqa: BLE001 — truncated/odd blob: keep what we got
        pass
    return out


def _note_text_from_blob(zdata: bytes) -> str:
    """gunzip + protobuf-walk an Apple Notes body to its plain text.

    Structure: top.field2 = document, document.field3 = note, note.field2 =
    UTF-8 note text. Falls back to the longest decodable string under `note`."""
    try:
        raw = gzip.decompress(zdata)
    except Exception:
        return ""  # password-protected/encrypted or not gzip — skip
    top = _walk(raw)
    doc = top.get(2)
    if not doc:
        return ""
    note = _walk(doc[0]).get(3)
    if not note:
        return ""
    fields = _walk(note[0])
    txt = fields.get(2)
    if txt and isinstance(txt[0], (bytes, bytearray)):
        return txt[0].decode("utf-8", "replace")
    # Fallback: pick the longest len-delimited field that decodes as text.
    best = ""
    for vals in fields.values():
        for v in vals:
            if isinstance(v, (bytes, bytearray)):
                try:
                    s = v.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if len(s) > len(best):
                    best = s
    return best


def _clean_note_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace(_ATTACHMENT_MARK, " ")
    return _clean_text(text)[:_BODY_MAX_CHARS]


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
def _cf_to_iso(cf: float | None) -> str | None:
    if not cf or cf <= 0:
        return None
    try:
        return datetime.fromtimestamp(cf + _CF_EPOCH, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _cf_to_unix(cf: float | None) -> float:
    return (cf + _CF_EPOCH) if (cf and cf > 0) else 0.0


def _iso_days(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp() / 86400.0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Evernote original-date oracle (authoritative created/updated by title)
# ---------------------------------------------------------------------------
class _StubNote:  # placeholder for pickled evernote.* thrift classes
    pass


class _NoteUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: D102
        if module.startswith("evernote."):
            return _StubNote
        return super().find_class(module, name)


def _load_en_note(blob: bytes):
    return _NoteUnpickler(io.BytesIO(lzma.decompress(blob))).load()


def _build_evernote_date_index(en_db: Path) -> dict[str, list[tuple[str, str]]]:
    """title(stripped) -> [(created_iso, updated_iso), ...] from en_backup.db."""
    index: dict[str, list[tuple[str, str]]] = {}
    try:
        conn = sqlite3.connect(f"file:{en_db}?mode=ro", uri=True)
    except Exception as e:  # noqa: BLE001
        _log(f"[notes] Evernote date oracle unavailable ({e}); using store dates.")
        return index

    def ms_iso(ms):
        if not ms:
            return None
        try:
            return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
        except Exception:
            return None

    n = 0
    for title, raw in conn.execute(
        "SELECT title, raw_note FROM notes WHERE raw_note IS NOT NULL AND is_active = 1"
    ):
        if not title:
            continue
        try:
            note = _load_en_note(raw)
        except Exception:
            continue
        created = ms_iso(getattr(note, "created", None))
        updated = ms_iso(getattr(note, "updated", None))
        if not created and not updated:
            continue
        index.setdefault(title.strip(), []).append((created or updated,
                                                     updated or created))
        n += 1
    conn.close()
    _log(f"[notes] Evernote date oracle: {n:,} notes across {len(index):,} titles.")
    return index


def _resolve_date_iso(title: str, created_iso: str | None, modified_iso: str | None,
                      en_index: dict[str, list[tuple[str, str]]]) -> str | None:
    """Best recency date for a note, immune to the two Apple-Notes traps:

    * IMPORTED notes had their *modification* date bumped to import-time, so for
      any note we can match to Evernote we use Evernote's authoritative
      ``updated`` (kills the false-recent bump).
    * NATIVE notes are often duplicated from an old template, which freezes the
      *creation* date years in the past even though the note is current — so for
      un-imported notes we use the store's ``modification`` date, which is clean
      (the import never touched native notes) and correctly goes recent when you
      edit the template.

    Priority: Evernote ``created`` (matched) → store modification → creation.

    Note we use Evernote ``created``, not ``updated``: the migration bumped the
    *updated* date on both sides (Apple modification AND Evernote updated landed
    on the import day), but ``created`` survived clean on both."""
    cands = en_index.get((title or "").strip())
    if cands:
        if len(cands) == 1 or not created_iso:
            return cands[0][0]  # Evernote created (clean)
        # Non-unique title: pick the Evernote note whose created is closest to
        # the store's clean creation date.
        target = _iso_days(created_iso)
        if target is not None:
            best = min(
                cands,
                key=lambda cu: abs((_iso_days(cu[0]) or 0.0) - target),
            )
            return best[0]
        return cands[0][0]
    return modified_iso or created_iso


# ---------------------------------------------------------------------------
# NoteStore access
# ---------------------------------------------------------------------------
def _copy_store(src: Path) -> Path | None:
    """Copy the (live/WAL) NoteStore + sidecars to temp; read the copy."""
    if not src.exists():
        return None
    tmp = Path(tempfile.mkdtemp(prefix="notescopy_"))
    try:
        shutil.copy2(src, tmp / src.name)
        for suffix in ("-wal", "-shm"):
            side = src.with_name(src.name + suffix)
            if side.exists():
                shutil.copy2(side, tmp / side.name)
        return tmp / src.name
    except Exception:
        return None


def _load_note_blobs(conn: sqlite3.Connection) -> dict[int, bytes]:
    """note Z_PK -> latest body blob (a note can have >1 ZICNOTEDATA row)."""
    blobs: dict[int, bytes] = {}
    for z_pk, znote, zdata in conn.execute(
        "SELECT Z_PK, ZNOTE, ZDATA FROM ZICNOTEDATA "
        "WHERE ZDATA IS NOT NULL ORDER BY Z_PK"
    ):
        if znote is not None:
            blobs[znote] = zdata  # ORDER BY Z_PK => last (latest) wins
    return blobs


def _folder_names(conn: sqlite3.Connection) -> dict[int, str]:
    out: dict[int, str] = {}
    for z_pk, name in conn.execute(
        "SELECT Z_PK, ZTITLE2 FROM ZICCLOUDSYNCINGOBJECT WHERE ZTITLE2 IS NOT NULL"
    ):
        out[z_pk] = name
    return out


def _trash_folder_pks(conn: sqlite3.Connection) -> set[int]:
    """Z_PKs of the 'Recently Deleted' folder(s). Apple leaves ZMARKEDFORDELETION
    = 0 on notes sitting in the trash, so the only reliable signal is the folder
    itself: ZFOLDERTYPE = 1 marks the system trash folder."""
    return {
        r[0] for r in conn.execute(
            "SELECT Z_PK FROM ZICCLOUDSYNCINGOBJECT WHERE ZFOLDERTYPE = 1"
        )
    }


def _existing_ingested_map(conn: sqlite3.Connection) -> dict[str, float]:
    return {
        r[0]: float(r[1])
        for r in conn.execute(
            "SELECT uuid, ingested_at FROM photo_meta WHERE uuid LIKE 'notes:%'"
        )
    }


def ingest(conn: sqlite3.Connection, note_store: Path, *, en_index: dict,
           force: bool = False, include_trashed: bool = False,
           dry_run: bool = False, progress_every: int = 500) -> tuple[int, int, int, int]:
    """Return (indexed_new, reindexed_edited, skipped, errors)."""
    copy = _copy_store(note_store)
    if not copy:
        _log(f"[notes] NoteStore not found or copy failed: {note_store}")
        _log("[notes] grant Full Disk Access to whatever runs this if permission-denied.")
        return 0, 0, 0, 0

    nconn = sqlite3.connect(str(copy))
    blobs = _load_note_blobs(nconn)
    folders = _folder_names(nconn)
    trash_folders = _trash_folder_pks(nconn)

    # One object row per note (those that have a body), keyed by Z_PK.
    obj: dict[int, sqlite3.Row] = {}
    nconn.row_factory = sqlite3.Row
    if blobs:
        id_list = ",".join(str(k) for k in blobs)   # note Z_PKs (integer literals)
        for r in nconn.execute(
            "SELECT Z_PK, ZIDENTIFIER, ZTITLE1, ZFOLDER, ZCREATIONDATE3, "
            "ZMODIFICATIONDATE1, IFNULL(ZMARKEDFORDELETION,0) AS trashed, "
            "IFNULL(ZISRECOVERINGFROMTRASH,0) AS recovering "
            f"FROM ZICCLOUDSYNCINGOBJECT WHERE Z_PK IN ({id_list})"
        ):
            obj[r["Z_PK"]] = r
    nconn.close()

    ingested_map = _existing_ingested_map(conn)
    new = edited = skipped = errors = 0
    batch = 0
    seen_ids: set[str] = set()
    seen = 0

    for z_pk, zdata in blobs.items():
        row = obj.get(z_pk)
        if row is None:
            continue
        seen += 1
        if not include_trashed and (row["trashed"] or row["recovering"]
                                    or row["ZFOLDER"] in trash_folders):
            skipped += 1
            continue

        ident = row["ZIDENTIFIER"]
        if not ident or ident in seen_ids:
            continue
        seen_ids.add(ident)
        uuid = f"notes:{ident}"

        created_iso = _cf_to_iso(row["ZCREATIONDATE3"])
        modified_iso = _cf_to_iso(row["ZMODIFICATIONDATE1"])
        updated_unix = _cf_to_unix(row["ZMODIFICATIONDATE1"])

        prev = ingested_map.get(uuid) or ingested_map.get(f"{uuid}#0")
        if prev is not None and not force:
            if updated_unix <= prev:
                skipped += 1
                continue
            is_edit = True
        else:
            is_edit = prev is not None

        title = (row["ZTITLE1"] or "(untitled)").strip()
        folder = folders.get(row["ZFOLDER"], "")
        body = _clean_note_text(_note_text_from_blob(zdata))
        if not body and title == "(untitled)":
            skipped += 1
            continue

        date_iso = _resolve_date_iso(title, created_iso, modified_iso, en_index)

        header_bits = [f"Title: {title}"]
        if folder:
            header_bits.append(f"Folder: {folder}")
        header = "\n".join(header_bits)

        filename_bits = [(date_iso or "unknown-date")[:10], title]
        if folder:
            filename_bits.append(f"[{folder}]")
        filename = " | ".join(filename_bits)

        if dry_run:
            new += 1 if not is_edit else 0
            edited += 1 if is_edit else 0
            if progress_every and seen % progress_every == 0:
                _log(f"  {seen:,} notes read | {new:,} new | {edited:,} edited")
            continue

        # Clean slate: a note may switch between single-row and chunked forms.
        conn.execute("DELETE FROM photo_meta WHERE uuid = ? OR uuid LIKE ?",
                     (uuid, f"{uuid}#%"))

        body_chunks = _chunk_text(body) if body else [""]
        n_chunks = len(body_chunks)
        for ci, chunk in enumerate(body_chunks):
            cu = uuid if n_chunks == 1 else f"{uuid}#{ci}"
            fn = filename if n_chunks == 1 else f"{filename} [part {ci + 1}/{n_chunks}]"
            upsert_photo(
                conn, uuid=cu, filename=fn, date_iso=date_iso,
                ocr_text=header + "\n\n" + chunk, vlm_text="",
                image_path_used="", open_url="", commit=False,
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
                 f"{skipped:,} skipped | {errors:,} err")

    if not dry_run:
        commit_ingest(conn)
    return new, edited, skipped, errors


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Ingest Apple Notes into the search index.")
    p.add_argument("--db", default=str(_DEFAULT_DB), help="Search index SQLite DB.")
    p.add_argument("--note-store", default=str(_DEFAULT_NOTE_STORE),
                   help="Apple Notes NoteStore.sqlite path.")
    p.add_argument("--evernote-db", default=str(_DEFAULT_EVERNOTE_DB),
                   help="en_backup.db used as the authoritative-date oracle.")
    p.add_argument("--no-evernote-dates", action="store_true",
                   help="Use only the Notes store's creation date (skip Evernote).")
    p.add_argument("--force", action="store_true", help="Re-index all notes.")
    p.add_argument("--include-trashed", action="store_true",
                   help="Also index trashed / recently-deleted notes.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report counts, write nothing.")
    p.add_argument("--progress-every", type=int, default=500)
    args = p.parse_args(argv)

    note_store = Path(args.note_store).expanduser()
    if not note_store.exists():
        _log(f"[notes] NoteStore not found: {note_store}")
        return

    en_index: dict = {}
    if not args.no_evernote_dates:
        en_db = Path(args.evernote_db)
        if en_db.exists():
            en_index = _build_evernote_date_index(en_db)
        else:
            _log(f"[notes] Evernote date oracle not found ({en_db}); using store dates.")

    conn = connect(Path(args.db))
    init_schema(conn)

    t0 = time.time()
    with global_ingest_lock():
        new, edited, skipped, errors = ingest(
            conn, note_store, en_index=en_index, force=args.force,
            include_trashed=args.include_trashed, dry_run=args.dry_run,
            progress_every=args.progress_every,
        )
    if (new or edited) and not args.dry_run:
        optimize(conn)
    dt = time.time() - t0
    tag = "[notes] DRY RUN — " if args.dry_run else "[notes] done in %.0fs — " % dt
    _log(f"{tag}{new:,} new | {edited:,} edited | {skipped:,} skipped | {errors:,} errors")
    if new and not args.dry_run:
        _log("Now embed:  PHOTO_INDEX_LLM_BACKEND=openai "
             "PHOTO_INDEX_LLM_BASE_URL=http://127.0.0.1:1234/v1 "
             f"python -m photo_index.embed_index --db {args.db}")


if __name__ == "__main__":
    main()
