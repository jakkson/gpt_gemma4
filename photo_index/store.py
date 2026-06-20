"""SQLite + FTS5 index for photo OCR + VLM text."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "latest",
    "me",
    "most",
    "my",
    "of",
    "on",
    "or",
    "recent",
    "show",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # Wait up to 2 minutes for our index DB if another process holds a write lock
    conn.execute("PRAGMA busy_timeout=120000")
    return conn


# photo_lex is an FTS5 *external-content* table: its rows are tokenized from
# photo_meta and referenced by rowid, so the source text is NOT duplicated into
# the FTS table (saves ~1.8 GB vs the old contentless 'doc' copy). The triggers
# below keep the index in sync on every insert/update/delete of photo_meta.
_FTS_CREATE_SQL = """
CREATE VIRTUAL TABLE photo_lex USING fts5(
    filename, date_iso, ocr_text, vlm_text,
    content='photo_meta',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
"""

_FTS_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS photo_meta_ai AFTER INSERT ON photo_meta BEGIN
    INSERT INTO photo_lex(rowid, filename, date_iso, ocr_text, vlm_text)
    VALUES (new.rowid, new.filename, new.date_iso, new.ocr_text, new.vlm_text);
END;
CREATE TRIGGER IF NOT EXISTS photo_meta_ad AFTER DELETE ON photo_meta BEGIN
    INSERT INTO photo_lex(photo_lex, rowid, filename, date_iso, ocr_text, vlm_text)
    VALUES ('delete', old.rowid, old.filename, old.date_iso, old.ocr_text, old.vlm_text);
END;
CREATE TRIGGER IF NOT EXISTS photo_meta_au AFTER UPDATE ON photo_meta BEGIN
    INSERT INTO photo_lex(photo_lex, rowid, filename, date_iso, ocr_text, vlm_text)
    VALUES ('delete', old.rowid, old.filename, old.date_iso, old.ocr_text, old.vlm_text);
    INSERT INTO photo_lex(rowid, filename, date_iso, ocr_text, vlm_text)
    VALUES (new.rowid, new.filename, new.date_iso, new.ocr_text, new.vlm_text);
END;
"""


def rebuild_fts(conn: sqlite3.Connection) -> None:
    """Repopulate the FTS index from photo_meta (the source of truth).

    Required after any operation that can renumber photo_meta rowids (e.g. VACUUM),
    since the external-content index references rows by rowid.
    """
    conn.execute("INSERT INTO photo_lex(photo_lex) VALUES('rebuild')")
    conn.commit()


def _migrate_fts_external(conn: sqlite3.Connection) -> None:
    """Replace a legacy (contentless 'doc') photo_lex with the external-content
    form. Safe to drop: photo_lex is fully derived from photo_meta."""
    conn.executescript(
        """
        DROP TRIGGER IF EXISTS photo_meta_ai;
        DROP TRIGGER IF EXISTS photo_meta_ad;
        DROP TRIGGER IF EXISTS photo_meta_au;
        DROP TABLE IF EXISTS photo_lex;
        """
    )
    conn.executescript(_FTS_CREATE_SQL)
    conn.executescript(_FTS_TRIGGERS_SQL)
    rebuild_fts(conn)


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS photo_meta (
            uuid TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            date_iso TEXT,
            ocr_text TEXT,
            vlm_text TEXT,
            image_path_used TEXT,
            ingested_at REAL NOT NULL
        );
        """
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(photo_meta)")}
    if "open_url" not in cols:
        conn.execute("ALTER TABLE photo_meta ADD COLUMN open_url TEXT NOT NULL DEFAULT ''")
    if "embedding" not in cols:
        # float32 vector as raw bytes; NULL until the embed backfill runs.
        conn.execute("ALTER TABLE photo_meta ADD COLUMN embedding BLOB")

    # FTS5: create fresh, or migrate a legacy contentless table to external content.
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='photo_lex'"
    ).fetchone()
    existing_sql = (row[0] if row else "") or ""
    if row is None:
        conn.executescript(_FTS_CREATE_SQL)
        conn.executescript(_FTS_TRIGGERS_SQL)
        rebuild_fts(conn)  # no-op when photo_meta is empty
    elif "content=" not in existing_sql:
        _migrate_fts_external(conn)
    else:
        # Already external-content; just make sure the sync triggers exist.
        conn.executescript(_FTS_TRIGGERS_SQL)
    conn.commit()


def fts_token_prefix_query(q: str) -> str:
    """Build an FTS5 AND query with prefix tokens (no phrase requirement)."""
    raw = re.findall(r"[\w'.-]+", q, re.UNICODE)
    parts: list[str] = []
    for r in raw:
        if len(r) < 1:
            continue
        # FTS5 treats "special" chars; keep alnum-heavy tokens
        safe = re.sub(r"[^\w.-]", "", r, flags=re.UNICODE)
        if not safe:
            continue
        low = safe.lower()
        # Drop very common words so natural language questions remain searchable.
        if low in _STOPWORDS:
            continue
        if len(low) < 3:
            continue
        parts.append(f"{safe}*")
    if not parts:
        return ""
    return " AND ".join(parts)


def already_indexed(conn: sqlite3.Connection, uuid: str) -> bool:
    row = conn.execute("SELECT 1 FROM photo_meta WHERE uuid = ?", (uuid,)).fetchone()
    return row is not None


def upsert_photo(
    conn: sqlite3.Connection,
    *,
    uuid: str,
    filename: str,
    date_iso: str | None,
    ocr_text: str,
    vlm_text: str,
    image_path_used: str,
    open_url: str = "",
    commit: bool = True,
) -> None:
    now = time.time()
    # FTS (photo_lex) is kept in sync automatically by triggers on photo_meta.
    conn.execute("DELETE FROM photo_meta WHERE uuid = ?", (uuid,))
    conn.execute(
        """INSERT INTO photo_meta
        (uuid, filename, date_iso, ocr_text, vlm_text, image_path_used, open_url, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (uuid, filename, date_iso, ocr_text, vlm_text, image_path_used, open_url or "", now),
    )
    if commit:
        conn.commit()


def delete_index_row(conn: sqlite3.Connection, uuid: str, *, commit: bool = True) -> None:
    """Remove one indexed row (used when Graph delta reports deletions).

    The photo_lex FTS row is removed automatically by the AFTER DELETE trigger.
    """
    conn.execute("DELETE FROM photo_meta WHERE uuid = ?", (uuid,))
    if commit:
        conn.commit()


def commit_ingest(conn: sqlite3.Connection) -> None:
    """Flush a batched ingest (call at checkpoints and end of run)."""
    conn.commit()


def search_meta(conn: sqlite3.Connection, fts_query: str, limit: int = 25) -> list[sqlite3.Row]:
    """BM25-ranked rows from FTS + meta join."""
    q = fts_token_prefix_query(fts_query)
    if not q:
        return []
    sql = """
    SELECT m.*, bm25(photo_lex) AS rank
    FROM photo_lex
    JOIN photo_meta m ON m.rowid = photo_lex.rowid
    WHERE photo_lex MATCH ?
    ORDER BY rank
    LIMIT ?
    """
    return list(conn.execute(sql, (q, limit)))


def search_meta_fallback_substring(
    conn: sqlite3.Connection, needle: str, limit: int = 25
) -> list[sqlite3.Row]:
    """
    If FTS MATCH fails or returns nothing, fall back to token LIKE.

    Natural language questions rarely exist as exact substrings in indexed rows.
    So we tokenize and OR-match meaningful terms (while preserving UUID/date order).
    """
    raw = re.findall(r"[\w'.-]+", needle or "", re.UNICODE)
    terms: list[str] = []
    for r in raw:
        safe = re.sub(r"[^\w.-]", "", r, flags=re.UNICODE).lower()
        if not safe or safe in _STOPWORDS or len(safe) < 3:
            continue
        if safe not in terms:
            terms.append(safe)
    if not terms:
        terms = [needle.strip()] if (needle or "").strip() else []
    if not terms:
        return []

    clauses: list[str] = []
    params: list[str | int] = []
    for t in terms:
        like = f"%{t}%"
        clauses.append("(lower(ocr_text) LIKE ? OR lower(vlm_text) LIKE ? OR lower(filename) LIKE ?)")
        params.extend([like, like, like])

    sql = f"""
    SELECT *, 0 AS rank
    FROM photo_meta
    WHERE {" OR ".join(clauses)}
    ORDER BY date_iso DESC, ingested_at DESC
    LIMIT ?
    """
    params.append(limit)
    return list(conn.execute(sql, params))


# Per-field character cap for prompt blocks. Document/PDF rows can store an
# entire file's OCR/VLM text; sending all of it for many rows can blow past the
# model context window. Long fields are truncated for the prompt only (the full
# text stays in the DB and FTS index).
_PROMPT_FIELD_CHAR_CAP = int(os.environ.get("PHOTO_INDEX_PROMPT_FIELD_CHARS", "2000"))
_PROMPT_LONG_TEXT_FIELDS = ("ocr_text", "vlm_text")


_PROMPT_SKIP_FIELDS = ("rank", "embedding", "score")


def row_to_prompt_block(row: sqlite3.Row, *, field_char_cap: int | None = None) -> str:
    cap = _PROMPT_FIELD_CHAR_CAP if field_char_cap is None else field_char_cap
    d = {k: row[k] for k in row.keys() if k not in _PROMPT_SKIP_FIELDS}
    if cap and cap > 0:
        for f in _PROMPT_LONG_TEXT_FIELDS:
            v = d.get(f)
            if isinstance(v, str) and len(v) > cap:
                d[f] = v[:cap] + f"\n…[truncated {len(v) - cap} chars for prompt]"
    return json.dumps(d, ensure_ascii=False, indent=2)


# --- Semantic search (embeddings) ---------------------------------------------

def row_embed_text(row: sqlite3.Row) -> str:
    """The text used to represent a row for embedding (filename + OCR + caption)."""
    parts = [
        str(row["filename"] or ""),
        str(row["ocr_text"] or ""),
        str(row["vlm_text"] or ""),
    ]
    return "\n".join(p for p in parts if p).strip()


def store_embedding(conn: sqlite3.Connection, uuid: str, vec: "list[float]", *, commit: bool = True) -> None:
    import numpy as np

    blob = np.asarray(vec, dtype=np.float32).tobytes()
    conn.execute("UPDATE photo_meta SET embedding = ? WHERE uuid = ?", (blob, uuid))
    if commit:
        conn.commit()


def count_rows_needing_embedding(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM photo_meta WHERE embedding IS NULL"
    ).fetchone()[0]


def iter_rows_needing_embedding(conn: sqlite3.Connection, batch: int = 256):
    """Yield batches of (uuid, embed_text) for rows lacking an embedding."""
    cur = conn.execute(
        "SELECT uuid, filename, ocr_text, vlm_text FROM photo_meta WHERE embedding IS NULL"
    )
    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break
        yield [(r["uuid"], row_embed_text(r)) for r in rows]


def load_embedding_matrix(conn: sqlite3.Connection):
    """Return (uuids: list[str], matrix: np.ndarray L2-normalized float32 [N, D]).

    Rows without an embedding are skipped. Returns ([], None) if none exist.
    """
    import numpy as np

    uuids: list[str] = []
    vecs: list[np.ndarray] = []
    for uuid, blob in conn.execute(
        "SELECT uuid, embedding FROM photo_meta WHERE embedding IS NOT NULL"
    ):
        if not blob:
            continue
        uuids.append(uuid)
        vecs.append(np.frombuffer(blob, dtype=np.float32))
    if not vecs:
        return [], None
    mat = np.vstack(vecs).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return uuids, mat


def count_embedded_rows(conn: sqlite3.Connection) -> int:
    """Cheap freshness signal for the embedding sidecar (rows with a vector)."""
    return conn.execute(
        "SELECT COUNT(*) FROM photo_meta WHERE embedding IS NOT NULL"
    ).fetchone()[0]


def _embedding_sidecar_paths(db_path: Path) -> tuple[Path, Path]:
    base = Path(db_path)
    return (
        base.with_name(base.name + ".emb_mat.npy"),
        base.with_name(base.name + ".emb_uuids.npy"),
    )


def invalidate_embedding_sidecar(db_path: Path) -> None:
    """Drop the on-disk embedding cache so it rebuilds on next load.

    Call after an embed backfill so searches pick up the new vectors immediately.
    """
    for p in _embedding_sidecar_paths(db_path):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def load_embedding_matrix_cached(conn: sqlite3.Connection, db_path: Path):
    """Like ``load_embedding_matrix`` but backed by an on-disk ``.npy`` sidecar.

    Scanning the embedding column out of the multi-GB ``photo_meta`` table costs
    several seconds; the normalized matrix is the same until the next ingest. We
    persist it next to the DB and rebuild only when the embedded-row count changes
    (or the sidecar is missing/corrupt). Returns ([], None) when no embeddings.
    """
    import numpy as np

    mat_path, uuid_path = _embedding_sidecar_paths(db_path)
    count = count_embedded_rows(conn)
    if count == 0:
        return [], None

    if mat_path.exists() and uuid_path.exists():
        try:
            uuids = list(np.load(uuid_path, allow_pickle=True))
            mat = np.load(mat_path)
            if mat.shape[0] == count == len(uuids):
                return uuids, mat
        except Exception:
            pass  # fall through to rebuild

    uuids, mat = load_embedding_matrix(conn)
    if mat is not None:
        try:
            np.save(mat_path, mat)
            np.save(uuid_path, np.asarray(uuids, dtype=object))
        except Exception:
            pass  # sidecar is an optimization; tolerate a read-only FS
    return uuids, mat
