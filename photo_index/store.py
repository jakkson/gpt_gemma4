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

        CREATE VIRTUAL TABLE IF NOT EXISTS photo_lex USING fts5(
            uuid UNINDEXED,
            doc,
            tokenize='porter unicode61'
        );
        """
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(photo_meta)")}
    if "open_url" not in cols:
        conn.execute("ALTER TABLE photo_meta ADD COLUMN open_url TEXT NOT NULL DEFAULT ''")
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
    conn.execute("DELETE FROM photo_meta WHERE uuid = ?", (uuid,))
    conn.execute("DELETE FROM photo_lex WHERE uuid = ?", (uuid,))
    conn.execute(
        """INSERT INTO photo_meta
        (uuid, filename, date_iso, ocr_text, vlm_text, image_path_used, open_url, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (uuid, filename, date_iso, ocr_text, vlm_text, image_path_used, open_url or "", now),
    )
    doc = "\n".join(
        [
            filename or "",
            date_iso or "",
            ocr_text or "",
            vlm_text or "",
        ]
    )
    conn.execute("INSERT INTO photo_lex (uuid, doc) VALUES (?, ?)", (uuid, doc))
    if commit:
        conn.commit()


def delete_index_row(conn: sqlite3.Connection, uuid: str, *, commit: bool = True) -> None:
    """Remove one indexed row (used when Graph delta reports deletions)."""
    conn.execute("DELETE FROM photo_meta WHERE uuid = ?", (uuid,))
    conn.execute("DELETE FROM photo_lex WHERE uuid = ?", (uuid,))
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
    JOIN photo_meta m ON m.uuid = photo_lex.uuid
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


def row_to_prompt_block(row: sqlite3.Row, *, field_char_cap: int | None = None) -> str:
    cap = _PROMPT_FIELD_CHAR_CAP if field_char_cap is None else field_char_cap
    d = {k: row[k] for k in row.keys() if k != "rank"}
    if cap and cap > 0:
        for f in _PROMPT_LONG_TEXT_FIELDS:
            v = d.get(f)
            if isinstance(v, str) and len(v) > cap:
                d[f] = v[:cap] + f"\n…[truncated {len(v) - cap} chars for prompt]"
    return json.dumps(d, ensure_ascii=False, indent=2)
