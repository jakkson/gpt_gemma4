#!/usr/bin/env python3
"""Re-chunk an alphabetized reference book BY ARTIST ENTRY.

Fixed-size chunking splits an artist's entry across passages, and only the
header passage carries the artist's name — so their hit list ends up in nameless
chunks and "list <artist>'s hits" can't retrieve them. This re-chunks the book
so each artist's *complete* entry stays together (name + all their songs), and
every continuation piece is prefixed with the artist name.

Tuned for Billboard/Whitburn "Top 40 Hits": artist headers are ALL-CAPS lines
(optionally "NAME, The"); the only other all-caps lines are the fixed field
labels (DATE / POSITION / WEEKS / LABEL & NO.) and record-label catalog lines.

Usage:
  python scripts/rechunk_reference_book.py \
      --epub "$HOME/LLM_Books/<the book>.epub" \
      --db data/photo_index.sqlite --book "Billboard Book of Top 40"
  python -m photo_index.embed_index --db data/photo_index.sqlite   # re-embed
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import time
from pathlib import Path

from photo_index.documents_ingest import extract_epub

_FIELD = {"DATE", "POSITION", "WEEKS", "LABEL & NO.", "LABEL & NO",
          "THE ARTISTS", "CONTENTS"}
_MAX_CHARS = 1800  # target size per piece


def _is_header(s: str) -> bool:
    if not s or s in _FIELD or len(s) > 55:
        return False
    if re.match(r"^[A-Z][A-Z&.\-]* \d{3,}$", s):   # record label + catalog no.
        return False
    # Group / single all-caps name (optionally "NAME, The"): e.g. CHILLIWACK,
    # "TEMPTATIONS, The", "EARTH, WIND & FIRE".
    core = re.sub(r",\s+(The|A|An)$", "", s).strip()
    letters = [c for c in core if c.isalpha()]
    if len(letters) >= 2 and all(c.isupper() for c in letters):
        return True
    # Solo artist "LASTNAME, Firstname" — caps last name, title-case first name.
    m = re.match(r"^([A-Z][A-Z0-9 &.'\-]+), [A-Z][a-z]", s)
    if m:
        ll = [c for c in m.group(1) if c.isalpha()]
        return bool(ll) and all(c.isupper() for c in ll)
    return False


def _natural(name: str) -> str:
    """Reorder inversions to natural order, but leave real comma-names alone:
    'TEMPTATIONS, The' -> 'The TEMPTATIONS', 'KEMP, Johnny' -> 'Johnny KEMP',
    but 'EARTH, WIND & FIRE' (all-caps after comma) stays put."""
    m = re.match(r"^(.*?),\s+(.+)$", name)
    if not m:
        return name
    first, second = m.group(1).strip(), m.group(2).strip()
    if second in ("The", "A", "An"):
        return f"{second} {first}"
    fl = [c for c in first if c.isalpha()]
    # solo artist: last name all-caps, first name Title-case (has lowercase)
    if fl and all(c.isupper() for c in fl) and re.match(r"^[A-Z][a-z]", second):
        return f"{second} {first}"
    return name


def _entries(text: str) -> list[tuple[str, str]]:
    """Split the book into (artist_name, entry_text) on artist headers."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[tuple[str, str]] = []
    name: str | None = None
    buf: list[str] = []
    for raw in lines:
        s = raw.strip()
        if _is_header(s):
            if name is not None and buf:
                out.append((name, "\n".join(buf).strip()))
            name = _natural(s)
            buf = [name]            # start the entry with the natural-order name
        elif name is not None:
            buf.append(raw)
    if name is not None and buf:
        out.append((name, "\n".join(buf).strip()))
    return out


def _chunks_for_entry(name: str, body: str) -> list[str]:
    """One chunk per short entry; long entries split with the name on each piece."""
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if len(body) <= _MAX_CHARS * 1.5:
        return [body]
    pieces, rest = [], body
    n = 0
    while rest:
        n += 1
        cut = rest[:_MAX_CHARS]
        # try to break on a newline near the limit
        nl = cut.rfind("\n")
        if nl > _MAX_CHARS // 2:
            cut = rest[:nl]
        rest = rest[len(cut):].lstrip("\n")
        prefix = "" if n == 1 else f"{name} (continued):\n"
        pieces.append(prefix + cut.strip())
    return pieces


def rechunk(epub: Path, db_path: Path, book: str) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=120000")
    tmpl = conn.execute(
        "SELECT uuid, filename, vlm_text, date_iso, image_path_used "
        "FROM photo_meta WHERE filename LIKE ? ORDER BY uuid LIMIT 1",
        (f"%{book}%",),
    ).fetchone()
    if not tmpl:
        print(f"No indexed rows for a book matching {book!r}.")
        return
    base_uuid = tmpl[0].split("#")[0]
    rel = re.sub(r" \[part \d+/\d+\]$", "", tmpl[1])
    vlm, date_iso, img = tmpl[2], tmpl[3], tmpl[4]

    text = extract_epub(epub)
    entries = _entries(text)
    chunks: list[str] = []
    for name, body in entries:
        chunks.extend(_chunks_for_entry(name, body))
    if not chunks:
        print("Parsed 0 artist entries — aborting (nothing written).")
        return

    conn.execute("DELETE FROM photo_meta WHERE uuid = ? OR uuid LIKE ?",
                 (base_uuid, base_uuid + "#%"))
    n = len(chunks)
    now = time.time()
    for i, ch in enumerate(chunks):
        conn.execute(
            """INSERT INTO photo_meta
               (uuid, filename, date_iso, ocr_text, vlm_text, image_path_used,
                open_url, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?, '', ?)""",
            (f"{base_uuid}#{i}", f"{rel} [part {i + 1}/{n}]", date_iso, ch,
             vlm, img, now),
        )
    conn.commit()
    conn.close()
    print(f"Re-chunked into {len(entries):,} artist entries -> {n:,} chunks "
          f"(was arbitrary passages).")
    print(f"Now run:  python -m photo_index.embed_index --db {db_path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epub", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--book", required=True, help="filename substring in the index")
    a = ap.parse_args(argv)
    rechunk(a.epub, a.db, a.book)


if __name__ == "__main__":
    main()
