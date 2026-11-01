#!/usr/bin/env python3
"""Clean up + re-chunk a personal contact-directory dump BY ENTRY.

Old address-book dumps ("a-z dirs.doc") are one giant run of
"NAME (notes)....PHONE" lines. Fixed-size chunking buries dozens of contacts per
passage, so the model can't extract a single person even when retrieval is
perfect. This splits the file into one small record per contact (name + notes +
phone + any detail lines), lightly cleans the dotted leaders, packs a few per
chunk, and re-ingests them — so "what info on <name>" pulls a clean, focused
record.

Entry boundary = a "contact line": has dotted leaders (....) or ends in a phone
number. Following lines without those are treated as that contact's details.

Usage:
  python scripts/rechunk_contact_directory.py \
      --doc "$HOME/Downloads/a-z dirs.doc" \
      --db data/photo_index.sqlite --file "a-z dirs"
  python -m photo_index.embed_index --db data/photo_index.sqlite
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import time
from pathlib import Path

from photo_index.documents_ingest import extract_auto

# One contact per chunk keeps each record's embedding purely about that person
# (packing several together dilutes it). A fixed preamble adds the words people
# actually type — "contact", "phone" — so keyword queries like "<name> phone
# number" match a record whose raw line only says "WK 1-818-...". The notes stay
# verbatim (we never guess at the cryptic abbreviations).
_PREAMBLE = "Personal contact directory entry (name, phone, notes):\n"
_MAX_CHARS = 1500  # only split a record if it's unusually long (rare)
_PHONE = re.compile(r"(?:WK|HM|FAX|CELL|H|W)?\s*1?[-\s(]?\d{3}[-)\s]\d{3}[-\s]\d{4}\s*$")


def _is_anchor(line: str) -> bool:
    return bool(re.search(r"\.{4,}", line) or _PHONE.search(line))


def _entries(text: str) -> list[str]:
    out: list[list[str]] = []
    cur: list[str] | None = None
    for raw in text.split("\n"):
        s = raw.rstrip()
        if not s.strip():
            continue
        if _is_anchor(s):
            if cur:
                out.append(cur)
            cur = [s]
        elif cur is not None:
            cur.append(s)
    if cur:
        out.append(cur)
    # clean each entry: collapse dotted leaders and runs of spaces
    cleaned = []
    for e in out:
        block = "\n".join(e)
        block = re.sub(r"\.{3,}", " — ", block)
        block = re.sub(r"[ \t]{3,}", "  ", block)
        cleaned.append(block.strip())
    return cleaned


def _chunks(entries: list[str]) -> list[str]:
    """One focused chunk per contact, each with the searchable preamble.

    A rare over-long record (many detail lines) is split, but the preamble and
    the contact's name lead every piece so continuations stay retrievable.
    """
    out: list[str] = []
    for e in entries:
        if len(e) + len(_PREAMBLE) <= _MAX_CHARS:
            out.append(_PREAMBLE + e)
            continue
        name = e.split("\n", 1)[0][:80]
        rest = e
        first = True
        while rest:
            cut = rest[:_MAX_CHARS - len(_PREAMBLE)]
            nl = cut.rfind("\n")
            if nl > _MAX_CHARS // 2:
                cut = rest[:nl]
            rest = rest[len(cut):].lstrip("\n")
            head = _PREAMBLE if first else f"{_PREAMBLE}{name} (continued):\n"
            out.append(head + cut.strip())
            first = False
    return out


def rechunk(doc: Path, db_path: Path, file_sub: str) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=120000")
    tmpl = conn.execute(
        "SELECT uuid, filename, vlm_text, date_iso, image_path_used "
        "FROM photo_meta WHERE filename LIKE ? ORDER BY uuid LIMIT 1",
        (f"%{file_sub}%",),
    ).fetchone()
    if not tmpl:
        print(f"No indexed rows for a file matching {file_sub!r}.")
        return
    base_uuid = tmpl[0].split("#")[0]
    rel = re.sub(r" \[part \d+/\d+\]$", "", tmpl[1])
    vlm, date_iso, img = tmpl[2], tmpl[3], tmpl[4]

    text, method, err = extract_auto(doc, doc.suffix.lower())
    if not text or not text.strip():
        print(f"Could not extract text from {doc} ({err or method}).")
        return
    entries = _entries(text)
    chunks = _chunks(entries)
    if not chunks:
        print("Parsed 0 entries — aborting (nothing written).")
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
    print(f"Cleaned + re-chunked into {len(entries):,} contact entries -> "
          f"{n:,} focused chunks (one contact each).")
    print(f"Now run:  python -m photo_index.embed_index --db {db_path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc", type=Path, required=True, help="the .doc/.txt file")
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--file", required=True, help="filename substring in the index")
    a = ap.parse_args(argv)
    rechunk(a.doc, a.db, a.file)


if __name__ == "__main__":
    main()
