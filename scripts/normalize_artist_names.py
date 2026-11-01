#!/usr/bin/env python3
"""Normalize "LASTNAME, The" reference-book inversions to natural order.

Alphabetized reference books (e.g. Billboard/Whitburn) list groups inverted for
sorting — "TEMPTATIONS, The", "ANIMALS, The" — so a natural-order search ("The
Temptations") matches poorly. This rewrites those to "The TEMPTATIONS" in the
indexed text so both orderings retrieve well.

SAFE by design: only ALL-CAPS artist headers followed by ", The/A/An" are
touched (that's the book's header format). Title-case commas in prose — e.g.
"...The Fighting Temptations, The Pink Panther..." (a movie list) — are left
alone. Changed rows have their embedding cleared so `embed_index` re-embeds them.

Usage:
  python scripts/normalize_artist_names.py --db data/photo_index.sqlite \
         --book "Billboard Book of Top 40"
  # then re-embed:
  python -m photo_index.embed_index --db data/photo_index.sqlite
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

# ALL-CAPS name (>=2 chars, may contain spaces/digits/&/.'-) then ", The|A|An".
_RX = re.compile(r"\b([A-Z][A-Z0-9][A-Z0-9 &.'-]{0,40}?), (The|A|An)\b")


def normalize(text: str) -> str:
    return _RX.sub(lambda m: f"{m.group(2)} {m.group(1)}", text or "")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--book", required=True,
                    help="substring of the book's filename to target")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args(argv)

    conn = sqlite3.connect(str(args.db))
    conn.execute("PRAGMA busy_timeout=120000")
    rows = conn.execute(
        "SELECT uuid, ocr_text FROM photo_meta WHERE filename LIKE ?",
        (f"%{args.book}%",),
    ).fetchall()
    if not rows:
        print(f"No rows match a book filename containing {args.book!r}.")
        return

    changed = subs = 0
    for uuid, text in rows:
        new = normalize(text)
        if new != text:
            subs += len(_RX.findall(text or ""))
            changed += 1
            if not args.dry_run:
                conn.execute(
                    "UPDATE photo_meta SET ocr_text = ?, embedding = NULL WHERE uuid = ?",
                    (new, uuid),
                )
    if not args.dry_run:
        conn.commit()
    conn.close()
    verb = "would change" if args.dry_run else "changed"
    print(f"{verb} {changed}/{len(rows)} rows (~{subs} names reordered).")
    if not args.dry_run and changed:
        print("Now run:  python -m photo_index.embed_index --db "
              f"{args.db}   (re-embeds the changed rows)")


if __name__ == "__main__":
    main()
