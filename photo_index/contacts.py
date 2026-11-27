#!/usr/bin/env python3
"""Resolve Apple Messages handles (phone numbers / emails) to contact names.

Apple Messages stores each conversation by *handle* — a phone number or email —
never the person's name (that lives in Contacts). So a search for a person by
name matches none of their texts, because the messages only contain the number.
This
module builds a handle->name map from a Google Contacts CSV export and:

  * `migrate` — stamps the contact name into existing `imsg:` rows' filename so
    name searches (FTS) find them, WITHOUT re-embedding (a plain UPDATE keeps the
    embedding; the FTS triggers re-index automatically).
  * `resolve` — used by messages_ingest so NEW messages get the name too.

The map (data/contacts_map.json) contains personal phone/name data, so it lives
under data/ (gitignored) and is never committed.

Usage:
  python -m photo_index.contacts --csv ~/path/to/contacts.csv --build
  python -m photo_index.contacts --db data/photo_index.sqlite --migrate
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path

_DEFAULT_MAP = Path(__file__).resolve().parent.parent / "data" / "contacts_map.json"


def normalize_handle(raw: str) -> str | None:
    """Canonicalize a phone number or email so both sides of the join match.

    Phones -> E.164-ish (+1XXXXXXXXXX for US 10-digit); emails -> lowercased.
    Returns None for anything too short to be a real handle."""
    s = (raw or "").strip()
    if not s or s.lower() == "unknown":
        return None
    if "@" in s:
        return s.lower()
    digits = re.sub(r"\D", "", s)
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits[0] == "1":
        return "+" + digits
    if len(digits) >= 8:
        return "+" + digits
    return None


def build_map_from_csv(csv_path: Path) -> dict[str, str]:
    """Parse a Google Contacts CSV export into {normalized_handle: name}."""
    handle_to_name: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        phone_cols = [c for c in cols if re.match(r"Phone \d+ - Value", c)]
        email_cols = [c for c in cols if re.match(r"E-?mail \d+ - Value", c, re.I)]
        for row in reader:
            name = (row.get("Name") or "").strip()
            if not name:
                name = " ".join(
                    p for p in (
                        (row.get("Given Name") or "").strip(),
                        (row.get("Family Name") or "").strip(),
                    ) if p
                ).strip()
            if not name:
                continue
            for col in phone_cols + email_cols:
                for part in re.split(r":::|;|\n", row.get(col) or ""):
                    h = normalize_handle(part)
                    if h:
                        # First contact to claim a handle wins (stable).
                        handle_to_name.setdefault(h, name)
    return handle_to_name


def load_map(path: Path = _DEFAULT_MAP) -> dict[str, str]:
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def resolve(handle: str, cmap: dict[str, str]) -> str | None:
    """Look up a raw message handle against the map."""
    h = normalize_handle(handle)
    return cmap.get(h) if h else None


def _handle_from_filename(filename: str) -> str | None:
    m = re.match(r"message:(.+?)(?:\s+\|\s+.*)?$", filename or "")
    return m.group(1).strip() if m else None


def migrate_index(db_path: Path, cmap: dict[str, str]) -> tuple[int, int]:
    """Stamp contact names into existing imsg rows' filename (idempotent).

    Returns (handles_resolved, rows_updated). Embeddings are untouched; the FTS
    external-content triggers re-index each updated row automatically.

    Done as ONE indexed UPDATE...FROM: build a temp table of the resolvable
    (base_filename -> new_filename) pairs actually present in the index, index
    it, and join. The naive version (one UPDATE per contact with a LIKE scan)
    was ~374 full scans of 400K rows and never finished."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=120000")

    # Distinct handles present in the message index (one scan).
    present: dict[str, str] = {}  # exact current filename -> new filename
    for (fn,) in conn.execute(
        "SELECT DISTINCT filename FROM photo_meta WHERE uuid LIKE 'imsg:%'"
    ):
        raw = _handle_from_filename(fn)
        if not raw:
            continue
        name = resolve(raw, cmap)
        if not name:
            continue
        new_fn = f"message:{raw} | {name}"
        if fn != new_fn:  # skip already-stamped (idempotent)
            present[fn] = new_fn

    if not present:
        conn.close()
        return 0, 0

    conn.execute("DROP TABLE IF EXISTS _contact_rename")
    conn.execute("CREATE TEMP TABLE _contact_rename (old TEXT PRIMARY KEY, new TEXT)")
    conn.executemany(
        "INSERT OR IGNORE INTO _contact_rename(old, new) VALUES (?, ?)",
        list(present.items()),
    )
    cur = conn.execute(
        """UPDATE photo_meta
           SET filename = (SELECT new FROM _contact_rename WHERE old = photo_meta.filename)
           WHERE uuid LIKE 'imsg:%'
             AND filename IN (SELECT old FROM _contact_rename)"""
    )
    rows_updated = cur.rowcount
    conn.execute("DROP TABLE IF EXISTS _contact_rename")
    conn.commit()
    conn.close()
    return len(present), rows_updated


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Contacts handle->name resolution.")
    ap.add_argument("--csv", type=Path, help="Google Contacts CSV export")
    ap.add_argument("--map", type=Path, default=_DEFAULT_MAP)
    ap.add_argument("--db", type=Path)
    ap.add_argument("--build", action="store_true", help="build the map from --csv")
    ap.add_argument("--migrate", action="store_true", help="stamp names into imsg rows")
    args = ap.parse_args(argv)

    if args.build:
        if not args.csv:
            ap.error("--build requires --csv")
        cmap = build_map_from_csv(args.csv)
        json.dump(cmap, open(args.map, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[contacts] built {len(cmap):,} handles -> {args.map}")

    if args.migrate:
        if not args.db:
            ap.error("--migrate requires --db")
        cmap = load_map(args.map)
        if not cmap:
            ap.error(f"no map at {args.map}; run --build first")
        n_handles, n_rows = migrate_index(args.db, cmap)
        print(f"[contacts] migrated: {n_handles:,} handles matched, "
              f"{n_rows:,} message rows stamped with a name")


if __name__ == "__main__":
    main()
