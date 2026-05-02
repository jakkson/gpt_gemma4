"""Delete already-indexed rows whose paths fall inside vendored / generated dirs.

Usage:
    python -m photo_index.prune_index --dry-run      # show what would go
    python -m photo_index.prune_index --yes          # actually delete
    python -m photo_index.prune_index --yes --vacuum # delete + reclaim disk space

The patterns mirror NOISE_DIR_NAMES / NOISE_DIR_SUFFIXES in
``photo_index.documents_ingest`` so the live ingest filter and this retroactive
cleanup stay in sync.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from photo_index.documents_ingest import NOISE_DIR_NAMES, NOISE_DIR_SUFFIXES
from photo_index.store import connect, init_schema

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"


def _build_match_clause() -> tuple[str, list[str]]:
    """Build a SQL WHERE fragment matching `filename` against any noise pattern.

    `filename` for documents-ingest rows is a relative path (e.g.
    ``Sad_Talker/env310/lib/python3.10/site-packages/foo/bar.py``), so segment
    matches are detected by surrounding the segment with ``/``.
    """
    clauses: list[str] = []
    params: list[str] = []
    for name in sorted(NOISE_DIR_NAMES):
        clauses.append(
            "(filename = ? OR filename LIKE ? OR filename LIKE ? OR filename LIKE ?)"
        )
        params.extend([
            name,
            f"{name}/%",
            f"%/{name}/%",
            f"%/{name}",
        ])
    for suf in NOISE_DIR_SUFFIXES:
        clauses.append("(filename LIKE ? OR filename LIKE ?)")
        params.extend([
            f"%{suf}/%",
            f"%{suf}",
        ])
    where = " OR ".join(clauses)
    return where, params


def find_noise_uuids(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    where, params = _build_match_clause()
    sql = f"SELECT uuid, filename FROM photo_meta WHERE {where}"
    return [(r["uuid"], r["filename"]) for r in conn.execute(sql, params)]


def delete_uuids(conn: sqlite3.Connection, uuids: list[str], chunk: int = 500) -> None:
    """Delete from photo_meta + photo_lex in batches to keep statements bounded."""
    for i in range(0, len(uuids), chunk):
        batch = uuids[i : i + chunk]
        placeholders = ",".join(["?"] * len(batch))
        conn.execute(f"DELETE FROM photo_meta WHERE uuid IN ({placeholders})", batch)
        conn.execute(f"DELETE FROM photo_lex  WHERE uuid IN ({placeholders})", batch)
    conn.commit()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=str, default=str(_DEFAULT_DB), help="SQLite DB path.")
    p.add_argument("--dry-run", action="store_true", help="Report counts only, no delete.")
    p.add_argument("--yes", action="store_true", help="Confirm deletion (required to delete).")
    p.add_argument(
        "--vacuum",
        action="store_true",
        help="Run VACUUM after delete to reclaim disk space (slow on large DBs).",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Print N example matches.",
    )
    args = p.parse_args(argv)

    db_path = Path(args.db)
    conn = connect(db_path)
    init_schema(conn)
    try:
        matches = find_noise_uuids(conn)
        total = len(matches)
        print(f"[prune] noise rows matched: {total}")
        for uuid, fn in matches[: max(0, args.sample)]:
            print(f"  - {uuid}  {fn}")
        if total > args.sample:
            print(f"  …and {total - args.sample} more")

        if total == 0 or args.dry_run:
            return 0
        if not args.yes:
            print("[prune] refusing to delete without --yes (use --dry-run for a preview).")
            return 1

        uuids = [u for (u, _) in matches]
        delete_uuids(conn, uuids)
        print(f"[prune] deleted {total} rows from photo_meta + photo_lex.")

        if args.vacuum:
            print("[prune] running VACUUM (this may take a while)…")
            conn.execute("VACUUM")
            print("[prune] VACUUM complete.")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
