#!/usr/bin/env python3
"""Self-maintenance "doctor" for the photo_index DB.

Reports health and runs SAFE maintenance so the index stays fast without manual
babysitting. Designed to run weekly (launchd) and leave a scannable log.

What it does (default):
  • REPORT: DB size, reclaimable free pages, FTS sync, embed backlog,
    oversize rows (data files kept single-row by policy), empty/awaiting-VLM
    rows, WAL size, quick integrity check.
  • MAINTAIN: PRAGMA optimize (query-planner stats) + FTS 'optimize'
    (merge search segments) + WAL checkpoint. All lightweight; no model needed.

VACUUM is NOT run automatically — it needs an exclusive lock and ~2x disk, and
is only worthwhile when a lot is reclaimable. The report flags when it's worth
doing; run with --vacuum to actually reclaim.

Lines that need attention are prefixed WARN so they're easy to grep in the log.

Usage:
  python scripts/doctor.py                # report + safe optimizes
  python scripts/doctor.py --report-only  # report only, no writes
  python scripts/doctor.py --vacuum       # also VACUUM (reclaim free pages)
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime
from pathlib import Path

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"
_VACUUM_SUGGEST_MB = 500      # suggest --vacuum when reclaimable exceeds this
_VACUUM_SUGGEST_PCT = 5.0


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def report(db: Path) -> dict:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    q = lambda s: c.execute(s).fetchone()[0]  # noqa: E731
    ps, pc, fl = q("PRAGMA page_size"), q("PRAGMA page_count"), q("PRAGMA freelist_count")
    reclaim_mb = ps * fl / 1e6
    pct = 100 * fl / pc if pc else 0
    meta = q("SELECT count(*) FROM photo_meta")
    lex = q("SELECT count(*) FROM photo_lex")
    need = q("SELECT count(*) FROM photo_meta WHERE embedding IS NULL")
    oversize = q("SELECT count(*) FROM photo_meta WHERE length(ocr_text)>12000")
    empty = q("SELECT count(*) FROM photo_meta WHERE ocr_text IS NULL OR ocr_text=''")
    await_vlm = q("SELECT count(*) FROM photo_meta WHERE (ocr_text IS NULL OR ocr_text='') "
                  "AND (vlm_text IS NULL OR vlm_text='')")
    c.close()
    wal = db.with_name(db.name + "-wal")
    wal_mb = wal.stat().st_size / 1e6 if wal.exists() else 0.0

    _log(f"DB size          : {ps*pc/1e9:.2f} GB ({pc:,} pages)")
    _log(f"Reclaimable      : {reclaim_mb:.0f} MB ({pct:.1f}% free pages)")
    _log(f"Rows (photo_meta): {meta:,}")
    _log(f"FTS (photo_lex)  : {lex:,}  {'in sync' if lex == meta else 'OUT OF SYNC'}")
    _log(f"Embed backlog    : {need:,} rows need embedding")
    _log(f"Oversize rows    : {oversize:,} (>12k chars — expected: data files kept single-row)")
    _log(f"Empty ocr_text   : {empty:,} ({empty-await_vlm:,} photos w/ captions, "
         f"{await_vlm:,} awaiting VLM)")
    _log(f"WAL file         : {wal_mb:.0f} MB")

    warns = []
    if lex != meta:
        warns.append("WARN: FTS out of sync with photo_meta — rebuild: "
                     "INSERT INTO photo_lex(photo_lex) VALUES('rebuild')")
    if need > 0:
        warns.append(f"WARN: {need:,} rows unembedded — run embed_index "
                     "(nightly normally clears this).")
    if reclaim_mb > _VACUUM_SUGGEST_MB or pct > _VACUUM_SUGGEST_PCT:
        warns.append(f"WARN: {reclaim_mb:.0f} MB reclaimable — worth a VACUUM "
                     "(run: python scripts/doctor.py --vacuum).")
    for w in warns:
        _log(w)
    if not warns:
        _log("OK: no attention items.")
    return {"reclaim_mb": reclaim_mb, "pct": pct, "in_sync": lex == meta,
            "need_embed": need, "warns": len(warns)}


def maintain(db: Path) -> None:
    c = sqlite3.connect(str(db))
    c.execute("PRAGMA busy_timeout=120000")
    t = time.time()
    c.execute("PRAGMA optimize")
    _log(f"maintain: PRAGMA optimize ({time.time()-t:.1f}s)")
    t = time.time()
    c.execute("INSERT INTO photo_lex(photo_lex) VALUES('optimize')")
    c.commit()
    _log(f"maintain: FTS segment-merge ({time.time()-t:.1f}s)")
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    c.close()
    _log("maintain: WAL checkpointed")


def vacuum(db: Path) -> None:
    c = sqlite3.connect(str(db))
    c.execute("PRAGMA busy_timeout=120000")
    t = time.time()
    _log("vacuum: running (exclusive lock; this can take a while)...")
    c.execute("VACUUM")
    c.close()
    _log(f"vacuum: done ({time.time()-t:.1f}s)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=_DEFAULT_DB)
    ap.add_argument("--report-only", action="store_true", help="no writes at all")
    ap.add_argument("--vacuum", action="store_true", help="also VACUUM (reclaim space)")
    a = ap.parse_args(argv)

    _log(f"===== doctor {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} : {a.db} =====")
    if not a.db.exists():
        _log(f"WARN: DB not found at {a.db}")
        return 1
    info = report(a.db)
    if not a.report_only:
        maintain(a.db)
        if a.vacuum:
            vacuum(a.db)
    _log("===== doctor done =====\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
