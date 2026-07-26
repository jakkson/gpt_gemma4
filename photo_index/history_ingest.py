#!/usr/bin/env python3
"""Index Safari + Chrome browsing history for "where / what did I see?" recall.

Tier 1 (metadata) ingest: reads each browser's local SQLite history DB
(page title + URL + last-visit time + visit count), filters obvious noise,
dedupes the same URL across browsers, and writes one row per URL as source
prefix **hist:**. The page TITLE is the primary "what I saw" signal (searchable
+ embedded); the URL adds domain/path tokens; `open_url` makes every hit
clickable. Fast and complete, and it reflects what you actually saw — no live
re-scraping. To also capture page *content* for select pages, feed their URLs
into `web_ingest`.

Read-only on the browsers: it COPIES each DB first (Chrome locks its History
file while running) and never writes to them. All personal — lands only in your
local index, never the public repo.

Stable-field design: filename/ocr_text/vlm_text (the embedded text) hold only
title + URL, which don't change on a re-visit, so re-runs skip re-embedding;
the volatile last-visit goes in `date_iso` (not embedded) and is refreshed in
place. So this is safe to run repeatedly / on a schedule.

Usage:
  python -m photo_index.history_ingest --dry-run              # report scope only
  python -m photo_index.history_ingest --min-visits 2 --embed
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urlparse

from .store import commit_ingest, connect, init_schema, upsert_photo

try:
    from .embed_index import run as _embed_run
except Exception:  # pragma: no cover
    _embed_run = None

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"
_SAFARI = Path.home() / "Library" / "Safari" / "History.db"
_CHROME_DIR = Path.home() / "Library" / "Application Support" / "Google" / "Chrome"

# Mac CFAbsoluteTime epoch (2001-01-01) and Chrome/WebKit epoch (1601-01-01).
_CF_EPOCH = 978_307_200
_CHROME_EPOCH = 11_644_473_600

_SKIP_SCHEMES = ("chrome://", "chrome-extension://", "about:", "edge://",
                 "data:", "javascript:", "blob:", "view-source:", "file:")
_SKIP_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "[::1]"}
_NOISE_SUBSTR = ("google.com/url?", "l.facebook.com/", "lm.facebook.com/",
                 "//t.co/", "out.reddit.com/", "google.com/search",
                 "bing.com/search", "duckduckgo.com/?q", "/searchresults")


def _iso(unix: float) -> str:
    return datetime.fromtimestamp(unix, tz=timezone.utc).isoformat()


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().split(":")[0]


def _is_noise(url: str, title: str | None) -> bool:
    u = (url or "").lower()
    if not u or any(u.startswith(s) for s in _SKIP_SCHEMES):
        return True
    host = _domain(url)
    if not host or host in _SKIP_HOSTS:
        return True
    return any(s in u for s in _NOISE_SUBSTR)


def _norm(url: str) -> str:
    return urldefrag(url)[0]


def _copy_db(src: Path) -> Path | None:
    """Copy a (possibly locked/WAL) browser DB + sidecars to temp; read the copy."""
    if not src.exists():
        return None
    tmp = Path(tempfile.mkdtemp(prefix="histcopy_"))
    dst = tmp / src.name
    try:
        shutil.copy2(src, dst)
        for suffix in ("-wal", "-shm"):
            side = src.with_name(src.name + suffix)
            if side.exists():
                shutil.copy2(side, tmp / side.name)
        return dst
    except Exception:
        return None


def _read_safari(log) -> list[dict]:
    copy = _copy_db(_SAFARI)
    if not copy:
        log("  [safari] no History.db (or copy failed) — skipping.")
        return []
    try:
        c = sqlite3.connect(str(copy))
        c.row_factory = sqlite3.Row
        rows = c.execute(
            """SELECT hi.url AS url, hi.visit_count AS vc,
                      MAX(hv.visit_time) AS vt,
                      (SELECT hv2.title FROM history_visits hv2
                        WHERE hv2.history_item = hi.id AND hv2.title IS NOT NULL
                        ORDER BY hv2.visit_time DESC LIMIT 1) AS title
               FROM history_items hi
               JOIN history_visits hv ON hv.history_item = hi.id
               GROUP BY hi.id""").fetchall()
        c.close()
    except Exception as e:  # noqa: BLE001
        log(f"  [safari] read failed: {e} "
            "(grant Terminal Full Disk Access if permission-denied).")
        return []
    out = []
    for r in rows:
        if not r["url"]:
            continue
        out.append({"url": r["url"], "title": r["title"],
                    "unix": (r["vt"] or 0) + _CF_EPOCH,
                    "visits": r["vc"] or 1, "browser": "safari"})
    log(f"  [safari] {len(out)} history items.")
    return out


def _read_chrome(log) -> list[dict]:
    if not _CHROME_DIR.exists():
        log("  [chrome] no Chrome profile dir — skipping.")
        return []
    out: list[dict] = []
    profiles = [p for p in sorted(_CHROME_DIR.glob("*/History"))
                if p.parent.name not in ("System Profile",)]
    for hist in profiles:
        copy = _copy_db(hist)
        if not copy:
            continue
        try:
            c = sqlite3.connect(str(copy))
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT url, title, visit_count, last_visit_time FROM urls "
                "WHERE hidden = 0").fetchall()
            c.close()
        except Exception as e:  # noqa: BLE001
            log(f"  [chrome:{hist.parent.name}] read failed: {e}")
            continue
        n = 0
        for r in rows:
            if not r["url"]:
                continue
            out.append({"url": r["url"], "title": r["title"],
                        "unix": (r["last_visit_time"] or 0) / 1e6 - _CHROME_EPOCH,
                        "visits": r["visit_count"] or 1, "browser": "chrome"})
            n += 1
        log(f"  [chrome:{hist.parent.name}] {n} history items.")
    return out


def _merge(entries: list[dict], min_visits: int, log) -> tuple[dict, int]:
    """Filter noise + dedupe by normalized URL. Returns (by_url, noise_count)."""
    by_url: dict[str, dict] = {}
    noise = 0
    for e in entries:
        if _is_noise(e["url"], e["title"]):
            noise += 1
            continue
        key = _norm(e["url"])
        cur = by_url.get(key)
        if cur is None:
            by_url[key] = {**e, "url": key, "browsers": {e["browser"]}}
        else:
            cur["visits"] += e["visits"]
            cur["unix"] = max(cur["unix"], e["unix"])
            cur["browsers"].add(e["browser"])
            if (e["title"] or "") and len(e["title"] or "") > len(cur["title"] or ""):
                cur["title"] = e["title"]
    if min_visits > 1:
        before = len(by_url)
        by_url = {k: v for k, v in by_url.items() if v["visits"] >= min_visits}
        log(f"  dropped {before - len(by_url)} URLs below --min-visits={min_visits}.")
    return by_url, noise


def ingest(db_path: Path, *, safari: bool, chrome: bool, min_visits: int,
           dry_run: bool, embed: bool, limit: int | None = None, log=print) -> dict:
    entries: list[dict] = []
    if safari:
        entries += _read_safari(log)
    if chrome:
        entries += _read_chrome(log)
    by_url, noise = _merge(entries, min_visits, log)
    items = list(by_url.values())
    if limit:
        items = items[:limit]

    if items:
        dates = [e["unix"] for e in items if e["unix"] > 0]
        span = f"{_iso(min(dates))[:10]} → {_iso(max(dates))[:10]}" if dates else "?"
        log(f"[history] {len(items)} unique URLs to index "
            f"(noise filtered: {noise}; date range {span}).")
    if dry_run:
        log("[history] DRY RUN — nothing written.")
        return {"urls": len(items), "noise": noise, "written": 0}

    conn = connect(db_path)
    init_schema(conn)
    written = refreshed = 0
    for e in items:
        url = e["url"]
        title = (e["title"] or _domain(url) or url).strip()
        ocr = title
        vlm = url  # stable: gives domain/path tokens without churning embeddings
        iso = _iso(e["unix"]) if e["unix"] > 0 else None
        uuid = "hist:" + hashlib.sha1(url.encode()).hexdigest()[:16]
        old = conn.execute(
            "SELECT filename, ocr_text, vlm_text FROM photo_meta WHERE uuid = ?",
            (uuid,)).fetchone()
        if old and old["filename"] == title and old["ocr_text"] == ocr \
                and old["vlm_text"] == vlm:
            # Same page seen again: refresh recency only, keep the embedding.
            conn.execute("UPDATE photo_meta SET date_iso = ? WHERE uuid = ?",
                         (iso, uuid))
            refreshed += 1
            continue
        upsert_photo(conn, uuid=uuid, filename=title, date_iso=iso, ocr_text=ocr,
                     vlm_text=vlm, image_path_used=url, open_url=url, commit=False)
        written += 1
    commit_ingest(conn)
    conn.close()
    log(f"[history] wrote/updated {written} new-or-changed rows, "
        f"refreshed {refreshed} unchanged. Source prefix: hist:")
    if embed and written:
        if _embed_run is None:
            log("[history] embed_index unavailable; run it manually.")
        else:
            log("[history] embedding new rows…")
            _embed_run(db_path, batch=64, limit=None)
    elif written:
        log("Now embed:  PHOTO_INDEX_LLM_BACKEND=openai "
            "PHOTO_INDEX_LLM_BASE_URL=http://127.0.0.1:1234/v1 "
            f"python -m photo_index.embed_index --db {db_path}")
    return {"urls": len(items), "noise": noise, "written": written,
            "refreshed": refreshed}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=_DEFAULT_DB)
    ap.add_argument("--no-safari", action="store_true", help="skip Safari history")
    ap.add_argument("--no-chrome", action="store_true", help="skip Chrome history")
    ap.add_argument("--min-visits", type=int, default=1,
                    help="only index URLs visited at least N times (cuts noise)")
    ap.add_argument("--limit", type=int, default=None, help="cap rows (testing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report scope (counts, date range) and write nothing")
    ap.add_argument("--embed", action="store_true",
                    help="run embed_index after (needs the LLM backend up)")
    a = ap.parse_args(argv)
    ingest(a.db, safari=not a.no_safari, chrome=not a.no_chrome,
           min_visits=a.min_visits, dry_run=a.dry_run, embed=a.embed, limit=a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
