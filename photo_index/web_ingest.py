#!/usr/bin/env python3
"""Scrape web pages you're allowed to read, review the text, then ingest it.

Two phases with a review gap so NOTHING reaches the DB until you've looked:

  1. fetch  — download + extract readable text, write one .md file per page into
              a staging folder (data/web_staging/<name>/). No DB writes.
  2.  ...you open the folder, read the files, DELETE the ones you don't want,
         and EDIT any you want to trim. commit ingests exactly what's left.
  3. commit — read the surviving staging files, chunk, insert `web:` rows, embed.

The heavy lifting (prose chunking, FTS, embeddings) is your existing pipeline:
fetch/extract is the only new stage, then rows go through `upsert_photo` and
`embed_index` like every other source.

Politeness (so "sites you're allowed to" stays true): robots.txt is honored by
default, requests are rate-limited, and a descriptive User-Agent is sent.

Examples:
  # a single page, peek at the text without staging anything
  python -m photo_index.web_ingest fetch --url https://example.com/post --dry-run

  # a whole blog via its sitemap, into a named staging run
  python -m photo_index.web_ingest fetch --sitemap https://blog.example.com/sitemap.xml --name blog
  #  ...review data/web_staging/blog/ , delete/edit files...
  python -m photo_index.web_ingest commit --name blog --embed

  # a bounded crawl from a starting page
  python -m photo_index.web_ingest fetch --crawl https://docs.example.com/ \
      --max-pages 40 --max-depth 2 --name docs
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests

from .documents_ingest import _chunk_text, _truncate
from .store import (commit_ingest, connect, init_schema,
                    invalidate_embedding_sidecar, upsert_photo)

try:  # embedding is optional at commit time (needs the LLM backend running)
    from .embed_index import run as _embed_run
except Exception:  # pragma: no cover
    _embed_run = None

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"
_STAGING_ROOT = Path(__file__).resolve().parent.parent / "data" / "web_staging"
_UA = ("gpt-local-gemma/web_ingest (personal offline RAG; contact: local user)")
_PAGE_CHAR_CAP = 200_000  # safety cap on a single page's extracted text
_HEADER_OPEN, _HEADER_CLOSE = "<!--web-ingest", "-->"


# --- fetching ---------------------------------------------------------------

def _http_get(url: str, *, timeout: int, ua: str) -> tuple[int, bytes, str, str]:
    r = requests.get(url, headers={"User-Agent": ua, "Accept": "text/html,*/*"},
                     timeout=timeout, allow_redirects=True)
    ctype = r.headers.get("Content-Type", "")
    return r.status_code, r.content, r.url, ctype


def _decode(content: bytes, ctype: str) -> str:
    m = re.search(r"charset=([\w\-]+)", ctype or "", re.I)
    if m:
        try:
            return content.decode(m.group(1), errors="replace")
        except Exception:
            pass
    for enc in ("utf-8", "latin-1"):
        try:
            return content.decode(enc)
        except Exception:
            continue
    return content.decode("utf-8", errors="replace")


def _render_html(url: str, *, timeout: int, ua: str) -> str | None:
    """Fetch a JS-rendered page via Playwright. Optional dependency."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("  [render] Playwright not installed. Install with:\n"
              "          pip install playwright && python -m playwright install chromium",
              file=sys.stderr)
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_context(user_agent=ua).new_page()
            page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
            html = page.content()
            browser.close()
            return html
    except Exception as e:  # noqa: BLE001
        print(f"  [render] failed: {e}", file=sys.stderr)
        return None


def _extract(html: str, url: str) -> tuple[str, str | None, str | None]:
    """Return (clean_text, title, date_iso). Prefer trafilatura; fall back to bs4."""
    try:
        import trafilatura
        text = trafilatura.extract(
            html, url=url, output_format="txt", favor_recall=True,
            include_comments=False, include_tables=True,
        ) or ""
        title = date = None
        try:
            meta = trafilatura.extract_metadata(html)
            title = getattr(meta, "title", None)
            date = getattr(meta, "date", None)
        except Exception:
            pass
        if text.strip():
            return text.strip(), title, date
    except Exception:
        pass
    # Fallback: strip boilerplate tags and take the visible text.
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "noscript"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else None
        main = soup.find("article") or soup.find("main") or soup.body or soup
        text = re.sub(r"\n{3,}", "\n\n", main.get_text("\n", strip=True))
        return text.strip(), title, None
    except Exception:
        return "", None, None


def _links(html: str, base_url: str) -> list[str]:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []
    out = []
    for a in soup.find_all("a", href=True):
        u, _ = urldefrag(urljoin(base_url, a["href"]))
        if u.startswith(("http://", "https://")):
            out.append(u)
    return out


# --- politeness -------------------------------------------------------------

class _Robots:
    """robots.txt gatekeeper, one parser cached per host."""

    def __init__(self, ua: str, ignore: bool):
        self.ua, self.ignore, self._cache = ua, ignore, {}

    def ok(self, url: str) -> bool:
        if self.ignore:
            return True
        p = urlparse(url)
        host = f"{p.scheme}://{p.netloc}"
        rp = self._cache.get(host)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(host + "/robots.txt")
            try:
                rp.read()
            except Exception:
                rp = None  # no robots reachable → allow
            self._cache[host] = rp
        return True if rp is None else rp.can_fetch(self.ua, url)


# --- sitemap / crawl target discovery --------------------------------------

def _sitemap_urls(url: str, *, timeout: int, ua: str, depth: int = 0) -> list[str]:
    if depth > 3:
        return []
    try:
        status, content, _, ctype = _http_get(url, timeout=timeout, ua=ua)
    except Exception as e:  # noqa: BLE001
        print(f"  [sitemap] {url}: {e}", file=sys.stderr)
        return []
    if url.endswith(".gz") or content[:2] == b"\x1f\x8b":
        try:
            content = gzip.decompress(content)
        except Exception:
            pass
    xml = _decode(content, ctype)
    locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml, re.I | re.S)
    if "<sitemapindex" in xml.lower():  # index of sitemaps → recurse
        out: list[str] = []
        for child in locs:
            out.extend(_sitemap_urls(child.strip(), timeout=timeout, ua=ua,
                                     depth=depth + 1))
        return out
    return [u.strip() for u in locs]


def _crawl_urls(start: str, *, max_pages: int, max_depth: int, robots: _Robots,
                delay: float, timeout: int, ua: str) -> list[str]:
    """Bounded, same-host BFS. Returns discovered page URLs (incl. the start)."""
    host = urlparse(start).netloc
    seen, ordered = {start}, [start]
    frontier = [(start, 0)]
    fetched_pages = 0
    while frontier and len(ordered) < max_pages * 4:
        url, d = frontier.pop(0)
        if d >= max_depth:
            continue
        if not robots.ok(url):
            continue
        try:
            status, content, final, ctype = _http_get(url, timeout=timeout, ua=ua)
            fetched_pages += 1
            time.sleep(delay)
        except Exception:
            continue
        if status != 200 or "html" not in ctype.lower():
            continue
        for link in _links(_decode(content, ctype), final):
            if urlparse(link).netloc != host or link in seen:
                continue
            seen.add(link)
            ordered.append(link)
            frontier.append((link, d + 1))
            if len(ordered) >= max_pages:
                return ordered[:max_pages]
    return ordered[:max_pages]


# --- staging ----------------------------------------------------------------

def _slug(title: str | None, url: str) -> str:
    base = title or urlparse(url).path.strip("/").replace("/", "-") or urlparse(url).netloc
    base = re.sub(r"[^\w\- ]", "", base).strip().replace(" ", "-").lower()
    return (base or "page")[:60]


def _stage_dir(name: str, root: Path) -> Path:
    return root / name


def _write_stage(dir_: Path, idx: int, url: str, title: str | None,
                 date_iso: str | None, text: str) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header = (
        f"{_HEADER_OPEN}\n"
        f"url: {url}\n"
        f"title: {title or ''}\n"
        f"date: {date_iso or ''}\n"
        f"fetched: {fetched}\n"
        f"chars: {len(text)}\n"
        f"{_HEADER_CLOSE}\n\n"
    )
    path = dir_ / f"{idx:04d}__{_slug(title, url)}.md"
    path.write_text(header + text, encoding="utf-8")
    return path


def _parse_stage(path: Path) -> tuple[dict, str] | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if _HEADER_OPEN not in raw:
        return None
    _, _, rest = raw.partition(_HEADER_OPEN)
    head, _, body = rest.partition(_HEADER_CLOSE)
    meta = {}
    for line in head.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, body.strip()


# --- commands ---------------------------------------------------------------

def _gather_targets(args) -> list[str]:
    urls: list[str] = []
    for u in args.url or []:
        urls.append(u)
    if args.urls_file:
        for line in Path(args.urls_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    if args.sitemap:
        found = _sitemap_urls(args.sitemap, timeout=args.timeout, ua=args.user_agent)
        print(f"[sitemap] {len(found)} URLs listed in {args.sitemap}")
        urls.extend(found)
    if args.crawl:
        robots = _Robots(args.user_agent, args.ignore_robots)
        found = _crawl_urls(args.crawl, max_pages=args.max_pages,
                            max_depth=args.max_depth, robots=robots,
                            delay=args.delay, timeout=args.timeout,
                            ua=args.user_agent)
        print(f"[crawl] {len(found)} URLs discovered from {args.crawl}")
        urls.extend(found)
    # de-dupe, keep order
    seen, out = set(), []
    for u in urls:
        u = urldefrag(u)[0]
        if u not in seen:
            seen.add(u)
            out.append(u)
    if args.limit:
        out = out[: args.limit]
    return out


def fetch_page(url: str, *, robots: _Robots, ua: str, timeout: int,
               render: bool, render_empty: bool, min_chars: int) -> dict:
    """Fetch + extract one page. Returns a dict; text is "" on skip/empty.

    status: HTTP code, or 'robots' / 'error'. Pure — writes nothing.
    """
    if not robots.ok(url):
        return {"url": url, "title": None, "date": None, "text": "",
                "status": "robots"}
    try:
        code, content, final, ctype = _http_get(url, timeout=timeout, ua=ua)
    except Exception as e:  # noqa: BLE001
        return {"url": url, "title": None, "date": None, "text": "",
                "status": f"error: {e}"}
    html = _decode(content, ctype)
    if render:
        r = _render_html(url, timeout=timeout, ua=ua)
        if r:
            html = r
    text, title, date = _extract(html, final)
    if (not text or len(text) < min_chars) and render_empty and not render:
        r = _render_html(url, timeout=timeout, ua=ua)
        if r:
            text, title, date = _extract(r, final)
    text = _truncate(text, _PAGE_CHAR_CAP)
    return {"url": final, "title": title, "date": date, "text": text,
            "status": code}


def stage_targets(targets: list[str], dir_: Path, *, robots: _Robots, ua: str,
                  timeout: int, delay: float, render: bool, render_empty: bool,
                  min_chars: int, dry_run: bool = False, log=print) -> list[dict]:
    """Fetch every target and (unless dry_run) write one .md per page to dir_.

    Returns a result dict per page: idx, url, title, date, chars, status,
    staged (Path|None), and text (populated on dry_run for previewing).
    Shared by the CLI and the GUI so both take the identical code path.
    """
    if not dry_run:
        dir_.mkdir(parents=True, exist_ok=True)
    manifest = None if dry_run else (dir_ / "_manifest.jsonl")
    results: list[dict] = []
    for i, url in enumerate(targets, 1):
        p = fetch_page(url, robots=robots, ua=ua, timeout=timeout,
                       render=render, render_empty=render_empty,
                       min_chars=min_chars)
        text = p["text"]
        row = {"idx": i, "url": p["url"], "title": p["title"], "date": p["date"],
               "chars": len(text), "status": p["status"], "staged": None,
               "text": text if dry_run else ""}
        shown = (p["title"] or p["url"])[:60]
        if isinstance(p["status"], str):  # robots / error
            log(f"{i:>4}  {'-':>7}  {p['status'][:6]:<6}  {shown}")
            results.append(row)
            continue
        note = "" if text else "  (empty — try Render)"
        log(f"{i:>4}  {len(text):>7}  {p['status']:>4}    {shown}{note}")
        if dry_run or not text:
            results.append(row)
            continue
        path = _write_stage(dir_, i, p["url"], p["title"], p["date"], text)
        row["staged"] = path
        with manifest.open("a", encoding="utf-8") as mf:
            mf.write(json.dumps({"idx": i, "url": p["url"], "title": p["title"],
                                 "date": p["date"], "chars": len(text)}) + "\n")
        results.append(row)
        time.sleep(delay)
    return results


def cmd_fetch(args) -> int:
    targets = _gather_targets(args)
    if not targets:
        print("No URLs to fetch. Give --url / --urls-file / --sitemap / --crawl.")
        return 1
    robots = _Robots(args.user_agent, args.ignore_robots)
    dir_ = _stage_dir(args.name, Path(args.staging_root))
    print(f"[fetch] {len(targets)} page(s) → "
          f"{'DRY RUN (stdout only)' if args.dry_run else dir_}")
    print(f"{'#':>4}  {'chars':>7}  status  title")
    results = stage_targets(
        targets, dir_, robots=robots, ua=args.user_agent, timeout=args.timeout,
        delay=args.delay, render=args.render, render_empty=args.render_empty,
        min_chars=args.min_chars, dry_run=args.dry_run)
    if args.dry_run:
        for r in results:
            if r["text"]:
                print("-" * 72)
                print(r["text"][:4000] +
                      ("\n… [truncated in dry-run preview]" if len(r["text"]) > 4000 else ""))
                print("-" * 72)
        return 0
    kept = sum(1 for r in results if r["staged"])
    print(f"\n[fetch] staged {kept} page(s) in {dir_}")
    print("Review them (delete/edit files you don't want), then:")
    print(f"  python -m photo_index.web_ingest commit --name {args.name} --embed")
    return 0


def rewrite_stage_body(path: Path, new_body: str) -> None:
    """Replace a staged page's text while keeping its header (for GUI edits)."""
    parsed = _parse_stage(path)
    if not parsed:
        return
    raw = path.read_text(encoding="utf-8", errors="replace")
    header, _, _ = raw.partition(_HEADER_CLOSE)
    path.write_text(f"{header}{_HEADER_CLOSE}\n\n{new_body.strip()}\n",
                    encoding="utf-8")


def commit_staging_dir(dir_: Path, db_path: Path, *, min_chars: int,
                       embed: bool, log=print) -> dict:
    """Ingest every .md still present in dir_ as web: rows. Returns counts.

    The set of files present IS the review decision — deleting a staging file
    (in Finder or the GUI) excludes that page. Re-commit is idempotent:
    DELETE web:<hash>#% before insert, so refreshing never duplicates.
    """
    files = sorted(p for p in dir_.glob("*.md"))
    if not files:
        log(f"No staged .md files in {dir_}.")
        return {"pages": 0, "rows": 0, "skipped": 0}
    conn = connect(db_path)
    init_schema(conn)
    pages = rows = skipped = 0
    for path in files:
        parsed = _parse_stage(path)
        if not parsed:
            log(f"  [skip] no header: {path.name}")
            continue
        meta, body = parsed
        url = meta.get("url", "").strip()
        if not url or len(body) < min_chars:
            skipped += 1
            continue
        title = meta.get("title") or url
        date_iso = meta.get("date") or datetime.now(timezone.utc).isoformat()
        base = "web:" + hashlib.sha1(url.encode()).hexdigest()[:16]
        conn.execute("DELETE FROM photo_meta WHERE uuid = ? OR uuid LIKE ?",
                     (base, base + "#%"))
        chunks = _chunk_text(body)
        n = len(chunks)
        vlm = f"web_ingest url={url}\nfetched={meta.get('fetched', '')}"
        for j, chunk in enumerate(chunks):
            cu = base if n == 1 else f"{base}#{j}"
            fn = title if n == 1 else f"{title} [part {j + 1}/{n}]"
            upsert_photo(conn, uuid=cu, filename=fn, date_iso=date_iso,
                         ocr_text=chunk, vlm_text=vlm, image_path_used=url,
                         open_url=url, commit=False)
            rows += 1
        pages += 1
    commit_ingest(conn)
    conn.close()
    log(f"[commit] ingested {pages} page(s) → {rows} rows "
        f"({skipped} skipped as too short). Source prefix: web:")
    if embed:
        if _embed_run is None:
            log("[commit] embed_index unavailable; run it manually.")
        else:
            log("[commit] embedding new rows…")
            _embed_run(db_path, batch=64, limit=None)
    return {"pages": pages, "rows": rows, "skipped": skipped}


def purge_domain(db_path: Path, needle: str, *, dry_run: bool = False,
                 log=print) -> dict:
    """Remove previously-ingested web: rows whose URL contains `needle`.

    Scoped to `uuid LIKE 'web:%'`, so it can only ever touch web-scraped pages —
    never mail/docs/photos. FTS entries drop via the AFTER DELETE trigger; the
    embedding sidecar is invalidated so the search matrix rebuilds. Fully
    reversible clean-up: re-fetch + commit puts the page back.
    """
    conn = connect(db_path)
    like = f"%{needle}%"
    where = "uuid LIKE 'web:%' AND (open_url LIKE ? OR image_path_used LIKE ?)"
    rows = conn.execute(f"SELECT count(*) FROM photo_meta WHERE {where}",
                        (like, like)).fetchone()[0]
    pages = conn.execute(
        f"SELECT count(DISTINCT open_url) FROM photo_meta WHERE {where}",
        (like, like)).fetchone()[0]
    if dry_run:
        conn.close()
        log(f"[purge] would remove {rows} row(s) across {pages} page(s) "
            f"matching {needle!r}.")
        return {"rows": rows, "pages": pages, "deleted": False}
    conn.execute(f"DELETE FROM photo_meta WHERE {where}", (like, like))
    conn.commit()
    conn.close()
    if rows:
        invalidate_embedding_sidecar(db_path)
    log(f"[purge] removed {rows} row(s) across {pages} page(s) "
        f"matching {needle!r}.")
    return {"rows": rows, "pages": pages, "deleted": True}


def cmd_purge(args) -> int:
    purge_domain(Path(args.db), args.domain, dry_run=args.dry_run)
    return 0


def cmd_commit(args) -> int:
    dir_ = _stage_dir(args.name, Path(args.staging_root))
    res = commit_staging_dir(dir_, Path(args.db), min_chars=args.min_chars,
                             embed=args.embed)
    if not args.embed and res["pages"]:
        print("Now embed the new rows:\n"
              "  PHOTO_INDEX_LLM_BACKEND=openai "
              "PHOTO_INDEX_LLM_BASE_URL=http://127.0.0.1:1234/v1 \\\n"
              f"    python -m photo_index.embed_index --db {args.db}")
    return 0


def _add_common(sp) -> None:
    sp.add_argument("--name", default="default", help="staging run name (folder)")
    sp.add_argument("--staging-root", default=str(_STAGING_ROOT))
    sp.add_argument("--db", default=str(_DEFAULT_DB))
    sp.add_argument("--user-agent", default=_UA)
    sp.add_argument("--timeout", type=int, default=30)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download + extract into the staging folder")
    _add_common(f)
    f.add_argument("--url", action="append", help="a page URL (repeatable)")
    f.add_argument("--urls-file", help="file with one URL per line (# = comment)")
    f.add_argument("--sitemap", help="a sitemap.xml URL (index files are followed)")
    f.add_argument("--crawl", help="start URL for a bounded same-host crawl")
    f.add_argument("--max-pages", type=int, default=25)
    f.add_argument("--max-depth", type=int, default=2)
    f.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    f.add_argument("--limit", type=int, default=None, help="cap total pages fetched")
    f.add_argument("--min-chars", type=int, default=200,
                   help="pages with less extracted text are treated as empty")
    f.add_argument("--render", action="store_true",
                   help="render with Playwright (JS pages); needs playwright installed")
    f.add_argument("--render-empty", action="store_true",
                   help="only render when static extraction comes back empty")
    f.add_argument("--ignore-robots", action="store_true",
                   help="skip robots.txt (use only for sites you're authorized on)")
    f.add_argument("--dry-run", action="store_true",
                   help="print extracted text to stdout; write nothing")
    f.set_defaults(func=cmd_fetch)

    c = sub.add_parser("commit", help="ingest the surviving staging files")
    _add_common(c)
    c.add_argument("--min-chars", type=int, default=200)
    c.add_argument("--embed", action="store_true",
                   help="run embed_index right after (needs the LLM backend up)")
    c.set_defaults(func=cmd_commit)

    pg = sub.add_parser("purge",
                        help="remove previously-ingested web pages by domain/URL")
    pg.add_argument("--db", default=str(_DEFAULT_DB))
    pg.add_argument("--domain", required=True,
                    help="domain or URL substring to remove, e.g. adcontrarian.blogspot.com")
    pg.add_argument("--dry-run", action="store_true",
                    help="show how many rows/pages would be removed, delete nothing")
    pg.set_defaults(func=cmd_purge)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
