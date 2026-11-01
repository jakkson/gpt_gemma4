#!/usr/bin/env python3
"""Read-only file bridge for cross-machine "Open local file" hits.

The index is built on the mini (source of truth), so hit rows store the mini's
filesystem paths. When the Gradio search UI runs on another Mac (the MacBook),
those paths don't resolve there and "Open local file" 404s. This tiny always-on
service lets the remote UI stream an indexed file from the mini over Tailscale
and view/download it in the browser — no model, trivial RAM, so the mini stays
ingest-only.

Security (personal tailnet, but defended anyway):
  * per-install token (constant-time compare) embedded in UI links,
  * the path must be a file THIS index actually ingested (image_path_used),
  * the caller must be loopback or a Tailscale address (100.64.0.0/10),
  * read-only: it only serves bytes, never writes or executes.
"""
from __future__ import annotations

import argparse
import ipaddress
import os
import secrets
import sqlite3
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse

_DB: Path | None = None
_TOKEN = ""


def _load_or_create_token() -> str:
    tok_path = Path(__file__).resolve().parent.parent / "data" / ".file_bridge_token"
    try:
        tok = tok_path.read_text(encoding="utf-8").strip()
        if len(tok) >= 16:
            return tok
    except OSError:
        pass
    tok = secrets.token_urlsafe(16)
    try:
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tok_path.write_text(tok, encoding="utf-8")
        os.chmod(tok_path, 0o600)
    except OSError:
        pass
    return tok


def _path_is_indexed(path: str) -> bool:
    if _DB is None:
        return False
    try:
        conn = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT 1 FROM photo_meta WHERE image_path_used = ? LIMIT 1", (path,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except Exception:
        return False


def _caller_allowed(req: Request) -> bool:
    host = (req.client.host if req.client else "") or ""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.is_loopback:
        return True
    # Tailscale CGNAT range (100.64.0.0/10).
    return ip in ipaddress.ip_network("100.64.0.0/10")


app = FastAPI(title="photo-index file bridge")


@app.get("/healthz")
def healthz() -> PlainTextResponse:
    return PlainTextResponse("ok")


@app.get("/file")
def serve_file(request: Request, path: str = "", t: str = "") -> FileResponse:
    if not _caller_allowed(request):
        raise HTTPException(status_code=403, detail="caller not on loopback/tailnet")
    if not secrets.compare_digest(t or "", _TOKEN):
        raise HTTPException(status_code=403, detail="bad or missing token")
    if not path:
        raise HTTPException(status_code=400, detail="missing path")
    if not _path_is_indexed(path):
        raise HTTPException(status_code=403, detail="path is not an indexed file")
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="not a regular file")
    # inline so PDFs/images render in the browser tab; others download.
    return FileResponse(p, filename=p.name, content_disposition_type="inline")


def main(argv: list[str] | None = None) -> None:
    global _DB, _TOKEN
    ap = argparse.ArgumentParser(description="Read-only indexed-file bridge.")
    default_db = Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"
    ap.add_argument("--db", default=str(default_db))
    ap.add_argument("--host", default="0.0.0.0", help="bind host (default 0.0.0.0; gated by token+IP)")
    ap.add_argument("--port", type=int, default=7870)
    args = ap.parse_args(argv)
    _DB = Path(os.path.abspath(args.db))
    _TOKEN = _load_or_create_token()
    print(f"[file_bridge] db={_DB} host={args.host} port={args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
