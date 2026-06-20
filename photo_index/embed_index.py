"""Backfill semantic-search embeddings for indexed rows.

Embeds every row missing an ``embedding`` (filename + OCR + caption) via the
configured LLM backend's embeddings endpoint (LM Studio nomic by default) and
stores the float32 vector in ``photo_meta.embedding``.

Resumable: only rows with NULL embedding are processed, and each batch is
committed, so Ctrl-C and re-run picks up where it left off.

Usage:
    python -m photo_index.embed_index            # embed all missing
    python -m photo_index.embed_index --limit 500  # smoke test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .ingest_lock import global_ingest_lock
from .llm_client import embed_model, embed_texts
from .store import (
    connect,
    count_rows_needing_embedding,
    init_schema,
    invalidate_embedding_sidecar,
    iter_rows_needing_embedding,
    store_embedding,
)

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"


def run(db_path: Path, *, batch: int, limit: int | None) -> None:
    conn = connect(db_path)
    init_schema(conn)
    try:
        todo = count_rows_needing_embedding(conn)
        if limit:
            todo = min(todo, limit)
        print(f"[embed] model={embed_model()} rows_needing_embedding={todo} batch={batch}", flush=True)
        if todo == 0:
            print("[embed] nothing to do — all rows already embedded.", flush=True)
            return

        done = 0
        errors = 0
        seen = 0  # rows attempted (success or fail) — drives the --limit guard
        t0 = time.time()
        for chunk in iter_rows_needing_embedding(conn, batch=batch):
            if limit and seen >= limit:
                break
            if limit:
                chunk = chunk[: max(0, limit - seen)]
            seen += len(chunk)
            uuids = [u for u, _ in chunk]
            texts = [t for _, t in chunk]
            try:
                vecs = embed_texts(texts, is_query=False)
            except Exception as e:  # noqa: BLE001 — keep going on a bad batch
                errors += len(chunk)
                print(f"[embed] batch failed ({len(chunk)} rows): {e}", flush=True)
                continue
            for uuid, vec in zip(uuids, vecs):
                store_embedding(conn, uuid, vec, commit=False)
            conn.commit()
            done += len(chunk)
            rate = done / max(1e-6, time.time() - t0)
            remaining = max(0, todo - done)
            eta = remaining / rate if rate > 0 else 0
            print(
                f"[embed] {done}/{todo} ({rate:.1f}/s, ETA {eta/60:.1f} min, errors={errors})",
                flush=True,
            )
        print(
            f"[embed] done: embedded={done} errors={errors} "
            f"wall={time.time()-t0:.0f}s",
            flush=True,
        )
        if done:
            # Force the search side to rebuild its on-disk matrix from the new vectors.
            invalidate_embedding_sidecar(db_path)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Backfill semantic-search embeddings.")
    p.add_argument("--db", type=Path, default=_DEFAULT_DB)
    p.add_argument("--batch", type=int, default=64, help="rows per embeddings request")
    p.add_argument("--limit", type=int, default=None, help="stop after N rows (testing)")
    p.add_argument("--no-lock", action="store_true", help="skip the shared ingest lock")
    args = p.parse_args(argv)

    if args.no_lock:
        run(args.db, batch=args.batch, limit=args.limit)
        return
    with global_ingest_lock() as acquired:
        if not acquired:
            print("[lock] Another content ingest is running; skipping embed backfill.", flush=True)
            return
        run(args.db, batch=args.batch, limit=args.limit)


if __name__ == "__main__":
    main()
