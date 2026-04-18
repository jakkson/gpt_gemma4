"""Build SQLite+FTS index from Photos (derivatives OK — no iCloud originals required)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"

import osxphotos
from ollama import chat

from photo_index.checkpoint import checkpoint_path_for_db, write_checkpoint
from photo_index.ollama_image import image_path_for_ollama
from photo_index.paths import PreferPath, resolve_local_image_path
from photo_index.store import already_indexed, commit_ingest, connect, init_schema, upsert_photo

_PHOTOS_ACCESS_HELP = """
macOS blocked read access to your Photos library.

Fix: System Settings → Privacy & Security → Full Disk Access → enable for Cursor or Terminal, then restart the app.
"""


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_ingest(
    *,
    db_path: Path,
    limit: int | None,
    force: bool,
    vlm_model: str,
    skip_vlm: bool,
    progress_every: int,
    prefer: PreferPath,
    commit_every: int,
    checkpoint_every: int,
) -> None:
    _log("[osxphotos] Opening Photos library…")
    try:
        photosdb = osxphotos.PhotosDB()
    except OSError as e:
        err = str(e).lower()
        if "operation not permitted" in err or "permission denied" in err:
            print(_PHOTOS_ACCESS_HELP.strip(), file=sys.stderr)
        raise

    conn = connect(db_path)
    init_schema(conn)

    photos = photosdb.photos(images=True, movies=False)
    if limit is not None:
        photos = photos[:limit]

    total = len(photos)
    ck_path = checkpoint_path_for_db(db_path)
    started_at = time.time()
    _log(
        f"[osxphotos] Indexing {total} image(s) (movies excluded). "
        f"prefer={prefer!r} VLM model={vlm_model!r} skip_vlm={skip_vlm} "
        f"commit_every={commit_every} checkpoint_every={checkpoint_every}"
    )

    ok = skip_no_path = skip_dup = errors = 0
    t0 = time.perf_counter()

    for i, photo in enumerate(photos, start=1):
        if progress_every and i % progress_every == 1:
            _log(f"[progress] {i}/{total} …")

        if not force and already_indexed(conn, photo.uuid):
            skip_dup += 1
            continue

        img_path = resolve_local_image_path(photo, prefer=prefer)
        if not img_path:
            skip_no_path += 1
            continue

        ocr_text = ""
        try:
            pairs = photo.detected_text()
            ocr_text = " ".join(str(t) for t, _conf in pairs if t)
        except Exception as e:
            _log(f"[warn] OCR failed for {photo.uuid}: {e}")

        vlm_text = ""
        if not skip_vlm:
            try:
                with image_path_for_ollama(img_path) as ollama_img:
                    response = chat(
                        model=vlm_model,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "Describe this image briefly for a personal search index. "
                                    "Focus on subjects, setting, text visible, and notable objects. "
                                    "2–4 sentences."
                                ),
                                "images": [ollama_img],
                            },
                        ],
                    )
                vlm_text = (response.message.content or "").strip()
            except Exception as e:
                _log(f"[warn] VLM failed for {photo.uuid}: {e}")
                errors += 1

        date_iso = None
        try:
            if photo.date:
                date_iso = photo.date.isoformat()
        except Exception:
            pass

        upsert_photo(
            conn,
            uuid=photo.uuid,
            filename=photo.filename or "",
            date_iso=date_iso,
            ocr_text=ocr_text,
            vlm_text=vlm_text,
            image_path_used=img_path,
            commit=False,
        )
        ok += 1

        flush_db = (
            commit_every <= 1
            or (ok % commit_every == 0)
            or (checkpoint_every > 0 and ok % checkpoint_every == 0)
        )
        if flush_db:
            commit_ingest(conn)

        if checkpoint_every > 0 and ok % checkpoint_every == 0:
            write_checkpoint(
                ck_path,
                db_path=db_path,
                prefer=prefer,
                total_candidates=total,
                processed_new_this_run=ok,
                last_uuid=photo.uuid,
                last_filename=photo.filename or "",
                started_at_unix=started_at,
            )
            _log(f"[checkpoint] wrote {ck_path} (indexed this run: {ok})")

    commit_ingest(conn)
    elapsed = time.perf_counter() - t0
    write_checkpoint(
        ck_path,
        db_path=db_path,
        prefer=prefer,
        total_candidates=total,
        processed_new_this_run=ok,
        last_uuid="",
        last_filename="",
        started_at_unix=started_at,
        elapsed_s=elapsed,
        finished=True,
    )
    _log(
        f"[done] indexed={ok} skipped_no_local_file={skip_no_path} "
        f"skipped_already={skip_dup} vlm_errors={errors} time={elapsed:.1f}s db={db_path}"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Index Photos library for Gemma search (uses previews when originals are absent).")
    p.add_argument(
        "--db",
        type=str,
        default=str(_DEFAULT_DB),
        help="SQLite database path (default: ./data/photo_index.sqlite)",
    )
    p.add_argument("--limit", type=int, default=None, help="Only process first N images (testing).")
    p.add_argument("--force", action="store_true", help="Re-index even if uuid already present.")
    p.add_argument(
        "--prefer",
        choices=("derivatives", "path"),
        default="derivatives",
        help="Which on-disk file to use: library previews first (default), or photo.path first for better quality when local.",
    )
    p.add_argument(
        "--commit-every",
        type=int,
        default=1,
        metavar="N",
        help="SQLite commit every N newly indexed photos (default: 1 = safest). Larger = faster, risk losing last batch on crash.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        metavar="N",
        help="Write checkpoint JSON beside the DB every N new indexes (0=off). Always commits at checkpoint.",
    )
    p.add_argument(
        "--vlm-model",
        default=os.environ.get("PHOTO_INDEX_VLM_MODEL", "gemma4:e2b"),
        help="Ollama vision model for captions (default: gemma4:e2b or PHOTO_INDEX_VLM_MODEL).",
    )
    p.add_argument("--skip-vlm", action="store_true", help="OCR only (Apple Vision); no Gemma captioning.")
    p.add_argument("--progress-every", type=int, default=50, help="Log progress every N images (0=off).")
    args = p.parse_args(argv)

    if args.commit_every < 1:
        p.error("--commit-every must be >= 1")

    db_path = Path(os.path.abspath(args.db))
    run_ingest(
        db_path=db_path,
        limit=args.limit,
        force=args.force,
        vlm_model=args.vlm_model,
        skip_vlm=args.skip_vlm,
        progress_every=args.progress_every,
        prefer=args.prefer,
        commit_every=args.commit_every,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
