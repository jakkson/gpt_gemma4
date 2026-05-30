"""OCR (Apple Vision) + VLM (Ollama) ingest for images and PDFs in a folder.

This is the visual-content counterpart to ``documents_ingest`` (which only
handles native text extraction via pypdf / textutil / etc.). For each image
file and each PDF under ``--root`` we:

1. Render PDF pages to JPEG (pypdfium2) — images are used as-is.
2. Run Apple Vision OCR on every page / image.
3. Run an Ollama vision model on the first (and optionally last) page or on
   the image itself for a 2–4 sentence caption.
4. Upsert one row into ``photo_meta`` keyed by ``doc:HASH(realpath)``.

The per-row ``vlm_text`` carries a ``[vlm_meta] mtime_ns=… size=…`` marker so
re-runs can skip files that haven't changed since the last successful pass.
This is the resumability story: Ctrl-C anytime, re-run, and it picks up
where it left off.

Why a separate command from ``documents_ingest``?

* It's much slower per-file (seconds, not milliseconds), so progress and
  checkpoint cadence look different.
* It needs Ollama to be running — text ingest doesn't.
* Different defaults: text ingest skips images by default; this one is
  *only* about images and PDFs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from ollama import chat

from photo_index.documents_ingest import (
    NOISE_DIR_NAMES,
    NOISE_DIR_SUFFIXES,
    is_noise_path,
)
from photo_index.ingest_lock import global_ingest_lock
from photo_index.keep_awake import start_keep_awake
from photo_index.ollama_image import image_path_for_ollama
from photo_index.pdf_render import render_pdf_pages
from photo_index.store import (
    commit_ingest,
    connect,
    init_schema,
    upsert_photo,
)
from photo_index.vision_ocr import vision_available, vision_ocr


def _log(msg: str) -> None:
    print(msg, flush=True)


_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"


def vlm_checkpoint_path(db_path: Path) -> Path:
    return db_path.with_name("documents_vlm_ingest.checkpoint.json")


_RASTER_IMAGE_EXT = frozenset(
    ".jpg .jpeg .png .gif .bmp .tif .tiff .heic .webp .ico".split()
)
_PDF_EXT = ".pdf"
_VLM_META_RE = re.compile(r"\[vlm_meta\]\s+mtime_ns=(\d+)\s+size=(\d+)")
_DOCMETA_RE = re.compile(r"\[docmeta\]\s+mtime_ns=(\d+)\s+size=(\d+)")
_VLM_PROMPT_DESCRIBE_ONLY = (
    "Describe this document/image briefly for a personal search index. "
    "Capture the document type (receipt, letter, screenshot, photo, scan, "
    "form, ID card, ticket, ad, etc.), any visible header / merchant / "
    "sender / date, and the main subject in 2-4 sentences. "
    "If text is the dominant content, summarize the topic rather than "
    "transcribing the text verbatim."
)

# Used when no separate OCR pass is producing the verbatim text — we ask
# the VLM to transcribe visible text in addition to describing. Slightly
# slower and lower quality than dedicated OCR, but covers the case where
# Apple Vision is unavailable on the host (e.g. macOS Tahoe IOSurface bug).
_VLM_PROMPT_DESCRIBE_AND_TRANSCRIBE = (
    "You are indexing a document/image for personal search.\n\n"
    "1) DESCRIBE the document in 2-3 sentences: type "
    "(receipt, letter, screenshot, photo, scan, form, ID card, ticket, ad, "
    "etc.), visible header / merchant / sender / date, and main subject.\n"
    "2) Then transcribe ALL visible text you can read, line by line, under a "
    "section heading 'TEXT:'. Preserve numbers, prices, dates, and names "
    "exactly. Skip pure decoration. If no readable text is present, write "
    "'TEXT: (none)'."
)


def doc_uuid(realpath_str: str) -> str:
    """Same scheme as documents_ingest.doc_uuid so rows align."""
    h = hashlib.sha256(realpath_str.encode("utf-8")).hexdigest()[:24]
    return f"doc:{h}"


def _existing_row(conn: sqlite3.Connection, uuid: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM photo_meta WHERE uuid = ?", (uuid,)
    ).fetchone()


def _vlm_meta_matches(vlm_text: str, mtime_ns: int, size: int) -> bool:
    if not vlm_text:
        return False
    m = _VLM_META_RE.search(vlm_text)
    if not m:
        return False
    return int(m.group(1)) == mtime_ns and int(m.group(2)) == size


def _existing_native_text(row: sqlite3.Row | None) -> tuple[str, str]:
    """Return (existing_ocr, existing_vlm_meta_block) from a prior text-only
    documents_ingest row, so we can preserve the pypdf text + docmeta header
    when augmenting with OCR + VLM."""
    if row is None:
        return "", ""
    ocr = row["ocr_text"] or ""
    vlm_old = row["vlm_text"] or ""
    # Preserve the original [docmeta] / extractor= header line(s) verbatim,
    # but drop any prior [vlm_meta] line (we'll write a fresh one).
    keep_lines: list[str] = []
    for line in vlm_old.splitlines():
        if line.startswith("[vlm_meta]"):
            continue
        keep_lines.append(line)
    return ocr, "\n".join(keep_lines).strip()


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 80] + "\n\n… [truncated for index size]\n"


def _chat_caption(vlm_model: str, image_path: str, *, ocr_in_pipeline: bool) -> str:
    """Run the VLM captioner. When OCR isn't part of the pipeline, ask the model
    to transcribe visible text too, so the resulting row still has searchable
    document content."""
    prompt = _VLM_PROMPT_DESCRIBE_ONLY if ocr_in_pipeline else _VLM_PROMPT_DESCRIBE_AND_TRANSCRIBE
    with image_path_for_ollama(image_path) as ollama_img:
        response = chat(
            model=vlm_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [ollama_img],
                }
            ],
        )
    return (response.message.content or "").strip()


def _process_image(
    p: Path,
    *,
    skip_vlm: bool,
    skip_ocr: bool,
    vlm_model: str,
) -> tuple[str, str, list[str]]:
    """Run OCR + VLM on a single image. Returns (ocr_text, vlm_text, errors)."""
    errors: list[str] = []
    ocr_text = ""
    if not skip_ocr:
        try:
            ocr_text = vision_ocr(p)
        except Exception as e:
            errors.append(f"OCR: {type(e).__name__}: {e}")

    vlm_text = ""
    if not skip_vlm:
        try:
            vlm_text = _chat_caption(
                vlm_model, str(p), ocr_in_pipeline=not skip_ocr
            )
        except Exception as e:
            errors.append(f"VLM: {type(e).__name__}: {e}")

    return ocr_text, vlm_text, errors


def _process_pdf(
    p: Path,
    *,
    skip_vlm: bool,
    skip_ocr: bool,
    vlm_model: str,
    pdf_render_dpi: int,
    pdf_max_vlm_pages: int,
    pdf_max_ocr_pages: int,
) -> tuple[str, str, list[str]]:
    """Render PDF pages, OCR each, VLM-caption first/last. Returns (ocr, vlm, errors)."""
    errors: list[str] = []
    ocr_chunks: list[str] = []
    vlm_chunks: list[str] = []

    # Render up to whichever cap is larger so we can serve both passes from
    # one rendered set. If both are skipped, we render nothing.
    render_cap: int | None
    if skip_ocr and skip_vlm:
        return "", "", []
    needed = []
    if not skip_ocr:
        needed.append(pdf_max_ocr_pages)
    if not skip_vlm:
        needed.append(pdf_max_vlm_pages)
    render_cap = max(needed) if needed else None
    # 0 means "all pages" in our CLI semantics.
    if render_cap is not None and render_cap <= 0:
        render_cap = None

    try:
        with render_pdf_pages(p, dpi=pdf_render_dpi, max_pages=render_cap) as pages:
            n_pages = len(pages)
            if not skip_ocr:
                ocr_pages = pages if pdf_max_ocr_pages <= 0 else pages[:pdf_max_ocr_pages]
                for i, page_img in enumerate(ocr_pages, start=1):
                    try:
                        text = vision_ocr(page_img)
                    except Exception as e:
                        errors.append(f"OCR page {i}: {type(e).__name__}: {e}")
                        text = ""
                    if text:
                        ocr_chunks.append(f"[Page {i}]\n{text}")

            if not skip_vlm and pages:
                # Always caption the first page; if pdf_max_vlm_pages >= 2
                # also caption the last rendered page (cheap insurance for
                # multi-page receipts whose total / signature lives at the end).
                indices = [0]
                if pdf_max_vlm_pages >= 2 and n_pages >= 2:
                    indices.append(n_pages - 1)
                for idx in indices:
                    page_img = pages[idx]
                    try:
                        cap = _chat_caption(
                            vlm_model, str(page_img), ocr_in_pipeline=not skip_ocr
                        )
                    except Exception as e:
                        errors.append(f"VLM page {idx + 1}: {type(e).__name__}: {e}")
                        cap = ""
                    if cap:
                        vlm_chunks.append(f"[Page {idx + 1}] {cap}")
    except Exception as e:
        errors.append(f"PDF render: {type(e).__name__}: {e}")

    return "\n\n".join(ocr_chunks), "\n\n".join(vlm_chunks), errors


def _format_combined_vlm_text(
    *,
    rel_path: str,
    mtime_ns: int,
    size: int,
    vlm_caption: str,
    preserved_meta: str,
    vlm_model: str,
    skip_vlm: bool,
    skip_ocr: bool,
) -> str:
    parts: list[str] = []
    if preserved_meta:
        parts.append(preserved_meta)
    # Fresh [vlm_meta] line — used by the resume check on next run.
    parts.append(
        f"[vlm_meta] mtime_ns={mtime_ns} size={size} "
        f"vlm_model={vlm_model if not skip_vlm else 'skip'} "
        f"ocr={'skip' if skip_ocr else 'apple_vision'} rel_path={rel_path}"
    )
    if vlm_caption:
        parts.append(f"[vlm_caption]\n{vlm_caption}")
    return "\n".join(parts)


def _format_combined_ocr_text(*, existing_ocr: str, new_ocr: str) -> str:
    """Merge prior pypdf-extracted text with our new Apple Vision OCR text."""
    pieces: list[str] = []
    if existing_ocr.strip():
        pieces.append(existing_ocr.strip())
    if new_ocr.strip():
        pieces.append("[vision_ocr]\n" + new_ocr.strip())
    return "\n\n".join(pieces)


def _native_text_already_substantial(row: sqlite3.Row | None, threshold: int) -> bool:
    """True when documents_ingest already extracted enough text via pypdf/etc.

    We use this to skip Apple Vision OCR on PDFs that already have a nice text
    layer (no point re-extracting). VLM caption is still produced for visual
    context.
    """
    if row is None:
        return False
    ocr = row["ocr_text"] or ""
    return len(ocr.strip()) >= threshold


def run_documents_vlm_ingest(
    *,
    root: Path,
    index_db_path: Path,
    limit: int | None,
    force: bool,
    vlm_model: str,
    skip_vlm: bool,
    skip_ocr: bool,
    skip_pdf: bool,
    skip_images: bool,
    min_image_bytes: int,
    pdf_render_dpi: int,
    pdf_max_vlm_pages: int,
    pdf_max_ocr_pages: int,
    pdf_native_text_threshold: int,
    progress_every: int,
    commit_every: int,
    checkpoint_every: int,
    max_chars_per_row: int,
) -> dict[str, int | float]:
    if not skip_ocr and not vision_available():
        raise RuntimeError(
            "Apple Vision OCR is not available; install pyobjc-framework-Vision "
            "and run on macOS, or pass --skip-ocr to run VLM only."
        )

    root = Path(os.path.abspath(root)).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    conn = connect(index_db_path)
    init_schema(conn)
    ck_path = vlm_checkpoint_path(index_db_path)

    walk = considered = 0
    indexed = skipped_unchanged = skipped_too_small = skipped_kind = 0
    skipped_noise = skipped_hidden = errors = 0
    last_rel_path = ""
    started_at_unix = time.time()
    t0 = time.perf_counter()

    def write_ck(finished: bool) -> None:
        ck_path.parent.mkdir(parents=True, exist_ok=True)
        elapsed = time.time() - started_at_unix
        rate = considered / elapsed if elapsed > 0 else 0.0
        payload = {
            "root": str(root),
            "db_path": str(index_db_path.resolve()),
            "vlm_model": vlm_model,
            "skip_vlm": skip_vlm,
            "skip_pdf": skip_pdf,
            "skip_images": skip_images,
            "min_image_bytes": min_image_bytes,
            "started_at_unix": started_at_unix,
            "updated_at_iso": datetime.now(timezone.utc).isoformat(),
            "ingest_finished": finished,
            "walk_files_seen": walk,
            "considered": considered,
            "indexed": indexed,
            "skipped_unchanged": skipped_unchanged,
            "skipped_too_small": skipped_too_small,
            "skipped_kind_filter": skipped_kind,
            "skipped_noise_dirs": skipped_noise,
            "skipped_hidden": skipped_hidden,
            "errors": errors,
            "last_rel_path": last_rel_path,
            "files_per_second": round(rate, 3),
            "elapsed_s": round(elapsed, 1),
        }
        ck_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _log(
        f"[vlm-docs] root={root} db={index_db_path} "
        f"vlm_model={vlm_model!r} skip_vlm={skip_vlm} skip_pdf={skip_pdf} "
        f"skip_images={skip_images} min_image_bytes={min_image_bytes} "
        f"noise_suffixes={NOISE_DIR_SUFFIXES}"
    )
    _log(f"[vlm-docs] checkpoint: {ck_path}")

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        walk += 1
        try:
            rp = p.resolve()
            rel_parts = rp.relative_to(root).parts
        except ValueError:
            continue
        if any(seg.startswith(".") for seg in rel_parts):
            skipped_hidden += 1
            continue
        if is_noise_path(rel_parts):
            skipped_noise += 1
            continue

        ext = p.suffix.lower()
        is_image = ext in _RASTER_IMAGE_EXT
        is_pdf = ext == _PDF_EXT
        if not (is_image or is_pdf):
            continue
        if is_image and skip_images:
            skipped_kind += 1
            continue
        if is_pdf and skip_pdf:
            skipped_kind += 1
            continue

        try:
            st = p.stat()
        except OSError:
            errors += 1
            continue

        if is_image and st.st_size < min_image_bytes:
            skipped_too_small += 1
            continue

        considered += 1
        rel_path = "/".join(rel_parts)
        last_rel_path = rel_path
        if limit is not None and considered > limit:
            break

        elapsed = time.perf_counter() - t0
        rate = considered / elapsed if elapsed > 0 else 0.0
        if progress_every and (considered - 1) % progress_every == 0:
            tail = rel_path if len(rel_path) <= 90 else "…" + rel_path[-87:]
            _log(
                f"[vlm-docs] {considered} | walk={walk} indexed={indexed} "
                f"unchanged={skipped_unchanged} too_small={skipped_too_small} "
                f"errors={errors} | {rate:.2f} files/s | {tail}"
            )

        real = os.path.realpath(str(p))
        uuid = doc_uuid(real)
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
        size_i = int(st.st_size)

        existing = _existing_row(conn, uuid)

        if not force and existing is not None and _vlm_meta_matches(
            existing["vlm_text"] or "", mtime_ns, size_i
        ):
            skipped_unchanged += 1
            continue

        # If a prior text-only ingest already extracted substantial native text
        # for a PDF, skip OCR but still produce a VLM caption. Also honor the
        # global --skip-ocr flag (set when Apple Vision is unavailable on
        # the host, e.g. macOS Tahoe IOSurface bug).
        eff_skip_ocr = skip_ocr
        if is_pdf and _native_text_already_substantial(
            existing, pdf_native_text_threshold
        ):
            eff_skip_ocr = True

        try:
            if is_image:
                ocr_new, vlm_caption, errs = _process_image(
                    p, skip_vlm=skip_vlm, skip_ocr=eff_skip_ocr, vlm_model=vlm_model
                )
            else:
                ocr_new, vlm_caption, errs = _process_pdf(
                    p,
                    skip_vlm=skip_vlm,
                    skip_ocr=eff_skip_ocr,
                    vlm_model=vlm_model,
                    pdf_render_dpi=pdf_render_dpi,
                    pdf_max_vlm_pages=pdf_max_vlm_pages,
                    pdf_max_ocr_pages=pdf_max_ocr_pages,
                )
        except Exception as e:
            errors += 1
            _log(f"[vlm-docs] ERROR processing {rel_path}: {type(e).__name__}: {e}")
            continue

        if errs:
            errors += len(errs)
            for e in errs:
                _log(f"[vlm-docs warn] {rel_path}: {e}")

        existing_ocr, preserved_meta = _existing_native_text(existing)
        combined_ocr = _format_combined_ocr_text(
            existing_ocr=existing_ocr, new_ocr=ocr_new
        )
        combined_ocr = _truncate(combined_ocr, max_chars_per_row)
        combined_vlm = _format_combined_vlm_text(
            rel_path=rel_path,
            mtime_ns=mtime_ns,
            size=size_i,
            vlm_caption=vlm_caption,
            preserved_meta=preserved_meta,
            vlm_model=vlm_model,
            skip_vlm=skip_vlm,
            skip_ocr=eff_skip_ocr,
        )

        date_iso = (
            datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
            if st.st_mtime > 0
            else None
        )

        try:
            upsert_photo(
                conn,
                uuid=uuid,
                filename=rel_path,
                date_iso=date_iso,
                ocr_text=combined_ocr,
                vlm_text=combined_vlm,
                image_path_used=str(p),
                commit=False,
            )
            indexed += 1
            if commit_every <= 1 or indexed % commit_every == 0:
                commit_ingest(conn)
        except Exception as e:
            errors += 1
            _log(f"[vlm-docs upsert warn] {rel_path}: {e}")

        if checkpoint_every > 0 and (indexed % checkpoint_every == 0 or considered % checkpoint_every == 0):
            write_ck(finished=False)

    commit_ingest(conn)
    write_ck(finished=True)
    elapsed = time.perf_counter() - t0
    _log(
        f"[vlm-docs done] considered={considered} indexed={indexed} "
        f"unchanged={skipped_unchanged} too_small={skipped_too_small} "
        f"kind_skipped={skipped_kind} noise={skipped_noise} "
        f"hidden={skipped_hidden} errors={errors} "
        f"elapsed={elapsed:.1f}s db={index_db_path}"
    )
    conn.close()
    return {
        "considered": considered,
        "indexed": indexed,
        "skipped_unchanged": skipped_unchanged,
        "skipped_too_small": skipped_too_small,
        "errors": errors,
        "elapsed": elapsed,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Apply Apple Vision OCR + Ollama VLM captions to images and PDFs "
            "under a folder, augmenting / creating rows in photo_meta."
        )
    )
    p.add_argument(
        "--root",
        type=str,
        default=str(Path.home() / "Dropbox" / "Documents"),
        help="Folder to walk recursively (default: ~/Dropbox/Documents).",
    )
    p.add_argument("--db", type=str, default=str(_DEFAULT_DB), help="Target SQLite DB path.")
    p.add_argument("--limit", type=int, default=None, help="Cap files considered (testing).")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-process even when [vlm_meta] mtime+size match the existing row.",
    )
    p.add_argument(
        "--vlm-model",
        default=os.environ.get("PHOTO_INDEX_DOC_VLM_MODEL", "gemma4:latest"),
        help="Ollama vision model for captions (default: gemma4:latest).",
    )
    p.add_argument(
        "--skip-vlm",
        action="store_true",
        help="OCR only — don't call the vision model. Much faster.",
    )
    p.add_argument(
        "--skip-ocr",
        action="store_true",
        help=(
            "Skip Apple Vision OCR entirely. Use when Vision is unavailable "
            "on the host (e.g. macOS Tahoe IOSurface allocation bug); the "
            "VLM prompt is automatically widened to also transcribe visible "
            "text."
        ),
    )
    p.add_argument(
        "--skip-images",
        action="store_true",
        help="Don't process raster image files (jpg/png/heic/etc.).",
    )
    p.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Don't process .pdf files.",
    )
    p.add_argument(
        "--min-image-bytes",
        type=int,
        default=10 * 1024,
        metavar="N",
        help=(
            "Skip image files smaller than N bytes (default: 10 KB). "
            "Filters out icons, sprites, and cached thumbnails."
        ),
    )
    p.add_argument(
        "--pdf-render-dpi",
        type=int,
        default=200,
        help="Render DPI for PDF pages before OCR (default: 200).",
    )
    p.add_argument(
        "--pdf-max-vlm-pages",
        type=int,
        default=1,
        metavar="N",
        help=(
            "How many pages per PDF to caption with the VLM. "
            "1 = first page only (fast, recommended). "
            "2 = first + last. <=0 = all (slow on long PDFs)."
        ),
    )
    p.add_argument(
        "--pdf-max-ocr-pages",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Max pages to OCR per PDF (default: 10). "
            "0 = no cap (all pages)."
        ),
    )
    p.add_argument(
        "--pdf-native-text-threshold",
        type=int,
        default=400,
        metavar="N",
        help=(
            "If a prior text-only ingest produced >=N characters of OCR text "
            "for this PDF, skip Apple Vision OCR and only run VLM. "
            "Default: 400."
        ),
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Log a progress line every N considered files (0=off).",
    )
    p.add_argument(
        "--commit-every",
        type=int,
        default=1,
        help="SQLite commit cadence (default: 1 — safest).",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Write checkpoint JSON every N indexed/considered files (0=off).",
    )
    p.add_argument(
        "--max-chars-per-row",
        type=int,
        default=400_000,
        help="Truncate combined OCR text beyond this character count.",
    )
    p.add_argument(
        "--no-keep-awake",
        action="store_true",
        help="Don't run caffeinate (long runs may sleep).",
    )
    p.add_argument(
        "--no-global-ingest-lock",
        action="store_true",
        help="Don't use shared content-ingest.lock (not recommended).",
    )
    args = p.parse_args(argv)

    if args.commit_every < 1:
        p.error("--commit-every must be >= 1")

    db_path = Path(os.path.abspath(args.db))
    root = Path(os.path.abspath(args.root))

    if not args.no_keep_awake:
        start_keep_awake(_log)

    def inner() -> None:
        run_documents_vlm_ingest(
            root=root,
            index_db_path=db_path,
            limit=args.limit,
            force=args.force,
            vlm_model=args.vlm_model,
            skip_vlm=args.skip_vlm,
            skip_ocr=args.skip_ocr,
            skip_pdf=args.skip_pdf,
            skip_images=args.skip_images,
            min_image_bytes=args.min_image_bytes,
            pdf_render_dpi=args.pdf_render_dpi,
            pdf_max_vlm_pages=args.pdf_max_vlm_pages,
            pdf_max_ocr_pages=args.pdf_max_ocr_pages,
            pdf_native_text_threshold=args.pdf_native_text_threshold,
            progress_every=args.progress_every,
            commit_every=args.commit_every,
            checkpoint_every=args.checkpoint_every,
            max_chars_per_row=args.max_chars_per_row,
        )

    if args.no_global_ingest_lock:
        inner()
        return

    with global_ingest_lock() as have_lock:
        if not have_lock:
            _log("[lock] Another content ingest is already running; skipping this run.")
            return
        inner()


if __name__ == "__main__":
    main()
