"""Render PDF pages to JPEG via pypdfium2 for OCR / VLM pipelines."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def render_pdf_pages(
    pdf_path: str | Path,
    *,
    dpi: int = 200,
    max_pages: int | None = None,
) -> Iterator[list[Path]]:
    """Yield a list of JPEG paths, one per rendered PDF page.

    Pages are written to a private temp dir that's wiped when the context
    manager exits, so callers don't need to clean up. Returns an empty list
    if the PDF has no pages or fails to open.

    Parameters
    ----------
    pdf_path : path to the source PDF.
    dpi      : render resolution (200 is a good OCR target — readable text,
               manageable file size).
    max_pages: cap the number of rendered pages (saves work on long PDFs
               when only the first few pages are needed for VLM captioning).
    """
    import pypdfium2 as pdfium

    src = Path(pdf_path)
    if not src.is_file():
        raise FileNotFoundError(str(src))

    pdf = None
    out_dir = Path(tempfile.mkdtemp(prefix="pivlm_pdf_"))
    paths: list[Path] = []
    try:
        pdf = pdfium.PdfDocument(str(src))
        n_pages = len(pdf)
        if max_pages is not None:
            n_pages = min(n_pages, max(0, max_pages))
        for i in range(n_pages):
            page = pdf[i]
            try:
                pil = page.render(scale=dpi / 72.0).to_pil()
                out = out_dir / f"page_{i + 1:04d}.jpg"
                pil.convert("RGB").save(out, "JPEG", quality=85)
                paths.append(out)
            finally:
                # Free pdfium's per-page buffers ASAP.
                page.close() if hasattr(page, "close") else None
        yield paths
    finally:
        try:
            if pdf is not None:
                pdf.close()
        except Exception:
            pass
        for p in paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            os.rmdir(out_dir)
        except Exception:
            pass


def pdf_page_count(pdf_path: str | Path) -> int:
    """Return the number of pages in a PDF (0 on open failure)."""
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            return len(pdf)
        finally:
            pdf.close()
    except Exception:
        return 0
