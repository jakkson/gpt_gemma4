"""Normalize arbitrary image files to JPEG paths Ollama vision can decode."""

from __future__ import annotations

import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


def _jpeg_via_pillow(src: Path, dest: Path) -> None:
    from PIL import Image, ImageOps

    im = Image.open(src)
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    im.save(dest, "JPEG", quality=90, optimize=True)


def _jpeg_via_sips(src: Path, dest: Path) -> None:
    """macOS ``sips`` decodes HEIC and formats Pillow may not have codecs for."""
    subprocess.run(
        ["sips", "-s", "format", "jpeg", str(src), "--out", str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )


@contextmanager
def image_path_for_ollama(src: str | Path):
    """
    Yield a path to a JPEG file suitable for ``ollama.chat(..., images=[...])``.

    Photos libraries often use HEIC, proprietary derivatives, or encodings the
    server reports as ``image: unknown format``; we re-encode to baseline JPEG
    via Pillow, with a macOS ``sips`` fallback.
    """
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(str(src))

    fd, tmp_name = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        try:
            _jpeg_via_pillow(src, tmp)
        except Exception as e1:
            tmp.unlink(missing_ok=True)
            fd2, tmp_name2 = tempfile.mkstemp(suffix=".jpg")
            os.close(fd2)
            tmp = Path(tmp_name2)
            try:
                _jpeg_via_sips(src, tmp)
            except Exception as e2:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"Could not decode image for Ollama (Pillow: {e1!r}; sips: {e2!r})") from e2
        yield str(tmp)
    finally:
        tmp.unlink(missing_ok=True)
