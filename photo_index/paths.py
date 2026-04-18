"""Resolve a local image file for Vision / VLM without requiring iCloud originals."""

from __future__ import annotations

import os
from typing import Literal

PreferPath = Literal["derivatives", "path"]


def resolve_local_image_path(photo, *, prefer: PreferPath = "derivatives") -> str | None:
    """
    Pick a file on disk for OCR/VLM.

    ``derivatives`` (default): use library previews first (``path_derivatives``, largest first),
    then ``photo.path``. Best when you want to avoid leaning on full local masters when a grid
    preview exists (typical iCloud “optimized library” case).

    ``path``: use ``photo.path`` first when it exists, then derivatives. Often better OCR/VLM
    quality when a full local file is already on disk.
    """
    derivs = getattr(photo, "path_derivatives", None) or []
    deriv_first = [p for p in derivs if p and os.path.isfile(p)]

    path = getattr(photo, "path", None) or None
    path_ok = path if (path and os.path.isfile(path)) else None

    if prefer == "path":
        if path_ok:
            return path_ok
        return deriv_first[0] if deriv_first else None

    # prefer == "derivatives"
    if deriv_first:
        return deriv_first[0]
    if path_ok:
        return path_ok
    return None
