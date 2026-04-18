"""Resolve a local image file for Vision / VLM without requiring iCloud originals."""

from __future__ import annotations

import os


def resolve_local_image_path(photo) -> str | None:
    """
    Pick a file on disk for OCR/VLM while avoiding full masters when a preview exists.

    osxphotos sorts ``path_derivatives`` with **largest file first** (best preview quality).
    We prefer those derivatives so iCloud-only assets that still have grid previews work,
    and we do not require downloading full originals from the cloud.

    If no derivative exists, we fall back to ``photo.path`` (local optimized/original file).
    """
    derivs = getattr(photo, "path_derivatives", None) or []
    for p in derivs:
        if p and os.path.isfile(p):
            return p

    path = getattr(photo, "path", None) or None
    if path and os.path.isfile(path):
        return path
    return None
