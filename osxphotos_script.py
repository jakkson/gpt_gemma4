"""Describe recent Photos library images with a local Ollama vision model."""

from __future__ import annotations

import sys
import time

import osxphotos
from ollama import ResponseError, chat

from photo_index.ollama_image import image_path_for_ollama
from photo_index.paths import resolve_local_image_path

_PHOTOS_ACCESS_HELP = """
macOS blocked read access to your Photos library (Photos.sqlite).

Fix:
  1. Open System Settings → Privacy & Security → Full Disk Access.
  2. Turn ON Full Disk Access for the app that runs this script:
       • Cursor (integrated terminal), or
       • Terminal.app / iTerm2 (if you run Python there).
  3. Quit and reopen that app, then run the script again.

If the Photos app is open and the library is locked, quit Photos and retry.
"""


def _log(msg: str) -> None:
    print(msg, flush=True)


def _photos_db():
    """Open the system Photos library; may raise on macOS privacy restrictions."""
    return osxphotos.PhotosDB()


def main() -> None:
    _log("[osxphotos] Opening Photos library…")
    t0 = time.perf_counter()
    try:
        photosdb = _photos_db()
    except OSError as e:
        err = str(e).lower()
        if "operation not permitted" in err or "permission denied" in err or getattr(e, "errno", None) in (1, 13):
            print(_PHOTOS_ACCESS_HELP.strip(), file=sys.stderr)
        raise

    _log(f"[osxphotos] OK — library path: {photosdb.library_path}")
    _log("[osxphotos] Loading image list (can take a while on large libraries)…")

    all_images = photosdb.photos(images=True, movies=False)
    _log(f"[osxphotos] OK — {len(all_images)} images in library (movies excluded)")

    batch = all_images[:10]
    _log(f"[osxphotos] Using first {len(batch)} images (previews preferred over full originals)")
    _log("[ollama] Images are re-encoded to JPEG for Gemma/Ollama (fixes HEIC / unknown format).")
    _log("-" * 60)

    ok = 0
    skipped_no_path = 0
    skipped_image = 0

    for i, photo in enumerate(batch, start=1):
        img_path = resolve_local_image_path(photo, prefer="derivatives")
        if not img_path:
            skipped_no_path += 1
            _log(
                f"[osxphotos] [{i}/{len(batch)}] skip (no local preview/file): "
                f"{photo.filename or photo.uuid}"
            )
            continue

        _log(f"[osxphotos] [{i}/{len(batch)}] Processing: {photo.filename}")
        _log(f"           source: {img_path}")

        try:
            with image_path_for_ollama(img_path) as ollama_img:
                response = chat(
                    model="gemma4:26b",
                    messages=[
                        {
                            "role": "user",
                            "content": "Describe this photo for my personal search index.",
                            "images": [ollama_img],
                        },
                    ],
                )
        except ResponseError as e:
            skipped_image += 1
            _log(f"[ollama] skip (vision error): {e}")
            _log("-" * 60)
            continue
        except (OSError, ValueError) as e:
            skipped_image += 1
            _log(f"[image] skip (could not decode): {e}")
            _log("-" * 60)
            continue

        ok += 1
        text = response.message.content or ""
        _log(f"[ollama] Gemma says: {text[:500]}{'…' if len(text) > 500 else ''}")
        _log("-" * 60)

    elapsed = time.perf_counter() - t0
    _log(
        f"[done] vision_ok={ok} skipped_no_path={skipped_no_path} "
        f"skipped_decode_or_ollama={skipped_image} time={elapsed:.1f}s"
    )

    if ok == 0 and skipped_no_path:
        _log(
            "[hint] No local preview/file for these items. "
            "Open Photos and let thumbnails generate, or enable iCloud downloads for a subset."
        )


if __name__ == "__main__":
    main()
