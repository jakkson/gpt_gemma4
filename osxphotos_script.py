"""Describe recent Photos library images with a local Ollama vision model."""

from __future__ import annotations

import sys
import time

import osxphotos
from ollama import chat

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
    _log(f"[osxphotos] Using first {len(batch)} images from that list (not necessarily newest)")
    _log("[ollama] Sending each image with a local path to gemma4:26b (slow per image)…")
    _log("-" * 60)

    processed = 0
    skipped_no_path = 0

    for i, photo in enumerate(batch, start=1):
        if not photo.path:
            skipped_no_path += 1
            _log(
                f"[osxphotos] [{i}/{len(batch)}] skip (no local file): "
                f"{photo.filename or photo.uuid} — often iCloud-only; download in Photos first"
            )
            continue

        processed += 1
        _log(f"[osxphotos] [{i}/{len(batch)}] Processing: {photo.filename}")
        _log(f"           path: {photo.path}")

        response = chat(
            model="gemma4:26b",
            messages=[
                {
                    "role": "user",
                    "content": "Describe this photo for my personal search index.",
                    "images": [photo.path],
                },
            ],
        )
        text = response.message.content or ""
        _log(f"[ollama] Gemma says: {text[:500]}{'…' if len(text) > 500 else ''}")
        _log("-" * 60)

    elapsed = time.perf_counter() - t0
    _log(f"[done] Processed with vision: {processed} image(s); skipped (no local path): {skipped_no_path}; time: {elapsed:.1f}s")

    if processed == 0 and skipped_no_path:
        _log(
            "[hint] All sampled images lacked a downloaded file on disk. "
            "Open Photos, select them, and use “Download Originals” (or similar), then rerun."
        )


if __name__ == "__main__":
    main()
