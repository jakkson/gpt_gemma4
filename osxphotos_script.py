"""Describe recent Photos library images with a local Ollama vision model."""

from __future__ import annotations

import sys

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


def _photos_db():
    """Open the system Photos library; may raise on macOS privacy restrictions."""
    return osxphotos.PhotosDB()


def main() -> None:
    # Loads photo metadata; [:10] only limits after the full list is built (slow on huge libraries).
    try:
        photosdb = _photos_db()
    except OSError as e:
        err = str(e).lower()
        if "operation not permitted" in err or "permission denied" in err or getattr(e, "errno", None) in (1, 13):
            print(_PHOTOS_ACCESS_HELP.strip(), file=sys.stderr)
        raise

    for photo in photosdb.photos(images=True, movies=False)[:10]:
        if not photo.path:
            continue
        print(f"Processing: {photo.filename}")

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
        print(f"Gemma says: {text}")


if __name__ == "__main__":
    main()
