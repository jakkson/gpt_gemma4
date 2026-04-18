"""Describe recent Photos library images with a local Ollama vision model."""

from __future__ import annotations

import osxphotos
from ollama import chat


def main() -> None:
    # Loads photo metadata; [:10] only limits after the full list is built (slow on huge libraries).
    photosdb = osxphotos.PhotosDB()
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
