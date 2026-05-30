"""Apple Vision OCR over arbitrary local image files.

This is the macOS-native equivalent of ``photo.detected_text()`` (which only
works on items inside the Photos library). Here we hand a file URL to
``VNImageRequestHandler`` directly so we can OCR loose JPEGs, PNGs, rendered
PDF pages, etc.
"""

from __future__ import annotations

from pathlib import Path

# Lazy / soft import so the rest of the package still imports on non-macOS
# machines (e.g. CI sanity checks). Everything below this fence is mac-only.
try:  # pragma: no cover - macOS-only path
    from Cocoa import NSURL  # type: ignore[import-not-found]
    from Vision import (  # type: ignore[import-not-found]
        VNImageRequestHandler,
        VNRecognizeTextRequest,
        VNRequestTextRecognitionLevelAccurate,
        VNRequestTextRecognitionLevelFast,
    )
    _VISION_AVAILABLE = True
    _VISION_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - non-macOS / pyobjc missing
    _VISION_AVAILABLE = False
    _VISION_IMPORT_ERROR = e


_DEFAULT_LANGUAGES = ("en-US",)


def vision_available() -> bool:
    """True if Apple Vision OCR can be used (macOS + pyobjc-framework-Vision)."""
    return _VISION_AVAILABLE


def _vision_ocr_url(
    path: Path,
    *,
    languages: tuple[str, ...],
    accurate: bool,
    use_language_correction: bool,
) -> str:
    url = NSURL.fileURLWithPath_(str(path))
    handler = VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    req = VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(
        VNRequestTextRecognitionLevelAccurate
        if accurate
        else VNRequestTextRecognitionLevelFast
    )
    req.setUsesLanguageCorrection_(use_language_correction)
    if languages:
        req.setRecognitionLanguages_(list(languages))

    success, error = handler.performRequests_error_([req], None)
    if not success:
        raise RuntimeError(f"Vision OCR failed for {path}: {error}")

    out: list[str] = []
    for obs in (req.results() or []):
        cands = obs.topCandidates_(1)
        if cands and len(cands) > 0:
            text = cands[0].string()
            if text:
                out.append(str(text))
    return "\n".join(out)


def vision_ocr(
    image_path: str | Path,
    *,
    languages: tuple[str, ...] = _DEFAULT_LANGUAGES,
    accurate: bool = True,
    use_language_correction: bool = True,
) -> str:
    """Run Apple Vision text recognition on a local image file.

    Returns the concatenated top-candidate strings from every text observation,
    joined by newlines. Empty string when nothing is detected.

    Falls back to a Pillow / sips JPEG normalization step when Vision can't
    decode the input directly (we've seen ``initWithURL_options_`` silently
    fail on certain PNGs / .PNG-with-uppercase-ext files; routing through
    a normalized JPEG fixes it).
    """
    if not _VISION_AVAILABLE:
        raise RuntimeError(
            "Apple Vision is not available on this system: "
            f"{_VISION_IMPORT_ERROR!r}"
        )
    p = Path(image_path)
    if not p.is_file():
        raise FileNotFoundError(str(p))

    try:
        return _vision_ocr_url(
            p,
            languages=languages,
            accurate=accurate,
            use_language_correction=use_language_correction,
        )
    except Exception:
        # Lazy import so vision_ocr stays self-contained on test paths
        # that don't touch the Pillow+sips machinery.
        from photo_index.ollama_image import image_path_for_ollama

        with image_path_for_ollama(p) as normalized:
            return _vision_ocr_url(
                Path(normalized),
                languages=languages,
                accurate=accurate,
                use_language_correction=use_language_correction,
            )
