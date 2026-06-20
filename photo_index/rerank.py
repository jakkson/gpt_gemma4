"""Cross-encoder reranking for retrieval.

The FTS5 (BM25) and nomic (bi-encoder) stages are both "blurry": they score the
query and each document *independently*. A cross-encoder reads each
(query, document) pair *together*, so it can tell whether a candidate actually
answers the question rather than merely sharing vocabulary. We run it as a
precision stage over the merged candidate pool, then blend its score into the
final ranking.

Env knobs:
  PHOTO_INDEX_RERANK            "1" (default on) / "0" to disable.
  PHOTO_INDEX_RERANK_MODEL      HF model id (default BAAI/bge-reranker-base).
  PHOTO_INDEX_RERANK_DEVICE     "mps" | "cpu" | "" (auto: mps if available).
  PHOTO_INDEX_RERANK_INPUT_CHARS per-document char cap fed to the encoder (1200).

Degrades gracefully: if the backend or model can't load, scoring returns {} and
the caller falls back to the existing FTS+semantic ranking. The failure is logged
once, never per-query.
"""

from __future__ import annotations

import os
import threading

_LOCK = threading.Lock()
# model: the loaded CrossEncoder (or None). tried: whether we've attempted a load
# (so a failed load doesn't retry on every query).
_STATE: dict[str, object] = {"model": None, "tried": False}

_DEFAULT_MODEL = "BAAI/bge-reranker-base"


def rerank_enabled() -> bool:
    return os.environ.get("PHOTO_INDEX_RERANK", "1").strip().lower() not in ("0", "false", "no")


def _resolve_device() -> str:
    dev = os.environ.get("PHOTO_INDEX_RERANK_DEVICE", "").strip()
    if dev:
        return dev
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_model():
    """Lazily load the cross-encoder once. Returns the model or None on failure."""
    if _STATE["tried"]:
        return _STATE["model"]
    with _LOCK:
        if _STATE["tried"]:
            return _STATE["model"]
        _STATE["tried"] = True
        try:
            from sentence_transformers import CrossEncoder

            model_name = os.environ.get("PHOTO_INDEX_RERANK_MODEL", _DEFAULT_MODEL)
            device = _resolve_device()
            model = CrossEncoder(model_name, max_length=512, device=device)
            _STATE["model"] = model
            print(f"[rerank] loaded {model_name} on {device}", flush=True)
        except Exception as e:  # noqa: BLE001 — reranking is best-effort
            print(f"[rerank] disabled (model load failed: {e})", flush=True)
            _STATE["model"] = None
        return _STATE["model"]


def rerank_scores(query: str, items: list[tuple[str, str]]) -> dict[str, float]:
    """Score each (id, text) pair against the query with the cross-encoder.

    Returns ``{id: normalized_score}`` with scores min-max scaled to [0, 1] across
    the candidate set (stable to blend against other ranking signals). Returns {}
    when disabled, when nothing to score, or when the encoder is unavailable.
    """
    if not rerank_enabled() or not items or not (query or "").strip():
        return {}
    model = _load_model()
    if model is None:
        return {}
    try:
        import numpy as np

        cap = int(os.environ.get("PHOTO_INDEX_RERANK_INPUT_CHARS", "1200"))
        pairs = [[query, (text or "")[:cap]] for _, text in items]
        scores = model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        s = np.asarray(scores, dtype=np.float32).reshape(-1)
        lo, hi = float(s.min()), float(s.max())
        s = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
        return {items[i][0]: float(s[i]) for i in range(len(items))}
    except Exception as e:  # noqa: BLE001
        print(f"[rerank] scoring failed: {e}", flush=True)
        return {}
