"""Lightweight query synonym expansion for retrieval."""

from __future__ import annotations

import json
import re
from pathlib import Path

_CANON_SYNONYMS: dict[str, tuple[str, ...]] = {
    "nytimes": ("nytimes", "nyt", "new york times", "ny times", "nytimes.com"),
    "netflix": ("netflix", "net flix"),
    "hbo max": ("hbo max", "max"),
    "uber": ("uber", "uber trip", "uber technologies"),
    "lyft": ("lyft",),
    "apple": ("apple", "apple.com", "itunes", "app store", "icloud"),
    "amazon": ("amazon", "amzn", "amazon.com"),
    "google": ("google", "gpay", "google pay", "youtube"),
    "subscription": ("subscription", "recurring", "monthly", "renewal", "autopay"),
    "charge": ("charge", "charged", "payment", "bill", "billing", "fee", "debit"),
    "price": ("price", "amount", "cost", "charge", "payment", "fee", "bill", "chrge"),
    "message": ("message", "text", "sms", "imessage", "imsg"),
}

_USER_SYNONYMS_PATH = Path(__file__).resolve().parent.parent / "data" / "synonyms.json"
_MERGED_CACHE: dict[str, tuple[str, ...]] | None = None


def _normalize(q: str) -> str:
    return " ".join((q or "").lower().split())


def _load_user_synonyms() -> dict[str, tuple[str, ...]]:
    if not _USER_SYNONYMS_PATH.exists():
        return {}
    try:
        obj = json.loads(_USER_SYNONYMS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, list):
            continue
        items = tuple(str(x).strip().lower() for x in v if str(x).strip())
        if items:
            out[k.strip().lower()] = items
    return out


def _merged_synonyms() -> dict[str, tuple[str, ...]]:
    global _MERGED_CACHE
    if _MERGED_CACHE is not None:
        return _MERGED_CACHE

    merged: dict[str, tuple[str, ...]] = dict(_CANON_SYNONYMS)
    user = _load_user_synonyms()
    for canon, syns in user.items():
        if canon in merged:
            existing = list(merged[canon])
            for s in syns:
                if s not in existing:
                    existing.append(s)
            merged[canon] = tuple(existing)
        else:
            merged[canon] = tuple(syns)
    _MERGED_CACHE = merged
    return merged


def reset_synonym_cache() -> None:
    """Clear merged synonym cache (useful for long-running app refresh hooks)."""
    global _MERGED_CACHE
    _MERGED_CACHE = None


def expand_query_terms(query: str, max_expansions: int = 6) -> list[str]:
    """
    Return a small set of expanded retrieval queries.

    First item is always the original query.
    """
    base = (query or "").strip()
    if not base:
        return [""]
    qn = _normalize(base)
    out: list[str] = [base]
    seen = {qn}

    for canon, synonyms in _merged_synonyms().items():
        all_terms = [canon, *synonyms]
        present_terms = [t for t in all_terms if _normalize(t) in qn]
        if not present_terms:
            continue

        # Generate alternatives by replacing the detected alias phrase with
        # sibling aliases, which broadens retrieval without adding AND tokens.
        for present in present_terms:
            present_norm = _normalize(present)
            for target in all_terms:
                target_norm = _normalize(target)
                if target_norm == present_norm:
                    continue
                # Phrase-aware replacement (case-insensitive).
                replaced = re.sub(
                    re.escape(present),
                    target,
                    base,
                    flags=re.IGNORECASE,
                )
                rn = _normalize(replaced)
                if rn not in seen:
                    out.append(replaced)
                    seen.add(rn)
                if len(out) >= max_expansions:
                    return out

    # simple abbreviation expansion: split tokens with dots, etc.
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'.-]+", base)
    if "nyt" in [t.lower() for t in tokens] and len(out) < max_expansions:
        expanded = f"{base} new york times"
        en = _normalize(expanded)
        if en not in seen:
            out.append(expanded)
    return out[:max_expansions]
