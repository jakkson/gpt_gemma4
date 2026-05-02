"""Local Gradio UI for querying the photo index with Gemma."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
    _LOCAL_TZ = ZoneInfo("America/Los_Angeles")
except Exception:  # pragma: no cover - fallback for stripped builds
    _LOCAL_TZ = timezone.utc
from pathlib import Path
from typing import Any

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from ollama import chat
from PIL import Image

from photo_index.ollama_image import image_path_for_ollama
from photo_index.query_expand import expand_query_terms, reset_synonym_cache

from photo_index.store import (
    connect,
    init_schema,
    row_to_prompt_block,
    search_meta,
    search_meta_fallback_substring,
)

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"
_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "gradio_search_cache.json"
_SYNONYMS_PATH = Path(__file__).resolve().parent.parent / "data" / "synonyms.json"
_CACHE_TTL_SECONDS = 24 * 60 * 60
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
_TERM_VOCAB_CACHE: dict[str, set[str]] = {}
_KEYBOARD_JS = """
() => {
  if (window.__photoSearchEnterBound) return;
  window.__photoSearchEnterBound = true;
  document.addEventListener(
    "keydown",
    (e) => {
      if (e.key !== "Enter" || e.shiftKey) return;
      const active = document.activeElement;
      const queryWrap = document.querySelector("#photo-query-input");
      const searchBtn = document.querySelector("#photo-search-btn button");
      if (!active || !queryWrap || !searchBtn) return;
      if (active.tagName !== "TEXTAREA") return;
      if (!queryWrap.contains(active)) return;
      e.preventDefault();
      e.stopPropagation();
      searchBtn.click();
    },
    true
  );
}
"""


def _format_local_dt(iso_str: str) -> str:
    """Render an ISO date string as `MM/DD/YYYY h:MM am/pm TZ` in America/Los_Angeles.

    Returns the original string unchanged if it cannot be parsed.
    """
    if not iso_str:
        return ""
    s = str(iso_str).strip()
    if not s:
        return ""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return s
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        dt = dt.astimezone(_LOCAL_TZ)
    except Exception:
        pass
    date_part = dt.strftime("%m/%d/%Y")
    hour = dt.hour % 12 or 12
    minute = dt.strftime("%M")
    ampm = "am" if dt.hour < 12 else "pm"
    tz_abbrev = dt.strftime("%Z") or "PT"
    return f"{date_part} {hour}:{minute} {ampm} {tz_abbrev}"


def _ui_version_stamp() -> str:
    p = Path(__file__).resolve()
    text = p.read_text(encoding="utf-8")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"{ts} / {digest}"


_LOCAL_INDEX_POLICY = """LOCAL PRIVATE INDEX (READ FIRST)
- Everything below is from this user's own machine: indexed files, OCR, captions, and messages they already own.
- When records mention health (colonoscopy, labs, prescriptions, etc.), REPORT what those records say:
  dates, procedure names, and short verbatim phrases. That is not telemedicine or a diagnosis;
  it is reading back text from their own documents.
- NEVER refuse with "I cannot access medical records", "I'm a chatbot without your health data", or
  "contact your doctor" if the indexed records below actually contain the answer.
- You MUST still avoid inventing facts not present in the records.

"""


def _build_prompt(question: str, rows: list[sqlite3.Row], *, aggregate: bool = False) -> str:
    blocks = [row_to_prompt_block(r) for r in rows]
    context = "\n\n---\n\n".join(blocks)
    if aggregate:
        return f"""You are answering questions about a single user's personal on-device index
(their own photos, OCR, VLM captions, and SMS/iMessage text).

{_LOCAL_INDEX_POLICY}
GROUND RULES
- Use ONLY the indexed records below. Do NOT use outside / general knowledge.
- Quote exact dollar amounts and dates from the records when relevant.
- Cite each record you use inline by its filename or imsg uuid.

REASONING ALLOWED (this is an aggregate / "how much per month" question)
1. Scan the records and list EVERY recurring/subscription/monthly charge you can find:
   merchant, amount, date, and the imsg uuid of the message.
2. Group by merchant if the same one shows up across months. Use the most recent
   amount for that merchant.
3. Sum the monthly amounts and give a per-month total.
4. If an amount is clearly annual (e.g. once-a-year), label it and divide by 12
   before adding it to the monthly total.
5. If you only have a few records and suspect more exist, say so explicitly:
   "Based on N indexed messages, your visible monthly subscriptions total $X.
   There may be more not yet indexed."

REFUSAL
- Reply "I don't see that in your indexed data yet." ONLY if the records contain
  no recurring charges at all.

Indexed records:
{context}

User question: {question}
"""

    return f"""You are answering questions about a single user's personal on-device index
(their own photos, OCR, VLM captions, and SMS/iMessage text).

{_LOCAL_INDEX_POLICY}
STRICT RULES
- Use ONLY the indexed records below. Do not use outside / general knowledge.
  Do not summarize what a product or company is in general.
- For money / price / payment / charge / subscription questions, quote the exact
  dollar amount and date(s) directly from the records, and cite the matching
  record (filename or imsg uuid) inline.
- You MAY add up, count, or compare amounts that are visible in the records.
- Prefer the most recent matching record when the user asks about "latest",
  "currently", or "right now".

REFUSAL
- Only say "I don't see that in your indexed data yet." if there are NO
  records at all that touch the topic. If there are partial matches, list what
  you found and explain what's missing.

Indexed records:
{context}

User question: {question}
"""


def _is_short_factual_query(question: str) -> bool:
    """Detect short who/what/when-style lookups that the small model handles well.

    A pure word-count check (e.g. <=6 words) is too aggressive: topic queries
    like "Valkyries media day 2026" are short but mixed-context, and the small
    Gemma tends to hallucinate ("Vegas event") instead of grounding in the
    retrieved records. Require BOTH a clear factual prefix AND short length.
    """
    q = " ".join((question or "").strip().lower().split())
    if not q:
        return False
    words = q.split()
    if not words or len(words) > 8:
        return False
    factual_starts = (
        "who",
        "what",
        "when",
        "where",
        "which",
        "find",
        "show",
        "list",
        "is ",
        "are ",
        "do ",
        "does ",
        "did ",
    )
    return q.startswith(factual_starts)


def _is_aggregate_finance_query(question: str) -> bool:
    """True for questions that need synthesis across many finance records,
    e.g. "how much am I paying in subscriptions per month?"."""
    q = " ".join((question or "").strip().lower().split())
    if not q:
        return False
    finance = (
        "subscription", "subscriptions", "recurring", "monthly", "per month",
        "each month", "every month", "annual", "annually", "per year",
        "yearly", "total", "sum", "spend", "spending", "paying", "pay",
        "bills", "charges", "payments", "fees",
    )
    aggregate = (
        "how much", "how many", "total", "sum", "altogether", "combined",
        "across", "average", "list all", "all my", "all of my", "everything",
    )
    return any(t in q for t in finance) and any(t in q for t in aggregate)


def _is_broad_or_ambiguous_query(question: str) -> bool:
    q = " ".join((question or "").strip().lower().split())
    if not q:
        return False
    broad_terms = (
        "summarize",
        "compare",
        "analyze",
        "story",
        "trends",
        "pattern",
        "why",
        "explain",
        "best",
        "overall",
        "across",
        "multiple",
    )
    words = q.split()
    return len(words) >= 14 or any(t in q for t in broad_terms)


def _low_confidence_answer(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    markers = (
        "i'm not sure",
        "i am not sure",
        "unsure",
        "cannot determine",
        "can't determine",
        "insufficient",
        "not enough information",
        "i don't know",
        "unknown",
    )
    return any(m in t for m in markers)


_POLICY_REFUSAL_MARKERS = (
    "cannot provide",
    "can't provide",
    "medical records",
    "private health",
    "health information",
    "i am an ai",
    "i'm an ai",
    "as an ai",
    "language model",
    "chatbot",
    "do not rely",
    "don't rely",
    "contact your doctor",
    "speak to your doctor",
    "talk to your doctor",
    "consult a healthcare",
    "consult your healthcare",
    "professional medical",
    "i do not have access to your personal",
    "don't have access to your personal",
)


def _policy_refusal_answer(answer: str) -> bool:
    """True if the model gave a generic safety brush-off instead of using context."""
    t = (answer or "").strip().lower()
    if not t:
        return True
    if "filename" in t or "imsg:" in t or "doc:" in t or "indexed" in t:
        return False
    return any(m in t for m in _POLICY_REFUSAL_MARKERS)


def _prompt_policy_retry_addon(question: str) -> str:
    return f"""
REMINDER (your last reply ignored the rules): The block above is ONLY this user's own indexed files/messages.
You must answer their question from those records if relevant text appears — including colonoscopy / lab / health dates.
Quote the date and cite filename or uuid. Do not refuse. If nothing matches, say only:
"I don't see that in your indexed data yet."
User question: {question}
"""


def _safe_chat(*, model: str, prompt: str) -> tuple[str, str | None]:
    try:
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.message.content or "").strip(), None
    except Exception as e:
        return "", str(e)


def _build_term_vocab(conn: sqlite3.Connection, limit_rows: int = 5000) -> set[str]:
    rows = conn.execute(
        """
        SELECT filename, ocr_text, vlm_text
        FROM photo_meta
        ORDER BY ingested_at DESC
        LIMIT ?
        """,
        (limit_rows,),
    ).fetchall()
    vocab: set[str] = set()
    for r in rows:
        blob = " ".join(
            [
                str(r["filename"] or ""),
                str(r["ocr_text"] or ""),
                str(r["vlm_text"] or ""),
            ]
        )
        for tok in _TOKEN_RE.findall(blob):
            vocab.add(tok.lower())
    return vocab


def _get_term_vocab(db_path: Path) -> set[str]:
    key = str(db_path)
    if key in _TERM_VOCAB_CACHE:
        return _TERM_VOCAB_CACHE[key]
    conn = connect(db_path)
    init_schema(conn)
    try:
        vocab = _build_term_vocab(conn)
    finally:
        conn.close()
    _TERM_VOCAB_CACHE[key] = vocab
    return vocab


def _suggest_query(original: str, db_path: Path) -> str:
    words = original.split()
    if not words:
        return original
    vocab = _get_term_vocab(db_path)
    if not vocab:
        return original
    vocab_list = list(vocab)
    fixed: list[str] = []
    changed = False
    for w in words:
        wl = w.lower()
        if len(wl) < 4 or not wl.isascii() or wl in vocab:
            fixed.append(w)
            continue
        best = difflib.get_close_matches(wl, vocab_list, n=1, cutoff=0.86)
        if best:
            fixed.append(best[0])
            changed = True
        else:
            fixed.append(w)
    return " ".join(fixed) if changed else original


def _installed_ollama_models() -> list[str]:
    try:
        out = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except Exception:
        return []
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return []
    models: list[str] = []
    for ln in lines[1:]:
        parts = ln.split()
        if parts:
            models.append(parts[0])
    return models


SORT_RELEVANT = "Most Relevant"
SORT_RECENT = "Most Recent"
SORT_OPTIONS = (SORT_RELEVANT, SORT_RECENT)

_BANK_ISSUERS = (
    "capital one", "chase", "wells fargo", "amex", "american express",
    "bank of america", "citi", "citibank", "discover", "venmo", "paypal",
    "apple cash", "apple card", "robinhood", "ally bank", "us bank",
    "synchrony", "barclays", "hsbc", "navy federal", "schwab", "fidelity",
)
_TRANSACTION_WORDS = (
    "chrge", "charge", "charged", "hold ", "transaction", "placed on your",
    "statement", "withdrawn", "debited", "auto-pay", "autopay", "payment of",
    "balance is", "bill of", "due", "posted",
)
_CURRENCY_RE = re.compile(r"\$\s?\d|\d+\.\d{2}")
_FINANCE_TRIGGER_TERMS = (
    "price", "charge", "charged", "payment", "bill", "billing", "fee",
    "subscription", "subscriptions", "cost", "pay", "paying", "paid",
    "amount", "money", "spend", "spending", "owe", "due", "rate",
    "monthly", "per month", "annual", "yearly",
)


def _is_bank_source(text: str) -> bool:
    """Return True if the text looks like an authoritative bank/credit-card
    transaction record (issuer + transaction word + currency figure)."""
    t = (text or "").lower()
    if not _CURRENCY_RE.search(t):
        return False
    if not any(w in t for w in _BANK_ISSUERS):
        return False
    return any(w in t for w in _TRANSACTION_WORDS)


def _is_finance_query(question: str) -> bool:
    q = " ".join((question or "").strip().lower().split())
    return bool(q) and any(t in q for t in _FINANCE_TRIGGER_TERMS)


def _retrieve_rows(
    db_path: Path,
    question: str,
    top_k: int,
    sort_by: str = SORT_RELEVANT,
    restrict_finance: bool = True,
) -> list[sqlite3.Row]:
    # Reload user-edited aliases for long-running UI sessions.
    reset_synonym_cache()
    conn = connect(db_path)
    init_schema(conn)
    try:
        merged: dict[str, sqlite3.Row] = {}
        candidate_limit = max(top_k * 4, 40)
        for expanded in expand_query_terms(question):
            try:
                hits = search_meta(conn, expanded, limit=candidate_limit)
            except sqlite3.OperationalError:
                hits = search_meta_fallback_substring(conn, expanded, limit=candidate_limit)
            for r in hits:
                merged[r["uuid"]] = r
        # Always merge fallback token-like matches too; FTS may return sparse/weak
        # results for natural-language questions.
        for expanded in expand_query_terms(question):
            hits = search_meta_fallback_substring(conn, expanded, limit=candidate_limit)
            for r in hits:
                merged[r["uuid"]] = r
            if len(merged) >= max(top_k * 6, 80):
                break
            # Explicit message "ping": run a message-only pass on individual
            # meaningful tokens so imsg rows are always considered for queries
            # that mention texts/billing/subscriptions/etc.
            for tok in re.findall(r"[a-z0-9'.-]+", expanded.lower()):
                if len(tok) < 3:
                    continue
                like = f"%{tok}%"
                msg_hits = conn.execute(
                    """
                    SELECT *, 0 AS rank
                    FROM photo_meta
                    WHERE uuid LIKE 'imsg:%'
                      AND (
                        lower(ocr_text) LIKE ?
                        OR lower(vlm_text) LIKE ?
                        OR lower(filename) LIKE ?
                      )
                    ORDER BY date_iso DESC, ingested_at DESC
                    LIMIT ?
                    """,
                    (like, like, like, candidate_limit),
                ).fetchall()
                for r in msg_hits:
                    merged[r["uuid"]] = r

        ql_for_finance = (question or "").lower()
        _finance_trigger = (
            "price", "charge", "charged", "payment", "bill", "billing", "fee",
            "subscription", "cost", "pay", "paying", "paid", "amount", "money",
            "spend", "spending", "owe", "due", "rate", "monthly", "per month",
        )
        if any(t in ql_for_finance for t in _finance_trigger):
            # Currency-bearing message sweep: pull every message that contains
            # a real $X.XX or $X figure. This guarantees Capital One charge
            # alerts surface for aggregate finance questions even if the query
            # didn't include words like "subscription" or merchant names.
            # Pull a broad pool of finance-looking message rows, then filter
            # in Python to those that actually contain a $X.XX figure.
            broad_msg_hits = conn.execute(
                """
                SELECT *, 0 AS rank
                FROM photo_meta
                WHERE uuid LIKE 'imsg:%'
                  AND (
                    ocr_text LIKE '%$%'
                    OR ocr_text LIKE '% chrge %'
                    OR ocr_text LIKE '% charge %'
                    OR ocr_text LIKE '% charged %'
                    OR ocr_text LIKE '% bill %'
                    OR ocr_text LIKE '% payment %'
                    OR ocr_text LIKE '%/month%'
                    OR ocr_text LIKE '%per month%'
                    OR ocr_text LIKE '%subscription%'
                    OR ocr_text LIKE '%recurring%'
                  )
                ORDER BY date_iso DESC, ingested_at DESC
                LIMIT ?
                """,
                (max(candidate_limit * 4, 200),),
            ).fetchall()
            _money_re = re.compile(r"\$\s?\d|\d+\.\d{2}")
            for r in broad_msg_hits:
                blob = (r["ocr_text"] or "") + " " + (r["vlm_text"] or "")
                if _money_re.search(blob):
                    merged[r["uuid"]] = r

        rows = list(merged.values())
        ql = (question or "").lower()

        # Restrict finance/subscription queries to authoritative bank or credit-card
        # statement rows. This excludes casual chat like "he was paying $30 for X".
        # If no bank rows exist in the candidate set, fall back to the unrestricted
        # set so the user still sees something.
        if restrict_finance and _is_finance_query(question):
            bank_rows = [
                r for r in rows
                if _is_bank_source(
                    f"{r['filename'] or ''} {r['ocr_text'] or ''} {r['vlm_text'] or ''}"
                )
            ]
            if bank_rows:
                rows = bank_rows
        finance_terms = (
            "price", "charge", "charged", "payment", "bill", "billing",
            "fee", "subscription", "cost",
            "pay", "paying", "paid", "amount", "money", "spend", "spending",
            "owe", "due", "rate",
        )
        wants_messages = ("text", "message", "sms", "imessage", "capital one", "bank", "statement")
        boost_messages = any(t in ql for t in finance_terms + wants_messages)
        wants_nyt = any(t in ql for t in ("ny times", "nytimes", "nyt", "new york times"))
        wants_money = any(t in ql for t in finance_terms)
        # Stopwords filtered out of overlap so generic words ("the", "what")
        # don't inflate noisy hits over targeted ones.
        _OVERLAP_STOP = {
            "the", "and", "for", "you", "what", "this", "that", "with", "your",
            "are", "was", "were", "from", "have", "has", "not", "but", "all",
            "any", "how", "much", "i'm", "i am", "now", "just",
        }

        def score(r: sqlite3.Row) -> tuple[int, float, float, str]:
            is_msg = 1 if str(r["uuid"]).startswith("imsg:") else 0
            # bm25 rank (lower is better); fallback rows use 0
            rank = float(r["rank"]) if "rank" in r.keys() and r["rank"] is not None else 0.0
            text = f"{r['filename'] or ''} {r['ocr_text'] or ''} {r['vlm_text'] or ''}".lower()
            overlap = 0.0
            for tok in re.findall(r"[a-z0-9'.-]+", ql):
                if len(tok) < 3 or tok in _OVERLAP_STOP:
                    continue
                if tok in text:
                    overlap += 1.0
            entity_bonus = 0.0
            has_nyt = any(k in text for k in ("new york times", "nytimes", "nyt ", "ny times"))
            has_dollar_figure = bool(_CURRENCY_RE.search(text))
            has_currency = has_dollar_figure or any(
                t in text for t in ("price", "chrge", "charge", "payment", "fee", "bill", "$")
            )
            is_billing_alert = is_msg and _is_bank_source(text)
            if wants_nyt and has_nyt:
                entity_bonus += 6.0
            if "subscription" in ql and "subscription" in text:
                entity_bonus += 2.0
            if wants_money and has_currency:
                entity_bonus += 3.0
            # Strong combo: query is finance-y AND record is a message that
            # mentions both the entity and an actual dollar figure. This is the
            # canonical "what am I paying for X?" hit.
            if wants_money and is_msg and has_currency and (not wants_nyt or has_nyt):
                entity_bonus += 8.0
            # Billing alerts dominate when the question is about money. They
            # are the highest-quality evidence by a wide margin.
            if wants_money and is_billing_alert:
                entity_bonus += 12.0
            msg_pref = 1 if (boost_messages and is_msg) else 0
            date_key = str(r["date_iso"] or "")
            # Tuple sorted desc: prefer messages, then higher overlap+bonus,
            # then better (lower) bm25 rank, then most-recent date_iso lex sort.
            return (msg_pref, overlap + entity_bonus, -rank, date_key)

        if sort_by == SORT_RECENT:
            def recency_key(r: sqlite3.Row) -> tuple[str, str]:
                return (str(r["date_iso"] or ""), str(r["ingested_at"] or ""))
            rows.sort(key=recency_key, reverse=True)
        else:
            rows.sort(key=score, reverse=True)
        return rows[:top_k]
    finally:
        conn.close()


def _rows_preview(rows: list[sqlite3.Row]) -> list[list[str]]:
    preview: list[list[str]] = []
    for r in rows:
        ocr = (r["ocr_text"] or "").replace("\n", " ").strip()
        vlm = (r["vlm_text"] or "").replace("\n", " ").strip()
        preview.append(
            [
                r["uuid"] or "",
                r["filename"] or "",
                r["date_iso"] or "",
                r["image_path_used"] or "",
                f"{r['rank']:.3f}" if "rank" in r.keys() else "0.000",
                ocr[:180] + ("..." if len(ocr) > 180 else ""),
                vlm[:180] + ("..." if len(vlm) > 180 else ""),
            ]
        )
    return preview


def _rows_to_hit_summary(rows: list[list[str]]) -> str:
    if not rows:
        return "No hits yet."
    parts: list[str] = ["### Search hits"]
    for i, r in enumerate(rows, start=1):
        uuid = r[0] if len(r) > 0 else ""
        filename = r[1] if len(r) > 1 else ""
        date_iso = r[2] if len(r) > 2 else ""
        image_path = r[3] if len(r) > 3 else ""
        ocr_excerpt = r[5] if len(r) > 5 else ""
        vlm_excerpt = r[6] if len(r) > 6 else ""
        is_msg = str(uuid).startswith("imsg:")
        is_doc = str(uuid).startswith("doc:")
        source = (
            "Messages"
            if is_msg
            else "Document"
            if is_doc
            else "Photos / Local file"
        )
        snippet = ocr_excerpt or vlm_excerpt or "(no snippet)"
        title = filename or uuid
        when = _format_local_dt(date_iso) or "n/a"
        # "Reference attachment" link: route the click through our own
        # /open-local-file endpoint so the OS default app actually opens
        # (browsers silently block `file://` navigation from `http://` pages,
        # which is why direct file:// links did nothing).
        if image_path:
            encoded = urllib.parse.quote(image_path, safe="")
            href = f"/open-local-file?path={encoded}"
            link_md = f"[Open local file]({href})"
            ref = f"`{image_path}`"
        elif is_msg:
            link_md = "Use **Open Messages.app** below to jump to your texts"
            ref = f"`{uuid}`"
        else:
            link_md = "(no local link)"
            ref = f"`{uuid}`"
        parts.append(
            f"**{i}. {title}**  \n"
            f"_{source} • {when}_  \n"
            f"{snippet}  \n"
            f"{link_md} — ref: {ref}"
        )
    return "\n\n".join(parts)


def _rows_to_gallery(rows: list[list[str]], max_items: int = 16) -> tuple[list[Any], list[str]]:
    gallery: list[Any] = []
    paths: list[str] = []
    for r in rows:
        image_path = r[3] if len(r) > 3 else ""
        if not image_path:
            continue
        p = Path(image_path)
        if not p.exists():
            continue
        try:
            img = _load_preview_image(p)
        except Exception:
            continue
        caption = f"{_format_local_dt(r[2]) or 'n/a'} | {r[1] or r[0]}"
        gallery.append((img, caption))
        paths.append(str(p))
        if len(gallery) >= max_items:
            break
    return gallery, paths


def _cache_key(
    *,
    question: str,
    db_path: Path,
    top_k: int,
    qa_model: str,
    qa_model_small: str,
    auto_route: bool,
    sort_by: str,
    restrict_finance: bool,
) -> str:
    q = " ".join((question or "").strip().lower().split())
    # Bind to UI version so any code change (which alters the file hash) auto-
    # invalidates previously-cached answers from older retrieval/ranking logic.
    version = _ui_version_stamp()
    return (
        f"{q}|{db_path}|{top_k}|{qa_model}|{qa_model_small}"
        f"|auto={int(auto_route)}|sort={sort_by}"
        f"|rf={int(restrict_finance)}|v={version}"
    )


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    tmp.replace(cache_path)


def _prune_cache(cache: dict, now: float) -> dict:
    keep: dict = {}
    for key, value in cache.items():
        if not isinstance(value, dict):
            continue
        ts = float(value.get("cached_at_unix", 0))
        if now - ts <= _CACHE_TTL_SECONDS:
            keep[key] = value
    return keep


def answer_question(
    question: str,
    db_path: Path,
    top_k: int,
    qa_model: str,
    qa_model_small: str,
    auto_route: bool,
    auto_correct: bool,
    sort_by: str = SORT_RELEVANT,
    restrict_finance: bool = True,
) -> tuple[str, list[list[str]], str, str, list[Any], list[str]]:
    q = (question or "").strip()
    if not q:
        return "Enter a question to search your photo index.", [], "Last search: n/a", "No hits yet.", [], []

    sort_by = sort_by if sort_by in SORT_OPTIONS else SORT_RELEVANT
    t0 = time.perf_counter()
    now = time.time()
    key = _cache_key(
        question=q,
        db_path=db_path,
        top_k=top_k,
        qa_model=qa_model,
        qa_model_small=qa_model_small,
        auto_route=auto_route,
        sort_by=sort_by,
        restrict_finance=restrict_finance,
    )
    cache = _prune_cache(_load_cache(_CACHE_PATH), now)
    cached = cache.get(key)
    if isinstance(cached, dict):
        age_s = int(now - float(cached.get("cached_at_unix", now)))
        answer = str(cached.get("answer", "")).strip() or "(No response text returned.)"
        rows = cached.get("rows")
        if isinstance(rows, list):
            elapsed = time.perf_counter() - t0
            used_model = str(cached.get("used_model", qa_model))
            route = str(cached.get("route", "cache"))
            stats = (
                f"Last search: cache hit ({age_s}s old), "
                f"total retrieval time {elapsed:.2f}s, top-k={top_k}, model={used_model}, "
                f"route={route}, sort={sort_by}, bank-only={int(restrict_finance)}"
            )
            hit_md = _rows_to_hit_summary(rows)
            gallery, gallery_paths = _rows_to_gallery(rows)
            return answer, rows, stats, hit_md, gallery, gallery_paths

    aggregate_mode = _is_aggregate_finance_query(q)
    # Aggregate questions need to see ALL recurring charges, not just top_k=15.
    effective_top_k = max(top_k, 40) if aggregate_mode else top_k

    rows = _retrieve_rows(
        db_path=db_path, question=q, top_k=effective_top_k,
        sort_by=sort_by, restrict_finance=restrict_finance,
    )
    effective_query = q
    autocorrect_note = ""
    if not rows and auto_correct:
        suggested = _suggest_query(q, db_path=db_path)
        if suggested != q:
            retry_rows = _retrieve_rows(
                db_path=db_path, question=suggested, top_k=effective_top_k,
                sort_by=sort_by, restrict_finance=restrict_finance,
            )
            if retry_rows:
                rows = retry_rows
                effective_query = suggested
                autocorrect_note = (
                    f"These are results for **{suggested}**.  \n"
                    f"Search instead for **{q}**."
                )
    if not rows:
        elapsed = time.perf_counter() - t0
        return (
            "No matches in index yet. Keep ingest running, then try again.",
            [],
            f"Last search: no matches, total retrieval time {elapsed:.2f}s, top-k={top_k}, sort={sort_by}",
            "No hits yet.",
            [],
            [],
        )

    prompt = _build_prompt(effective_query, rows, aggregate=aggregate_mode)
    route = "large_direct"
    first_model = qa_model
    if aggregate_mode:
        first_model = qa_model
        route = "aggregate_finance->large"
    elif auto_route:
        if _is_short_factual_query(q):
            first_model = qa_model_small
            route = "small_first_factual"
        elif _is_broad_or_ambiguous_query(q):
            first_model = qa_model
            route = "large_first_broad"
        else:
            # Default to the LARGE model. The small Gemma was prone to topical
            # drift on mixed-context retrieval (e.g. answering a Valkyries query
            # as if it were about Vegas because the index also contained heavy
            # WNBA / Aces transcripts). Better slow + correct than fast + wrong.
            first_model = qa_model
            route = "large_default"

    answer, err = _safe_chat(model=first_model, prompt=prompt)
    used_model = first_model
    if err:
        if first_model != qa_model:
            retry_text, retry_err = _safe_chat(model=qa_model, prompt=prompt)
            if retry_err:
                elapsed = time.perf_counter() - t0
                return (
                    f"Search failed: small model `{first_model}` and fallback `{qa_model}` both errored.\n\n"
                    f"small error: {err}\n\nfallback error: {retry_err}",
                    [],
                    f"Last search: error after {elapsed:.2f}s, top-k={top_k}",
                    "No hits due to error.",
                    [],
                    [],
                )
            answer = retry_text or "(No response text returned.)"
            used_model = qa_model
            route = f"{route}->large_fallback_on_error"
        else:
            elapsed = time.perf_counter() - t0
            return (
                f"Search failed with model `{qa_model}`: {err}",
                [],
                f"Last search: error after {elapsed:.2f}s, top-k={top_k}, model={qa_model}",
                "No hits due to error.",
                [],
                [],
            )
    answer = answer or "(No response text returned.)"
    if auto_route and first_model != qa_model and _low_confidence_answer(answer):
        retry_text, retry_err = _safe_chat(model=qa_model, prompt=prompt)
        if retry_text and not retry_err:
            answer = retry_text
            used_model = qa_model
            route = f"{route}->large_retry"
    if rows and _policy_refusal_answer(answer):
        retry_p = prompt + _prompt_policy_retry_addon(effective_query)
        retry_text, retry_err = _safe_chat(model=qa_model, prompt=retry_p)
        if retry_text and not retry_err and not _policy_refusal_answer(retry_text):
            answer = retry_text
            used_model = qa_model
            route = f"{route}->policy_retry_large"
    if autocorrect_note:
        answer = f"{autocorrect_note}\n\n---\n\n{answer}"
    preview_rows = _rows_preview(rows)

    cache[key] = {
        "cached_at_unix": now,
        "answer": answer,
        "rows": preview_rows,
        "used_model": used_model,
        "route": route,
    }
    _save_cache(_CACHE_PATH, cache)

    elapsed = time.perf_counter() - t0
    stats = (
        f"Last search: cache miss, total retrieval time {elapsed:.2f}s, "
        f"top-k={top_k}, model={used_model}, route={route}, sort={sort_by}, "
        f"bank-only={int(restrict_finance)}"
    )
    hit_md = _rows_to_hit_summary(preview_rows)
    gallery, gallery_paths = _rows_to_gallery(preview_rows)
    return answer, preview_rows, stats, hit_md, gallery, gallery_paths


def recheck_with_large_only(
    question: str,
    db_path: Path,
    top_k: int,
    qa_model: str,
    qa_model_small: str,
    sort_by: str = SORT_RELEVANT,
    restrict_finance: bool = True,
) -> tuple[str, list[list[str]], str, str, list[Any], list[str]]:
    q = (question or "").strip()
    if not q:
        return "Enter a query first, then use Re-check with 26b only.", [], "Last search: n/a", "No hits yet.", [], []
    answer, rows, stats, hit_md, gallery, gallery_paths = answer_question(
        question=q,
        db_path=db_path,
        top_k=top_k,
        qa_model=qa_model,
        qa_model_small=qa_model_small,
        auto_route=False,
        auto_correct=False,
        sort_by=sort_by,
        restrict_finance=restrict_finance,
    )
    return answer, rows, f"{stats} [double-check: large model only]", hit_md, gallery, gallery_paths


def _extract_row(rows, row_idx: int) -> list[str]:
    # Gradio can provide table data as list[list[str]] or as a pandas DataFrame.
    if hasattr(rows, "iloc"):
        values = rows.iloc[row_idx].tolist()
        return [str(v) if v is not None else "" for v in values]
    selected = rows[row_idx]
    return [str(v) if v is not None else "" for v in selected]


def _load_preview_image(image_path: Path):
    # Return an in-memory RGB image so Gradio doesn't need to serve arbitrary filesystem paths.
    try:
        with Image.open(image_path) as im:
            return im.convert("RGB")
    except Exception:
        # Fall back to our robust converter (handles HEIC/odd encodings via Pillow+sips).
        with image_path_for_ollama(image_path) as tmp_jpeg:
            with Image.open(tmp_jpeg) as im:
                return im.convert("RGB")


def preview_selected(rows, evt: gr.SelectData):
    if rows is None or evt is None or evt.index is None:
        return None, "Select a result row to preview the image.", ""
    if len(rows) == 0:
        return None, "Select a result row to preview the image.", ""
    row_idx = int(evt.index[0]) if isinstance(evt.index, (tuple, list)) else int(evt.index)
    if row_idx < 0 or row_idx >= len(rows):
        return None, "Selected row is out of range.", ""
    selected = _extract_row(rows, row_idx)
    image_path = selected[3] if len(selected) > 3 else ""
    if not image_path:
        return None, "No image path stored for this row.", ""
    p = Path(image_path)
    if not p.exists():
        return None, f"Image path not found on disk: {image_path}", image_path
    try:
        preview_img = _load_preview_image(p)
    except Exception as e:
        return None, f"Could not render preview for {selected[1]}: {e}", image_path
    return preview_img, f"Previewing: {selected[1]} ({selected[0]})", image_path


def reveal_in_finder(selected_path: str) -> str:
    p = Path((selected_path or "").strip())
    if not selected_path:
        return "Select a row first, then click Reveal in Finder."
    if not p.exists():
        return f"Cannot reveal missing file: {selected_path}"
    try:
        subprocess.run(["open", "-R", str(p)], check=True)
    except Exception as e:
        return f"Reveal failed: {e}"
    return f"Revealed in Finder: {p.name}"


def open_messages_app() -> str:
    try:
        subprocess.run(["open", "-a", "Messages"], check=True)
    except Exception as e:
        return f"Could not open Messages.app: {e}"
    return "Opened Messages.app — search there for the matching conversation."


def clear_search_cache() -> str:
    try:
        if _CACHE_PATH.exists():
            _CACHE_PATH.unlink()
        return "Last search: cache cleared. Run a new search to repopulate."
    except Exception as e:
        return f"Failed to clear cache: {e}"


def _maybe_wipe_cache(should_wipe: bool) -> None:
    """Silently delete the on-disk search cache before a fresh search.
    Used by the 'Always run fresh' UI toggle. No outputs."""
    if not should_wipe:
        return
    try:
        if _CACHE_PATH.exists():
            _CACHE_PATH.unlink()
    except Exception:
        pass


def clear_search_outputs() -> tuple[str, list, None, str, str, str, str, list[Any], list[str]]:
    return "", [], None, "Running search...", "", "Last search: running...", "Running search...", [], []


def on_gallery_select(evt: gr.SelectData, gallery_paths: list[str]):
    if evt is None or evt.index is None:
        return None, "Select a thumbnail to preview.", ""
    idx = int(evt.index)
    if idx < 0 or idx >= len(gallery_paths):
        return None, "Selected thumbnail is out of range.", ""
    p = Path(gallery_paths[idx])
    if not p.exists():
        return None, f"Image path not found on disk: {p}", str(p)
    try:
        img = _load_preview_image(p)
    except Exception as e:
        return None, f"Could not render selected thumbnail: {e}", str(p)
    return img, f"Previewing selected hit: {p.name}", str(p)


def load_alias_json() -> tuple[str, str]:
    if not _SYNONYMS_PATH.exists():
        return "{}", f"Alias file missing; expected at `{_SYNONYMS_PATH}`"
    try:
        text = _SYNONYMS_PATH.read_text(encoding="utf-8")
    except Exception as e:
        return "{}", f"Failed to read aliases: {e}"
    return text, f"Loaded aliases from `{_SYNONYMS_PATH}`"


def save_alias_json(raw_json: str) -> str:
    text = (raw_json or "").strip()
    if not text:
        return "Alias JSON is empty; nothing saved."
    try:
        obj = json.loads(text)
    except Exception as e:
        return f"Invalid JSON: {e}"
    if not isinstance(obj, dict):
        return "Alias JSON must be an object/dictionary at top level."

    normalized: dict[str, list[str]] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        if k.startswith("_"):
            # keep comments/meta keys untouched if they are strings/lists
            normalized[k] = v if isinstance(v, list) else [str(v)]
            continue
        if not isinstance(v, list):
            return f"Alias value for '{k}' must be a list of strings."
        vals = [str(x).strip().lower() for x in v if str(x).strip()]
        if vals:
            normalized[k.strip().lower()] = vals

    _SYNONYMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _SYNONYMS_PATH.write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as e:
        return f"Failed to save aliases: {e}"

    # Make updated aliases visible immediately in this long-running process.
    reset_synonym_cache()
    return f"Saved aliases to `{_SYNONYMS_PATH}` (changes active immediately)."


def _parse_alias_json(raw_json: str) -> tuple[dict, str | None]:
    text = (raw_json or "").strip() or "{}"
    try:
        obj = json.loads(text)
    except Exception as e:
        return {}, f"Invalid JSON: {e}"
    if not isinstance(obj, dict):
        return {}, "Alias JSON must be an object/dictionary."
    return obj, None


def upsert_alias_entry(raw_json: str, canonical: str, aliases_csv: str) -> tuple[str, str]:
    obj, err = _parse_alias_json(raw_json)
    if err:
        return raw_json, err
    key = (canonical or "").strip().lower()
    if not key:
        return raw_json, "Canonical term is required."

    aliases: list[str] = []
    for chunk in (aliases_csv or "").split(","):
        v = chunk.strip().lower()
        if v:
            aliases.append(v)
    if not aliases:
        return raw_json, "Provide at least one alias (comma-separated)."

    existing = obj.get(key)
    merged: list[str] = []
    if isinstance(existing, list):
        merged.extend(str(x).strip().lower() for x in existing if str(x).strip())
    for a in aliases:
        if a not in merged:
            merged.append(a)
    obj[key] = merged
    new_text = json.dumps(obj, indent=2, ensure_ascii=False)
    return new_text, f"Updated alias entry `{key}` ({len(merged)} alias values). Click Save aliases to persist."


def remove_alias_entry(raw_json: str, canonical: str) -> tuple[str, str]:
    obj, err = _parse_alias_json(raw_json)
    if err:
        return raw_json, err
    key = (canonical or "").strip().lower()
    if not key:
        return raw_json, "Canonical term is required."
    if key not in obj:
        return raw_json, f"No alias entry named `{key}` found."
    del obj[key]
    new_text = json.dumps(obj, indent=2, ensure_ascii=False)
    return new_text, f"Removed alias entry `{key}`. Click Save aliases to persist."


def build_app(
    *,
    db_path: Path,
    top_k: int,
    qa_model: str,
    qa_model_small: str,
    auto_route: bool,
    auto_correct: bool,
    installed_models: list[str],
) -> gr.Blocks:
    installed = ", ".join(installed_models) if installed_models else "(could not detect)"
    version = _ui_version_stamp()
    custom_css = """
    /* Constrain readable content width and increase typography size,
       Google-style narrow column. Inputs/dataframes stay full width. */
    .gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
    #pi-answer, #pi-hits { max-width: 720px; }
    #pi-answer p, #pi-hits p, #pi-answer li, #pi-hits li {
        font-size: 1.08rem;
        line-height: 1.55;
    }
    #pi-answer h1, #pi-answer h2, #pi-answer h3,
    #pi-hits h1, #pi-hits h2, #pi-hits h3 {
        font-size: 1.25rem;
        margin-top: 0.6em;
    }
    #pi-answer code, #pi-hits code { font-size: 0.95rem; }
    #pi-stats { color: #555; font-size: 0.95rem; }
    """
    with gr.Blocks(title="Personal Index Search", css=custom_css) as demo:
        gr.Markdown("## Personal Index Search (Gemma + SQLite FTS)")
        gr.Markdown(f"UI version: `{version}`")
        with gr.Accordion("App config / model info", open=False):
            gr.Markdown(
                f"Using DB: `{db_path}`  \nLarge model: `{qa_model}`  \nSmall model: `{qa_model_small}`  \nTop-K retrieval: `{top_k}`  \nAuto-route: `{auto_route}`  \nAuto-correct: `{auto_correct}`"
            )
            gr.Markdown(
                "Routing reference: short/factual queries -> small model, broad/ambiguous queries -> large model, "
                "and low-confidence small-model responses auto-retry on large model."
            )
            gr.Markdown(
                f"Installed Ollama models detected: `{installed}`  \n"
                "Recommended small-model candidates: `gemma4:latest`, `qwen2.5:3b` (if installed)."
            )
        with gr.Accordion("Alias Manager (synonyms.json)", open=False):
            canonical = gr.Textbox(
                label="Canonical term",
                placeholder="e.g. new york times",
            )
            aliases_csv = gr.Textbox(
                label="Aliases (comma-separated)",
                placeholder="e.g. nyt, nytimes, nytimes.com, ny times",
            )
            with gr.Row():
                alias_upsert_btn = gr.Button("Add/Update alias entry")
                alias_remove_btn = gr.Button("Remove alias entry")
            alias_json = gr.Textbox(
                label="Alias JSON",
                lines=12,
                placeholder='{\n  "new york times": ["nyt", "nytimes", "ny times"]\n}',
            )
            alias_status = gr.Markdown("Edit aliases and click Save.")
            with gr.Row():
                alias_load_btn = gr.Button("Load aliases")
                alias_save_btn = gr.Button("Save aliases")
        recheck_btn = gr.Button("Re-check with 26b only")
        stats = gr.Markdown("Last search: n/a", elem_id="pi-stats")

        question = gr.Textbox(
            label="Ask about your data (photos, messages, etc.)",
            placeholder="e.g. latest NY Times subscription charge from my messages",
            lines=2,
            elem_id="photo-query-input",
        )
        with gr.Row():
            sort_choice = gr.Radio(
                choices=list(SORT_OPTIONS),
                value=SORT_RELEVANT,
                label="Sort hits by",
                info="Most Relevant uses entity/keyword scoring. Most Recent ignores ranking and sorts by date.",
            )
            ask = gr.Button("Search", elem_id="photo-search-btn")
        restrict_finance_cb = gr.Checkbox(
            value=True,
            label="Restrict finance answers to bank/credit-card statements",
            info="When ON, money/subscription queries ignore casual chat and only use bank or credit-card transaction messages (Capital One, Chase, Apple Cash, etc.).",
        )
        always_fresh_cb = gr.Checkbox(
            value=False,
            label="Always run fresh (clear cache on every new search)",
            info="When ON, each new search wipes the 24h cache before running so you always see fresh retrieval. Slower for repeat queries. Does NOT affect chat context (each search is independent). The manual 'Clear search cache' button is still available.",
        )

        answer = gr.Markdown(label="Answer", elem_id="pi-answer")
        hit_summary = gr.Markdown("No hits yet.", elem_id="pi-hits")
        hits = gr.Dataframe(
            label="Retrieved index rows",
            headers=["uuid", "filename", "date_iso", "image_path_used", "rank", "ocr_excerpt", "vlm_excerpt"],
            datatype=["str", "str", "str", "str", "str", "str", "str"],
            wrap=True,
        )
        preview = gr.Image(label="Selected result preview")
        hit_gallery = gr.Gallery(label="Hit Thumbnails", columns=4, height=260, object_fit="contain")
        hit_gallery_paths = gr.State([])
        preview_note = gr.Markdown("Select a result row to preview the image.")
        selected_path = gr.Textbox(label="Selected image path", interactive=False)
        with gr.Row():
            reveal_btn = gr.Button("Reveal in Finder")
            open_messages_btn = gr.Button("Open Messages.app")
            clear_cache_btn = gr.Button("Clear search cache")

        search_event = ask.click(
            fn=_maybe_wipe_cache,
            inputs=[always_fresh_cb],
            outputs=[],
            queue=False,
        ).then(
            fn=clear_search_outputs,
            outputs=[answer, hits, preview, preview_note, selected_path, stats, hit_summary, hit_gallery, hit_gallery_paths],
            queue=False,
        ).then(
            fn=lambda q, s, rf: answer_question(
                q,
                db_path=db_path,
                top_k=top_k,
                qa_model=qa_model,
                qa_model_small=qa_model_small,
                auto_route=auto_route,
                auto_correct=auto_correct,
                sort_by=s,
                restrict_finance=bool(rf),
            ),
            inputs=[question, sort_choice, restrict_finance_cb],
            outputs=[answer, hits, stats, hit_summary, hit_gallery, hit_gallery_paths],
        )
        question.submit(
            fn=_maybe_wipe_cache,
            inputs=[always_fresh_cb],
            outputs=[],
            queue=False,
        ).then(
            fn=clear_search_outputs,
            outputs=[answer, hits, preview, preview_note, selected_path, stats, hit_summary, hit_gallery, hit_gallery_paths],
            queue=False,
        ).then(
            fn=lambda q, s, rf: answer_question(
                q,
                db_path=db_path,
                top_k=top_k,
                qa_model=qa_model,
                qa_model_small=qa_model_small,
                auto_route=auto_route,
                auto_correct=auto_correct,
                sort_by=s,
                restrict_finance=bool(rf),
            ),
            inputs=[question, sort_choice, restrict_finance_cb],
            outputs=[answer, hits, stats, hit_summary, hit_gallery, hit_gallery_paths],
        )
        hits.select(
            fn=preview_selected,
            inputs=[hits],
            outputs=[preview, preview_note, selected_path],
        )
        reveal_btn.click(fn=reveal_in_finder, inputs=[selected_path], outputs=[preview_note])
        open_messages_btn.click(fn=open_messages_app, outputs=[preview_note])
        clear_cache_btn.click(fn=clear_search_cache, outputs=[stats])
        recheck_btn.click(
            fn=_maybe_wipe_cache,
            inputs=[always_fresh_cb],
            outputs=[],
            queue=False,
        ).then(
            fn=clear_search_outputs,
            outputs=[answer, hits, preview, preview_note, selected_path, stats, hit_summary, hit_gallery, hit_gallery_paths],
            queue=False,
        ).then(
            fn=lambda q, s, rf: recheck_with_large_only(
                q,
                db_path=db_path,
                top_k=top_k,
                qa_model=qa_model,
                qa_model_small=qa_model_small,
                sort_by=s,
                restrict_finance=bool(rf),
            ),
            inputs=[question, sort_choice, restrict_finance_cb],
            outputs=[answer, hits, stats, hit_summary, hit_gallery, hit_gallery_paths],
        )
        hit_gallery.select(
            fn=on_gallery_select,
            inputs=[hit_gallery_paths],
            outputs=[preview, preview_note, selected_path],
        )
        alias_upsert_btn.click(
            fn=upsert_alias_entry,
            inputs=[alias_json, canonical, aliases_csv],
            outputs=[alias_json, alias_status],
        )
        alias_remove_btn.click(
            fn=remove_alias_entry,
            inputs=[alias_json, canonical],
            outputs=[alias_json, alias_status],
        )
        alias_load_btn.click(fn=load_alias_json, outputs=[alias_json, alias_status])
        alias_save_btn.click(fn=save_alias_json, inputs=[alias_json], outputs=[alias_status])

        # Inject keyboard binding (Enter-to-search) on every page load. This used
        # to live on `Blocks.launch(js=...)`, but we now bypass `launch()` to
        # mount Gradio onto our own FastAPI app (so /open-local-file works), and
        # `Blocks.__init__` doesn't take `js`. `load(js=...)` runs the same JS.
        demo.load(fn=lambda: None, js=_KEYBOARD_JS)

    return demo


def _open_local_file_handler(path: str) -> Response:
    """Open ``path`` in the OS default app (macOS ``open`` / Linux ``xdg-open``).

    The hit summary in the UI links to ``/open-local-file?path=<encoded>``.
    Browsers won't navigate from a localhost HTTP page directly to ``file://``
    URIs, so we route the click through this server-side handler. Returns
    HTTP 204 No Content on success so the browser stays on the current page
    while the file pops open in its native app.
    """
    if not path:
        raise HTTPException(status_code=400, detail="missing path")
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail=f"not a regular file: {path}")
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        elif sys.platform == "win32":
            os.startfile(str(p))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(p)])
    except Exception as e:  # pragma: no cover - depends on host environment
        raise HTTPException(status_code=500, detail=f"open failed: {e}")
    return Response(status_code=204)


def _find_free_port(host: str, start: int, attempts: int = 10) -> int:
    """Probe ports starting at ``start``, return the first that's bindable."""
    last_err: OSError | None = None
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError as e:
                last_err = e
                continue
    raise OSError(
        f"No free port in {start}..{start + attempts - 1} on {host}: {last_err}"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run local Gradio UI for photo index search.")
    p.add_argument("--db", default=str(_DEFAULT_DB), help="SQLite database path")
    p.add_argument(
        "--qa-model",
        default=os.environ.get("PHOTO_INDEX_QA_MODEL", "gemma4:26b"),
        help="Primary/large Ollama model used for answer generation.",
    )
    p.add_argument(
        "--qa-model-small",
        default=os.environ.get("PHOTO_INDEX_QA_MODEL_SMALL", "gemma4:latest"),
        help="Smaller/faster Ollama model used by auto-routing.",
    )
    p.add_argument("--top-k", type=int, default=15, help="How many retrieved rows to send to Gemma.")
    p.add_argument("--host", default="127.0.0.1", help="Host to bind (default localhost).")
    p.add_argument("--port", type=int, default=7860, help="Port to bind.")
    p.add_argument(
        "--no-auto-route",
        action="store_true",
        help="Disable model auto-routing; always use --qa-model.",
    )
    p.add_argument(
        "--no-auto-correct",
        action="store_true",
        help="Disable typo auto-correct suggestion when a query returns no results.",
    )
    args = p.parse_args(argv)

    db_path = Path(os.path.abspath(args.db))
    installed_models = _installed_ollama_models()
    blocks = build_app(
        db_path=db_path,
        top_k=args.top_k,
        qa_model=args.qa_model,
        qa_model_small=args.qa_model_small,
        auto_route=not args.no_auto_route,
        auto_correct=not args.no_auto_correct,
        installed_models=installed_models,
    )
    # Build our own FastAPI app so we can register /open-local-file (the
    # server-side helper that backs the "Open local file" hit links) alongside
    # the mounted Gradio routes. Gradio's own launch() doesn't expose a way to
    # add arbitrary HTTP routes, so we bypass it.
    api_app = FastAPI(title="photo-index")
    api_app.add_api_route(
        "/open-local-file",
        _open_local_file_handler,
        methods=["GET"],
        include_in_schema=False,
    )
    port = _find_free_port(args.host, args.port, attempts=10)
    gr.mount_gradio_app(api_app, blocks, path="", server_name=args.host, server_port=port)
    uvicorn.run(api_app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
