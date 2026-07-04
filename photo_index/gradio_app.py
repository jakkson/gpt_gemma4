"""Local Gradio UI for querying the photo index with Gemma."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
import os
import re
import secrets
import socket
import sqlite3
import subprocess
import sys
import threading
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
from PIL import Image

from photo_index.llm_client import (
    chat_completion_text,
    chat_user_prompt,
    embed_query,
    inference_opts_for_model,
    is_big_qa_model,
    list_llm_models,
    llm_backend,
)
from photo_index.ollama_image import image_path_for_ollama
from photo_index.query_expand import expand_query_terms, reset_synonym_cache

from photo_index.store import (
    connect,
    count_embedded_rows,
    init_schema,
    load_embedding_matrix_cached,
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
# In-process cache of the embedding matrix (loaded from the .npy sidecar). On TTL
# expiry we re-check the embedded-row count (a cheap COUNT) and only rebuild when
# it changed, so a running backfill / new ingest is picked up without the old
# multi-second full-table rescan on every refresh.
_EMB_CACHE: dict[str, Any] = {"uuids": None, "mat": None, "loaded_at": 0.0, "count": -1}
_EMB_CACHE_TTL = float(os.environ.get("PHOTO_INDEX_EMB_CACHE_TTL", "180"))
_SEMANTIC_ENABLED = os.environ.get("PHOTO_INDEX_SEMANTIC", "1").strip().lower() not in ("0", "false", "no")
# Weight applied to cosine similarity (~0.3-0.8) when blending into the rank score.
_SEMANTIC_WEIGHT = float(os.environ.get("PHOTO_INDEX_SEMANTIC_WEIGHT", "25"))
# Cross-encoder rerank: a precision stage over the merged candidate pool. ON by
# default for testing — set PHOTO_INDEX_RERANK=0 to A/B back to the old ranking.
_RERANK_ENABLED = os.environ.get("PHOTO_INDEX_RERANK", "1").strip().lower() not in ("0", "false", "no")
# Weight on the normalized [0,1] cross-encoder score (comparable to _SEMANTIC_WEIGHT).
_RERANK_WEIGHT = float(os.environ.get("PHOTO_INDEX_RERANK_WEIGHT", "20"))
# Cap candidates sent to the cross-encoder to bound latency (pre-trimmed by sem/bm25).
_RERANK_MAX_CANDIDATES = int(os.environ.get("PHOTO_INDEX_RERANK_MAX_CANDIDATES", "100"))
_PAGE_LOAD_JS = """
() => {
  if (!window.__photoSearchEnterBound) {
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
  if (!window.__photoOpenLocalBound) {
    window.__photoOpenLocalBound = true;
    document.addEventListener(
      "click",
      (ev) => {
        const t = ev.target && ev.target.closest && ev.target.closest(".pi-open-local-file");
        if (!t) return;
        ev.preventDefault();
        ev.stopPropagation();
        ev.stopImmediatePropagation();
        const u = t.getAttribute("data-open-href");
        if (!u) return;
        fetch(u, {
          method: "GET",
          credentials: "same-origin",
          cache: "no-store",
          priority: "low",
        })
          .then((r) => {
            if (!r.ok) {
              alert(
                "Could not open file (HTTP " +
                  r.status +
                  "). File missing or no permission."
              );
            }
          })
          .catch(() => {
            alert("Could not reach server to open file.");
          });
      },
      true
    );
  }
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


def _build_prompt(
    question: str,
    rows: list[sqlite3.Row],
    *,
    aggregate: bool = False,
    scope_month: tuple[int, int] | None = None,
    field_char_cap: int | None = None,
    conversational: bool = False,
) -> str:
    blocks = [row_to_prompt_block(r, field_char_cap=field_char_cap) for r in rows]
    context = "\n\n---\n\n".join(blocks)
    if conversational:
        style_block = """
TONE & STYLE (important)
- Write like a helpful assistant talking with the user — not like a search
  engine returning results. Address them as "you".
- Lead with the direct answer in the first sentence, then weave supporting
  details into short, natural paragraphs. Avoid bullet-point dumps unless the
  user explicitly asks for a list, table, or breakdown.
- Refer to sources conversationally in-sentence — "your Capital One alert from
  June 21", "an email from Fidelity in January 2019" — the exact records are
  already listed below your answer, so never paste raw uuids, filenames, or
  file paths into the prose.
- If it feels natural, close with one short follow-up offer (e.g. "Want me to
  break that down by month?"). Keep it to one sentence.
"""
        cite_rule = (
            "- Attribute each fact to its source conversationally (who sent it and "
            "when); do NOT quote raw uuids or filenames in the answer text."
        )
    else:
        style_block = ""
        cite_rule = "- Cite each record you use inline by its filename or imsg uuid."
    month_scope = ""
    if scope_month:
        label = _month_label(scope_month)
        month_scope = f"""
DATE SCOPE (critical — read before answering)
- The user asked specifically about **{label}**.
- Include ONLY records whose date falls in that calendar month ({label}).
- If none of the records below are dated {label}, reply: "No charges or payments
  dated {label} are in your indexed data yet." Do NOT list or summarize
  transactions from any other month or year.
"""
    if aggregate:
        return f"""You are answering questions about a single user's personal on-device index
(their own photos, OCR, VLM captions, and SMS/iMessage text).

{_LOCAL_INDEX_POLICY}
GROUND RULES
- Use ONLY the indexed records below. Do NOT use outside / general knowledge.
- Quote exact dollar amounts and dates from the records when relevant.
{cite_rule}
{month_scope}{style_block}
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
  dollar amount and date(s) directly from the records.
{cite_rule}
- You MAY add up, count, or compare amounts that are visible in the records.
- Prefer the most recent matching record when the user asks about "latest",
  "currently", or "right now".
{month_scope}{style_block}
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


def _query_contains_term(q: str, term: str) -> bool:
    """Match multi-word phrases as substrings; single words use word boundaries
    so e.g. ``sum`` does not match ``summary``."""
    if " " in term:
        return term in q
    return bool(re.search(rf"\b{re.escape(term)}\b", q))


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
        "summary of all", "all charges", "all payments", "tally",
    )
    return any(_query_contains_term(q, t) for t in finance) and any(
        _query_contains_term(q, t) for t in aggregate
    )


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
    if "filename" in t or "imsg:" in t or "m365:" in t or "doc:" in t or "indexed" in t:
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


def _prompt_field_cap_for_model(model: str) -> int | None:
    if is_big_qa_model(model):
        return int(os.environ.get("PHOTO_INDEX_PROMPT_FIELD_CHARS_BIG", "900"))
    return None


def _safe_chat(*, model: str, prompt: str) -> tuple[str, str | None]:
    opts = inference_opts_for_model(model)
    try:
        return (
            chat_user_prompt(
                model=model,
                prompt=prompt,
                timeout=float(opts["timeout"]),
                max_tokens=int(opts["max_tokens"]),
                stream=bool(opts["stream"]),
            ),
            None,
        )
    except Exception as e:
        return "", str(e)


# --- Follow-up conversation ---------------------------------------------------
#
# A search is a one-shot RAG lookup. To let the user reply to the answer ("yes,
# break it down") without re-searching for the literal words, we stash the last
# search's retrieved records + Q/A here, and the follow-up chat reuses those same
# records as context (no fresh retrieval) plus the running conversation history.
_LAST_CONVO: dict[str, Any] = {"context": "", "turns": []}


def _set_last_convo(context: str, question: str, answer: str) -> None:
    _LAST_CONVO["context"] = context or ""
    _LAST_CONVO["turns"] = [(question, answer)] if (question or answer) else []


def _seed_chat_from_last() -> list:
    """Seed the Chatbot with the last search's Q/A so follow-ups continue it."""
    return list(_LAST_CONVO.get("turns") or [])


def _chat_system_prompt(context: str) -> str:
    return (
        "You are having a running conversation with the user about their own "
        "on-device personal index (their photos, OCR, captions, messages, email, "
        "and notes).\n\n"
        f"{_LOCAL_INDEX_POLICY}"
        "GROUND RULES\n"
        "- Use ONLY the indexed records below plus what has already been said in "
        "this conversation. Do not use outside general knowledge.\n"
        "- Answer in a natural, conversational tone; refer to sources in-sentence "
        "and never paste raw uuids or filenames.\n"
        "- If the user asks for something the records and prior turns don't "
        "support, say so briefly rather than inventing it.\n\n"
        f"Indexed records:\n{context}\n"
    )


def _safe_chat_messages(*, model: str, messages: list[dict]) -> tuple[str, str | None]:
    opts = inference_opts_for_model(model)
    try:
        return (
            chat_completion_text(
                model=model,
                messages=messages,
                timeout=float(opts["timeout"]),
                max_tokens=int(opts["max_tokens"]),
                stream=False,
            ),
            None,
        )
    except Exception as e:
        return "", str(e)


def chat_follow_up(user_msg: str, history: list, model: str):
    """Continue the conversation about the last search's records. Returns
    (updated_history, cleared_input)."""
    user_msg = (user_msg or "").strip()
    history = list(history or [])
    if not user_msg:
        return history, ""
    context = str(_LAST_CONVO.get("context") or "")
    if not context:
        history.append((user_msg, "Run a search first — then I can answer "
                                  "follow-ups about that result."))
        return history, ""
    messages = [{"role": "system", "content": _chat_system_prompt(context)}]
    for u, a in history:
        messages.append({"role": "user", "content": str(u)})
        messages.append({"role": "assistant", "content": str(a)})
    messages.append({"role": "user", "content": user_msg})
    reply, err = _safe_chat_messages(model=model, messages=messages)
    if not reply:
        reply = f"(follow-up failed: {err})" if err else "(no response)"
    history.append((user_msg, reply))
    return history, ""


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
    "price", "charge", "charged", "charges", "payment", "payments",
    "bill", "billing", "bills", "fee", "fees",
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
    return bool(q) and any(_query_contains_term(q, t) for t in _FINANCE_TRIGGER_TERMS)


def _is_finance_hit_row(r: sqlite3.Row, *, restrict_finance: bool) -> bool:
    """Whether a row belongs in a finance/charge/payment query result set."""
    if restrict_finance:
        return _is_transaction_row(r)
    uid = str(r["uuid"] or "")
    blob = f"{r['filename'] or ''} {r['ocr_text'] or ''} {r['vlm_text'] or ''}"
    if uid.startswith(("imsg:", "m365:")):
        return _is_transaction_text(blob)
    # Documents/photos stay strict even when the checkbox is off — a colonoscopy
    # prep PDF or ticket image is not a charge.
    return _is_transaction_row(r)


# Real money amount: must carry a '$' sign (or explicit USD). A bare "3.14" or
# version string must NOT count, or long documents match as "charges".
_MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d{1,2})?|\busd\s?\d", re.I)

# Phrases that mark an actual transaction/billing event (not just any text that
# happens to mention money).
_TXN_PHRASES = (
    "charged", "charge of", "was charged", "transaction", "withdrawn",
    "debited", "auto-pay", "autopay", "payment of", "amount due", "amount of",
    "balance is", "new statement", "statement for", "posted", "purchase of",
    "you paid", "was placed on", "placed on your", "pay $", "due $",
    "charge on", "charged to", "bill of", "billed",
)

_SUBSCRIPTION_PHRASES = (
    "subscription", "recurring charge", "recurring payment", "/month",
    "per month", "monthly charge", "auto-renew", "auto renew", "renews on",
    "renewal of", "membership to", "membership for", "invoice", "receipt for",
)


def _is_transaction_text(text: str) -> bool:
    """Strict: text contains a real $ amount AND a transaction/subscription/issuer
    signal. Tight enough that ordinary documents do not register as charges."""
    t = (text or "").lower()
    if not _MONEY_RE.search(t):
        return False
    if any(w in t for w in _BANK_ISSUERS):
        return True
    if any(w in t for w in _TXN_PHRASES):
        return True
    return any(w in t for w in _SUBSCRIPTION_PHRASES)


# Filename hints that a document really is a statement/receipt (not just a long
# document that happens to mention a bank or a dollar figure).
_DOC_STATEMENT_HINTS = (
    "statement", "invoice", "receipt", "1099", "remittance", "e-statement",
    "estatement", "billing", "transactions", "account summary", "paystub",
    "pay stub", "payslip",
)


def _is_transaction_row(r: sqlite3.Row) -> bool:
    """Authoritative charge/payment record. Messages/mail qualify on a strict
    transaction-text match. Documents/photos must both look like a bank source
    (issuer + transaction word + currency) AND carry a statement-like filename,
    so articles, summaries, and tax worksheets don't leak into a spending tally."""
    uid = str(r["uuid"] or "")
    blob = f"{r['filename'] or ''} {r['ocr_text'] or ''} {r['vlm_text'] or ''}"
    if uid.startswith(("imsg:", "m365:")):
        return _is_transaction_text(blob)
    if not _is_bank_source(blob):
        return False
    fn = (r["filename"] or "").lower()
    return any(h in fn for h in _DOC_STATEMENT_HINTS)


# Stopwords filtered out of token overlap so generic words ("the", "what")
# don't inflate noisy hits over targeted ones.
_OVERLAP_STOP = frozenset({
    "the", "and", "for", "you", "what", "this", "that", "with", "your",
    "are", "was", "were", "from", "have", "has", "not", "but", "all",
    "any", "how", "much", "i'm", "i am", "now", "just", "did", "spend",
    "spent", "tally", "total", "many",
})


def _query_token_overlap(text: str, ql: str) -> int:
    t = (text or "").lower()
    n = 0
    for tok in re.findall(r"[a-z0-9'.-]+", ql or ""):
        if len(tok) < 3 or tok in _OVERLAP_STOP:
            continue
        if tok in t:
            n += 1
    return n


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _query_month_year(question: str) -> tuple[int, int] | None:
    """Extract an explicit (year, month) like 'april 2026' for date-scoped sort."""
    ql = (question or "").lower()
    ym = re.search(r"\b(20\d{2})\b", ql)
    if not ym:
        return None
    year = int(ym.group(1))
    for name, num in _MONTHS.items():
        if re.search(rf"\b{name}\b", ql):
            return (year, num)
    return None


_MONTH_NAMES = (
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _month_label(year_month: tuple[int, int]) -> str:
    year, month = year_month
    return f"{_MONTH_NAMES[month]} {year}"


def _finance_empty_message(question: str, *, restrict_finance: bool) -> str:
    scoped_my = _query_month_year(question)
    if scoped_my:
        label = _month_label(scoped_my)
        if restrict_finance:
            return (
                f"No bank or credit-card charges/payments dated {label} are in your "
                f"index. (Finance filter is ON — only transaction records count.) "
                f"Try another month or run a fresh messages ingest if you expect a "
                f"charge that isn't indexed yet."
            )
        return (
            f"No charges or payments dated {label} are in your index. "
            f"Try another month or run a fresh messages ingest if you expect "
            f"records that aren't indexed yet."
        )
    if restrict_finance:
        return (
            "No bank or credit-card transaction records matched. The finance filter is "
            "ON, so casual mentions and documents are excluded — turn it off to "
            "search more broadly, or run a fresh messages ingest."
        )
    return "No matches in index yet. Keep ingest running, then try again."


# Words stripped before FTS / substring retrieval. Natural questions like
# "find photos of Paris" built an FTS AND over find + photos + Paris; captions
# rarely contain the word "photos", so every real Paris hit was dropped.
_FTS_RETRIEVAL_BOILERPLATE = frozenset({
    "find", "show", "list", "display", "search", "look", "looking", "locate",
    "get", "give", "want", "need", "please", "tell", "help", "me", "my",
    "some", "any", "all", "also", "really", "actually", "just", "like",
    "photo", "photos", "photograph", "photographs", "picture", "pictures",
    "pic", "pics", "image", "images", "snapshot", "snapshots", "shot", "shots",
    "gallery", "album", "camera", "jpeg", "jpg", "png", "heic",
    # Glue words still picked up by our tokenizer:
    "of", "to", "in", "at", "on", "for", "with", "from", "about", "near",
    "around", "into", "that", "those", "these", "this",
    # Temporal fillers that match OCR captions ("most recent update…") and bury SMS hits:
    "most", "recent", "recently", "latest", "last", "newest",
})


def _slim_question_for_retrieval(question: str) -> str:
    """Drop command/filler tokens so FTS/substring search targets entities."""
    raw = re.findall(r"[\w'.-]+", (question or "").lower())
    kept = [w for w in raw if w not in _FTS_RETRIEVAL_BOILERPLATE and len(w) >= 2]
    if not kept:
        return (question or "").strip()
    return " ".join(kept)


_IMAGE_PATH_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".webp",
    ".ico", ".dng", ".cr2", ".nef", ".arw",
)


def _row_is_photo_library_or_image_file(r: sqlite3.Row) -> bool:
    """True for Photos-ingest rows or other indexed rows backed by a raster path."""
    uid = str(r["uuid"] or "")
    if uid.startswith(("doc:", "imsg:", "m365:")):
        return False
    path = (
        (r["image_path_used"] or "")
        + "\n"
        + (r["filename"] or "")
    ).lower()
    return any(path.endswith(ext) for ext in _IMAGE_PATH_SUFFIXES)


_VISUAL_QUERY_TERMS = (
    "photo", "photos", "photograph", "picture", "pictures", "image", "images",
    "snapshot", "snapshots", "shot", "shots", "gallery", "album",
)

_MESSAGE_DISCOVERY_RE = re.compile(
    r"\b(?:sms|imessage)\b|\btext\s+messages?\b|\bmessages?\b|\btexts?\b|"
    r"\bemail\b|\boutlook\b|\bmailbox\b|\bexchange\b|\bm365\b",
    re.I,
)


def _message_discovery_query(question: str) -> bool:
    """True for SMS/iMessage *or* Outlook-style mail discovery queries."""
    q = (question or "").strip()
    if not q:
        return False
    return bool(_MESSAGE_DISCOVERY_RE.search(q.lower()))


def _merge_imessage_like_tokens(
    conn: sqlite3.Connection,
    merged: dict[str, sqlite3.Row],
    token_sources: list[str],
    candidate_limit: int,
) -> None:
    """Broaden ``merged`` with ``imsg:`` / ``m365:`` rows matching query tokens."""
    seen: set[str] = set()
    for src in token_sources:
        for tok in re.findall(r"[a-z0-9'.-]+", (src or "").lower()):
            if len(tok) < 3 or tok in seen:
                continue
            seen.add(tok)
            like = f"%{tok}%"
            msg_hits = conn.execute(
                """
                SELECT *, 0 AS rank
                FROM photo_meta
                WHERE (uuid LIKE 'imsg:%' OR uuid LIKE 'm365:%')
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


def _get_embedding_matrix(conn: sqlite3.Connection, db_path: Path):
    """Return cached (uuids, normalized matrix). (None, None) if no embeddings.

    Within the TTL the in-process matrix is returned directly. After the TTL we do
    a cheap COUNT to detect new vectors; only a changed count triggers a reload
    (from the .npy sidecar — no full-table scan unless the sidecar is stale)."""
    now = time.time()
    if (
        _EMB_CACHE["mat"] is not None
        and (now - _EMB_CACHE["loaded_at"]) < _EMB_CACHE_TTL
    ):
        return _EMB_CACHE["uuids"], _EMB_CACHE["mat"]

    try:
        count = count_embedded_rows(conn)
    except Exception:
        count = -1
    if _EMB_CACHE["mat"] is not None and count == _EMB_CACHE["count"]:
        # Nothing new since last load; keep the matrix, reset the TTL clock.
        _EMB_CACHE["loaded_at"] = now
        return _EMB_CACHE["uuids"], _EMB_CACHE["mat"]

    try:
        uuids, mat = load_embedding_matrix_cached(conn, db_path)
    except Exception:
        uuids, mat = [], None
    _EMB_CACHE["uuids"] = uuids
    _EMB_CACHE["mat"] = mat
    _EMB_CACHE["loaded_at"] = now
    _EMB_CACHE["count"] = count
    return uuids, mat


def _semantic_scores(conn: sqlite3.Connection, question: str, k: int, db_path: Path) -> dict[str, float]:
    """Cosine-similarity scores for the top-``k`` rows nearest the query embedding.

    Returns {} when semantic search is disabled, no embeddings exist yet, or the
    embedder is unreachable — callers then fall back to keyword retrieval only.
    """
    if not _SEMANTIC_ENABLED or k <= 0:
        return {}
    uuids, mat = _get_embedding_matrix(conn, db_path)
    if mat is None or not uuids:
        return {}
    try:
        import numpy as np

        qv = embed_query(question)
        if not qv:
            return {}
        q = np.asarray(qv, dtype=np.float32)
        n = np.linalg.norm(q)
        if n == 0:
            return {}
        q = q / n
        sims = mat @ q  # matrix is pre-normalized → cosine similarity
        k = min(k, sims.shape[0])
        top_idx = np.argpartition(-sims, k - 1)[:k]
        return {uuids[i]: float(sims[i]) for i in top_idx}
    except Exception:
        return {}


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
        rq = _slim_question_for_retrieval(question)
        if len(rq.strip()) < 2:
            rq = question

        ql = (question or "").lower()
        msg_disc = _message_discovery_query(question)
        finance_terms = (
            "price", "charge", "charged", "payment", "bill", "billing",
            "fee", "subscription", "cost",
            "pay", "paying", "paid", "amount", "money", "spend", "spending",
            "owe", "due", "rate", "monthly", "per month",
        )
        wants_messages = (
            "text", "message", "sms", "imessage", "capital one", "bank", "statement",
            "email", "outlook", "mailbox",
        )
        boost_messages = any(t in ql for t in finance_terms + wants_messages)
        wants_money = any(t in ql for t in finance_terms)
        wants_visual = any(t in ql for t in _VISUAL_QUERY_TERMS)
        # Photo-discovery: user asked for photos/images and it isn't a
        # message/mail or finance lookup. Seed recent photo-library rows
        # directly (FTS can't find them: "photo"/"recent" are stripped as
        # boilerplate and Photos rows store captions, not the word "photo").
        photo_disc = wants_visual and not msg_disc and not wants_money

        merged: dict[str, sqlite3.Row] = {}
        candidate_limit = max(top_k * 4, 40)
        for expanded in expand_query_terms(rq):
            try:
                hits = search_meta(conn, expanded, limit=candidate_limit)
            except sqlite3.OperationalError:
                hits = search_meta_fallback_substring(conn, expanded, limit=candidate_limit)
            for r in hits:
                merged[r["uuid"]] = r
        # Always merge fallback token-like matches too; FTS may return sparse/weak
        # results for natural-language questions.
        for expanded in expand_query_terms(rq):
            hits = search_meta_fallback_substring(conn, expanded, limit=candidate_limit)
            for r in hits:
                merged[r["uuid"]] = r
            if len(merged) >= max(top_k * 6, 80):
                break

        _finance_trigger = (
            "price", "charge", "charged", "payment", "bill", "billing", "fee",
            "subscription", "cost", "pay", "paying", "paid", "amount", "money",
            "spend", "spending", "owe", "due", "rate", "monthly", "per month",
        )
        if any(t in ql for t in _finance_trigger):
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
                WHERE (uuid LIKE 'imsg:%' OR uuid LIKE 'm365:%')
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

        # Message-discovery queries must always pull ``imsg:`` candidates: substring
        # FTS matches words like "text" inside OCR blobs from unrelated PDFs, and
        # ``Most Recent`` sorting used to float those above real texts.
        if msg_disc:
            seed_lim = max(candidate_limit, min(300, top_k * 25))
            seed_rows = conn.execute(
                """
                SELECT *, 0 AS rank
                FROM photo_meta
                WHERE uuid LIKE 'imsg:%' OR uuid LIKE 'm365:%'
                ORDER BY date_iso DESC, ingested_at DESC
                LIMIT ?
                """,
                (seed_lim,),
            ).fetchall()
            for r in seed_rows:
                merged[r["uuid"]] = r

        # Photo-discovery seeding: pull recent Photos-library / image-file rows
        # directly by date so visual queries have real photo candidates to rank
        # (mirrors the message-discovery seed above).
        if photo_disc:
            seed_lim = max(candidate_limit, min(300, top_k * 25))
            photo_seed = conn.execute(
                """
                SELECT *, 0 AS rank
                FROM photo_meta
                WHERE uuid NOT LIKE 'doc:%'
                  AND uuid NOT LIKE 'imsg:%'
                  AND uuid NOT LIKE 'm365:%'
                ORDER BY date_iso DESC, ingested_at DESC
                LIMIT ?
                """,
                (seed_lim,),
            ).fetchall()
            for r in photo_seed:
                merged[r["uuid"]] = r

        if boost_messages or msg_disc:
            src_order = list(dict.fromkeys([*(expand_query_terms(rq)), rq, question]))
            _merge_imessage_like_tokens(conn, merged, src_order, candidate_limit)

        # Semantic (embedding) candidates: meaning-based matches keyword FTS misses
        # (e.g. "Paris" finding a caption that only says "Eiffel Tower"). Uses the
        # full question for richer embedding. Empty {} if embeddings/embedder absent.
        sem_scores = _semantic_scores(conn, question, max(top_k * 4, 40), db_path)
        missing = [u for u in sem_scores if u not in merged]
        for i in range(0, len(missing), 400):
            chunk = missing[i : i + 400]
            ph = ",".join("?" * len(chunk))
            for r in conn.execute(
                f"SELECT *, 0 AS rank FROM photo_meta WHERE uuid IN ({ph})", chunk
            ):
                merged[r["uuid"]] = r

        rows = list(merged.values())

        query_my = _query_month_year(question)
        query_month_prefix = f"{query_my[0]:04d}-{query_my[1]:02d}" if query_my else None
        is_fin = _is_finance_query(question)

        # Finance restrict checkbox: filter to transaction/bank rows only.
        # Explicit month/year in the question ALWAYS hard-scopes results, even
        # when the checkbox is off — otherwise "May 2026" queries leak Jan/Feb.
        if is_fin:
            rows = [r for r in rows if _is_finance_hit_row(r, restrict_finance=restrict_finance)]
            if query_month_prefix:
                rows = [
                    r for r in rows
                    if str(r["date_iso"] or "").startswith(query_month_prefix)
                ]

        wants_nyt = any(t in ql for t in ("ny times", "nytimes", "nyt", "new york times"))

        # Cross-encoder rerank (relevance mode only): re-score the surviving
        # candidates by reading each (question, row) pair jointly, then blend the
        # normalized score into `score()` below. Best-effort — {} on any failure.
        rerank_map: dict[str, float] = {}
        if _RERANK_ENABLED and rows and sort_by == SORT_RELEVANT:
            def _bm25(r: sqlite3.Row) -> float:
                return float(r["rank"]) if ("rank" in r.keys() and r["rank"] is not None) else 0.0

            cand = rows
            if len(cand) > _RERANK_MAX_CANDIDATES:
                cand = sorted(
                    cand,
                    key=lambda r: (sem_scores.get(str(r["uuid"]), 0.0), -_bm25(r)),
                    reverse=True,
                )[:_RERANK_MAX_CANDIDATES]
            items = [
                (
                    str(r["uuid"]),
                    f"{r['filename'] or ''}\n{r['ocr_text'] or ''}\n{r['vlm_text'] or ''}",
                )
                for r in cand
            ]
            try:
                from photo_index.rerank import rerank_scores

                rerank_map = rerank_scores(question, items)
            except Exception:
                rerank_map = {}

        def score(r: sqlite3.Row) -> tuple[int, float, float, str]:
            uid = str(r["uuid"] or "")
            is_imsg = uid.startswith("imsg:")
            is_m365 = uid.startswith("m365:")
            is_chat_mail = is_imsg or is_m365
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
            is_billing_alert = is_imsg and _is_bank_source(text)
            if query_month_prefix and is_fin:
                if str(r["date_iso"] or "").startswith(query_month_prefix):
                    entity_bonus += 25.0
                else:
                    entity_bonus -= 60.0
            if wants_visual and _row_is_photo_library_or_image_file(r):
                entity_bonus += 14.0
            # Down-rank generic documents when the user clearly asked for photos.
            if wants_visual and str(r["uuid"]).startswith("doc:"):
                entity_bonus -= 5.0
            if wants_nyt and has_nyt:
                entity_bonus += 6.0
            if "subscription" in ql and "subscription" in text:
                entity_bonus += 2.0
            if wants_money and has_currency:
                entity_bonus += 3.0
            # Strong combo: query is finance-y AND record is a message that
            # mentions both the entity and an actual dollar figure. This is the
            # canonical "what am I paying for X?" hit.
            if wants_money and is_chat_mail and has_currency and (not wants_nyt or has_nyt):
                entity_bonus += 8.0
            # Billing alerts dominate when the question is about money. They
            # are the highest-quality evidence by a wide margin.
            if wants_money and is_billing_alert:
                entity_bonus += 12.0
            if msg_disc:
                if is_chat_mail:
                    entity_bonus += 14.0
                elif uid.startswith("doc:"):
                    entity_bonus -= 12.0
            # Semantic similarity: strong signal for meaning-based relevance.
            entity_bonus += sem_scores.get(uid, 0.0) * _SEMANTIC_WEIGHT
            # Cross-encoder rerank: highest-precision relevance signal when present.
            entity_bonus += rerank_map.get(uid, 0.0) * _RERANK_WEIGHT
            msg_pref = 1 if (boost_messages and is_chat_mail) else 0
            date_key = str(r["date_iso"] or "")
            # Tuple sorted desc: prefer chat/mail rows, then higher overlap+bonus,
            # then better (lower) bm25 rank, then most-recent date_iso lex sort.
            return (msg_pref, overlap + entity_bonus, -rank, date_key)

        if sort_by == SORT_RECENT:
            month_year = _query_month_year(question)
            month_prefix = (
                f"{month_year[0]:04d}-{month_year[1]:02d}" if month_year else None
            )

            def recency_tuple(r: sqlite3.Row) -> tuple[str, str]:
                return (str(r["date_iso"] or ""), str(r["ingested_at"] or ""))

            def month_recency_key(r: sqlite3.Row) -> tuple[int, str, str]:
                in_month = (
                    1
                    if month_prefix and str(r["date_iso"] or "").startswith(month_prefix)
                    else 0
                )
                return (in_month, str(r["date_iso"] or ""), str(r["ingested_at"] or ""))

            if msg_disc:
                msgs_only = [
                    r
                    for r in rows
                    if str(r["uuid"]).startswith("imsg:") or str(r["uuid"]).startswith("m365:")
                ]
                non_msg = [
                    r
                    for r in rows
                    if not (
                        str(r["uuid"]).startswith("imsg:")
                        or str(r["uuid"]).startswith("m365:")
                    )
                ]
                msgs_only.sort(key=recency_tuple, reverse=True)
                non_msg.sort(key=recency_tuple, reverse=True)
                rows = msgs_only[:top_k]
                if len(rows) < top_k:
                    rows.extend(non_msg[: top_k - len(rows)])
            elif photo_disc:
                photos_only = [r for r in rows if _row_is_photo_library_or_image_file(r)]
                non_photo = [r for r in rows if not _row_is_photo_library_or_image_file(r)]
                photos_only.sort(key=recency_tuple, reverse=True)
                non_photo.sort(key=recency_tuple, reverse=True)
                rows = photos_only[:top_k]
                if len(rows) < top_k:
                    rows.extend(non_photo[: top_k - len(rows)])
            elif _is_finance_query(question):
                # Finance queries: surface real transaction/billing rows first,
                # date-sorted (and within the asked-for month, if any), so a tally
                # doesn't get buried by an unrelated recent document. When a month
                # was named, the restrict block already scoped rows to it.
                fin = [r for r in rows if _is_finance_hit_row(r, restrict_finance=restrict_finance)]
                fin_ids = {str(r["uuid"]) for r in fin}
                other = [r for r in rows if str(r["uuid"]) not in fin_ids]
                fin.sort(key=month_recency_key, reverse=True)
                other.sort(key=recency_tuple, reverse=True)
                rows = fin[:top_k]
                if len(rows) < top_k and not month_prefix:
                    rows.extend(other[: top_k - len(rows)])
            else:
                # Keep rows that actually match the query ahead of unrelated recent
                # rows, so a recent-but-irrelevant document can't top the list.
                # A strong semantic match counts as "matched" even with no keyword overlap.
                matched = [r for r in rows if (
                    _query_token_overlap(
                        f"{r['filename'] or ''} {r['ocr_text'] or ''} {r['vlm_text'] or ''}", ql
                    ) > 0
                    or sem_scores.get(str(r["uuid"]), 0.0) >= 0.5
                )]
                matched_ids = {str(r["uuid"]) for r in matched}
                unmatched = [r for r in rows if str(r["uuid"]) not in matched_ids]
                matched.sort(key=month_recency_key, reverse=True)
                unmatched.sort(key=recency_tuple, reverse=True)
                rows = matched[:top_k]
                if len(rows) < top_k:
                    rows.extend(unmatched[: top_k - len(rows)])
        else:
            rows.sort(key=score, reverse=True)
        return rows[:top_k]
    finally:
        conn.close()


def _safe_https_browser_url(url: str) -> str | None:
    """Allow only https URLs for links embedded in hit summaries (e.g. Graph webLink)."""
    u = (url or "").strip()
    if not u.startswith("https://"):
        return None
    try:
        parts = urllib.parse.urlparse(u)
        if parts.scheme != "https" or not parts.netloc:
            return None
        return u
    except Exception:
        return None


def _rows_preview(rows: list[sqlite3.Row]) -> list[list[str]]:
    preview: list[list[str]] = []
    for r in rows:
        ocr = (r["ocr_text"] or "").replace("\n", " ").strip()
        vlm = (r["vlm_text"] or "").replace("\n", " ").strip()
        ou = ""
        if "open_url" in r.keys():
            ou = str(r["open_url"] or "").strip()
        preview.append(
            [
                r["uuid"] or "",
                r["filename"] or "",
                r["date_iso"] or "",
                r["image_path_used"] or "",
                f"{r['rank']:.3f}" if "rank" in r.keys() else "0.000",
                ocr[:180] + ("..." if len(ocr) > 180 else ""),
                vlm[:180] + ("..." if len(vlm) > 180 else ""),
                ou,
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
        open_url = r[7] if len(r) > 7 else ""
        is_msg = str(uuid).startswith("imsg:")
        is_m365 = str(uuid).startswith("m365:")
        is_mail = str(uuid).startswith("mail:")
        is_evernote = str(uuid).startswith("evernote:")
        is_doc = str(uuid).startswith("doc:")
        source = (
            "Messages"
            if is_msg
            else "Outlook / Microsoft 365"
            if is_m365
            else "Apple Mail"
            if is_mail
            else "Evernote"
            if is_evernote
            else "Document"
            if is_doc
            else "Photos / Local file"
        )
        snippet = ocr_excerpt or vlm_excerpt or "(no snippet)"
        title = filename or uuid
        when = _format_local_dt(date_iso) or "n/a"
        # "Reference attachment": route open via /open-local-file (fetch + capture-
        # phase handler in _PAGE_LOAD_JS). Buttons avoid hash navigation from href="#".
        web_open = _safe_https_browser_url(open_url)
        if image_path and not image_path.startswith("https://"):
            encoded = urllib.parse.quote(image_path, safe="")
            req_path = f"/open-local-file?path={encoded}&t={_OPEN_LOCAL_FILE_TOKEN}"
            data_attr = html.escape(req_path, quote=True)
            link_md = (
                '<button type="button" class="pi-open-local-file" '
                f'data-open-href="{data_attr}">Open local file</button>'
            )
            ref = f"`{image_path}`"
        elif is_msg:
            link_md = "Use **Open Messages.app** below to jump to your texts"
            ref = f"`{uuid}`"
        elif is_m365 and web_open:
            esc = html.escape(web_open, quote=True)
            link_md = (
                f'<a href="{esc}" target="_blank" rel="noopener noreferrer">'
                "Open in Outlook (web)</a>"
            )
            ref = f"`{uuid}`"
        elif is_m365:
            link_md = (
                "No saved Outlook link for this row — run mail ingest again to refresh. "
                "Meanwhile open [**Outlook on the web**](https://outlook.office.com/mail/)."
            )
            ref = f"`{uuid}`"
        else:
            link_md = "(no local link)"
            ref = f"`{uuid}`"
        # Escape indexed text (email subjects, note titles are external input)
        # so it can't smuggle HTML into the rendered hit summary.
        parts.append(
            f"**{i}. {html.escape(str(title))}**  \n"
            f"_{source} • {when}_  \n"
            f"{html.escape(str(snippet))}  \n"
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
    conversational: bool = False,
) -> str:
    q = " ".join((question or "").strip().lower().split())
    # Bind to UI version so any code change (which alters the file hash) auto-
    # invalidates previously-cached answers from older retrieval/ranking logic.
    version = _ui_version_stamp()
    return (
        f"{q}|{db_path}|{top_k}|{qa_model}|{qa_model_small}"
        f"|auto={int(auto_route)}|sort={sort_by}"
        f"|rf={int(restrict_finance)}|conv={int(conversational)}|v={version}"
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
    conversational: bool = True,
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
        conversational=conversational,
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
                f"route={route}, sort={sort_by}, finance_cb={int(restrict_finance)}"
            )
            hit_md = _rows_to_hit_summary(rows)
            gallery, gallery_paths = _rows_to_gallery(rows)
            # Seed follow-up chat from the cached records too.
            _set_last_convo(str(cached.get("context", "")), q, answer)
            return answer, rows, stats, hit_md, gallery, gallery_paths

    aggregate_mode = _is_aggregate_finance_query(q)
    scoped_my = _query_month_year(q)
    # Broad aggregate pulls need more rows; month-scoped queries stay at top_k
    # (often zero hits — skip sending 40 rows to a slow 32B model).
    effective_top_k = max(top_k, 40) if aggregate_mode and not scoped_my else top_k

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
        if _is_finance_query(q):
            no_match_msg = _finance_empty_message(q, restrict_finance=restrict_finance)
        else:
            no_match_msg = "No matches in index yet. Keep ingest running, then try again."
        return (
            no_match_msg,
            [],
            f"Last search: no matches, total retrieval time {elapsed:.2f}s, top-k={top_k}, sort={sort_by}, finance_cb={int(restrict_finance)}",
            "No hits yet.",
            [],
            [],
        )

    route = "direct"
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
            first_model = qa_model
            route = "large_default"

    prompt = _build_prompt(
        effective_query,
        rows,
        aggregate=aggregate_mode,
        scope_month=scoped_my,
        field_char_cap=_prompt_field_cap_for_model(first_model),
        conversational=conversational,
    )

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

    # Stash the retrieved records + this Q/A so follow-up chat can continue
    # without re-retrieving. Same field cap as the answer prompt used.
    cap = _prompt_field_cap_for_model(first_model)
    context_block = "\n\n---\n\n".join(row_to_prompt_block(r, field_char_cap=cap) for r in rows)
    _set_last_convo(context_block, effective_query, answer)

    cache[key] = {
        "cached_at_unix": now,
        "answer": answer,
        "rows": preview_rows,
        "used_model": used_model,
        "route": route,
        "context": context_block,
    }
    _save_cache(_CACHE_PATH, cache)

    elapsed = time.perf_counter() - t0
    stats = (
        f"Last search: cache miss, total retrieval time {elapsed:.2f}s, "
        f"top-k={top_k}, model={used_model}, route={route}, sort={sort_by}, "
        f"finance_cb={int(restrict_finance)}"
    )
    hit_md = _rows_to_hit_summary(preview_rows)
    gallery, gallery_paths = _rows_to_gallery(preview_rows)
    return answer, preview_rows, stats, hit_md, gallery, gallery_paths


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
    uuid = selected[0] if len(selected) > 0 else ""
    image_path = selected[3] if len(selected) > 3 else ""
    open_url = selected[7] if len(selected) > 7 else ""
    if str(uuid).startswith("m365:"):
        ou = _safe_https_browser_url(open_url)
        if ou:
            esc = html.escape(ou, quote=True)
            return (
                None,
                f"Outlook message — open in browser: <a href=\"{esc}\" "
                'target="_blank" rel="noopener noreferrer">Open in Outlook (web)</a>',
                "",
            )
        return (
            None,
            "Outlook message — no **webLink** stored yet; run `outlook_graph_ingest` again "
            "or search hits above for the generic Outlook link.",
            "",
        )
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


# Ad-hoc document analysis (upload box), independent of the indexed corpus.
_UPLOAD_TEXT_CHAR_CAP = int(os.environ.get("PHOTO_INDEX_UPLOAD_CHAR_CAP", "32000"))


def analyze_uploaded_file(file_path: str | None, question: str, qa_model: str) -> str:
    """Extract text from an uploaded document and answer/summarize it with the LLM.

    Separate from index search: this works only on the one uploaded file, not the
    SQLite corpus. Text documents only (the answer model has no vision/OCR).
    """
    if not file_path:
        return "Drop or browse a document first (PDF, Word, PowerPoint, Excel, text…)."
    from photo_index.documents_ingest import extract_auto

    path = Path(file_path)
    ext = path.suffix.lower()
    try:
        text, method, err = extract_auto(path, ext)
    except Exception as e:  # noqa: BLE001
        return f"Could not read **{path.name}**: {e}"
    if err or not text or not text.strip():
        return (
            f"No extractable text in **{path.name}** ({err or 'empty'}). "
            "Scanned / image-only PDFs need OCR, which this text model can't do."
        )
    text = text.strip()
    truncated = len(text) > _UPLOAD_TEXT_CHAR_CAP
    if truncated:
        text = text[:_UPLOAD_TEXT_CHAR_CAP]
    task = (question or "").strip() or "Summarize this document concisely, capturing the key points."
    prompt = f"""You are analyzing a single document the user uploaded.
Use ONLY the document text below. Do not use outside knowledge.

Document: {path.name}

--- DOCUMENT START ---
{text}
--- DOCUMENT END ---

Task: {task}
"""
    answer, chat_err = _safe_chat(model=qa_model, prompt=prompt)
    if chat_err:
        return f"Model error analyzing **{path.name}**: {chat_err}"
    note = (
        f"\n\n*(Document truncated to {_UPLOAD_TEXT_CHAR_CAP:,} chars to fit the model context.)*"
        if truncated
        else ""
    )
    return f"**{path.name}** · extracted via `{method}`\n\n{answer}{note}"


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
    button.pi-open-local-file {
      background: none;
      border: none;
      padding: 0;
      margin: 0;
      font: inherit;
      color: #2563eb;
      text-decoration: underline;
      cursor: pointer;
      display: inline;
    }
    button.pi-open-local-file:hover { color: #1d4ed8; }
    """
    with gr.Blocks(title="Personal Index Search", css=custom_css) as demo:
        gr.Markdown("## Personal Index Search (local LLM + SQLite FTS)")
        gr.Markdown(f"UI version: `{version}`")
        with gr.Accordion("📄 Summarize / query an uploaded document", open=False):
            gr.Markdown(
                "Drop a file (PDF, Word, PowerPoint, Excel, text) to summarize or ask "
                "about it directly — this is separate from your index. Text documents "
                "only (no scanned/image OCR)."
            )
            upload_file = gr.File(
                label="Drop a document here or browse",
                file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".txt", ".md", ".rtf", ".csv"],
                type="filepath",
            )
            upload_question = gr.Textbox(
                label="What should I do with it?",
                placeholder="Leave blank to summarize, or ask e.g. 'What are the payment terms?'",
                lines=2,
            )
            upload_btn = gr.Button("Analyze document", variant="primary")
            upload_answer = gr.Markdown("No document analyzed yet.", elem_id="pi-answer", sanitize_html=False)
        with gr.Accordion("App config / model info", open=False):
            gr.Markdown(
                f"Using DB: `{db_path}`  \nAnswer model: `{qa_model}`  \nTop-K retrieval: `{top_k}`  \nAuto-correct: `{auto_correct}`  \nRerank (cross-encoder): `{'on' if _RERANK_ENABLED else 'off'}`"
            )
            gr.Markdown(
                f"LLM backend: `{llm_backend()}`  \n"
                f"Models detected on backend: `{installed}`  \n"
                "Retrieval: hybrid semantic (embeddings) + keyword (SQLite FTS). "
                "Vision ingest still uses Ollama. Answers use PHOTO_INDEX_LLM_BACKEND "
                "(ollama or openai for LM Studio)."
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
            answer_style = gr.Radio(
                choices=["Conversational", "Precise (citations)"],
                value="Conversational",
                label="Answer style",
                info=(
                    "Conversational reads like a chat reply — sources woven into "
                    "sentences, no raw uuids. Precise keeps audit-style inline "
                    "citations (best for double-checking money questions)."
                ),
            )
            with gr.Column(scale=0, min_width=140):
                ask = gr.Button("Search", elem_id="photo-search-btn")
                stop_search = gr.Button(
                    "Stop Search",
                    elem_id="photo-stop-search-btn",
                    variant="stop",
                    size="md",
                )
        restrict_finance_cb = gr.Checkbox(
            value=True,
            label="Restrict finance answers to bank/credit-card statements",
            info=(
                "When ON, money/subscription queries ignore casual chat and only use "
                "bank or credit-card transaction messages. When you name a month "
                "(e.g. May 2026), results are always scoped to that month regardless "
                "of this setting."
            ),
        )
        always_fresh_cb = gr.Checkbox(
            value=False,
            label="Always run fresh (clear cache on every new search)",
            info="When ON, each new search wipes the 24h cache before running so you always see fresh retrieval. Slower for repeat queries. Does NOT affect chat context (each search is independent). The manual 'Clear search cache' button is still available.",
        )

        answer = gr.Markdown(
            label="Answer", elem_id="pi-answer", sanitize_html=False
        )

        # Follow-up chat: continue the conversation about the answer above without
        # re-searching. Reuses the records the last search already retrieved.
        with gr.Accordion("Continue the conversation (follow-up questions)", open=True):
            followup_chat = gr.Chatbot(
                label="Follow-up chat", elem_id="pi-followup", height=280,
            )
            with gr.Row():
                followup_box = gr.Textbox(
                    placeholder='Reply to the answer above — e.g. "yes, break down the ingredients"',
                    label="Your follow-up", scale=5, lines=1,
                )
                with gr.Column(scale=0, min_width=110):
                    followup_send = gr.Button("Send", variant="primary")
                    followup_clear = gr.Button("New topic", size="sm")
            gr.Markdown(
                "_Follow-ups reuse the records from your last search. For a brand-new "
                "topic, use the Search box above._",
                elem_id="pi-followup-note",
            )

        hit_summary = gr.Markdown(
            "No hits yet.", elem_id="pi-hits", sanitize_html=False
        )
        hits = gr.Dataframe(
            label="Retrieved index rows",
            headers=[
                "uuid",
                "filename",
                "date_iso",
                "image_path_used",
                "rank",
                "ocr_excerpt",
                "vlm_excerpt",
                "open_url",
            ],
            datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
            wrap=True,
        )
        preview = gr.Image(label="Selected result preview")
        hit_gallery = gr.Gallery(label="Hit Thumbnails", columns=4, height=260, object_fit="contain")
        hit_gallery_paths = gr.State([])
        preview_note = gr.Markdown(
            "Select a result row to preview the image.", sanitize_html=False
        )
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
            fn=lambda q, s, rf, style: answer_question(
                q,
                db_path=db_path,
                top_k=top_k,
                qa_model=qa_model,
                qa_model_small=qa_model_small,
                auto_route=auto_route,
                auto_correct=auto_correct,
                sort_by=s,
                restrict_finance=bool(rf),
                conversational=(style == "Conversational"),
            ),
            inputs=[question, sort_choice, restrict_finance_cb, answer_style],
            outputs=[answer, hits, stats, hit_summary, hit_gallery, hit_gallery_paths],
            queue=True,
        )
        search_event.then(fn=_seed_chat_from_last, outputs=[followup_chat], queue=False)
        submit_event = question.submit(
            fn=_maybe_wipe_cache,
            inputs=[always_fresh_cb],
            outputs=[],
            queue=False,
        ).then(
            fn=clear_search_outputs,
            outputs=[answer, hits, preview, preview_note, selected_path, stats, hit_summary, hit_gallery, hit_gallery_paths],
            queue=False,
        ).then(
            fn=lambda q, s, rf, style: answer_question(
                q,
                db_path=db_path,
                top_k=top_k,
                qa_model=qa_model,
                qa_model_small=qa_model_small,
                auto_route=auto_route,
                auto_correct=auto_correct,
                sort_by=s,
                restrict_finance=bool(rf),
                conversational=(style == "Conversational"),
            ),
            inputs=[question, sort_choice, restrict_finance_cb, answer_style],
            outputs=[answer, hits, stats, hit_summary, hit_gallery, hit_gallery_paths],
            queue=True,
        )
        submit_event.then(fn=_seed_chat_from_last, outputs=[followup_chat], queue=False)
        stop_search.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[search_event, submit_event],
            queue=False,
        )
        followup_send.click(
            fn=lambda msg, hist: chat_follow_up(msg, hist, qa_model),
            inputs=[followup_box, followup_chat],
            outputs=[followup_chat, followup_box],
            queue=True,
        )
        followup_box.submit(
            fn=lambda msg, hist: chat_follow_up(msg, hist, qa_model),
            inputs=[followup_box, followup_chat],
            outputs=[followup_chat, followup_box],
            queue=True,
        )
        followup_clear.click(fn=lambda: [], outputs=[followup_chat], queue=False)
        hits.select(
            fn=preview_selected,
            inputs=[hits],
            outputs=[preview, preview_note, selected_path],
        )
        reveal_btn.click(fn=reveal_in_finder, inputs=[selected_path], outputs=[preview_note])
        open_messages_btn.click(fn=open_messages_app, outputs=[preview_note])
        clear_cache_btn.click(fn=clear_search_cache, outputs=[stats])
        upload_btn.click(
            fn=lambda f, q: analyze_uploaded_file(f, q, qa_model),
            inputs=[upload_file, upload_question],
            outputs=[upload_answer],
            queue=True,
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

        # Inject Enter-to-search and delegated "open local file" clicks on load.
        demo.load(fn=lambda: None, js=_PAGE_LOAD_JS)

    return demo


def _spawn_open_default_app(path: Path) -> None:
    """Background worker: spawn OS open after HTTP response (Friendlier to tunnels)."""

    def _run() -> None:
        try:
            if sys.platform == "darwin":
                # ``open -g`` avoids activating the app — foreground switches were
                # disrupting screen sharing / remote browser tabs for several minutes.
                fg = os.environ.get("PHOTO_INDEX_OPEN_FOREGROUND", "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                )
                cmd = ["open", str(path)] if fg else ["open", "-g", str(path)]
                subprocess.Popen(cmd)
            elif sys.platform == "win32":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as e:  # pragma: no cover
            print(f"photo_index.gradio_app: open failed for {path}: {e}", file=sys.stderr)

    threading.Thread(target=_run, daemon=True).start()


# Per-run CSRF token for /open-local-file. Without it, any web page the user
# visits could fire a simple GET at 127.0.0.1:7860 (no CORS preflight applies)
# and make this process `open` arbitrary local files. The token is embedded in
# the served page's links, so only our own UI can construct valid requests.
_OPEN_LOCAL_FILE_TOKEN = secrets.token_urlsafe(16)
# Set in main(); used to verify requested paths are actually indexed files.
_OPEN_LOCAL_FILE_DB: Path | None = None


def _path_is_indexed(path: str) -> bool:
    """True if ``path`` is stored as an indexed file (image_path_used) in the DB."""
    if _OPEN_LOCAL_FILE_DB is None:
        return False
    try:
        conn = sqlite3.connect(f"file:{_OPEN_LOCAL_FILE_DB}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT 1 FROM photo_meta WHERE image_path_used = ? LIMIT 1", (path,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except Exception:
        return False


def _open_local_file_handler(path: str, t: str = "") -> Response:
    """Open ``path`` in the OS default app (macOS ``open`` / Linux ``xdg-open``).

    Invoked via ``fetch()`` from hit-summary controls (delegated click handler in
    ``_PAGE_LOAD_JS``) so the Gradio SPA never navigates. The handler returns
    immediately after scheduling ``open`` so proxies/tunnels finish the request
    before any foreground UI churn.

    Security: requires the per-run token AND the path to be a file this index
    actually ingested — both checks stop cross-site "open anything" requests.
    """
    if not secrets.compare_digest(t or "", _OPEN_LOCAL_FILE_TOKEN):
        raise HTTPException(status_code=403, detail="bad or missing token")
    if not path:
        raise HTTPException(status_code=400, detail="missing path")
    if not _path_is_indexed(path):
        raise HTTPException(status_code=403, detail="path is not an indexed file")
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail=f"not a regular file: {path}")
    _spawn_open_default_app(p)
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
        help="Primary/large model for answers (Ollama tag or LM Studio model id).",
    )
    p.add_argument(
        "--qa-model-small",
        default=os.environ.get("PHOTO_INDEX_QA_MODEL_SMALL", "gemma4:latest"),
        help="Smaller/faster model for auto-routing (only used when auto-route is on).",
    )
    p.add_argument("--top-k", type=int, default=15, help="How many retrieved rows to send to the model.")
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
    global _OPEN_LOCAL_FILE_DB
    _OPEN_LOCAL_FILE_DB = db_path  # lets /open-local-file verify requested paths
    installed_models = list_llm_models()
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
