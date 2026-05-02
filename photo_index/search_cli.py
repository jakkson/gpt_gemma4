"""FTS search + Gemma answers over indexed photo text."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"

from ollama import chat

from photo_index.query_expand import expand_query_terms, reset_synonym_cache
from photo_index.store import (
    connect,
    init_schema,
    row_to_prompt_block,
    search_meta,
    search_meta_fallback_substring,
)


def run_search(
    *,
    db_path,
    question: str,
    top_k: int,
    qa_model: str,
) -> None:
    reset_synonym_cache()
    conn = connect(db_path)
    init_schema(conn)

    rows: list[sqlite3.Row] = []
    merged: dict[str, sqlite3.Row] = {}
    for expanded in expand_query_terms(question):
        try:
            hits = search_meta(conn, expanded, limit=top_k)
        except sqlite3.OperationalError as e:
            print(f"[warn] FTS query issue: {e}; using substring fallback.", file=sys.stderr)
            hits = search_meta_fallback_substring(conn, expanded, limit=top_k)
        for r in hits:
            merged[r["uuid"]] = r
    rows = list(merged.values())[:top_k]

    for expanded in expand_query_terms(question):
        hits = search_meta_fallback_substring(conn, expanded, limit=top_k)
        for r in hits:
            merged[r["uuid"]] = r
        if len(merged) >= max(top_k * 2, top_k):
            break
    rows = list(merged.values())[:top_k]

    if not rows:
        print("No matches in index. Run: python -m photo_index.ingest", file=sys.stderr)
        sys.exit(1)

    blocks = [row_to_prompt_block(r) for r in rows]
    context = "\n\n---\n\n".join(blocks)

    prompt = f"""You are helping search a personal photo library.
The records below are from on-device previews (not necessarily full-resolution originals), plus OCR and short vision descriptions.

Use only this evidence. If unsure, say so.

Indexed records:
{context}

User question: {question}
"""

    response = chat(
        model=qa_model,
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.message.content or "")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Ask Gemma about indexed photos.")
    p.add_argument("question", nargs="+", help="Natural language question")
    p.add_argument(
        "--db",
        type=str,
        default=str(_DEFAULT_DB),
        help="SQLite database path",
    )
    p.add_argument("--top-k", type=int, default=15, help="How many FTS hits to pass to Gemma.")
    p.add_argument(
        "--qa-model",
        default=os.environ.get("PHOTO_INDEX_QA_MODEL", "gemma4:26b"),
        help="Ollama model for answering (default: gemma4:26b or PHOTO_INDEX_QA_MODEL).",
    )
    args = p.parse_args(argv)

    q = " ".join(args.question).strip()
    if not q:
        p.error("question required")

    db_path = Path(os.path.abspath(args.db))
    run_search(db_path=db_path, question=q, top_k=args.top_k, qa_model=args.qa_model)


if __name__ == "__main__":
    main()
