#!/usr/bin/env python3
"""Standalone Gradio GUI for web_ingest — paste a URL, pick a mode, review, ingest.

A visual front-end over photo_index.web_ingest: it calls the exact same
fetch/stage/commit functions the CLI uses, so behavior is identical. The review
step (delete/edit staged pages before they enter the DB) becomes point-and-click
instead of managing files in Finder.

Flow in the window:
  1. Paste URL(s), choose a mode (Single / Sitemap / Crawl), set options.
  2. Fetch → each page's extracted text is staged and listed in a table.
  3. Review → untick "Keep" to drop a page; select a page to read/trim its text.
  4. Commit → ingests only the kept pages (optionally embeds right after).

Run:
  PHOTO_INDEX_LLM_BACKEND=openai PHOTO_INDEX_LLM_BASE_URL=http://127.0.0.1:1234/v1 \
      python -m photo_index.web_ingest_gui --host 127.0.0.1 --port 7861
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from . import web_ingest as W

_MODES = ["Single page(s)", "Whole site (sitemap)", "Bounded crawl"]


def _targets_for(mode: str, urls_text: str, max_pages: int, max_depth: int,
                 ignore_robots: bool, delay: float, limit: int) -> list[str]:
    lines = [ln.strip() for ln in (urls_text or "").splitlines() if ln.strip()]
    ns = SimpleNamespace(
        url=lines if mode == _MODES[0] else None,
        urls_file=None,
        sitemap=lines[0] if (mode == _MODES[1] and lines) else None,
        crawl=lines[0] if (mode == _MODES[2] and lines) else None,
        max_pages=int(max_pages), max_depth=int(max_depth),
        timeout=30, user_agent=W._UA, ignore_robots=ignore_robots,
        delay=float(delay), limit=(int(limit) or None),
    )
    return W._gather_targets(ns)


def do_fetch(urls_text, mode, max_pages, max_depth, limit, name, dry_run,
             render, ignore_robots, delay):
    """Fetch + stage. Returns log, table, dropdown, states, preview, reveal, text."""
    logs: list[str] = []
    targets = _targets_for(mode, urls_text, max_pages, max_depth,
                           ignore_robots, delay, limit)
    if not targets:
        return ("No URLs found. Paste at least one URL above.", [],
                gr.update(choices=[], value=None), "", {}, "",
                gr.update(visible=False), "")
    logs.append(f"{len(targets)} target(s). "
                f"{'PREVIEW — nothing will be staged.' if dry_run else ''}")
    logs.append(f"{'#':>4}  {'chars':>7}  status  title")
    dir_ = W._stage_dir(name or "gui", W._STAGING_ROOT)
    robots = W._Robots(W._UA, ignore_robots)
    results = W.stage_targets(
        targets, dir_, robots=robots, ua=W._UA, timeout=30, delay=float(delay),
        render=render, render_empty=False, min_chars=200, dry_run=dry_run,
        log=logs.append)

    if dry_run:
        preview = "\n\n".join(
            f"### {r['title'] or r['url']}\n{r['url']}\n\n{r['text'][:4000]}"
            for r in results if r["text"])
        return ("\n".join(logs), [], gr.update(choices=[], value=None), "", {},
                preview or "(no text extracted)", gr.update(visible=False), "")

    table, idx_to_path, choices = [], {}, []
    for r in results:
        if not r["staged"]:
            continue
        table.append([True, r["idx"], r["chars"], (r["title"] or "")[:80], r["url"]])
        idx_to_path[str(r["idx"])] = str(r["staged"])
        choices.append(f"{r['idx']}: {(r['title'] or r['url'])[:70]}")
    # Auto-load the first staged page so the captured text is visible at once.
    first_text = ""
    if table:
        parsed = W._parse_stage(Path(idx_to_path[str(table[0][1])]))
        first_text = parsed[1] if parsed else ""
    logs.append(f"\nStaged {len(table)} page(s) in {dir_}. The first page's text "
                "is shown below — use the dropdown to read others, untick 'Keep' "
                "to drop a page.")
    return ("\n".join(logs), table,
            gr.update(choices=choices, value=(choices[0] if choices else None)),
            str(dir_), idx_to_path, "", gr.update(visible=bool(table)), first_text)


def load_page(selected, idx_to_path):
    if not selected or not idx_to_path:
        return ""
    idx = selected.split(":", 1)[0].strip()
    path = idx_to_path.get(idx)
    if not path:
        return ""
    parsed = W._parse_stage(Path(path))
    return parsed[1] if parsed else ""


def save_page(selected, new_text, idx_to_path):
    if not selected or not idx_to_path:
        return "Nothing selected."
    idx = selected.split(":", 1)[0].strip()
    path = idx_to_path.get(idx)
    if not path:
        return "Page not found."
    W.rewrite_stage_body(Path(path), new_text)
    return f"Saved edits to page {idx} ({len(new_text)} chars)."


def do_commit(table, embed, run_dir, idx_to_path, db_path):
    if not run_dir:
        return "Fetch some pages first."
    logs: list[str] = []
    dropped = 0
    rows = table or []
    # Gradio may hand back a DataFrame; normalize to list-of-lists.
    if hasattr(rows, "values"):
        rows = rows.values.tolist()
    for row in rows:
        keep = bool(row[0])
        idx = str(int(float(row[1]))) if row[1] not in ("", None) else None
        if not keep and idx and idx in idx_to_path:
            p = Path(idx_to_path[idx])
            if p.exists():
                p.unlink()
                dropped += 1
    if dropped:
        logs.append(f"Dropped {dropped} page(s) you unticked.")
    res = W.commit_staging_dir(Path(run_dir), Path(db_path), min_chars=200,
                               embed=embed, log=logs.append)
    if not embed and res["pages"]:
        logs.append("Not embedded yet. Turn on 'Embed now' (needs LM Studio up) "
                    "or run embed_index later.")
    return "\n".join(logs)


def do_purge(domain, db_path, apply):
    if not (domain or "").strip():
        return "Enter a domain or URL substring (e.g. adcontrarian.blogspot.com)."
    logs: list[str] = []
    W.purge_domain(Path(db_path), domain.strip(), dry_run=not apply, log=logs.append)
    if not apply:
        logs.append("Preview only — nothing deleted. Click **Purge** to remove.")
    return "\n".join(logs)


def build(db_default: str) -> gr.Blocks:
    with gr.Blocks(title="Web Ingest") as demo:
        gr.Markdown("## Web Ingest — scrape permitted sites into your index\n"
                    "Fetch → review (drop/trim pages) → commit. Nothing enters "
                    "the database until you press **Commit kept pages**.")
        run_dir = gr.State("")
        idx_map = gr.State({})

        with gr.Row():
            with gr.Column(scale=2):
                urls = gr.Textbox(
                    label="URL(s)", lines=3, placeholder="https://example.com/post\n"
                    "(one per line for Single mode; the start URL for Sitemap/Crawl)")
            with gr.Column(scale=1):
                mode = gr.Radio(_MODES, value=_MODES[0], label="Mode")
        with gr.Group(visible=False) as crawl_opts:
            with gr.Row():
                max_pages = gr.Number(value=25, label="Max pages", precision=0)
                max_depth = gr.Number(value=2, label="Max depth", precision=0)
        with gr.Row():
            name = gr.Textbox(value="gui", label="Staging run name")
            limit = gr.Number(value=25, precision=0, label="Max pages (0 = all)")
            delay = gr.Slider(0, 5, value=1.5, step=0.5, label="Delay between requests (s)")
        with gr.Row():
            dry_run = gr.Checkbox(label="Preview only (don't stage)")
            render = gr.Checkbox(label="Render JS pages (Playwright)")
            ignore_robots = gr.Checkbox(label="Ignore robots.txt (authorized sites only)")
        fetch_btn = gr.Button("Fetch", variant="primary")

        log = gr.Textbox(label="Log", lines=8, interactive=False)
        preview = gr.Markdown(visible=True)

        with gr.Group(visible=False) as review:
            gr.Markdown("### Review — untick **Keep** to drop a page before commit")
            table = gr.Dataframe(
                headers=["Keep", "#", "Chars", "Title", "URL"],
                datatype=["bool", "number", "number", "str", "str"],
                type="array", interactive=True, label="Staged pages")
            with gr.Row():
                pick = gr.Dropdown(choices=[], label="Read / trim a page", scale=2)
                save_btn = gr.Button("Save edits to this page", scale=1)
            page_text = gr.Textbox(
                label="Captured text — read it here; edit to trim, then Save",
                lines=18)
            save_status = gr.Markdown()
            with gr.Row():
                db_path = gr.Textbox(value=db_default, label="Database", scale=3)
                embed = gr.Checkbox(label="Embed now (needs LM Studio)", value=True, scale=1)
            commit_btn = gr.Button("Commit kept pages", variant="primary")
            commit_log = gr.Textbox(label="Commit result", lines=6, interactive=False)

        with gr.Accordion("Remove ingested web pages (undo)", open=False):
            gr.Markdown("Delete previously-committed pages by domain or URL "
                        "substring. Only affects **web:** pages — never your mail, "
                        "docs, or photos. Preview first to see the count.")
            with gr.Row():
                purge_domain = gr.Textbox(
                    label="Domain / URL substring",
                    placeholder="adcontrarian.blogspot.com", scale=3)
                purge_db = gr.Textbox(value=db_default, label="Database", scale=3)
            with gr.Row():
                preview_btn = gr.Button("Preview (count only)")
                purge_btn = gr.Button("Purge", variant="stop")
            purge_log = gr.Textbox(label="Result", lines=3, interactive=False)

        mode.change(lambda m: gr.update(visible=(m == _MODES[2])),
                    inputs=mode, outputs=crawl_opts)
        fetch_btn.click(
            do_fetch,
            inputs=[urls, mode, max_pages, max_depth, limit, name, dry_run,
                    render, ignore_robots, delay],
            outputs=[log, table, pick, run_dir, idx_map, preview, review,
                     page_text])
        pick.change(load_page, inputs=[pick, idx_map], outputs=page_text)
        save_btn.click(save_page, inputs=[pick, page_text, idx_map],
                       outputs=save_status)
        commit_btn.click(do_commit, inputs=[table, embed, run_dir, idx_map, db_path],
                         outputs=commit_log)
        preview_btn.click(lambda d, db: do_purge(d, db, False),
                          inputs=[purge_domain, purge_db], outputs=purge_log)
        purge_btn.click(lambda d, db: do_purge(d, db, True),
                        inputs=[purge_domain, purge_db], outputs=purge_log)
    return demo


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Web Ingest GUI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--db", default=str(W._DEFAULT_DB))
    a = ap.parse_args(argv)
    build(a.db).launch(server_name=a.host, server_port=a.port, inbrowser=True)


if __name__ == "__main__":
    main()
