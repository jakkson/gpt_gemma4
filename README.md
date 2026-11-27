# gpt-local-gemma

A fully local, private retrieval-augmented assistant over your own Apple data.
It indexes **Photos, Messages, Apple Mail, Documents & books, Apple Notes, Calendar,
and browser history** into a single SQLite index, then answers questions and
summarizes documents with a **local LLM** — nothing leaves the machine.

Everything runs on-device: a local LLM serves answers, a local embedding model
powers semantic search, and a local vision model captions images during ingest.

---

## Architecture at a glance

| Piece | What it is | Runs on |
|---|---|---|
| **Answer model** | Qwen3-30B-A3B (instruct) via **LM Studio** (OpenAI-compatible API on `:1234`) | local |
| **Embeddings** | `nomic-embed-text-v1.5` (768-dim) for semantic search | local (LM Studio / Ollama) |
| **Reranker** | `bge-reranker-base` cross-encoder (precision stage, MPS on Apple Silicon) | local |
| **Vision (ingest only)** | Ollama VLM + Apple Vision OCR for images/PDFs | local |
| **Index** | SQLite + FTS5 (external-content) with an embedding BLOB column | `data/photo_index.sqlite` |
| **UI** | Gradio web app on `:7860` | local |

**Retrieval is hybrid:** nomic embeddings (meaning) blended with FTS5 BM25
(keywords), then bge reranking, plus a self-correcting query-rewrite pass and
follow-up chat that reuses the prior answer's records.

**Source tags** — every indexed row is prefixed by origin so results stay
attributable and de-collide: `imsg:` (Messages), `mail:` (Apple Mail),
`doc:` (Documents/books), `notes:` (Apple Notes), `cal:` (Calendar), `hist:` (browser
history), `web:` (scraped pages); Photos rows carry the raw Photos UUID.

### Optional two-machine setup

This stack is designed to run on one Mac, but also supports a **source-of-truth
+ daily-driver** split:

- A always-on Mac (e.g. a Mac mini) holds the **canonical index** and runs the
  nightly ingest, mail rules, and the search app.
- A higher-RAM Mac (e.g. a MacBook Pro) is the **daily driver** for answers,
  pulling a fresh copy of the DB on demand with `scripts/sync_db_from_mini.sh`.

Remote access is left to you (SSH / Tailscale / etc.).

---

## First-time setup

1. **Virtualenv + deps**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **LM Studio (answers + embeddings)**
   - Install [LM Studio](https://lmstudio.ai/) and download a **Qwen3-30B-A3B instruct** build
     (MLX on Apple Silicon is fastest) plus the **nomic-embed-text-v1.5** embedding model.
   - Load the answer model, then start the server: **Developer/Local Server → Start** (`http://127.0.0.1:1234`).
   - ⚠️ **Load the answer model at 16384+ context.** If it JIT-loads at the 4096 default,
     long prompts are silently truncated and answers confabulate. Verify with `lms ps`.
   - Point the app at it:
     ```bash
     export PHOTO_INDEX_LLM_BACKEND=openai
     export PHOTO_INDEX_LLM_BASE_URL=http://127.0.0.1:1234/v1
     export PHOTO_INDEX_QA_MODEL=<the model id shown in LM Studio>
     export PHOTO_INDEX_QA_MODEL_SMALL=<same id>
     ```

3. **Ollama (vision ingest only)**
   - Install [Ollama](https://ollama.com/) and pull a vision model (e.g. `ollama pull gemma3` or a Qwen-VL).
   - Only used to caption images/PDFs during ingest; answers do not use Ollama.

4. **macOS privacy permissions (required)**
   - Grant **Full Disk Access** to the Python binary running this code
     (`.venv/bin/python`) in **System Settings → Privacy & Security**.
   - Needed to read Apple Photos, Messages `chat.db`, Apple Mail, Calendar, and browser history.
     A background nightly can't click the per-app TCC prompt, so this one-time grant is essential.

5. **Start the search UI**
   ```bash
   ./start_search.sh        # loads the answer model at proper context + launches the app
   ```
   Open <http://127.0.0.1:7860>.

---

## What gets indexed

Each source has its own ingest module. The **nightly** (`scripts/run_nightly_ingest.sh`)
runs them incrementally; you can also run any one by hand.

| Source | Module | Notes |
|---|---|---|
| Photos | `photo_index.ingest` | OCR (Apple Vision) + Ollama VLM captions |
| Messages | `photo_index.messages_ingest` | `chat.db`; handles resolved to contact names |
| Apple Mail | `photo_index.mail_ingest` | `.emlx` across accounts & folders |
| Outlook / M365 | `photo_index.outlook_graph_ingest` | Microsoft Graph, delegated `Mail.Read` (optional) |
| Documents & books | `photo_index.documents_ingest` | PDF/Word/PowerPoint/Excel/RTF/text/epub; long docs chunked into passages |
| Document OCR/VLM | `photo_index.documents_vlm_ingest` | Apple Vision + Ollama VLM on images/PDFs (slow) |
| Apple Notes | `photo_index.notes_ingest` | reads NoteStore directly; creation-date resolved (Evernote-original when known) |
| Calendar | `photo_index.calendar_ingest` | Apple iCal events, calendar-aware retrieval |
| Browser history | `photo_index.history_ingest` | Safari + Chrome (title + URL + last-visit) |
| Web pages | `photo_index.web_ingest` | scrape permitted sites (robots + rate-limit), with a review gap |
| Embedding pass | `photo_index.embed_index` | embeds new/changed rows after text ingest |

Config knobs used by the nightly:
`PHOTO_INDEX_BOOKS_ROOT` (offline library, default `~/LLM_Books`),
`EXTRA_DOC_ROOTS` (extra document folders), and `--exclude` for noisy subfolders.
The primary document root defaults to `~/Dropbox/Documents`.

---

## Answering & the search UI

`python -m photo_index.gradio_app` (or `./start_search.sh`) serves the web UI:

- Hybrid semantic + keyword search with **bge reranking**.
- **Self-correcting retrieval** — if a named anchor is missing from the top rows,
  the LLM rewrites the query and retries (`PHOTO_INDEX_SELF_CORRECT=1`).
- **Follow-up chat** — replies drill into the prior answer's records instead of
  re-searching, unless you introduce a new topic.
- **Upload / summarize box** — drop one or more documents for a one-shot
  summary/analysis. This is **ad-hoc: uploaded files are NOT added to the index.**
  To make a file searchable, put it in an ingest root and let the nightly index it.
- **`about_me.md`** — an optional personal profile in `data/` that is prepended to
  every answer for context.
- Click-to-preview image row and Reveal in Finder.

`python -m photo_index.search_cli "your question"` gives the same retrieval from the CLI.

---

## Nightly ingest, health check, boot start

- **Nightly ingest** — `scripts/run_nightly_ingest.sh`, scheduled via launchd for
  **22:00 nightly (Mon–Sat)**. It takes the search app offline during ingest,
  frees the answer model to avoid GPU memory over-commit, hard-stops on a time
  budget, and auto-restores the app afterward. Install/uninstall:
  ```bash
  ./install_photo_nightly_launchd.sh
  ./uninstall_photo_nightly_launchd.sh
  ```
- **Weekly doctor** — `scripts/doctor.py`, **Sunday 22:00**: DB health report + safe optimizes.
- **Boot auto-start** — `scripts/boot_search.sh` (launchd `RunAtLoad`) brings LM
  Studio + the answer model + the search app back up after a reboot.

Run everything by hand:
```bash
python -m photo_index.nightly                       # all incremental ingests
python -m photo_index.nightly --skip-photos --skip-messages   # docs only
./update_documents.sh            # documents, text only
./update_documents.sh --ocr-vlm  # documents + OCR/VLM (matches nightly)
```

---

## Mail rules & the Mail Rule Dropper

An automation that files or trashes Inbox mail by rule:

- **`data/mail_rule.py`** — moves senders to folders (`data/mail_routes.json`) or
  trashes by keyword (`data/trash_keywords.txt`), driving Mail via AppleScript.
  Folder routes win over trash. (These `data/` files are personal and gitignored.)
- **`scripts/mail_rules_tick.sh`** — one launchd job (every 60s) that runs the
  route/keyword/trump passes **sequentially** with a lock, so Mail is never driven
  by concurrent AppleScript clients (which caused beachballs).
- **`scripts/mail_dropper.py`** — a small GUI: grab/drop a sender, choose Trash or
  a folder, Apply. Adding a rule is instant; the per-minute job files matching mail
  (newest ~50, read or unread) within a minute. Shows existing rules for a sender
  with per-rule Delete buttons.

---

## Other tools

- **`photo_index.web_ingest_gui`** — standalone Gradio front-end for web scraping (`:7862`).
- **`scripts/sync_db_from_mini.sh`** — pull the source-of-truth DB to a daily-driver Mac.
- **`scripts/rechunk_contact_directory.py` / `rechunk_reference_book.py` /
  `normalize_artist_names.py`** — re-chunk reference dumps (address books,
  alphabetized books) into one record per entry for reliable retrieval.
- **`photo_index.prune_index`** — remove rows by source/prefix.

---

## Core data files (all under `data/`, gitignored)

- `photo_index.sqlite` — the index (+ `.emb_mat.npy` / `.emb_uuids.npy` embedding cache)
- `synonyms.json` — user-editable retrieval aliases
- `about_me.md` — personal profile prepended to answers
- `mail_routes.json`, `trash_keywords.txt` — mail-rule config (personal)
- `query_log.jsonl` — durable query log
- `gradio_search_cache.json` — cached answers

---

## Common commands & health checks

```bash
# Row counts by source
sqlite3 data/photo_index.sqlite \
  "SELECT substr(uuid,1,instr(uuid,':')-1) src, COUNT(*) FROM photo_meta \
   WHERE instr(uuid,':')>0 GROUP BY src ORDER BY 2 DESC;"

# Is an ingest running?
pgrep -fl "photo_index" || echo "not running"

# Nightly job + logs
launchctl list | grep com.gptlocalgemma
tail -f data/nightly_ingest.log

# Confirm the answer model's loaded context (must be >= 16384)
lms ps
```

---

## Concurrency safety

Ingest modules share a global lock (`data/content_ingest.lock` via
`photo_index.ingest_lock.global_ingest_lock()`) so photo/doc/message/mail/embedding
jobs never run simultaneously.

---

## Operational notes / gotchas

- **LM Studio context:** always load the answer model at **16384+** (ideally 32K–64K
  for document summarization). The 4096 default silently truncates. Verify with `lms ps`.
- **GPU memory (Apple Silicon unified memory):** the answer model and the vision
  model can over-commit a small machine and trigger a GPU fault. The nightly frees
  the answer model during vision ingest; keep only one large model resident at a time.
- **Full Disk Access** for the venv Python is required for unattended ingest to read
  protected app data; re-grant it if you upgrade Python (the binary path changes).
