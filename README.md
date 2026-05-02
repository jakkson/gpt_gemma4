# gpt-local-gemma

Local personal-search stack for Photos + Messages using Ollama/Gemma, SQLite FTS, and Gradio.

## First-Time Setup

1) Create and activate virtualenv:

- `python3 -m venv .venv`
- `source .venv/bin/activate`

2) Install dependencies:

- `pip install -r requirements.txt`

3) Make sure Ollama is running and models are available:

- `ollama list`
- Example pulls:
  - `ollama pull gemma4:26b`
  - `ollama pull qwen2.5:3b`

4) macOS privacy permissions (required):

- Enable **Full Disk Access** for the app running Python (Cursor/Terminal/iTerm).
- Required for:
  - Apple Photos library access
  - Apple Messages `chat.db` access

5) Start Gradio UI:

- `python -m photo_index.gradio_app --qa-model gemma4:26b --qa-model-small qwen2.5:3b --top-k 15`
- Open `http://127.0.0.1:7860`

## What Each Python App Does

- `python -m photo_index.ingest`
  - Main Photos indexer.
  - Reads Apple Photos library items, runs OCR + VLM captioning, writes to `data/photo_index.sqlite`.
  - Skips already indexed UUIDs by default.

- `python -m photo_index.messages_ingest`
  - Messages indexer.
  - Reads `~/Library/Messages/chat.db` text messages and writes them into the same SQLite/FTS index.
  - Uses `imsg:` UUID prefix to avoid collisions with photo UUIDs.

- `python -m photo_index.search_cli "your question"`
  - CLI retrieval + Gemma answer over indexed content.
  - Uses FTS with synonym expansion and fallback substring search.

- `python -m photo_index.gradio_app`
  - Local web UI (`http://127.0.0.1:7860`) for search and Q&A.
  - Includes:
    - click-to-preview image row
    - Reveal in Finder
    - auto-routing (small vs large model + retry-on-low-confidence)
    - typo autocorrect fallback
    - synonym expansion
    - Alias Manager UI for editing `data/synonyms.json`
    - "Re-check with 26b only" button

- `python -m photo_index.nightly`
  - One-pass runner intended for scheduler usage (launchd job).

- `python osxphotos_script.py`
  - Small demo script (first 10 photos) for quick vision checks.
  - Not the full production ingest.

## Core Data Files

- `data/photo_index.sqlite` - main index DB
- `data/photo_index.checkpoint.json` - latest ingest checkpoint status
- `data/synonyms.json` - user-editable aliases for retrieval expansion
- `data/gradio_search_cache.json` - cached Gradio search answers

## Common Commands

- Start Gradio app:
  - `python -m photo_index.gradio_app --qa-model gemma4:26b --qa-model-small qwen2.5:3b --top-k 15`

- Run Photos ingest now:
  - `python -m photo_index.ingest --vlm-model gemma4:26b --progress-every 1`

- Run Messages ingest now:
  - `python -m photo_index.messages_ingest`

- Install nightly 2:00 AM photo ingest (launchd):
  - `./install_photo_nightly_launchd.sh`

## Concurrency Safety

Ingest modules use a shared global lock (`data/content_ingest.lock`) so different ingest jobs (photos/docs/messages/email) do not run at the same time, as long as they use `photo_index.ingest_lock.global_ingest_lock()`.

## Daily Workflow

- Open search UI:
  - `source .venv/bin/activate`
  - `python -m photo_index.gradio_app --qa-model gemma4:26b --qa-model-small qwen2.5:3b --top-k 15`
  - Browse at `http://127.0.0.1:7860`

- Manually ingest new photos:
  - `python -m photo_index.ingest --vlm-model gemma4:26b --progress-every 1`

- Manually ingest new Messages:
  - `python -m photo_index.messages_ingest`

- Quick health checks:
  - Row count:
    - `sqlite3 data/photo_index.sqlite "SELECT COUNT(*) FROM photo_meta;"`
  - Last checkpoint:
    - `cat data/photo_index.checkpoint.json`
  - Ingest running?:
    - `pgrep -fl "photo_index.ingest" || echo "not running"`

- Nightly job status (if installed):
  - `launchctl list | rg com.gptlocalgemma.photoindex.nightly`
  - Logs:
    - `tail -f data/nightly_ingest.log`
    - `tail -f data/nightly_ingest.error.log`

