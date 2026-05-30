# gpt-local-gemma

Local personal-search stack for Photos + Messages using Ollama/Gemma, SQLite FTS, and Gradio.

## First-Time Setup

1) Create and activate virtualenv:

- `python3 -m venv .venv`
- `source .venv/bin/activate`

2) Install dependencies:

- `pip install -r requirements.txt`

3) LLM for **answers** (pick one):

**Option A — LM Studio (recommended for faster Q&A):**

- Install [LM Studio](https://lmstudio.ai/), download **Qwen2.5-3B-Instruct** (GGUF).
- In LM Studio: load the model → **Local Server** tab → Start server (default `http://127.0.0.1:1234`).
- Set env (add to your shell profile or run before Gradio):

```bash
export PHOTO_INDEX_LLM_BACKEND=openai
export PHOTO_INDEX_LLM_BASE_URL=http://127.0.0.1:1234/v1
export PHOTO_INDEX_QA_MODEL=qwen2.5-3b-instruct
export PHOTO_INDEX_QA_MODEL_SMALL=qwen2.5-3b-instruct
```

Use the exact model id shown in LM Studio’s server UI if the name differs. With one model loaded, any id often works.

**Option B — Ollama only (legacy):**

- `ollama list`
- Example pulls: `ollama pull gemma4:26b`, `ollama pull qwen2.5:3b`
- Leave `PHOTO_INDEX_LLM_BACKEND` unset (defaults to `ollama`).

**Vision captioning during ingest always uses Ollama** (`gemma4` etc.) unless you change ingest separately. A **VL** model in LM Studio (e.g. Qwen2.5-VL) is not used by this repo yet.

4) macOS privacy permissions (required):

- Enable **Full Disk Access** for the app running Python (Cursor/Terminal/iTerm).
- Required for:
  - Apple Photos library access
  - Apple Messages `chat.db` access

5) Start Gradio UI:

**Quickest (LM Studio):** start LM Studio's Developer-tab server, then:

```bash
./start_search.sh          # fast 3B for everything
./start_search.sh --big    # 3B default, escalate to 32B on broad queries
```

Override model ids with env vars if your LM Studio shows different names
(e.g. `PHOTO_INDEX_QA_MODEL`, `PHOTO_INDEX_QA_MODEL_SMALL`).

**Manual equivalent:**

```bash
export PHOTO_INDEX_LLM_BACKEND=openai
export PHOTO_INDEX_LLM_BASE_URL=http://127.0.0.1:1234/v1
export PHOTO_INDEX_QA_MODEL=qwen2.5-vl-3b-instruct
export PHOTO_INDEX_QA_MODEL_SMALL=qwen2.5-vl-3b-instruct
python -m photo_index.gradio_app --no-auto-route --top-k 15
```

(`--no-auto-route` is fine when large and small are the same model.)

**With Ollama (answers):**

- `python -m photo_index.gradio_app --qa-model gemma4:26b --qa-model-small qwen2.5:3b --top-k 15`

Open `http://127.0.0.1:7860`

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
    - "Re-check with big model" button — re-runs the current query on
      `--qa-model-big` (env `PHOTO_INDEX_QA_MODEL_BIG`, default
      `qwen2.5-vl-32b-instruct`) regardless of the launch model. With LM Studio,
      enable Just-In-Time model loading so the big model loads on demand.

- `python -m photo_index.nightly`
  - One-pass runner for launchd: incremental **new photos**, **new messages**, then
    **new/changed documents** under `~/Dropbox/Documents` (text extract, then Apple
    Vision OCR + Ollama VLM on PDF/PNG/JPG). Skips unchanged files by mtime+size; no
    `--force`. Requires **Ollama** for photo and document VLM. Opt out of heavy OCR:
    `--skip-documents-vlm`.

- `python -m photo_index.documents_ingest`
  - Index PDF / Office / text files under a folder (default: `~/Dropbox/Documents`).
  - Skips unchanged files already in the DB (same path + mtime + size). Use `--force`
    to re-read everything.

- `python -m photo_index.documents_vlm_ingest`
  - Apple Vision OCR + Ollama VLM for **images and PDFs** in a folder (slow). Nightly
    runs this after `documents_ingest` unless `--skip-documents-vlm`.

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

- Run Documents ingest now (incremental — new/changed files only):
  - `./update_documents.sh` (text only)
  - `./update_documents.sh --ocr-vlm` (text + OCR/VLM on PDF/PNG/JPG; matches nightly)
  - Or: `python -m photo_index.documents_ingest`
  - Override folder: `python -m photo_index.documents_ingest --root ~/Dropbox/Documents`

- Run all incremental ingests (photos + messages + documents + document OCR/VLM):
  - `python -m photo_index.nightly`
  - Documents only: `python -m photo_index.nightly --skip-photos --skip-messages`
  - Documents text only (no OCR/VLM): add `--skip-documents-vlm`

- Run Outlook / Microsoft 365 mail ingest (Microsoft Graph, delegated **Mail.Read**):
  - Register an app in Entra ID, add redirect `http://localhost`, grant Graph delegated **Mail.Read** (and **offline_access**).
  - Or run **`./scripts/register_entra_personal_photo_index_mail.sh`** after **`az login --allow-no-subscriptions`** (no Azure subscription needed) or **`az login`**, to create **`personal-photo-index-mail`** automatically.
  - `export GRAPH_CLIENT_ID='your-client-id'`
  - `python -m photo_index.outlook_graph_ingest --auth interactive`
  - First run opens a browser to sign in; later runs use `data/graph_mail_token_cache.json`. Incremental sync uses `data/graph_mail_delta.json`.
  - **Azure CLI login crashes** (no Azure subscription; error in **Tenant and subscription selection** such as `'NoneType' object has no attribute 'get'`): run **`brew upgrade azure-cli`**, then **`az login --allow-no-subscriptions --tenant YOUR_TENANT_ID`** (copy **Tenant ID** from **Microsoft Entra admin center → Overview**). Or **`az login --use-device-code --allow-no-subscriptions --tenant YOUR_TENANT_ID`**. If CLI keeps failing, create the app only in the portal (**App registrations → New registration**, name **`personal-photo-index-mail`**, **Mobile and desktop** redirect **`http://localhost`**, API permissions **Mail.Read** + **offline_access**, **Grant admin consent**), then **`export GRAPH_CLIENT_ID='…'`** from **Application (client) ID**.

- Install nightly 2:00 AM photo + messages + documents ingest (text + OCR/VLM; launchd):
  - `./install_photo_nightly_launchd.sh`
  - Also installs a **wake scheduler** (root LaunchDaemon): wakes the Mac at **1:55 AM**
    daily so ingest starts at 2:00 AM. Requires your admin password once.
  - Check wake status: `.venv/bin/python -m photo_index.nightly_wake status`
  - Uninstall: `./uninstall_photo_nightly_launchd.sh`
  - **Laptop:** plug in overnight; closed lid on battery may block wake. **Desktop:** leave
    “Prevent computer from sleeping” or rely on the scheduled wake + `caffeinate` during ingest.

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

- Manually ingest Outlook / Microsoft 365 (after `GRAPH_CLIENT_ID` is set):
  - `python -m photo_index.outlook_graph_ingest --auth interactive`

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

