# AGENTS.md — startup guide for any coding session

Read this first in every new chat before making code changes.

## Project
Local personal-search stack: Apple Photos + Messages + documents + Outlook mail,
indexed into SQLite/FTS, searched via a Gradio UI and CLI. Python in `photo_index/`.

## Required workflow for every coding task
1. **Make the change**, then **verify** before claiming done:
   - Syntax check edited Python: `python3 -m py_compile <files>`.
   - Check linter diagnostics on edited files.
2. **Commit + push after each logical batch** (one feature/fix/refactor/docs unit):
   - `git add <files>` (group related files; keep unrelated work in separate commits).
   - `git commit -m "<clear message>"` — concise, explains the "why".
   - `git push origin main` (canonical repo: https://github.com/jakkson/gpt_gemma4.git).
   - Never leave large mixed uncommitted changes when switching tasks.
   - See `.cursor/rules/git-commit-workflow.mdc` (always applied).
3. **Only commit when a batch is complete.** If asked to commit, follow the steps above.

## LLM backends (important)
- **Answers / Q&A** (`gradio_app.py`, `search_cli.py`) go through `photo_index/llm_client.py`.
  - `PHOTO_INDEX_LLM_BACKEND=ollama` (default) or `openai` (LM Studio / llama.cpp).
  - LM Studio: start its Developer-tab server, then set `PHOTO_INDEX_LLM_BASE_URL`,
    `PHOTO_INDEX_QA_MODEL`, `PHOTO_INDEX_QA_MODEL_SMALL`.
- **Vision captioning during ingest** (`ingest.py`, `documents_vlm_ingest.py`,
  `osxphotos_script.py`) still uses **Ollama** directly. Do not assume LM Studio handles images.
- **Nightly** (`nightly.py`): photos → messages → documents text → documents OCR/VLM
  (PDF/PNG/JPG). `--skip-documents-vlm` to skip the heavy pass; needs Ollama running.
- **Nightly wake** (`nightly_wake.py` + root LaunchDaemon from `install_photo_nightly_launchd.sh`):
  schedules `pmset wake` at 1:55 AM; ingest wrapper uses `caffeinate -s`.

## Conventions
- Don't add narration comments; explain only non-obvious intent.
- Prefer editing existing files; avoid creating new docs unless asked.
- Long prompt fields are capped and auto-shrunk on context overflow in `llm_client.py`
  / `store.py` — keep that behavior when touching prompt construction.

## Outlook / Graph mail
- `outlook_graph_ingest.py` uses delegated `Mail.Read` + `User.Read`.
- A `#EXT#` UPN means an invited/external identity; mail APIs often 401 even when
  `/me` works. That is an account/tenant issue, not a code bug.
