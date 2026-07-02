#!/usr/bin/env bash
# launchd entry: keep the Mac awake for the full nightly ingest run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PHOTO_INDEX_PYTHON:-$ROOT/.venv/bin/python}"
export PHOTO_INDEX_LLM_BACKEND="${PHOTO_INDEX_LLM_BACKEND:-openai}"
CAFFEINATE="$(command -v caffeinate || true)"

# Refresh the Evernote backup (incremental; only new/changed notes) before the
# python ingest reads it. Best-effort: a rate-limit or network error here must
# not abort the rest of the nightly run, so we don't let it trip `set -e`.
# NOTE: evernote-backup lives in the system Python (homebrew), NOT the venv —
# its click pin conflicts with gradio's typer. Do not install it in .venv.
EN_BACKUP="${EVERNOTE_BACKUP_BIN:-$(command -v evernote-backup || echo /opt/homebrew/bin/evernote-backup)}"
EN_DB="$ROOT/data/evernote/en_backup.db"
if [[ -x "$EN_BACKUP" && -f "$EN_DB" ]]; then
  echo "[nightly] evernote-backup sync ..."
  "$EN_BACKUP" sync --database "$EN_DB" || echo "[nightly warn] evernote-backup sync incomplete (rate limit / network); will resume next run"
else
  echo "[nightly] skipping evernote-backup sync (binary or db missing)"
fi

ARGS=(
  -m photo_index.nightly
  --db "$ROOT/data/photo_index.sqlite"
  --vlm-model "${PHOTO_INDEX_VLM_MODEL:-gemma4:26b}"
  --progress-every 50
)

if [[ -n "$CAFFEINATE" ]]; then
  exec "$CAFFEINATE" -s -w $$ "$PYTHON_BIN" "${ARGS[@]}"
fi

exec "$PYTHON_BIN" "${ARGS[@]}"
