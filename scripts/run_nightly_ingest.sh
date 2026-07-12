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

# Keep the Mac awake for the WHOLE script (ingest + embed), not just nightly.
# (-w $$ ties the assertion to this shell's lifetime.)
if [[ -n "$CAFFEINATE" ]]; then
  "$CAFFEINATE" -s -w $$ &
fi

rc=0
"$PYTHON_BIN" "${ARGS[@]}" || rc=$?

# Offline library (books etc.): a local-only folder that Dropbox/iCloud never
# turn into online-only placeholders, so files are always readable. Same
# documents_ingest (text extract) — epub/pdf/docx/txt. Runs before embed so new
# books are searchable that night.
BOOKS_ROOT="${PHOTO_INDEX_BOOKS_ROOT:-$HOME/LLM_Books}"
if [[ -d "$BOOKS_ROOT" ]]; then
  echo "[nightly] ingesting offline library: $BOOKS_ROOT"
  "$PYTHON_BIN" -m photo_index.documents_ingest \
    --db "$ROOT/data/photo_index.sqlite" --root "$BOOKS_ROOT" --progress-every 50 \
    || echo "[nightly warn] library ingest failed"
fi

# Apple Calendar (iCal) events: personal calendars in full, holidays only for
# the next 365 days, subscribed feeds skipped. Reads a copy of the Calendar
# store so a running Calendar app can't lock it. Runs before embed so new/
# changed events are searchable that night. Best-effort.
echo "[nightly] ingesting Apple Calendar events ..."
"$PYTHON_BIN" -m photo_index.calendar_ingest \
  --db "$ROOT/data/photo_index.sqlite" --progress-every 2000 \
  || echo "[nightly warn] calendar ingest failed"

# Embed any rows the ingest just added — otherwise new content is invisible to
# semantic search until someone runs embed_index by hand. Requires LM Studio
# (:1234) for the nomic model; a failure here must not mask the ingest result.
echo "[nightly] embedding new rows (embed_index) ..."
"$PYTHON_BIN" -m photo_index.embed_index --db "$ROOT/data/photo_index.sqlite" \
  || echo "[nightly warn] embed_index failed (is LM Studio running?); rows stay queued for next run"

exit "$rc"
