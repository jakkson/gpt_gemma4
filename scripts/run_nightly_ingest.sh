#!/usr/bin/env bash
# launchd entry: run the incremental ingest overnight, then restore search.
#
# Schedule (see the launchd plist): starts at 22:00 and HARD-STOPS at 04:30 no
# matter what (a ~6.5 h window, 7 days a week) — the ingest is fully incremental,
# so a 04:30 kill just resumes the next night. The hard stop guarantees the vision ingest can never bleed
# into the day: on a 32 GB Mac the answer model (~17 GB) and the vision model
# (~19 GB) cannot coexist, and a daytime overlap caused GPU-OOM crashes and a
# kernel panic. For the same reason we take the SEARCH APP OFFLINE for the whole
# run (a running UI would JIT-reload the answer model on top of the vision model
# and crash) and relaunch it at the end.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PHOTO_INDEX_PYTHON:-$ROOT/.venv/bin/python}"
export PHOTO_INDEX_LLM_BACKEND="${PHOTO_INDEX_LLM_BACKEND:-openai}"
CAFFEINATE="$(command -v caffeinate || true)"

LMS_BIN="${LMS_BIN:-$HOME/.lmstudio/bin/lms}"
QA_MODEL="${PHOTO_INDEX_QA_MODEL:-qwen3-30b-a3b-instruct-2507}"
OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama || echo /opt/homebrew/bin/ollama)}"

# --- cleanup: always restore search, however we exit --------------------------
restore_search() {
  # Runs on ANY exit (normal finish, 04:30 hard stop, or error). Frees the
  # vision model and relaunches the search app, which reloads the answer model
  # (parallel=1) and serves the UI again. The launchd plist sets
  # AbandonProcessGroup=true so this detached app survives after the job exits.
  set +e
  [ -n "${DEADLINE_PID:-}" ] && kill "$DEADLINE_PID" 2>/dev/null
  if [[ -x "$OLLAMA_BIN" ]]; then
    "$OLLAMA_BIN" ps 2>/dev/null | awk 'NR>1{print $1}' | while read -r _m; do
      [[ -n "$_m" ]] && "$OLLAMA_BIN" stop "$_m" >/dev/null 2>&1 || true
    done
  fi
  echo "[nightly] restoring search app"
  nohup "$ROOT/start_search.sh" </dev/null >"$ROOT/data/search_app.log" 2>&1 &
}
trap restore_search EXIT
trap 'exit 143' TERM INT

# --- take search offline for the ingest window --------------------------------
echo "[nightly] taking search app + answer model offline for the ingest"
pkill -f "photo_index.gradio_app" 2>/dev/null || true
[[ -x "$LMS_BIN" ]] && "$LMS_BIN" unload "$QA_MODEL" >/dev/null 2>&1 || true

# --- 04:30 hard stop ----------------------------------------------------------
# Schedule: launchd fires at 22:00; this kills every ingest worker at the NEXT
# 04:30 wall-clock (a ~6.5 h window). Computed as the upcoming 04:30 — if today's
# 04:30 has already passed (any evening start, incl. the 22:00 norm), use
# tomorrow's. A 7 h safety cap keeps a wrong-time catch-up run (e.g. a daytime
# wake-from-sleep launch) from bleeding through the whole day.
NOW=$(date +%s)
HARD_STOP=$(date -v4H -v30M -v0S +%s 2>/dev/null || echo 0)
if [ "$HARD_STOP" -le "$NOW" ]; then
  HARD_STOP=$(date -v+1d -v4H -v30M -v0S +%s 2>/dev/null || echo $((NOW + 23400)))
fi
MAX_END=$((NOW + 25200))   # 7 h safety cap
if [ "$HARD_STOP" -gt "$MAX_END" ]; then HARD_STOP="$MAX_END"; fi
# Test hook: PHOTO_INDEX_NIGHTLY_MAX_SECONDS=<n> forces a short window so the
# whole cycle (app-down -> deadline kill -> app restore) can be verified quickly.
if [ -n "${PHOTO_INDEX_NIGHTLY_MAX_SECONDS:-}" ]; then
  HARD_STOP=$((NOW + PHOTO_INDEX_NIGHTLY_MAX_SECONDS))
fi
past_deadline() { [ "$(date +%s)" -ge "$HARD_STOP" ]; }

(
  sleep $((HARD_STOP - NOW))
  echo "[nightly] hard stop ($(date -r "$HARD_STOP" '+%H:%M')) — ending ingest"
  for _ in 1 2 3; do
    for m in nightly documents_ingest documents_vlm_ingest messages_ingest \
             mail_ingest evernote_ingest calendar_ingest history_ingest embed_index; do
      pkill -f "photo_index.$m" 2>/dev/null
    done
    sleep 3
  done
) &
DEADLINE_PID=$!

# Keep the Mac awake for the whole run (-w $$ ties it to this shell's lifetime).
if [[ -n "$CAFFEINATE" ]]; then
  "$CAFFEINATE" -s -w $$ &
fi

# --- ingest steps (each skipped if we've passed the 04:30 deadline) -----------
# Refresh the Evernote backup (incremental) before the python ingest reads it.
# evernote-backup lives in the system Python (homebrew), NOT the venv — its
# click pin conflicts with gradio's typer. Do not install it in .venv.
EN_BACKUP="${EVERNOTE_BACKUP_BIN:-$(command -v evernote-backup || echo /opt/homebrew/bin/evernote-backup)}"
EN_DB="$ROOT/data/evernote/en_backup.db"
if ! past_deadline && [[ -x "$EN_BACKUP" && -f "$EN_DB" ]]; then
  echo "[nightly] evernote-backup sync ..."
  "$EN_BACKUP" sync --database "$EN_DB" || echo "[nightly warn] evernote-backup sync incomplete; resumes next run"
fi

ARGS=(
  -m photo_index.nightly
  --db "$ROOT/data/photo_index.sqlite"
  --vlm-model "${PHOTO_INDEX_VLM_MODEL:-gemma4:26b}"
  --progress-every 50
)

rc=0
past_deadline || "$PYTHON_BIN" "${ARGS[@]}" || rc=$?

# Offline library (books etc.): a local-only folder Dropbox/iCloud never turn
# into online-only placeholders. Same documents_ingest (epub/pdf/docx/txt).
BOOKS_ROOT="${PHOTO_INDEX_BOOKS_ROOT:-$HOME/LLM_Books}"
if ! past_deadline && [[ -d "$BOOKS_ROOT" ]]; then
  echo "[nightly] ingesting offline library: $BOOKS_ROOT"
  "$PYTHON_BIN" -m photo_index.documents_ingest \
    --db "$ROOT/data/photo_index.sqlite" --root "$BOOKS_ROOT" --progress-every 50 \
    || echo "[nightly warn] library ingest failed"
fi

# Extra document folders outside ~/Dropbox/Documents (text extraction only, same
# as the library step — images/video/audio are not captioned here). Add more
# paths to the array as needed.
EXTRA_DOC_ROOTS=(
  "$HOME/Dropbox/KPIX"
)
# Subfolders to skip within the extra roots (raw data we'll parse with a
# dedicated tool later, not free-text). KPIX Ratings = ~150K near-duplicate
# ratings-table chunks; excluded from generic text ingest.
EXTRA_DOC_EXCLUDES=(
  "$HOME/Dropbox/KPIX/KPIX Ratings"
)
for extra in "${EXTRA_DOC_ROOTS[@]}"; do
  if ! past_deadline && [[ -d "$extra" ]]; then
    echo "[nightly] ingesting extra folder: $extra"
    excl_args=()
    for ex in "${EXTRA_DOC_EXCLUDES[@]}"; do excl_args+=(--exclude "$ex"); done
    "$PYTHON_BIN" -m photo_index.documents_ingest \
      --db "$ROOT/data/photo_index.sqlite" --root "$extra" --progress-every 50 \
      "${excl_args[@]}" \
      || echo "[nightly warn] extra folder ingest failed: $extra"
  fi
done

# Apple Calendar events (personal cals in full, holidays next 365 d, subscribed
# skipped). Reads a copy of the store so a running Calendar app can't lock it.
if ! past_deadline; then
  echo "[nightly] ingesting Apple Calendar events ..."
  "$PYTHON_BIN" -m photo_index.calendar_ingest \
    --db "$ROOT/data/photo_index.sqlite" --progress-every 2000 \
    || echo "[nightly warn] calendar ingest failed"
fi

# Browser history (Safari + Chrome): title/URL metadata for "where/what did I
# see?". Read-only on the browsers (copies each DB first). Incremental: re-runs
# refresh recency and only re-embed genuinely new/changed pages.
if ! past_deadline; then
  echo "[nightly] ingesting browser history (Safari + Chrome) ..."
  "$PYTHON_BIN" -m photo_index.history_ingest \
    --db "$ROOT/data/photo_index.sqlite" \
    || echo "[nightly warn] history ingest failed"
fi

# Embed new rows (nomic in LM Studio — independent of the unloaded answer model).
if ! past_deadline; then
  echo "[nightly] embedding new rows (embed_index) ..."
  "$PYTHON_BIN" -m photo_index.embed_index --db "$ROOT/data/photo_index.sqlite" \
    || echo "[nightly warn] embed_index failed; rows stay queued for next run"
fi

exit "$rc"
