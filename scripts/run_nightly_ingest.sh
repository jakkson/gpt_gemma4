#!/usr/bin/env bash
# launchd entry: keep the Mac awake for the full nightly ingest run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PHOTO_INDEX_PYTHON:-$ROOT/.venv/bin/python}"
CAFFEINATE="$(command -v caffeinate || true)"

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
