#!/usr/bin/env bash
# Root LaunchDaemon helper: ensure pmset has a wake before the 2:00 AM ingest.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PHOTO_INDEX_PYTHON:-$ROOT/.venv/bin/python}"

exec "$PYTHON_BIN" -m photo_index.nightly_wake schedule
