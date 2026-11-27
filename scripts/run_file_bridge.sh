#!/usr/bin/env bash
# Always-on, read-only file bridge for the mini (ingest-only box, source of truth).
# Lets the Gradio search UI running on another Mac stream/open an indexed file over
# Tailscale. No model, trivial RAM. Launched + kept alive by
# com.gptlocalgemma.filebridge; token + tailnet-IP gated (see photo_index/file_bridge.py).
set -uo pipefail
REPO="$HOME/gpt-local-gemma"
cd "$REPO" || { echo "[file_bridge] repo not found: $REPO"; exit 1; }
exec .venv/bin/python -m photo_index.file_bridge \
  --host 0.0.0.0 --port "${PHOTO_INDEX_FILE_BRIDGE_PORT:-7870}"
