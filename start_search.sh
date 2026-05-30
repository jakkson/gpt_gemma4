#!/usr/bin/env bash
# Launch the Personal Index Search UI against LM Studio (answers).
# Vision ingest still uses Ollama separately.
#
# Usage:
#   ./start_search.sh            # fast 3B for answers (no auto-route)
#   ./start_search.sh --big      # 3B default, escalate to 32B (auto-route)
#
# Either way, the "Re-check with big model" button uses PHOTO_INDEX_QA_MODEL_BIG
# (default qwen2.5-vl-32b-instruct); LM Studio loads it on demand (JIT).
#
# Prereq: LM Studio Developer-tab server running on :1234 with the models loaded
# (enable Just-In-Time model loading so the 32B can load when the button is used).
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PHOTO_INDEX_LLM_BACKEND="${PHOTO_INDEX_LLM_BACKEND:-openai}"
export PHOTO_INDEX_LLM_BASE_URL="${PHOTO_INDEX_LLM_BASE_URL:-http://127.0.0.1:1234/v1}"
export PHOTO_INDEX_QA_MODEL_SMALL="${PHOTO_INDEX_QA_MODEL_SMALL:-qwen2.5-vl-3b-instruct}"
export PHOTO_INDEX_QA_MODEL_BIG="${PHOTO_INDEX_QA_MODEL_BIG:-qwen2.5-vl-32b-instruct}"

ROUTE_FLAG="--no-auto-route"
if [[ "${1:-}" == "--big" ]]; then
  export PHOTO_INDEX_QA_MODEL="${PHOTO_INDEX_QA_MODEL:-qwen2.5-vl-32b-instruct}"
  ROUTE_FLAG=""
else
  export PHOTO_INDEX_QA_MODEL="${PHOTO_INDEX_QA_MODEL:-qwen2.5-vl-3b-instruct}"
fi

echo "[start_search] backend=$PHOTO_INDEX_LLM_BACKEND url=$PHOTO_INDEX_LLM_BASE_URL"
echo "[start_search] launch=$PHOTO_INDEX_QA_MODEL small=$PHOTO_INDEX_QA_MODEL_SMALL big=$PHOTO_INDEX_QA_MODEL_BIG"
echo "[start_search] open http://127.0.0.1:7860 once Uvicorn reports running"

exec python -m photo_index.gradio_app --top-k 15 --host 127.0.0.1 --port 7860 ${ROUTE_FLAG}
