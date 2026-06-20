#!/usr/bin/env bash
# Launch the Personal Index Search UI against LM Studio (answers).
# Vision ingest still uses Ollama separately.
#
# Usage:
#   ./start_search.sh            # Qwen3-30B-A3B-Instruct answers (single model, no auto-route)
#
# Override any model with env vars, e.g. PHOTO_INDEX_QA_MODEL=other-model ./start_search.sh
#
# Prereq: LM Studio Developer-tab server running on :1234 with
# qwen3-30b-a3b-instruct-2507 (MLX 4-bit) loaded, context set to 16384.
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Single answer model: Qwen3-30B-A3B-Instruct-2507 (MLX 4-bit, ~17 GB — fits 32 GB
# without swap). MoE (~3B active = fast) and NON-reasoning, so no chain-of-thought
# latency and no blank-content/thinking-leak issues.
export PHOTO_INDEX_LLM_BACKEND="${PHOTO_INDEX_LLM_BACKEND:-openai}"
export PHOTO_INDEX_LLM_BASE_URL="${PHOTO_INDEX_LLM_BASE_URL:-http://127.0.0.1:1234/v1}"
export PHOTO_INDEX_QA_MODEL_SMALL="${PHOTO_INDEX_QA_MODEL_SMALL:-qwen3-30b-a3b-instruct-2507}"
export PHOTO_INDEX_LLM_TIMEOUT_BIG="${PHOTO_INDEX_LLM_TIMEOUT_BIG:-1800}"
export PHOTO_INDEX_LLM_MAX_TOKENS_BIG="${PHOTO_INDEX_LLM_MAX_TOKENS_BIG:-4096}"
export PHOTO_INDEX_PROMPT_FIELD_CHARS_BIG="${PHOTO_INDEX_PROMPT_FIELD_CHARS_BIG:-900}"

export PHOTO_INDEX_QA_MODEL="${PHOTO_INDEX_QA_MODEL:-qwen3-30b-a3b-instruct-2507}"

# Cross-encoder reranking (precision stage over FTS+semantic candidates). ON for
# testing; set PHOTO_INDEX_RERANK=0 to A/B back to the previous ranking.
export PHOTO_INDEX_RERANK="${PHOTO_INDEX_RERANK:-1}"

ROUTE_FLAG="--no-auto-route"

echo "[start_search] backend=$PHOTO_INDEX_LLM_BACKEND url=$PHOTO_INDEX_LLM_BASE_URL"
echo "[start_search] answer model=$PHOTO_INDEX_QA_MODEL  rerank=$PHOTO_INDEX_RERANK"
echo "[start_search] open http://127.0.0.1:7860 once Uvicorn reports running"

exec python -m photo_index.gradio_app --top-k 15 --host 127.0.0.1 --port 7860 ${ROUTE_FLAG}
