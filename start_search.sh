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

# Guard: the answer model MUST be loaded with a large context. LM Studio's
# default JIT-load uses 4096, which silently truncates RAG prompts — the
# overflow-shrink path then feeds the model a fraction of the records and it
# confabulates amounts/merchants (took a long debug session to find). Reload
# with 16384 whenever the loaded context is too small.
LMS_BIN="${LMS_BIN:-$HOME/.lmstudio/bin/lms}"
WANT_CTX=16384
if [[ -x "$LMS_BIN" ]]; then
  loaded_ctx=$("$LMS_BIN" ps 2>/dev/null | awk -v m="$PHOTO_INDEX_QA_MODEL" '$1==m {print $5}')
  if [[ -n "${loaded_ctx:-}" && "$loaded_ctx" =~ ^[0-9]+$ && "$loaded_ctx" -lt "$WANT_CTX" ]]; then
    echo "[start_search] $PHOTO_INDEX_QA_MODEL loaded with context=$loaded_ctx (<$WANT_CTX); reloading..."
    "$LMS_BIN" unload "$PHOTO_INDEX_QA_MODEL" >/dev/null 2>&1 || true
    "$LMS_BIN" load "$PHOTO_INDEX_QA_MODEL" --context-length "$WANT_CTX" >/dev/null 2>&1 \
      && echo "[start_search] reloaded with context=$WANT_CTX" \
      || echo "[start_search warn] could not reload model; answers may truncate records"
  elif [[ -z "${loaded_ctx:-}" ]]; then
    echo "[start_search] $PHOTO_INDEX_QA_MODEL not loaded; loading with context=$WANT_CTX..."
    "$LMS_BIN" load "$PHOTO_INDEX_QA_MODEL" --context-length "$WANT_CTX" >/dev/null 2>&1 \
      || echo "[start_search warn] could not load model via lms; LM Studio may JIT-load at 4096"
  fi
fi

echo "[start_search] backend=$PHOTO_INDEX_LLM_BACKEND url=$PHOTO_INDEX_LLM_BASE_URL"
echo "[start_search] answer model=$PHOTO_INDEX_QA_MODEL  rerank=$PHOTO_INDEX_RERANK"
echo "[start_search] open http://127.0.0.1:7860 once Uvicorn reports running"

exec python -m photo_index.gradio_app --top-k 15 --host 127.0.0.1 --port 7860 ${ROUTE_FLAG}
