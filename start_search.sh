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
export PHOTO_INDEX_PROMPT_FIELD_CHARS_BIG="${PHOTO_INDEX_PROMPT_FIELD_CHARS_BIG:-2000}"

export PHOTO_INDEX_QA_MODEL="${PHOTO_INDEX_QA_MODEL:-qwen3-30b-a3b-instruct-2507}"

# Cross-encoder reranking (precision stage over FTS+semantic candidates). ON for
# testing; set PHOTO_INDEX_RERANK=0 to A/B back to the previous ranking.
export PHOTO_INDEX_RERANK="${PHOTO_INDEX_RERANK:-1}"

# Optional app login for remote (Tailscale) access. Credential lives in a LOCAL,
# gitignored file (data/.search_auth: PHOTO_INDEX_AUTH="user:pass") so it never
# enters the public repo. Absent file = no login gate (local-only behaviour).
if [[ -f data/.search_auth ]]; then
  # shellcheck disable=SC1091
  source data/.search_auth
fi

ROUTE_FLAG="--no-auto-route"

# Guard: the answer model MUST be loaded with a large context. LM Studio's
# default JIT-load uses 4096, which silently truncates RAG prompts — the
# overflow-shrink path then feeds the model a fraction of the records and it
# confabulates amounts/merchants (took a long debug session to find). Reload
# with 16384 whenever the loaded context is too small.
# Free the vision model (Ollama gemma, ~10 GB on the GPU) before loading the
# answer model. On a 32 GB Mac, gemma (10 GB) + Qwen (~17 GB) co-resident blows
# past unified memory and the answer model crashes mid-generation with a Metal
# "Insufficient Memory" abort. Search never needs gemma; the nightly ingest
# reloads it automatically. Best-effort — never block the UI on this.
OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama || echo /opt/homebrew/bin/ollama)}"
if [[ -x "$OLLAMA_BIN" ]]; then
  "$OLLAMA_BIN" ps 2>/dev/null | awk 'NR>1{print $1}' | while read -r _m; do
    [[ -n "$_m" ]] && "$OLLAMA_BIN" stop "$_m" >/dev/null 2>&1 || true
  done
fi

LMS_BIN="${LMS_BIN:-$HOME/.lmstudio/bin/lms}"
WANT_CTX=16384
# parallel=1: a single user needs one KV-cache slot, not four. parallel=4 (the
# default) reserves a full 16384-token KV cache PER slot — ~4x the GPU memory —
# which is what tipped the machine into the OOM crash under load.
WANT_PARALLEL=1
if [[ -x "$LMS_BIN" ]]; then
  # NB: `read` returns non-zero at EOF (when no model is loaded, awk prints
  # nothing) — the `|| true` stops `set -e` from aborting the whole script.
  loaded_ctx=""; loaded_par=""
  read -r loaded_ctx loaded_par < <(
    "$LMS_BIN" ps 2>/dev/null | awk -v m="$PHOTO_INDEX_QA_MODEL" '$1==m {print $5, $6}'
  ) || true
  # Reload if not loaded, context too small, OR parallel too high (all three are
  # memory/quality correctness conditions).
  need_reload=0
  if [[ -z "${loaded_ctx:-}" ]]; then
    need_reload=1
  elif [[ "$loaded_ctx" =~ ^[0-9]+$ && "$loaded_ctx" -lt "$WANT_CTX" ]]; then
    need_reload=1
  elif [[ "${loaded_par:-}" =~ ^[0-9]+$ && "$loaded_par" -gt "$WANT_PARALLEL" ]]; then
    need_reload=1
  fi
  if [[ "$need_reload" == "1" ]]; then
    echo "[start_search] loading $PHOTO_INDEX_QA_MODEL (context=$WANT_CTX parallel=$WANT_PARALLEL)..."
    "$LMS_BIN" unload "$PHOTO_INDEX_QA_MODEL" >/dev/null 2>&1 || true
    "$LMS_BIN" load "$PHOTO_INDEX_QA_MODEL" --context-length "$WANT_CTX" --parallel "$WANT_PARALLEL" >/dev/null 2>&1 \
      && echo "[start_search] loaded (context=$WANT_CTX parallel=$WANT_PARALLEL)" \
      || echo "[start_search warn] could not load model via lms; answers may truncate or OOM"
  fi
fi

echo "[start_search] backend=$PHOTO_INDEX_LLM_BACKEND url=$PHOTO_INDEX_LLM_BASE_URL"
echo "[start_search] answer model=$PHOTO_INDEX_QA_MODEL  rerank=$PHOTO_INDEX_RERANK"
echo "[start_search] open http://127.0.0.1:7860 once Uvicorn reports running"

exec python -m photo_index.gradio_app --top-k 15 --host 127.0.0.1 --port 7860 ${ROUTE_FLAG}
