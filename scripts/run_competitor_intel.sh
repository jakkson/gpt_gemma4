#!/usr/bin/env bash
# Nightly competitor-intel refresh for all RETAINED engagements.
#
# Scheduled ~05:00 by com.gptlocalgemma.competitor.nightly — AFTER the 22:00-04:30
# ingest window, so the Qwen answer model (synthesis) never fights the Ollama vision
# model for GPU memory (17GB + 10GB > 32GB → OOM/kernel panic; see start_search.sh).
#
# No `set -e`: one failing engagement must not abort the rest.
set -uo pipefail

REPO="$HOME/gpt-local-gemma"
cd "$REPO" || { echo "[competitor_intel] repo not found at $REPO"; exit 1; }

export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

# --- Answer model via LM Studio (OpenAI-compatible), mirroring start_search.sh ----
export PHOTO_INDEX_LLM_BACKEND=openai
export PHOTO_INDEX_LLM_BASE_URL="http://127.0.0.1:1234/v1"
export PHOTO_INDEX_QA_MODEL="${PHOTO_INDEX_QA_MODEL:-qwen3-30b-a3b-instruct-2507}"

LMS_BIN="${LMS_BIN:-$HOME/.lmstudio/bin/lms}"
OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama || echo /opt/homebrew/bin/ollama)}"

# The mini is ingest-only for search; nothing else here keeps the 17 GB answer
# model resident. Release it on every exit path so this nightly batch doesn't
# leave it loaded (it has no TTL) and refill the box until the next reboot.
cleanup() { [[ -x "$LMS_BIN" ]] && "$LMS_BIN" unload "$PHOTO_INDEX_QA_MODEL" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "[competitor_intel] $(date '+%F %T') starting nightly run"

# Free the Ollama vision model (gemma, ~10GB) so Qwen (~17GB) fits without OOM.
if [[ -x "$OLLAMA_BIN" ]]; then
  "$OLLAMA_BIN" ps 2>/dev/null | awk 'NR>1{print $1}' | while read -r _m; do
    [[ -n "$_m" ]] && "$OLLAMA_BIN" stop "$_m" >/dev/null 2>&1 || true
  done
fi

# Ensure LM Studio is up and the answer model is loaded ONCE at a safe context and
# parallel=1 (unload+load guarantees a single clean instance — no double KV cache).
if [[ -x "$LMS_BIN" ]]; then
  "$LMS_BIN" server start >/dev/null 2>&1 || true
  "$LMS_BIN" unload "$PHOTO_INDEX_QA_MODEL" >/dev/null 2>&1 || true
  if "$LMS_BIN" load "$PHOTO_INDEX_QA_MODEL" --context-length 16384 --parallel 1 >/dev/null 2>&1; then
    echo "[competitor_intel] model loaded: $PHOTO_INDEX_QA_MODEL (ctx=16384 parallel=1)"
  else
    echo "[competitor_intel warn] lms load failed; synthesis may 404/truncate"
  fi
fi

[[ -f .venv/bin/activate ]] && source .venv/bin/activate

# Only RETAINED engagements run nightly; one-shot prospecting engagements are skipped.
CLIENTS=$(python -c "import json,pathlib; d=json.loads(pathlib.Path('competitor_intel/engagements.json').read_text()); print(' '.join(k for k,v in d.items() if v.get('retained')))")
if [[ -z "${CLIENTS// /}" ]]; then
  echo "[competitor_intel] no retained engagements — nothing to do"
  exit 0
fi

for c in $CLIENTS; do
  echo "[competitor_intel] === engagement: $c ==="
  python -m competitor_intel.run run --client "$c" \
    || echo "[competitor_intel] engagement '$c' failed (continuing)"
  # Email the fresh report (skips cleanly if SMTP isn't configured in .env).
  python -m competitor_intel.run deliver --client "$c" \
    || echo "[competitor_intel] deliver '$c' failed (continuing)"
done

echo "[competitor_intel] $(date '+%F %T') done"
