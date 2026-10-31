#!/bin/bash
# Boot auto-start for the mini's LLM search stack. Launched once at login by the
# LaunchAgent com.gptlocalgemma.search.boot (RunAtLoad), so after an auto-login
# reboot/crash the chat comes back with no human touch.
#
# Loads ONLY the answer model (Qwen3-30B-A3B-Instruct-2507) + the tiny nomic
# embedder — never the 35B or the VL models (which would over-commit the 32 GB
# and panic the GPU). start_search.sh pins the 30B at context 16384 / parallel 1.
#
# No KeepAlive on the agent: the nightly ingest deliberately unloads the model
# and kills the app for its window; KeepAlive would fight that. The nightly's own
# restore relaunches the app afterward.
REPO="$HOME/gpt-local-gemma"
LMS="$HOME/.lmstudio/bin/lms"
QA_MODEL="qwen3-30b-a3b-instruct-2507"
EMBED_MODEL="text-embedding-nomic-embed-text-v1.5"

echo "[boot_search] $(date '+%F %T') starting..."

# Make sure LM Studio's backend is up (it manages model loading).
open -ga "LM Studio" 2>/dev/null || true
"$LMS" server start >/dev/null 2>&1 || true

# Wait for the server to answer before loading anything (up to ~90 s).
for _ in $(seq 1 45); do
  curl -s -o /dev/null "http://127.0.0.1:1234/v1/models" && break
  sleep 2
done

# Force-load ONLY the intended models. Unload anything else that somehow came up
# (e.g. a restored session) so the 32 GB is never over-committed.
"$LMS" ps 2>/dev/null | awk 'NR>1 && $1!="IDENTIFIER"{print $1}' | while read -r m; do
  case "$m" in
    "$QA_MODEL"|"$EMBED_MODEL"|"$EMBED_MODEL":*) : ;;   # keep
    "" ) : ;;
    * ) echo "[boot_search] unloading stray model: $m"; "$LMS" unload "$m" >/dev/null 2>&1 || true ;;
  esac
done

# Embedder (tiny) so nightly embed works even with JIT loading disabled.
"$LMS" load "$EMBED_MODEL" >/dev/null 2>&1 || true

# The mini is INGEST-ONLY now: keep the tiny embedder resident for the nightly
# embed step, but do NOT load the 17 GB answer model or launch the search UI —
# that frees ~17 GB on the 32 GB box. To search on the mini on demand, run the
# "Start LLM (MINI).command" launcher (it loads the 30B at context 16384 and
# starts the app). Set MINI_SEARCH_AUTOSTART=1 to restore the old behavior.
cd "$REPO" || { echo "[boot_search] repo not found: $REPO"; exit 1; }
if [[ "${MINI_SEARCH_AUTOSTART:-0}" == "1" ]]; then
  echo "[boot_search] MINI_SEARCH_AUTOSTART=1 -> start_search.sh (loads $QA_MODEL)"
  exec ./start_search.sh
fi
# Ingest-only: make sure the 17 GB answer model is NOT resident, even if LM Studio
# restored it from a previous session on launch (autoStartOnLaunch). The stray
# loop above intentionally keeps $QA_MODEL, so unload it explicitly here.
"$LMS" unload "$QA_MODEL" >/dev/null 2>&1 || true
echo "[boot_search] ingest-only: embedder loaded, 30B NOT loaded (unloaded if restored)."
echo "[boot_search] run 'Start LLM (MINI).command' to search on the mini on demand."
