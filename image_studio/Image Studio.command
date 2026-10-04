#!/bin/bash
# Qwen-Image Studio launcher — MacBook only. Copy this file to ~/Launchers (or the
# Desktop) and double-click. First launch runs setup.sh (installs mflux); the first
# image downloads ~33 GB of weights.
REPO="${GPT_LOCAL_GEMMA:-$HOME/gpt-local-gemma}"
DIR="$REPO/image_studio"
cd "$DIR" || { echo "Not found: $DIR"; read -r -p "Press Return to close "; exit 1; }

if [ ! -x ./.venv/bin/python ]; then
  echo "First run — setting up the image-studio venv (installs mflux)…"
  bash setup.sh || { read -r -p "Setup failed. Press Return to close "; exit 1; }
fi

echo "Starting Qwen-Image Studio → http://127.0.0.1:7863"
echo "(Unload the big LLM in LM Studio first — they don't fit in memory together.)"
( sleep 4; open "http://127.0.0.1:7863" >/dev/null 2>&1 ) &
exec ./.venv/bin/python app.py
