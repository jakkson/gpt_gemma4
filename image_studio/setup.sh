#!/usr/bin/env bash
# One-time setup for Qwen-Image Studio — RUN ON THE MACBOOK (not the 32 GB mini).
# Creates an isolated venv, installs mflux + gradio, and makes the output folder.
# The first image you generate downloads ~33 GB of Qwen-Image weights.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
PY="${PYTHON:-python3}"

echo "[image_studio] Python: $("$PY" --version 2>&1)"
echo "[image_studio] creating venv in $DIR/.venv ..."
"$PY" -m venv .venv
./.venv/bin/pip install --upgrade pip wheel

echo "[image_studio] installing mflux + gradio (pulls MLX; a few minutes) ..."
./.venv/bin/pip install -r requirements.txt

mkdir -p "${IMAGE_STUDIO_OUT:-$HOME/Generated Images}"

# Sanity: did the Qwen command install? (name can vary by mflux version)
if ./.venv/bin/python -c "import shutil,sys; sys.exit(0 if shutil.which('mflux-generate-qwen-2.1') else 1)"; then
  echo "[image_studio] mflux-generate-qwen-2.1 is available."
else
  echo "[image_studio] NOTE: mflux-generate-qwen-2.1 not on PATH inside the venv."
  echo "               Check 'ls ./.venv/bin | grep mflux' and, if the Qwen command"
  echo "               is named differently, run the app with"
  echo "               MFLUX_QWEN_CMD=<that-name> ./.venv/bin/python app.py"
fi

echo
echo "[image_studio] done. Start it with:"
echo "    cd $DIR && ./.venv/bin/python app.py"
echo "then open http://127.0.0.1:7863"
