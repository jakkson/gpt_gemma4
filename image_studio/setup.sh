#!/usr/bin/env bash
# One-time setup for Image Studio — RUN ON THE MACBOOK (not the 32 GB mini).
# mflux needs Python >= 3.11; macOS's default python3 is 3.9 and will NOT work.
# Creates an isolated venv, installs mflux + gradio, makes the output folder.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# --- find a Python >= 3.11 (mflux requirement). Override with PYTHON=/path ------
pick_python() {
  local c ver major minor
  for c in "${PYTHON:-}" python3.12 python3.11 python3.13 \
           /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 \
           "$HOME/gpt-local-gemma/.venv/bin/python" python3; do
    [ -z "$c" ] && continue
    command -v "$c" >/dev/null 2>&1 || [ -x "$c" ] || continue
    ver=$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo 0.0)
    major=${ver%%.*}; minor=${ver#*.}
    if [ "${major:-0}" -eq 3 ] && [ "${minor:-0}" -ge 11 ]; then echo "$c"; return 0; fi
  done
  return 1
}

if ! PY="$(pick_python)"; then
  echo "[image_studio] ERROR: need Python 3.11+ (mflux requirement); found only older."
  echo "  Install one:   brew install python@3.12"
  echo "  Then re-run:   PYTHON=python3.12 bash setup.sh"
  exit 1
fi
echo "[image_studio] using $("$PY" --version 2>&1)  ($(command -v "$PY" || echo "$PY"))"

# --- (re)create the venv — wipe any earlier one built with the wrong Python -----
if [ -d .venv ]; then
  echo "[image_studio] removing existing .venv (rebuilding) ..."
  rm -rf .venv
fi
"$PY" -m venv .venv
./.venv/bin/pip install --upgrade pip wheel

echo "[image_studio] installing mflux + gradio (pulls MLX; a few minutes) ..."
if ! ./.venv/bin/pip install -r requirements.txt; then
  echo
  echo "[image_studio] pip install failed. If it was an mflux/mlx wheel error, your"
  echo "  Python may be too new for mflux yet — use 3.12:  brew install python@3.12"
  echo "  then  PYTHON=python3.12 bash setup.sh"
  exit 1
fi

mkdir -p "${IMAGE_STUDIO_OUT:-$HOME/Generated Images}"

echo
echo "[image_studio] done. Start it with:"
echo "    cd $DIR && ./.venv/bin/python app.py"
echo "then open http://127.0.0.1:7863"
