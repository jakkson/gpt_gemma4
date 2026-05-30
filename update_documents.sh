#!/usr/bin/env bash
# Incrementally index new/changed documents under ~/Dropbox/Documents (or --root).
# Skips files already in the DB with the same mtime+size (no --force).
#
# Usage:
#   ./update_documents.sh              # text extract only (fast)
#   ./update_documents.sh --ocr-vlm    # text then Apple Vision OCR + Ollama VLM on PDF/PNG/JPG
#   ./update_documents.sh --root /path/to/folder
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

ROOT="${PHOTO_INDEX_DOCUMENTS_ROOT:-$HOME/Dropbox/Documents}"
OCR_VLM=0
EXTRA=()

for arg in "$@"; do
  case "$arg" in
    --ocr-vlm)
      OCR_VLM=1
      ;;
    *)
      EXTRA+=("$arg")
      ;;
  esac
done

python -m photo_index.documents_ingest --root "$ROOT" --progress-every 500 "${EXTRA[@]}"

if [[ "$OCR_VLM" -eq 1 ]]; then
  python -m photo_index.documents_vlm_ingest --root "$ROOT" --progress-every 10 "${EXTRA[@]}"
fi
