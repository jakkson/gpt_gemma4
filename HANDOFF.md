# Handoff notes (cross-machine)

This file is the durable "memory" that travels with the repo. Claude Code
sessions and per-machine memory do **not** sync across computers — git does.
On another machine, open Claude Code in this repo and say *"read HANDOFF.md"* to
get oriented, then point it at the task below.

## Pre-ingest folder audit (`scripts/folder_audit.py`)

**Purpose:** before ingesting a new folder into the index, see what's inside and
offload bulky / non-ingestable items (video, audio, raw media, archives, model
weights) so they don't bloat the run. Read-only by default; nothing moves until
you ask.

**Run the audit (any folder, any machine):**
```bash
python scripts/folder_audit.py /path/to/folder
```
Outputs a printed category summary plus a per-extension "target list" CSV
(`<folder>_audit.csv`) sorted by total size — the space hogs float to the top.
Open it in Numbers, review the `OFFLOAD` column (pre-filled `x` for bulky
categories: Video, Audio, RawMedia, Archives, CodeData), adjust as desired.

**Generate a dry-run move script** (offloads the bulky categories to a holding
folder; reversible):
```bash
python scripts/folder_audit.py /path/to/folder --emit-move ~/Offload > move.sh
# review move.sh — every line is `echo mv ...` (a no-op dry run)
# remove the `echo` to actually move, then: bash move.sh
```

**Workflow we settled on (mirrors the email cleanup):**
1. Audit → review the target-list CSV in Numbers.
2. Decide which extensions/categories to offload.
3. Dry-run the move, eyeball it, then execute.
4. Ingest what remains.

**Categories** are defined in `CATEGORY_EXT` at the top of the script — add
extensions there if something lands in "Other" that should be grouped.

## Stack context (for a fresh Claude)

- Local personal-data RAG index: SQLite + FTS5 + nomic embeddings + cross-encoder
  rerank; LM Studio (Qwen3-30B-A3B) for answers; Ollama for vision ingest.
- Ingest sources: Apple Photos, Apple Messages, documents (`~/Dropbox/Documents`),
  and Apple Mail `.emlx` (mail ingest module still to be built — see below).
- This repo is being copied to a 128 GB MacBook Pro to test GLM 5.2.

## Open threads (not yet done)
- **Apple Mail `.emlx` ingest** — build `photo_index/mail_ingest.py` (parse
  `.emlx` → strip HTML → `upsert_photo` with `uuid=mail:<message-id>`), wire into
  `photo_index/nightly.py`. Mail lives at `~/Library/Mail/V10/<acct-UUID>/*.mbox`.
- **Mailbox cleanup tool** — `data/mail_kill.py` drives Mail.app via AppleScript
  to move flagged senders to Trash (reads a senders CSV with a KILL column).
  Used to clear ~24k newsletters; retry on the "mailbox temporarily unavailable"
  Exchange hiccup (it's transient — re-run delete, verify via Envelope Index).
