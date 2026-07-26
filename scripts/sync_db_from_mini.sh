#!/usr/bin/env bash
# Manual, on-demand pull of the index DB from the Mac mini to THIS machine (MBP).
#
# Both machines run the same stack, but the mini is where the nightly/web/history
# ingests + doctor run, so its DB is the source of truth. Run this on the MacBook
# whenever you want the MacBook to catch up to the mini.
#
# How it stays safe:
#   • VACUUM INTO on the mini = a consistent, WAL-free, defragmented snapshot even
#     while the mini's DB is live (no torn copy).
#   • The transferred file is integrity-checked and row-count-matched BEFORE it
#     replaces anything.
#   • The current MacBook DB is kept as ONE rolling backup (…​.bak) for rollback.
#   • Derived files (-wal/-shm and the .npy embedding sidecar) are dropped so the
#     search app rebuilds them from the new vectors.
#
# One-time setup on the MINI: System Settings > General > Sharing > Remote Login = ON.
# One-time on the MacBook (passwordless):  ssh-copy-id jackpoormanmini4@Jacks-Mac-mini.local
#
# Override the target if you reach the mini differently (e.g. over Tailscale):
#   MINI_SSH=jackpoormanmini4@100.84.222.14 scripts/sync_db_from_mini.sh
set -euo pipefail

MINI_SSH="${MINI_SSH:-jackpoormanmini4@Jacks-Mac-mini.local}"
MINI_REPO="${MINI_REPO:-/Users/jackpoormanmini4/gpt-local-gemma}"
LOCAL_REPO="${LOCAL_REPO:-$HOME/gpt-local-gemma}"
DB="$LOCAL_REPO/data/photo_index.sqlite"
INCOMING="$LOCAL_REPO/data/_incoming.sqlite"
SNAP_REMOTE="$MINI_REPO/data/_sync_snapshot.sqlite"
SQLITE="/usr/bin/sqlite3"

say() { printf '[sync] %s\n' "$*"; }

# 1. Reachable?
say "checking SSH to $MINI_SSH ..."
if ! ssh -o ConnectTimeout=8 -o BatchMode=yes "$MINI_SSH" true 2>/dev/null; then
  say "ERROR: cannot SSH to $MINI_SSH."
  say "  • On the mini: System Settings > General > Sharing > Remote Login = ON."
  say "  • First time (avoid password prompt): ssh-copy-id $MINI_SSH"
  say "  • Different address? MINI_SSH=user@host $0"
  exit 1
fi

# 2. Consistent snapshot on the mini.
say "making a consistent snapshot on the mini (VACUUM INTO, ~1 min) ..."
ssh "$MINI_SSH" "rm -f '$SNAP_REMOTE'; $SQLITE '$MINI_REPO/data/photo_index.sqlite' \"VACUUM INTO '$SNAP_REMOTE'\""
rows_remote=$(ssh "$MINI_SSH" "$SQLITE '$SNAP_REMOTE' 'SELECT count(*) FROM photo_meta'")
say "snapshot ready on mini: $rows_remote rows."

# 3. Transfer.
say "transferring to the MacBook ..."
rsync -ah --info=progress2 "$MINI_SSH:$SNAP_REMOTE" "$INCOMING"

# 4. Verify BEFORE touching the live DB.
say "verifying transferred DB ..."
chk=$("$SQLITE" "$INCOMING" 'PRAGMA quick_check' 2>/dev/null | head -1 || echo "fail")
rows_local=$("$SQLITE" "$INCOMING" 'SELECT count(*) FROM photo_meta' 2>/dev/null || echo 0)
if [[ "$chk" != "ok" || "$rows_local" != "$rows_remote" ]]; then
  say "ERROR: verification failed (check=$chk rows=$rows_local vs $rows_remote). Current DB untouched."
  rm -f "$INCOMING"; ssh "$MINI_SSH" "rm -f '$SNAP_REMOTE'" || true
  exit 1
fi
say "verified: $rows_local rows, integrity ok."

# 5. Stop the local search app for a clean swap (it caches the embedding matrix).
app_was_up=0
if pgrep -f "photo_index.gradio_app" >/dev/null 2>&1; then
  say "stopping the local search app for the swap ..."
  pkill -f "photo_index.gradio_app" 2>/dev/null || true
  app_was_up=1
  sleep 1
fi

# 6. Atomic swap + drop derived files (one rolling backup kept).
if [[ -f "$DB" ]]; then rm -f "$DB.bak"; mv "$DB" "$DB.bak"; fi
mv "$INCOMING" "$DB"
rm -f "$DB-wal" "$DB-shm" "$DB.emb_mat.npy" "$DB.emb_uuids.npy"
say "swapped in the new DB (previous kept as $DB.bak)."

# 7. Remove the mini's snapshot.
ssh "$MINI_SSH" "rm -f '$SNAP_REMOTE'" || true

# 8. Bring the search app back if we stopped it.
LAUNCHER="$HOME/Desktop/Start LLM Search.command"
if [[ "$app_was_up" == "1" ]]; then
  if [[ -f "$LAUNCHER" ]]; then
    say "restarting the search app ..."
    nohup bash "$LAUNCHER" </dev/null >"$LOCAL_REPO/data/search_app.log" 2>&1 &
  else
    say "restart your search app to load the new DB."
  fi
fi

say "done — MacBook now matches the mini ($rows_local rows). Backup: $DB.bak"
