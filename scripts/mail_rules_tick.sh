#!/bin/bash
# Single per-minute mail-rule tick.
#
# Replaces three separate launchd jobs (route / keyword-list / trump) that each
# drove Mail.app via AppleScript every 60s. Their timers drifted and sometimes
# fired together; Mail can only service ONE AppleScript client at a time, so two
# or three concurrent clients made Mail beachball (the "every 3rd rule" freeze).
#
# Here the three passes run SEQUENTIALLY, so Mail is driven by at most one client
# at a time. A mkdir-atomic lock makes a slow tick SKIP the next minute instead of
# overlapping it (and a stale lock from a killed run is cleared after 30 min).
set -u
REPO="/Users/jackpoormanmini4/gpt-local-gemma"
PY="$REPO/.venv/bin/python"
RULE="$REPO/data/mail_rule.py"
KW="$REPO/data/trash_keywords.txt"
WM="$REPO/data/mail_rule_trump.watermark"
LOCK="$REPO/data/mail_rules_tick.lock"

# Clear a stale lock left by a tick that was killed before its trap ran.
if [ -d "$LOCK" ] && [ -n "$(find "$LOCK" -maxdepth 0 -mmin +30 2>/dev/null)" ]; then
  rmdir "$LOCK" 2>/dev/null
fi
# Atomic lock: mkdir succeeds only if the dir does not already exist.
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[tick] previous run still active — skipping this minute ($(date))"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

cd "$REPO" || exit 1
echo "===== tick $(date) ====="

# 1) Folder routes first (folder rules win over trash).
"$PY" "$RULE" route  --recent-minutes 5 --last 50
# 2) Keyword-list trash (sender/subject, space/punct-insensitive).
"$PY" "$RULE" delete --keyword-file "$KW" --recent-minutes 5 --last 50
# 3) Single-keyword "trump" trash (subject/BODY), incremental via watermark.
#    Body scanning is expensive (~40s at --last 50 vs ~3s for the sender/subject
#    passes above), so this one keeps a narrower --last 20 window. Its --since-file
#    watermark already catches every NEW message incrementally; --last is only the
#    already-read safety net, which doesn't need the full 50 here.
"$PY" "$RULE" delete --since-file "$WM" --recent-minutes 5 --last 20
