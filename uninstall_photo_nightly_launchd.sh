#!/usr/bin/env bash
set -euo pipefail

PLIST_PATH="$HOME/Library/LaunchAgents/com.gptlocalgemma.photoindex.nightly.plist"
WAKE_DAEMON_PLIST="/Library/LaunchDaemons/com.gptlocalgemma.photoindex.nightly-wake.plist"

launchctl unload "$PLIST_PATH" >/dev/null 2>&1 || true
rm -f "$PLIST_PATH"
echo "Removed nightly ingest LaunchAgent: $PLIST_PATH"

if [[ -f "$WAKE_DAEMON_PLIST" ]]; then
  echo "Removing nightly wake scheduler (requires administrator password)..."
  sudo launchctl bootout system/com.gptlocalgemma.photoindex.nightly-wake >/dev/null 2>&1 || true
  sudo rm -f "$WAKE_DAEMON_PLIST"
  echo "Removed wake LaunchDaemon: $WAKE_DAEMON_PLIST"
  echo "Note: existing one-off pmset wake events are left in place; they expire after firing."
else
  echo "Wake LaunchDaemon not installed ($WAKE_DAEMON_PLIST)."
fi
