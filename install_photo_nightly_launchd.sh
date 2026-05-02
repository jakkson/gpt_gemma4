#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/Users/jackpoormanmini4/gpt-local-gemma"
PYTHON_BIN="$WORKDIR/.venv/bin/python"
PLIST_PATH="$HOME/Library/LaunchAgents/com.gptlocalgemma.photoindex.nightly.plist"
OUT_LOG="$WORKDIR/data/nightly_ingest.log"
ERR_LOG="$WORKDIR/data/nightly_ingest.error.log"

mkdir -p "$HOME/Library/LaunchAgents" "$WORKDIR/data"

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.gptlocalgemma.photoindex.nightly</string>

  <key>WorkingDirectory</key>
  <string>$WORKDIR</string>

  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON_BIN</string>
    <string>-m</string>
    <string>photo_index.nightly</string>
    <string>--db</string>
    <string>$WORKDIR/data/photo_index.sqlite</string>
    <string>--vlm-model</string>
    <string>gemma4:26b</string>
    <string>--progress-every</string>
    <string>50</string>
  </array>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>2</integer>
    <key>Minute</key>
    <integer>0</integer>
  </dict>

  <key>StandardOutPath</key>
  <string>$OUT_LOG</string>
  <key>StandardErrorPath</key>
  <string>$ERR_LOG</string>

  <key>RunAtLoad</key>
  <false/>
</dict>
</plist>
EOF

launchctl unload "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl load "$PLIST_PATH"

echo "Installed nightly 2:00 AM job:"
echo "  $PLIST_PATH"
echo "Logs:"
echo "  $OUT_LOG"
echo "  $ERR_LOG"
echo
echo "The shared content ingest lock prevents overlap with future doc/email ingests"
echo "as long as those ingesters also use photo_index.ingest_lock.global_ingest_lock()."
