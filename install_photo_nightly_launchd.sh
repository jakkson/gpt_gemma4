#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/Users/jackpoormanmini4/gpt-local-gemma"
RUN_SCRIPT="$WORKDIR/scripts/run_nightly_ingest.sh"
WAKE_SCRIPT="$WORKDIR/scripts/schedule_nightly_wake.sh"
PLIST_PATH="$HOME/Library/LaunchAgents/com.gptlocalgemma.photoindex.nightly.plist"
WAKE_DAEMON_PLIST="/Library/LaunchDaemons/com.gptlocalgemma.photoindex.nightly-wake.plist"
OUT_LOG="$WORKDIR/data/nightly_ingest.log"
ERR_LOG="$WORKDIR/data/nightly_ingest.error.log"

chmod +x "$RUN_SCRIPT" "$WAKE_SCRIPT"

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
    <string>/bin/bash</string>
    <string>$RUN_SCRIPT</string>
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

echo "Installed nightly 2:00 AM ingest job (LaunchAgent):"
echo "  $PLIST_PATH"
echo "Logs:"
echo "  $OUT_LOG"
echo "  $ERR_LOG"
echo

echo "Installing nightly wake scheduler (requires administrator password)..."
echo "  Wakes the Mac at 1:55 AM so ingest can start at 2:00 AM."
echo "  Re-schedules at login and at 8:00 / 12:00 / 18:00 / 23:00 while you are awake."
echo

sudo tee "$WAKE_DAEMON_PLIST" >/dev/null <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.gptlocalgemma.photoindex.nightly-wake</string>

  <key>WorkingDirectory</key>
  <string>$WORKDIR</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$WAKE_SCRIPT</string>
  </array>

  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>12</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>18</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>23</integer><key>Minute</key><integer>0</integer></dict>
  </array>

  <key>RunAtLoad</key>
  <true/>

  <key>StandardOutPath</key>
  <string>$WORKDIR/data/nightly_wake.log</string>
  <key>StandardErrorPath</key>
  <string>$WORKDIR/data/nightly_wake.error.log</string>
</dict>
</plist>
EOF

sudo launchctl bootout system/com.gptlocalgemma.photoindex.nightly-wake >/dev/null 2>&1 || true
sudo launchctl bootstrap system "$WAKE_DAEMON_PLIST"
sudo launchctl kickstart -k system/com.gptlocalgemma.photoindex.nightly-wake

echo
echo "Wake scheduler installed:"
echo "  $WAKE_DAEMON_PLIST"
echo "  log: $WORKDIR/data/nightly_wake.log"
echo
echo "Check status anytime:"
echo "  .venv/bin/python -m photo_index.nightly_wake status"
echo
echo "The ingest wrapper uses caffeinate -s so the Mac stays awake during the run."
echo "Plug in overnight on a laptop; a closed lid on battery may still block wake/ingest."
