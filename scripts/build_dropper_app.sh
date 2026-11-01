#!/usr/bin/env bash
# Build "Mail Rule Dropper.app" — a double-clickable macOS app bundle that runs
# scripts/mail_dropper.py in the repo's venv. Not a frozen binary: it wraps the
# existing venv, so it stays in sync with the code and needs no py2app/pyinstaller.
#
# Usage:  bash scripts/build_dropper_app.sh            # -> ~/Applications
#         bash scripts/build_dropper_app.sh /Applications
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$HOME/Applications}"
APP="$DEST/Mail Rule Dropper.app"

mkdir -p "$DEST"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Mail Rule Dropper</string>
  <key>CFBundleDisplayName</key><string>Mail Rule Dropper</string>
  <key>CFBundleIdentifier</key><string>com.gptlocalgemma.mailruledropper</string>
  <key>CFBundleExecutable</key><string>MailRuleDropper</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSUIElement</key><false/>
  <key>NSAppleEventsUsageDescription</key>
  <string>Mail Rule Dropper reads the sender of a selected or dropped email and files or trashes mail from that sender.</string>
</dict>
</plist>
PLIST

printf 'APPL????' > "$APP/Contents/PkgInfo"

# macOS 26 (Tahoe) refuses to launch a bundle whose main executable is a shell
# SCRIPT — it must be a compiled Mach-O. Compile a tiny stub that execs the
# venv Python on scripts/mail_dropper.py (paths baked in at build time).
STUB_C="$(mktemp /tmp/mailruledropper_stub.XXXXXX.c)"
cat > "$STUB_C" <<STUB
#include <unistd.h>
#include <stdlib.h>
int main(void) {
    chdir("$REPO");
    setenv("PHOTO_INDEX_LLM_BACKEND", "openai", 0);
    execl("$REPO/.venv/bin/python", "python",
          "$REPO/scripts/mail_dropper.py", (char *)0);
    return 1;
}
STUB
clang -O2 -o "$APP/Contents/MacOS/MailRuleDropper" "$STUB_C"
rm -f "$STUB_C"
chmod +x "$APP/Contents/MacOS/MailRuleDropper"
# Ad-hoc sign so Tahoe accepts the locally-built bundle.
codesign --force --sign - "$APP" 2>/dev/null || true

# Refresh Launch Services so the app appears in Spotlight/Finder immediately.
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
  -f "$APP" 2>/dev/null || true

echo "Built: $APP"
echo "Double-click it, or drag it to your Dock. First run will ask permission to control Mail."
