"""Move UNREAD Inbox messages matching a keyword (subject or body) to Trash.

Drives Mail.app via AppleScript. Built for a daily launchd schedule, but always
run `count` first to preview what a keyword catches — `delete` moves matches to
the Trash (recoverable) but it is unattended and keyword matching is substring
(so "trump" also matches "trumpet", "trumped up", a person named "Trumbull").

Usage:
  python mail_rule.py count               # dry run: how many unread match
  python mail_rule.py count --list        # dry run + list matching subjects
  python mail_rule.py delete              # move matches to Trash
  python mail_rule.py count --keyword X   # override keyword (default: trump)

Env:
  PHOTO_MAIL_RULE_KEYWORD   default keyword if --keyword not passed
  PHOTO_MAIL_RULE_ACCOUNT   Mail account name (default: Exchange)
  PHOTO_MAIL_RULE_FOLDER    mailbox name (default: Inbox)
"""
import argparse
import os
import re
import subprocess
import time

ACCT = os.environ.get("PHOTO_MAIL_RULE_ACCOUNT", "Exchange")
FOLDER = os.environ.get("PHOTO_MAIL_RULE_FOLDER", "Inbox")
DEFAULT_KEYWORD = os.environ.get("PHOTO_MAIL_RULE_KEYWORD", "trump")


def as_quote(s: str) -> str:
    """Escape a value for an AppleScript double-quoted string (injection-safe)."""
    s = str(s).replace("\\", "\\\\").replace('"', '\\"')
    return re.sub(r"[\x00-\x1f\x7f]", "", s)


def build_script(keyword: str, mode: str, list_subjects: bool) -> str:
    kw = as_quote(keyword)
    acct = as_quote(ACCT)
    folder = as_quote(FOLDER)
    # Gather unread messages first (fast, indexed), then test subject/body. This
    # avoids Mail fetching the body of every message in a large mailbox.
    if mode == "delete":
        # Capture stable message ids in one scan, THEN delete each by re-resolving
        # its id. Holding message references across deletes fails on Exchange
        # (-1728): moving a message re-indexes the mailbox and staled references
        # abort the loop mid-batch. The per-item `try` also means a single id that
        # can't be resolved (already moved, thread-collapsed) is skipped, not fatal
        # — anything missed is simply caught on the next run.
        action = """
      set idList to {}
      repeat with m in uMsgs
        if ((subject of m) contains kw) or ((content of m) contains kw) then
          set end of idList to (id of m)
        end if
      end repeat
      set movedCount to 0
      repeat with theID in idList
        try
          delete (first message of mb whose id is theID)
          set movedCount to movedCount + 1
        end try
      end repeat
      return movedCount as string"""
    else:
        collect = """
      set matchList to {}
      set subjOut to ""
      repeat with m in uMsgs
        if ((subject of m) contains kw) or ((content of m) contains kw) then
          set end of matchList to m
          set subjOut to subjOut & "  - " & (subject of m) & linefeed
        end if
      end repeat"""
        if list_subjects:
            action = collect + """
      return ((count of matchList) as string) & linefeed & subjOut"""
        else:
            action = collect + """
      return (count of matchList) as string"""
    return f'''with timeout of 1800 seconds
tell application "Mail"
  set kw to "{kw}"
  set acct to first account whose name is "{acct}"
  set mb to (first mailbox of acct whose name is "{folder}")
  set uMsgs to (messages of mb whose read status is false){action}
end tell
end timeout'''


def run(keyword: str, mode: str, list_subjects: bool):
    script = build_script(keyword, mode, list_subjects)
    t0 = time.time()
    p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    dt = time.time() - t0
    out = (p.stdout or p.stderr).strip()
    return out, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["count", "delete"], help="count = dry run; delete = move to Trash")
    ap.add_argument("--keyword", default=DEFAULT_KEYWORD, help=f"keyword to match (default: {DEFAULT_KEYWORD!r})")
    ap.add_argument("--list", action="store_true", help="in count mode, also print matching subjects")
    args = ap.parse_args()

    out, dt = run(args.keyword, args.mode, args.list)
    verb = "would move" if args.mode == "count" else "moved to Trash"
    # First line of AppleScript output is the count.
    lines = out.splitlines()
    count = lines[0] if lines else "0"
    print(f"[mail_rule] account={ACCT} folder={FOLDER} keyword={args.keyword!r} "
          f"unread-only  ({dt:.0f}s)")
    print(f"[mail_rule] {count} unread message(s) {verb}.")
    if args.list and len(lines) > 1:
        print("\n".join(lines[1:]))


if __name__ == "__main__":
    main()
