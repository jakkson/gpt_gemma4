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


# --- Sender/subject keyword-list path (space+punctuation-insensitive) ---------
#
# Matches a LIST of keywords against the sender address OR subject line only
# (never the body). Matching normalizes both sides to lowercase alphanumerics,
# so "car tablet" becomes "cartablet" and matches "CarTablet", "car-tablet", or
# "car tablet" — but NOT "car" alone, "tablet" alone, or the two words far apart
# (they must be adjacent to form the contiguous run). Fields are matched
# separately so a keyword can't span the sender/subject boundary.

_FS = "\x1f"  # unit separator between id/sender/subject
_RS = "\x1e"  # record separator between messages


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def load_keywords(path: str) -> list[str]:
    """One keyword per line; blanks and #-comments ignored. Returns normalized."""
    kws: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            norm = _normalize(line)
            if norm:
                kws.append(norm)
    return kws


def _fetch_unread_sender_subject() -> list[tuple[str, str, str]]:
    """Return [(id, sender, subject), ...] for unread Inbox messages."""
    acct = as_quote(ACCT)
    folder = as_quote(FOLDER)
    script = f'''with timeout of 1800 seconds
tell application "Mail"
  set fs to (ASCII character 31)
  set rs to (ASCII character 30)
  set acct to first account whose name is "{acct}"
  set mb to (first mailbox of acct whose name is "{folder}")
  set uMsgs to (messages of mb whose read status is false)
  set out to ""
  repeat with m in uMsgs
    set out to out & (id of m) & fs & (sender of m) & fs & (subject of m) & rs
  end repeat
  return out
end tell
end timeout'''
    p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    out = p.stdout or ""
    rows: list[tuple[str, str, str]] = []
    for rec in out.split(_RS):
        if not rec.strip():
            continue
        parts = rec.split(_FS)
        if len(parts) >= 3:
            rows.append((parts[0].strip(), parts[1], parts[2]))
    return rows


def _matches(sender: str, subject: str, keywords: list[str]) -> bool:
    ns, nsub = _normalize(sender), _normalize(subject)
    return any((kw in ns) or (kw in nsub) for kw in keywords)


def _delete_ids(ids: list[str]) -> int:
    if not ids:
        return 0
    acct = as_quote(ACCT)
    folder = as_quote(FOLDER)
    id_literal = "{" + ", ".join(str(int(i)) for i in ids) + "}"
    script = f'''with timeout of 1800 seconds
tell application "Mail"
  set acct to first account whose name is "{acct}"
  set mb to (first mailbox of acct whose name is "{folder}")
  set movedCount to 0
  repeat with theID in {id_literal}
    try
      delete (first message of mb whose id is theID)
      set movedCount to movedCount + 1
    end try
  end repeat
  return movedCount as string
end tell
end timeout'''
    p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    try:
        return int((p.stdout or "0").strip())
    except ValueError:
        return 0


def run_keyword_file(path: str, mode: str, list_subjects: bool):
    keywords = load_keywords(path)
    t0 = time.time()
    rows = _fetch_unread_sender_subject()
    matched = [(mid, sender, subject) for (mid, sender, subject) in rows
               if _matches(sender, subject, keywords)]
    if mode == "delete":
        moved = _delete_ids([m[0] for m in matched])
    else:
        moved = len(matched)
    dt = time.time() - t0
    return keywords, matched, moved, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["count", "delete"], help="count = dry run; delete = move to Trash")
    ap.add_argument("--keyword", default=DEFAULT_KEYWORD, help=f"single keyword vs subject/body (default: {DEFAULT_KEYWORD!r})")
    ap.add_argument("--keyword-file", help="file of keywords matched (space/punct-insensitive) vs SENDER or SUBJECT only")
    ap.add_argument("--list", action="store_true", help="in count mode, also print matching subjects")
    args = ap.parse_args()

    # Sender/subject keyword-list path (space+punctuation-insensitive adjacency).
    if args.keyword_file:
        keywords, matched, moved, dt = run_keyword_file(args.keyword_file, args.mode, args.list)
        verb = "would move" if args.mode == "count" else "moved to Trash"
        print(f"[mail_rule] account={ACCT} folder={FOLDER} keyword-file={args.keyword_file} "
              f"({len(keywords)} keywords, sender/subject) unread-only  ({dt:.0f}s)")
        print(f"[mail_rule] {moved} unread message(s) {verb}.")
        if args.list and matched:
            for _mid, sender, subject in matched:
                print(f"  - {subject}   [from: {sender}]")
        return

    # Single-keyword subject/body path (the Trump rule).
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
