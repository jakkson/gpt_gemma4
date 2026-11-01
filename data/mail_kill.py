"""Generate + run AppleScript to count or trash mail from flagged senders.

Reads the KILL-marked rows from email_senders_top500.csv, finds which folders
contain those senders (via Mail's Envelope Index), and drives Mail.app to
either COUNT (dry run) or DELETE (move to Trash) matching messages.

Usage:
  python mail_kill.py count          # dry run, all target folders
  python mail_kill.py count <folder> # dry run, one folder
  python mail_kill.py delete <folder># move to Trash, one folder
  python mail_kill.py delete         # move to Trash, all target folders
"""
import sqlite3, csv, re, sys, subprocess, time, urllib.parse
from pathlib import Path

DB = '/Users/jackpoormanmini4/Library/Mail/V10/MailData/Envelope Index'
import os as _os
CSV = _os.environ.get('PHOTO_MAIL_CSV',
                      '/Users/jackpoormanmini4/gpt-local-gemma/data/email_senders_top500.csv')
ACCT = "Exchange"
ACCT_UUID = "B3D3A9D3"
EXCLUDE = {'Sent Items','Sent Messages','Drafts','Outbox',
           'Deleted Items','Deleted Messages','Junk','Junk Email','Bulk Mail'}


def kill_addresses():
    lines = Path(CSV).read_text().splitlines(keepends=True)
    start = 1 if not lines[0].startswith('rank,') else 0
    r = csv.DictReader(lines[start:])
    kc = [c for c in r.fieldnames if 'KILL' in c.upper()][0]
    return set(row['sender_address'].strip().lower()
               for row in r if (row.get(kc) or '').strip().lower() == 'x')


def folder_address_map(kill):
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    mbx = {}
    for rowid, url in conn.execute(
            "SELECT ROWID,url FROM mailboxes WHERE url LIKE ?", (f'%{ACCT_UUID}%',)):
        mbx[rowid] = urllib.parse.unquote(re.sub(r'.*/', '', url or ''))
    ph = ','.join('?' * len(mbx))
    q = (f"SELECT m.mailbox, a.address, COUNT(*) FROM messages m "
         f"JOIN addresses a ON a.ROWID=m.sender WHERE m.mailbox IN ({ph}) "
         f"GROUP BY m.mailbox, a.address")
    fmap = {}   # folder -> {addr: count}
    for mb, addr, n in conn.execute(q, list(mbx)):
        a = (addr or '').strip().lower()
        if a in kill:
            nm = mbx.get(mb, '')
            if nm in EXCLUDE:
                continue
            fmap.setdefault(nm, {})[a] = n
    return fmap


def as_quote(s):
    """Escape a value for an AppleScript double-quoted string.

    Sender addresses arrive from arbitrary inbound mail; a crafted address
    containing a quote could otherwise inject AppleScript into the delete pass.
    """
    s = str(s).replace("\\", "\\\\").replace('"', '\\"')
    # Control chars have no place in an address or folder name — drop them.
    return re.sub(r"[\x00-\x1f\x7f]", "", s)


def predicate(addrs):
    terms = [f'sender contains "{as_quote(a)}"' for a in addrs]
    return "(" + " or ".join(terms) + ")"


def run_folder(folder, addrs, mode):
    folder = as_quote(folder)
    pred = predicate(addrs)
    verb = ("count (messages of mb whose " + pred + ")") if mode == "count" \
        else ("delete (messages of mb whose " + pred + ")\n    return -1")
    script = f'''with timeout of 3600 seconds
tell application "Mail"
  set acct to first account whose name is "{ACCT}"
  set mb to (first mailbox of acct whose name is "{folder}")
  {verb}
end tell
end timeout'''
    t0 = time.time()
    p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    dt = time.time() - t0
    out = (p.stdout or p.stderr).strip()
    return out, dt


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "count"
    only = sys.argv[2] if len(sys.argv) > 2 else None
    kill = kill_addresses()
    fmap = folder_address_map(kill)
    # smallest folders first (by expected count)
    order = sorted(fmap.items(), key=lambda kv: sum(kv[1].values()))
    if only:
        order = [(f, a) for f, a in order if f == only]
        if not order:
            print(f"folder not found among targets: {only}")
            return
    grand = 0
    print(f"mode={mode}  folders={len(order)}  kill_senders={len(kill)}")
    for folder, addrs in order:
        expected = sum(addrs.values())
        out, dt = run_folder(folder, list(addrs), mode)
        print(f"  [{folder}] expected~{expected}  result={out}  ({dt:.0f}s)")
        try:
            grand += int(out) if mode == "count" else expected
        except ValueError:
            pass
    label = "counted" if mode == "count" else "moved to Trash"
    print(f"TOTAL {label}: {grand:,}")


if __name__ == "__main__":
    main()
