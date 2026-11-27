#!/usr/bin/env python3
"""Mail Rule Dropper — drag an email in, pick its sender's fate, forever.

Drop a message (dragged from Apple Mail or a .eml/.emlx from Finder) onto the
window — or click "Grab selected email in Mail" — and choose:

  • Trash forever  -> sender is appended to data/trash_keywords.txt, which the
    per-minute launchd keyword job already enforces (unread + last-5-min read).
  • File to folder -> sender -> folder is added to data/mail_routes.json and a
    per-minute launchd route job moves matching Inbox mail there from now on.

Either way the rule is applied immediately to what's in the Inbox right now.

Run:  .venv/bin/python scripts/mail_dropper.py   (or the .command launcher)
"""
from __future__ import annotations

import email
import email.header
import email.utils
import json
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
KEYWORDS_TXT = DATA / "trash_keywords.txt"
ROUTES_JSON = DATA / "mail_routes.json"
MAIL_RULE = DATA / "mail_rule.py"
PYTHON = REPO / ".venv" / "bin" / "python"
ACCT = "Exchange"

try:
    from tkinterdnd2 import DND_FILES, DND_TEXT, TkinterDnD
    _DND = True
except Exception:  # app still works via the "grab selected" button
    _DND = False


def sender_from_message_url(url: str) -> tuple[str, str] | None:
    """Resolve a dragged Apple Mail 'message://<id>' URL to (address, name)
    via the Envelope Index — fast, and avoids poking Mail's scripting."""
    import sqlite3
    import urllib.parse

    m = re.search(r"message:/*(.+)$", url.strip())
    if not m:
        return None
    mid = urllib.parse.unquote(m.group(1)).strip().strip("/")
    if not mid:
        return None
    if not mid.startswith("<"):
        mid = f"<{mid}>"
    db = Path.home() / "Library/Mail/V10/MailData/Envelope Index"
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT a.address, a.comment FROM messages m "
                "JOIN message_global_data g ON g.message_id = m.message_id "
                "JOIN addresses a ON a.ROWID = m.sender "
                "WHERE g.message_id_header = ? LIMIT 1",
                (mid,),
            ).fetchone()
        finally:
            conn.close()
    except Exception:
        return None
    if not row:
        return None
    return (str(row[0] or "").strip(), str(row[1] or "").strip())


# --- sender extraction ---------------------------------------------------------

def sender_from_message_file(path: Path) -> tuple[str, str]:
    """Return (address, display_name) from a .eml or .emlx file."""
    raw = path.read_bytes()
    if path.suffix.lower() == ".emlx":
        nl = raw.find(b"\n")
        try:
            count = int(raw[:nl].strip())
            raw = raw[nl + 1 : nl + 1 + count]
        except (ValueError, TypeError):
            raw = raw[nl + 1 :]
    msg = email.message_from_bytes(raw)
    from_hdr = msg.get("From") or ""
    decoded = "".join(
        (s.decode(enc or "utf-8", errors="replace") if isinstance(s, bytes) else s)
        for s, enc in email.header.decode_header(from_hdr)
    )
    name, addr = email.utils.parseaddr(decoded)
    return (addr or decoded).strip(), (name or "").strip()


def sender_from_mail_selection() -> tuple[str, str]:
    """Sender of the message currently selected in Mail.app."""
    script = '''tell application "Mail"
  set sel to selection
  if (count of sel) is 0 then return ""
  return sender of item 1 of sel
end tell'''
    p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    raw = (p.stdout or "").strip()
    if not raw:
        raise RuntimeError("No email selected in Mail (or Mail not running).")
    name, addr = email.utils.parseaddr(raw)
    return (addr or raw).strip(), (name or "").strip()


_ENVELOPE_DB = Path.home() / "Library/Mail/V10/MailData/Envelope Index"
_ACCT_UUID = "B3D3A9D3"  # Exchange (jack@jackpoorman.com)
_FOLDER_CACHE = DATA / ".mail_folders_cache.json"
_FOLDER_SKIP = {"inbox", "sent items", "sent messages", "drafts", "outbox",
                "deleted items", "deleted messages", "junk", "junk email",
                "bulk mail", "conversation history", "sync issues", "recovered messages"}


def _folders_from_index() -> list[str]:
    """Fast path via Mail's Envelope Index — needs Full Disk Access."""
    import sqlite3
    import urllib.parse

    conn = sqlite3.connect(f"file:{_ENVELOPE_DB}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT url FROM mailboxes WHERE url LIKE ?", (f"%{_ACCT_UUID}%",)
        ).fetchall()
    finally:
        conn.close()
    names = set()
    for (url,) in rows:
        leaf = urllib.parse.unquote(re.sub(r".*/", "", url or "")).strip()
        if leaf and leaf.lower() not in _FOLDER_SKIP:
            names.add(leaf)
    return sorted(names, key=str.lower)


def _folders_from_applescript() -> list[str]:
    """Automation path — works without Full Disk Access (app already has
    Automation permission from the Grab button)."""
    script = f'''tell application "Mail"
  set nms to name of every mailbox of (first account whose name is "{ACCT}")
  set {{tid, AppleScript's text item delimiters}} to {{AppleScript's text item delimiters, "\\n"}}
  set out to nms as string
  set AppleScript's text item delimiters to tid
  return out
end tell'''
    p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=90)
    names = {n.strip() for n in (p.stdout or "").splitlines() if n.strip()}
    return sorted((n for n in names if n.lower() not in _FOLDER_SKIP), key=str.lower)


def load_folder_cache() -> list[str]:
    try:
        return list(json.loads(_FOLDER_CACHE.read_text(encoding="utf-8")))
    except Exception:
        return []


def list_account_folders() -> list[str]:
    """Folder names for the dropdown. Try the fast index (FDA), then AppleScript
    (Automation, no FDA needed), then the on-disk cache. Cache any live success
    so future launches are instant."""
    for fn in (_folders_from_index, _folders_from_applescript):
        try:
            folders = fn()
        except Exception:
            folders = []
        if folders:
            try:
                _FOLDER_CACHE.write_text(json.dumps(folders), encoding="utf-8")
            except OSError:
                pass
            return folders
    return load_folder_cache()


# --- rule persistence ----------------------------------------------------------

def add_trash_rule(addr: str) -> str:
    existing = KEYWORDS_TXT.read_text(encoding="utf-8") if KEYWORDS_TXT.exists() else ""
    if addr.lower() in existing.lower():
        return f"{addr} is already on the trash list."
    with KEYWORDS_TXT.open("a", encoding="utf-8") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(addr.lower() + "\n")
    return f"Added {addr} to trash list (enforced every minute)."


def add_folder_rule(addr: str, folder: str) -> str:
    routes = {}
    if ROUTES_JSON.exists():
        try:
            routes = json.loads(ROUTES_JSON.read_text(encoding="utf-8"))
        except ValueError:
            routes = {}
    prev = routes.get(addr.lower())
    routes[addr.lower()] = folder
    ROUTES_JSON.write_text(json.dumps(routes, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    if prev and prev != folder:
        return f"Updated: {addr} now files to “{folder}” (was “{prev}”)."
    return f"Added: {addr} files to “{folder}” from now on."


def remove_trash_keywords(keywords) -> int:
    """Delete the given keyword lines from trash_keywords.txt. Match is exact
    (case-insensitive) against the stored line, so only the rules shown are
    removed. Returns how many lines were removed."""
    if not KEYWORDS_TXT.exists() or not keywords:
        return 0
    targets = {str(k).strip().lower() for k in keywords}
    kept, removed = [], 0
    for line in KEYWORDS_TXT.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and s.lower() in targets:
            removed += 1
            continue
        kept.append(line)
    KEYWORDS_TXT.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    return removed


def remove_routes(needles) -> int:
    """Delete the given route keys from mail_routes.json (case-insensitive on the
    key). Returns how many routes were removed."""
    if not ROUTES_JSON.exists() or not needles:
        return 0
    try:
        routes = json.loads(ROUTES_JSON.read_text(encoding="utf-8"))
    except ValueError:
        return 0
    targets = {str(n).strip().lower() for n in needles}
    removed = 0
    for key in list(routes.keys()):
        if str(key).strip().lower() in targets:
            del routes[key]
            removed += 1
    ROUTES_JSON.write_text(json.dumps(routes, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    return removed


# NOTE: The dropper deliberately does NOT run an enforcement pass when a rule is
# added — that would drive Mail.app synchronously and freeze the user's Mail. The
# per-minute launchd jobs (mail_rule.py, with --last 50) do the moving in the
# background. For a one-off retroactive sweep of the WHOLE Inbox (e.g. to catch
# older buried read mail beyond the newest 50), run manually:
#   .venv/bin/python data/mail_rule.py route   --recent-minutes 5256000
#   .venv/bin/python data/mail_rule.py delete  --keyword-file data/trash_keywords.txt --recent-minutes 5256000


# --- UI --------------------------------------------------------------------------

class DropperApp:
    def __init__(self):
        root = (TkinterDnD.Tk() if _DND else tk.Tk())
        self.root = root
        root.title("Mail Rule Dropper")
        root.geometry("520x560")
        # Always-on-top: keeps the dropper visible over Mail for easy side-by-side
        # selection of an email and grabbing/dropping its sender.
        root.attributes("-topmost", True)

        self.addr = tk.StringVar(value="")
        self.action = tk.StringVar(value="trash")
        self._name = ""
        self.status = tk.StringVar(
            value=("Drop an email here, or use the button below."
                   if _DND else "Drag-drop unavailable — use the button below."))

        drop = tk.Label(root, text="📥  Drag an email here\n(from Mail, or a .eml/.emlx file)",
                        relief="ridge", bd=2, height=5, bg="#eef2ff",
                        font=("Helvetica", 15))
        drop.pack(fill="x", padx=14, pady=(14, 6))
        self._drop = drop
        if _DND:
            # Register BOTH files and text/URLs: Mail delivers a message:// URL
            # (as text/URL), not a real file, so DND_FILES alone catches nothing.
            try:
                drop.drop_target_register(DND_FILES, DND_TEXT)
            except tk.TclError:
                drop.drop_target_register(DND_FILES)
            drop.dnd_bind("<<Drop>>", self.on_drop)
            drop.dnd_bind("<<DropEnter>>", lambda e: (drop.config(bg="#c7d2fe"), e.action))
            drop.dnd_bind("<<DropLeave>>", lambda e: drop.config(bg="#eef2ff"))

        tk.Button(root, text="Grab sender from email selected in Mail",
                  command=self.on_grab).pack(pady=4)
        # Right below the Grab button: shows whether the captured sender is already
        # covered by trash keyword(s) or folder route(s), with a Delete button per
        # rule (and Delete ALL when there's more than one) so a "double rule" can be
        # cleaned up before re-applying. Rebuilt on each capture via _refresh_rule_ui.
        self.rules_frame = tk.Frame(root)
        self.rules_frame.pack(fill="x", padx=14, pady=(0, 2))

        sf = tk.LabelFrame(root, text="Sender", padx=10, pady=6)
        sf.pack(fill="x", padx=14, pady=6)
        tk.Label(sf, textvariable=self.addr, font=("Helvetica", 13, "bold"),
                 fg="#1d4ed8").pack(anchor="w")

        af = tk.LabelFrame(root, text="Rule (applies now AND to all future mail)",
                           padx=10, pady=6)
        af.pack(fill="x", padx=14, pady=6)
        tk.Radiobutton(af, text="Move to Trash — now and forever",
                       variable=self.action, value="trash").pack(anchor="w")
        row = tk.Frame(af); row.pack(anchor="w", fill="x")
        tk.Radiobutton(row, text="File into folder:", variable=self.action,
                       value="folder").pack(side="left")
        self.folder_box = ttk.Combobox(row, values=[], width=30)
        self.folder_box.pack(side="left", padx=6)
        cached = load_folder_cache()
        if cached:
            self.folder_box.configure(values=cached)   # instant from cache
        else:
            self.folder_box.set("loading…")

        tk.Button(root, text="✅  Apply rule", font=("Helvetica", 13, "bold"),
                  command=self.on_apply).pack(pady=8)
        tk.Label(root, textvariable=self.status, wraplength=480,
                 fg="#444").pack(fill="x", padx=14, pady=(0, 10))

        threading.Thread(target=self._load_folders, daemon=True).start()

    def _load_folders(self):
        folders = list_account_folders()

        def _apply():
            if folders:
                cur = self.folder_box.get()
                self.folder_box.configure(values=folders)
                if cur == "loading…":
                    self.folder_box.set("")
            elif not load_folder_cache():
                self.folder_box.set("")
                self.status.set("Couldn't list folders. Trash rules still work. "
                                "For folder filing, grant this app Full Disk Access "
                                "(System Settings ▸ Privacy & Security), or make sure "
                                "Mail is open.")
        self.root.after(0, _apply)

    def on_drop(self, event):
        if self._drop is not None:
            self._drop.config(bg="#eef2ff")
        data = (event.data or "").strip()
        # tkdnd brace-wraps items containing spaces; split into tokens.
        tokens = [b or s for b, s in re.findall(r"\{([^}]*)\}|(\S+)", data)]

        # 1) message:// URL dragged from Apple Mail -> resolve via Envelope Index
        for tok in tokens:
            if tok.lower().startswith("message:"):
                got = sender_from_message_url(tok)
                if got and got[0]:
                    self._set_sender(*got, how="dragged from Mail")
                    return
                self.status.set("Dragged a Mail message, but its sender wasn't in "
                                "the local index. Try the Grab button instead.")
                return

        # 2) a real .eml/.emlx file (dragged from Finder)
        for tok in tokens:
            p = Path(tok)
            if p.suffix.lower() in (".eml", ".emlx") and p.exists():
                try:
                    addr, name = sender_from_message_file(p)
                    self._set_sender(addr, name, how=f"from {p.name}")
                    return
                except Exception as e:
                    self.status.set(f"Could not read {p.name}: {e}")
                    return

        # 3) a bare email address as text
        m = re.search(r"[\w.+-]+@[\w.-]+\.\w+", data)
        if m:
            self._set_sender(m.group(0), "", how="from dropped text")
            return

        # nothing recognized — show the raw payload so we can debug
        self.status.set("Couldn't read that drop. Use the Grab button, or tell "
                        f"me what this shows: {data[:120]!r}")

    def _matching_rules(self, addr: str, name: str = ""):
        """All rules this sender already matches, using the SAME matching the
        enforcement (data/mail_rule.py) uses: trash = normalized (alnum-only)
        keyword as a substring of the full sender field; routes = lowercase
        substring of the sender. Returns (trash_keywords, [(needle, folder), ...])
        — ALL hits, not just the first, so each can be deleted individually."""
        addr = (addr or "").strip()
        trash, routes_hit = [], []
        if not addr:
            return trash, routes_hit
        sender = f"{name} <{addr}>".strip() if name else addr
        low = sender.lower()
        norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
        nsender = norm(sender)
        if KEYWORDS_TXT.exists():
            for line in KEYWORDS_TXT.read_text(encoding="utf-8").splitlines():
                kw = line.strip()
                if not kw or kw.startswith("#"):
                    continue
                nk = norm(kw)
                if nk and nk in nsender:
                    trash.append(kw)
        if ROUTES_JSON.exists():
            try:
                routes = json.loads(ROUTES_JSON.read_text(encoding="utf-8"))
            except ValueError:
                routes = {}
            for needle, folder in routes.items():
                n = str(needle).strip().lower()
                if n and n in low:
                    routes_hit.append((str(needle), str(folder)))
        return trash, routes_hit

    def _refresh_rule_ui(self, addr: str = "", name: str = ""):
        """Rebuild the rules panel: one Delete button per existing rule, plus a
        Delete-ALL when the sender matches more than one (the 'double rule' case)."""
        frame = self.rules_frame
        for w in frame.winfo_children():
            w.destroy()
        addr = (addr or "").strip()
        if not addr:
            return
        trash, routes_hit = self._matching_rules(addr, name)
        if not trash and not routes_hit:
            tk.Label(frame, text="✓ Not on any rule list yet — this sender is new.",
                     fg="#15803d", justify="left").pack(anchor="w")
            return
        tk.Label(frame, text="Already in rules — Delete any you want to change:",
                 fg="#b45309", justify="left").pack(anchor="w")
        for kw in trash:
            r = tk.Frame(frame); r.pack(anchor="w", fill="x", pady=1)
            tk.Button(r, text="Delete", fg="#b91c1c",
                      command=lambda k=kw: self._delete_rules([k], [])).pack(side="left")
            tk.Label(r, text=f"  🗑  TRASH  (matches “{kw}”)", fg="#555").pack(side="left")
        for needle, folder in routes_hit:
            r = tk.Frame(frame); r.pack(anchor="w", fill="x", pady=1)
            tk.Button(r, text="Delete", fg="#b91c1c",
                      command=lambda n=needle: self._delete_rules([], [n])).pack(side="left")
            tk.Label(r, text=f"  📁  → “{folder}”  (matches “{needle}”)",
                     fg="#555").pack(side="left")
        if len(trash) + len(routes_hit) >= 2:
            tk.Button(frame, text="Delete ALL rules for this sender", fg="#b91c1c",
                      command=lambda: self._delete_rules(
                          list(trash), [n for n, _ in routes_hit])
                      ).pack(anchor="w", pady=(3, 0))

    def _delete_rules(self, keywords, needles):
        n = remove_trash_keywords(keywords) + remove_routes(needles)
        self.status.set(f"Removed {n} rule(s). The background jobs stop applying "
                        f"them within a minute.")
        self._refresh_rule_ui(getattr(self, "_addr", ""), self._name)

    def _set_sender(self, addr: str, name: str, how: str = ""):
        addr = (addr or "").strip()
        self._addr = addr
        self._name = (name or "").strip()
        self.addr.set(addr + (f"   ({name})" if name else ""))
        self.status.set(f"Sender captured {how} — choose a rule and Apply.".strip())
        self._refresh_rule_ui(addr, name)

    def on_grab(self):
        try:
            addr, name = sender_from_mail_selection()
            self.addr.set(f"{addr}" + (f"   ({name})" if name else ""))
            self._addr = addr
            self._name = (name or "").strip()
            self.status.set("Sender captured — choose the rule and Apply.")
            self._refresh_rule_ui(addr, name)
        except Exception as e:
            self.status.set(str(e))
            self._refresh_rule_ui("")

    def on_apply(self):
        addr = getattr(self, "_addr", "").strip()
        if not addr:
            self.status.set("Capture a sender first (drop an email or Grab).")
            return
        mode = self.action.get()
        if mode == "folder":
            folder = (self.folder_box.get() or "").strip()
            if not folder or folder.startswith("("):
                self.status.set("Pick a destination folder first.")
                return
            msg = add_folder_rule(addr, folder)
        else:
            msg = add_trash_rule(addr)
        # Rule is written instantly — no blocking enforcement pass here. The old
        # apply_now drove Mail.app through a full-Inbox scan synchronously, which
        # froze Mail for ~30s+. Instead, the per-minute launchd job scans the newest
        # ~50 Inbox messages (read or unread) and files matches within a minute, so
        # adding a rule stays instant and never interrupts your Mail use.
        self.status.set(msg + "  The 1-min background scan will file matching Inbox mail (newest ~50).")
        self._refresh_rule_ui(addr, self._name)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    if not PYTHON.exists():
        print(f"venv python not found at {PYTHON}", file=sys.stderr)
        sys.exit(1)
    DropperApp().run()
