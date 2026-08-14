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


def apply_now(mode: str) -> str:
    """Run one immediate enforcement pass (full unread scan)."""
    args = [str(PYTHON), str(MAIL_RULE)]
    args += (["delete", "--keyword-file", str(KEYWORDS_TXT)] if mode == "trash"
             else ["route"])
    p = subprocess.run(args, capture_output=True, text=True, timeout=1800)
    out = (p.stdout or p.stderr or "").strip().splitlines()
    return out[-1] if out else "(no output)"


# --- UI --------------------------------------------------------------------------

class DropperApp:
    def __init__(self):
        root = (TkinterDnD.Tk() if _DND else tk.Tk())
        self.root = root
        root.title("Mail Rule Dropper")
        root.geometry("520x470")
        # (No always-on-top: the window behaves like a normal window so it can go
        # behind others without needing to be minimized.)

        self.addr = tk.StringVar(value="")
        self.action = tk.StringVar(value="trash")
        self.rule_status_var = tk.StringVar(value="")
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
        # Shows, right below the Grab button, whether the captured sender is
        # already covered by a trash keyword or a folder route.
        tk.Label(root, textvariable=self.rule_status_var, wraplength=490,
                 justify="left", fg="#555").pack(fill="x", padx=14, pady=(0, 2))

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

    def _rule_membership(self, addr: str, name: str = "") -> str:
        """Is this sender already covered by a trash keyword or a folder route?
        Uses the SAME matching the enforcement (data/mail_rule.py) uses:
        trash = normalized (alnum-only) keyword as a substring of the full sender
        field; routes = lowercase substring of the sender."""
        addr = (addr or "").strip()
        if not addr:
            return ""
        sender = f"{name} <{addr}>".strip() if name else addr
        low = sender.lower()
        norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
        nsender = norm(sender)

        hits = []
        # Trash keyword list.
        if KEYWORDS_TXT.exists():
            for line in KEYWORDS_TXT.read_text(encoding="utf-8").splitlines():
                kw = line.strip()
                if not kw or kw.startswith("#"):
                    continue
                nk = norm(kw)
                if nk and nk in nsender:
                    hits.append(f"🗑  on the TRASH list (matches “{kw}”)")
                    break
        # Folder routes.
        if ROUTES_JSON.exists():
            try:
                routes = json.loads(ROUTES_JSON.read_text(encoding="utf-8"))
            except ValueError:
                routes = {}
            for needle, folder in routes.items():
                n = str(needle).strip().lower()
                if n and n in low:
                    hits.append(f"📁  routes to “{folder}” (matches “{needle}”)")
                    break

        if not hits:
            return "✓ Not on any rule list yet — this sender is new."
        return "Already in rules:\n    " + "\n    ".join(hits)

    def _set_sender(self, addr: str, name: str, how: str = ""):
        addr = (addr or "").strip()
        self._addr = addr
        self._name = (name or "").strip()
        self.addr.set(addr + (f"   ({name})" if name else ""))
        self.status.set(f"Sender captured {how} — choose a rule and Apply.".strip())
        self.rule_status_var.set(self._rule_membership(addr, name))

    def on_grab(self):
        try:
            addr, name = sender_from_mail_selection()
            self.addr.set(f"{addr}" + (f"   ({name})" if name else ""))
            self._addr = addr
            self._name = (name or "").strip()
            self.status.set("Sender captured — choose the rule and Apply.")
            self.rule_status_var.set(self._rule_membership(addr, name))
        except Exception as e:
            self.status.set(str(e))
            self.rule_status_var.set("")

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
        self.status.set(msg + "  Applying to current Inbox…")

        def _bg():
            try:
                result = apply_now("trash" if mode == "trash" else "route")
            except Exception as e:
                result = f"apply failed: {e}"
            self.root.after(0, lambda: self.status.set(f"{msg}\n{result}"))
        threading.Thread(target=_bg, daemon=True).start()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    if not PYTHON.exists():
        print(f"venv python not found at {PYTHON}", file=sys.stderr)
        sys.exit(1)
    DropperApp().run()
