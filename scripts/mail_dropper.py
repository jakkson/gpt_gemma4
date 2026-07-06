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
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND = True
except Exception:  # app still works via the "grab selected" button
    _DND = False


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


def list_account_folders() -> list[str]:
    """Names of mailboxes on the account, for the folder dropdown.

    Read from Mail's Envelope Index (SQLite, read-only) instead of AppleScript:
    Mail's scripting interface stalls for minutes while it syncs / is driven by
    the per-minute rule jobs, but the index is always instantly readable."""
    import sqlite3
    import urllib.parse

    conn = sqlite3.connect(f"file:{_ENVELOPE_DB}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT url FROM mailboxes WHERE url LIKE ?", (f"%{_ACCT_UUID}%",)
        ).fetchall()
    finally:
        conn.close()
    skip = {"inbox", "sent items", "sent messages", "drafts", "outbox",
            "deleted items", "deleted messages", "junk", "junk email",
            "bulk mail", "archive", "conversation history", "sync issues"}
    names = set()
    for (url,) in rows:
        leaf = urllib.parse.unquote(re.sub(r".*/", "", url or "")).strip()
        if leaf and leaf.lower() not in skip:
            names.add(leaf)
    return sorted(names, key=str.lower)


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
        root.geometry("520x420")
        root.attributes("-topmost", True)

        self.addr = tk.StringVar(value="")
        self.action = tk.StringVar(value="trash")
        self.status = tk.StringVar(
            value=("Drop an email here, or use the button below."
                   if _DND else "Drag-drop unavailable — use the button below."))

        drop = tk.Label(root, text="📥  Drop email here\n(drag from Mail or a .eml/.emlx file)",
                        relief="ridge", bd=2, height=5,
                        font=("Helvetica", 15))
        drop.pack(fill="x", padx=14, pady=(14, 6))
        if _DND:
            drop.drop_target_register(DND_FILES)
            drop.dnd_bind("<<Drop>>", self.on_drop)

        tk.Button(root, text="Grab sender from email selected in Mail",
                  command=self.on_grab).pack(pady=4)

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
        self.folder_box = ttk.Combobox(row, values=["(loading folders…)"], width=30)
        self.folder_box.pack(side="left", padx=6)

        tk.Button(root, text="✅  Apply rule", font=("Helvetica", 13, "bold"),
                  command=self.on_apply).pack(pady=8)
        tk.Label(root, textvariable=self.status, wraplength=480,
                 fg="#444").pack(fill="x", padx=14, pady=(0, 10))

        threading.Thread(target=self._load_folders, daemon=True).start()

    def _load_folders(self):
        try:
            folders = list_account_folders()
            self.root.after(0, lambda: self.folder_box.configure(values=folders))
        except Exception as e:
            self.root.after(0, lambda: self.status.set(f"Folder list failed: {e}"))

    def on_drop(self, event):
        # tkdnd delivers paths brace-wrapped when they contain spaces.
        paths = re.findall(r"\{([^}]+)\}|(\S+)", event.data or "")
        for brace, bare in paths:
            p = Path(brace or bare)
            if p.suffix.lower() in (".eml", ".emlx") and p.exists():
                try:
                    addr, name = sender_from_message_file(p)
                    self.addr.set(f"{addr}" + (f"   ({name})" if name else ""))
                    self._addr = addr
                    self.status.set("Sender captured — choose the rule and Apply.")
                    return
                except Exception as e:
                    self.status.set(f"Could not read {p.name}: {e}")
                    return
        self.status.set("That didn't look like an email file. Tip: drag from Mail "
                        "to the Desktop first if a direct drop doesn't land, or "
                        "use the Grab button.")

    def on_grab(self):
        try:
            addr, name = sender_from_mail_selection()
            self.addr.set(f"{addr}" + (f"   ({name})" if name else ""))
            self._addr = addr
            self.status.set("Sender captured — choose the rule and Apply.")
        except Exception as e:
            self.status.set(str(e))

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
