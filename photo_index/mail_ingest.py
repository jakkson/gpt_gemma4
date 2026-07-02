"""Ingest Apple Mail .emlx files into the shared SQLite/FTS index.

Walks the local Mail store (~/.Library/Mail/V10/<account-uuid>/) for each
configured account, parses each .emlx file (RFC 2822 email), extracts plain
text from the body, and upserts it into photo_meta via upsert_photo().

UUID per message: "mail:<message-id-header>" so re-runs are idempotent.

Excluded folders (Trash, Junk, Sent, Drafts, etc.) are skipped at walk time.

Usage:
  python -m photo_index.mail_ingest
  python -m photo_index.mail_ingest --accounts B3D3A9D3 93F75222
  python -m photo_index.mail_ingest --progress-every 500 --db /path/to/index.sqlite
"""
from __future__ import annotations

import argparse
import email
import email.header
import email.policy
import email.utils
import hashlib
import html.parser
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from photo_index.ingest_lock import global_ingest_lock
from photo_index.store import (
    already_indexed,
    commit_ingest,
    connect,
    init_schema,
    optimize,
    upsert_photo,
)

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"
_MAIL_ROOT = Path.home() / "Library" / "Mail" / "V10"

# Account UUIDs: jack@jackpoorman.com (Exchange) + iCloud
_DEFAULT_ACCOUNTS = [
    "B3D3A9D3-A469-41EB-B66B-FBF4B32593E8",
    "93F75222-1FC0-4708-8C04-C7E86D57AC6C",
]

# Mbox folder names to skip (case-insensitive). Checked against the .mbox
# directory name at the top level of each account folder.
_EXCLUDE_FOLDERS = {
    "trash", "deleted messages", "deleted items", "deleted",
    "junk", "junk email", "bulk mail", "spam",
    "sent", "sent messages", "sent items",
    "drafts", "outbox", "sendlater",
    "[airmail]", "-airmail-",
}

_BODY_MAX_CHARS = 8_000
_COMMIT_EVERY = 200


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class _HTMLStripper(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._skip = True
        elif tag in ("br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse whitespace runs but preserve paragraph breaks
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _strip_html(html_text: str) -> str:
    s = _HTMLStripper()
    try:
        s.feed(html_text)
        return s.get_text()
    except Exception:
        # Last resort: just remove tags
        return re.sub(r"<[^>]+>", " ", html_text).strip()


def _clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# emlx parsing
# ---------------------------------------------------------------------------

def _extract_body(msg: email.message.Message) -> str:
    """Return plain-text body, capped at _BODY_MAX_CHARS."""
    charset_fallback = "utf-8"

    def _decode_part(part: email.message.Message) -> str:
        payload = part.get_payload(decode=True)
        if not payload:
            return ""
        charset = part.get_content_charset() or charset_fallback
        return payload.decode(charset, errors="replace")

    if msg.is_multipart():
        # Prefer text/plain first
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    return _clean_text(_decode_part(part))[:_BODY_MAX_CHARS]
                except Exception:
                    pass
        # Fallback: text/html
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                try:
                    return _strip_html(_decode_part(part))[:_BODY_MAX_CHARS]
                except Exception:
                    pass
        return ""
    else:
        try:
            ct = msg.get_content_type()
            text = _decode_part(msg)
            if ct == "text/html":
                return _strip_html(text)[:_BODY_MAX_CHARS]
            return _clean_text(text)[:_BODY_MAX_CHARS]
        except Exception:
            return ""


def _parse_emlx(path: Path) -> dict | None:
    """Parse one .emlx file and return a dict of fields, or None on failure."""
    try:
        raw = path.read_bytes()
        nl = raw.find(b"\n")
        if nl < 0:
            return None
        try:
            byte_count = int(raw[:nl].strip())
        except ValueError:
            return None
        email_bytes = raw[nl + 1 : nl + 1 + byte_count]
        msg = email.message_from_bytes(email_bytes)

        message_id = (msg.get("Message-ID") or "").strip().strip("<>").strip()
        if not message_id:
            # Stable fallback: hash first 512 bytes of content
            message_id = "hash:" + hashlib.sha1(email_bytes[:512]).hexdigest()

        from_header = msg.get("From") or ""
        subject = (msg.get("Subject") or "(no subject)").strip()
        # Decode RFC 2047 encoded words in subject/from
        subject = email.header.decode_header(subject)
        subject = "".join(
            (s.decode(enc or "utf-8", errors="replace") if isinstance(s, bytes) else s)
            for s, enc in subject
        )
        from_decoded = email.header.decode_header(from_header)
        from_str = "".join(
            (s.decode(enc or "utf-8", errors="replace") if isinstance(s, bytes) else s)
            for s, enc in from_decoded
        ).strip()

        date_iso: str | None = None
        date_str = msg.get("Date") or ""
        if date_str:
            try:
                dt = email.utils.parsedate_to_datetime(date_str)
                date_iso = dt.isoformat()
            except Exception:
                pass

        body = _extract_body(msg)

        return {
            "message_id": message_id,
            "from_str": from_str,
            "subject": subject,
            "date_iso": date_iso,
            "body": body,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Walk helpers
# ---------------------------------------------------------------------------

def _folder_name_from_path(path: Path, account_root: Path) -> str:
    """Return the top-level .mbox folder name for this emlx path."""
    try:
        rel = path.relative_to(account_root)
        for part in rel.parts:
            if part.endswith(".mbox"):
                return part[:-5]
    except ValueError:
        pass
    return ""


def _should_skip_folder(folder_name: str) -> bool:
    return folder_name.lower() in _EXCLUDE_FOLDERS


def _iter_emlx(account_root: Path):
    """Yield (emlx_path, folder_name) for all eligible .emlx files."""
    for mbox_dir in sorted(account_root.iterdir()):
        if not mbox_dir.name.endswith(".mbox"):
            continue
        folder_name = mbox_dir.name[:-5]
        if _should_skip_folder(folder_name):
            continue
        for emlx_path in mbox_dir.rglob("*.emlx"):
            if "partial.emlx" in emlx_path.name:
                continue
            yield emlx_path, folder_name


# ---------------------------------------------------------------------------
# Main ingest
# ---------------------------------------------------------------------------

def _indexed_mail_paths(conn) -> set[str]:
    """All emlx paths already in the index (one query).

    .emlx files are immutable — Mail writes a new file per message and never
    edits it — so a path match means the message is already ingested and the
    file need not be opened or parsed again. This turns the nightly re-run
    from ~4.5 min of parsing 160k files into a fast directory walk.
    """
    return {
        r[0]
        for r in conn.execute(
            "SELECT image_path_used FROM photo_meta "
            "WHERE uuid LIKE 'mail:%' AND image_path_used != ''"
        )
    }


def ingest_account(
    conn,
    account_root: Path,
    *,
    progress_every: int = 1000,
    force: bool = False,
) -> tuple[int, int, int]:
    """Ingest one account. Returns (indexed, skipped_dup, errors)."""
    indexed = skipped_dup = errors = 0
    batch = 0
    seen_paths = set() if force else _indexed_mail_paths(conn)

    for emlx_path, folder_name in _iter_emlx(account_root):
        # Path-level skip: emlx files are immutable, so an already-indexed
        # path never needs re-parsing. (Message-id dedup below still guards
        # against the same message appearing under a new path.)
        if str(emlx_path) in seen_paths:
            skipped_dup += 1
            continue

        data = _parse_emlx(emlx_path)
        if data is None:
            errors += 1
            continue

        uuid = f"mail:{data['message_id']}"

        if not force and already_indexed(conn, uuid):
            skipped_dup += 1
            continue

        subject = data["subject"]
        from_str = data["from_str"]
        date_iso = data["date_iso"]
        body = data["body"]

        if not body and not subject:
            skipped_dup += 1  # nothing to index
            continue

        # filename carries subject + sender so FTS covers them even if body is short
        filename = f"{date_iso or 'unknown-date'} | {subject} | {from_str} [{folder_name}]"
        ocr_text = f"Subject: {subject}\nFrom: {from_str}\n\n{body}"

        # message:// URL lets the Gradio UI open Mail.app directly to this email
        mid = data["message_id"]
        if not mid.startswith("hash:"):
            import urllib.parse
            open_url = "message://%3C" + urllib.parse.quote(mid, safe="") + "%3E"
        else:
            open_url = ""

        upsert_photo(
            conn,
            uuid=uuid,
            filename=filename,
            date_iso=date_iso,
            ocr_text=ocr_text,
            vlm_text="",
            image_path_used=str(emlx_path),
            open_url=open_url,
            commit=False,
        )
        indexed += 1
        batch += 1

        if batch >= _COMMIT_EVERY:
            commit_ingest(conn)
            batch = 0

        total = indexed + skipped_dup + errors
        if progress_every and total % progress_every == 0:
            _log(f"  [{account_root.name[:8]}] {total:,} seen | "
                 f"{indexed:,} new | {skipped_dup:,} dup | {errors:,} err | "
                 f"folder={folder_name}")

    commit_ingest(conn)
    return indexed, skipped_dup, errors


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Ingest Apple Mail .emlx files into the index.")
    p.add_argument("--db", default=str(_DEFAULT_DB))
    p.add_argument(
        "--accounts",
        nargs="+",
        default=_DEFAULT_ACCOUNTS,
        metavar="UUID",
        help="Account UUID folders under ~/Library/Mail/V10/ to ingest.",
    )
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--force", action="store_true", help="Re-index already-indexed messages.")
    args = p.parse_args(argv)

    db_path = Path(args.db)
    conn = connect(db_path)
    init_schema(conn)

    t_start = time.time()
    total_indexed = total_dup = total_err = 0

    with global_ingest_lock():
        for acct_id in args.accounts:
            # Accept both full UUID and prefix (e.g. "B3D3A9D3")
            matches = list(_MAIL_ROOT.glob(f"{acct_id}*"))
            if not matches:
                _log(f"[mail_ingest] account not found: {acct_id}")
                continue
            account_root = matches[0]
            _log(f"[mail_ingest] ingesting {account_root.name} ...")

            idx, dup, err = ingest_account(
                conn,
                account_root,
                progress_every=args.progress_every,
                force=args.force,
            )
            total_indexed += idx
            total_dup += dup
            total_err += err
            _log(f"[mail_ingest] {account_root.name[:8]}: "
                 f"{idx:,} new | {dup:,} already indexed | {err:,} errors")

    if total_indexed:
        optimize(conn)
    elapsed = time.time() - t_start
    _log(f"[mail_ingest] done in {elapsed:.0f}s — "
         f"{total_indexed:,} indexed | {total_dup:,} skipped | {total_err:,} errors")


if __name__ == "__main__":
    main()
