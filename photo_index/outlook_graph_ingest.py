"""Ingest Microsoft 365 mail through Microsoft Graph into ``photo_index`` SQLite/FTS.

Azure Entra ID setup (one-time)
-----------------------------
1. **Entra admin center → Applications → App registrations → New registration.**
   - Name: e.g. ``personal-photo-index-mail``.
   - Redirect URI: *Mobile and desktop applications* → add ``http://localhost``.
2. **API permissions → Microsoft Graph → Delegated permissions:**
   - ``Mail.Read`` (required).
   - ``offline_access`` (recommended so refresh tokens are issued).
   - Grant consent for your org / admin consent if policy requires it.
3. Copy **Application (client) ID** into ``GRAPH_CLIENT_ID``.

Environment variables
---------------------
``GRAPH_CLIENT_ID``  
    Required unless passed as ``--client-id``.

``GRAPH_TENANT_ID``  
    Optional. Defaults to ``organizations`` (work/school accounts). Use ``common``
    for mixed consumer/work, or your tenant GUID for a single tenant.

``GRAPH_MAILBOX_UPN``  
    Optional default mailbox user principal name for ``--mailbox``. Overrides ``/me``
    when set.

Runs share ``data/content_ingest.lock`` with other ingest scripts unless you pass
``--no-global-ingest-lock``.

Token cache path: ``data/graph_mail_token_cache.json`` (under repo ``data/``).
Delta checkpoint path: ``data/graph_mail_delta.json``.
Indexed UUID prefix: ``m365:{graph_message_id}``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

import msal
import requests
import sqlite3

from photo_index.ingest_lock import global_ingest_lock
from photo_index.store import commit_ingest, connect, delete_index_row, init_schema, upsert_photo

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "photo_index.sqlite"
_DEFAULT_TOKEN_CACHE = Path(__file__).resolve().parent.parent / "data" / "graph_mail_token_cache.json"
_DEFAULT_DELTA_PATH = Path(__file__).resolve().parent.parent / "data" / "graph_mail_delta.json"

_GRAPH_ROOT = "https://graph.microsoft.com/v1.0"
_SCOPES = ["https://graph.microsoft.com/Mail.Read", "offline_access"]

_BR_HTML_RE = re.compile(r"<\s*br\s*/?>", re.I)
_TAG_HTML_RE = re.compile(r"<[^>]+>")
_FILENAME_SAFE_RE = re.compile(r'[/\\:*?"<>|\x00-\x1f]')


def _log(msg: str) -> None:
    print(msg, flush=True)


class _JsonTokenCache(msal.SerializableTokenCache):
    """Serializable MSAL cache backed by a single UTF-8 JSON file."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        if path.exists():
            try:
                self.deserialize(path.read_text(encoding="utf-8"))
            except Exception:
                pass

    def flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(self.serialize(), encoding="utf-8")


def _authority_host(tenant: str) -> str:
    t = (tenant or "organizations").strip()
    return f"https://login.microsoftonline.com/{t}"


def _mailbox_segment(mailbox_upn: str | None) -> str:
    if mailbox_upn and mailbox_upn.strip():
        return f"/users/{urllib.parse.quote(mailbox_upn.strip(), safe='')}"
    return "/me"


def _delta_entry_url(mailbox_segment: str, folder_well_known: str | None) -> str:
    if folder_well_known and folder_well_known.strip():
        fk = urllib.parse.quote(folder_well_known.strip(), safe="")
        return f"{_GRAPH_ROOT}{mailbox_segment}/mailFolders/{fk}/messages/delta"
    return f"{_GRAPH_ROOT}{mailbox_segment}/messages/delta"


def _normalize_mail_scope(mailbox_upn: str | None, folder_well_known: str | None) -> str:
    mb = (mailbox_upn or "").strip() or "me"
    fd = (folder_well_known or "").strip() or "all"
    return f"{mb}|{fd}"


def _load_delta_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_delta_state(path: Path, *, delta_link: str | None, mail_scope: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"delta_link": delta_link, "mail_scope": mail_scope}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _recipient_line(rows: list[dict[str, Any]] | None) -> str:
    parts: list[str] = []
    for row in rows or []:
        ea = row.get("emailAddress") or {}
        addr = (ea.get("address") or "").strip()
        name = (ea.get("name") or "").strip()
        if not addr:
            continue
        parts.append(f"{name} <{addr}>" if name else addr)
    return "; ".join(parts)


def _plain_body(body: dict[str, Any] | None, preview: str) -> str:
    if not body:
        return (preview or "").strip()
    ctype = str(body.get("contentType") or "").lower()
    content = str(body.get("content") or "")
    if ctype == "text":
        return content.strip()
    if ctype == "html":
        t = _BR_HTML_RE.sub("\n", content)
        t = _TAG_HTML_RE.sub(" ", t)
        return "\n".join(line.strip() for line in t.splitlines() if line.strip()).strip()
    return (preview or "").strip() or content.strip()


def _safe_mail_filename(subject: str) -> str:
    raw = (subject or "").strip() or "(no subject)"
    raw = _FILENAME_SAFE_RE.sub("_", raw).strip()
    return raw[:200] if len(raw) > 200 else raw


def _graph_get(session: requests.Session, url: str, *, timeout: float = 120.0) -> dict[str, Any]:
    backoff = 3.0
    for attempt in range(8):
        resp = session.get(url, timeout=timeout)
        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            try:
                sleep_s = float(ra) if ra else backoff
            except ValueError:
                sleep_s = backoff
            _log(f"[graph] 429 rate limited; sleeping {sleep_s:.1f}s …")
            time.sleep(sleep_s)
            backoff = min(backoff * 2, 120.0)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Microsoft Graph: too many 429 retries")


def _acquire_token(
    *,
    client_id: str,
    tenant: str,
    cache_path: Path,
    auth_mode: str,
) -> str:
    cache = _JsonTokenCache(cache_path)
    app = msal.PublicClientApplication(
        client_id.strip(),
        authority=_authority_host(tenant),
        token_cache=cache,
    )
    accounts = app.get_accounts()
    result = None
    if accounts:
        result = app.acquire_token_silent(_SCOPES, account=accounts[0])
    if result and result.get("access_token"):
        cache.flush()
        return str(result["access_token"])

    auth_mode = auth_mode.strip().lower()
    if auth_mode == "device":
        flow = app.initiate_device_flow(scopes=_SCOPES)
        if "user_code" not in flow:
            raise RuntimeError(f"Device flow failed: {flow}")
        _log(flow["message"])
        result = app.acquire_token_by_device_flow(flow)
    elif auth_mode == "interactive":
        result = app.acquire_token_interactive(scopes=_SCOPES)
    else:
        raise ValueError(f"Unknown --auth mode: {auth_mode}")

    cache.flush()
    if not result or not result.get("access_token"):
        err = (result or {}).get("error_description") or (result or {}).get("error") or "unknown"
        raise RuntimeError(f"Authentication failed: {err}")
    return str(result["access_token"])


def _process_message_item(
    conn: sqlite3.Connection, item: dict[str, Any], *, commit_every: int, counters: dict[str, int]
) -> None:
    if item.get("@removed"):
        mid = item.get("id")
        if not mid:
            return
        uuid = f"m365:{mid}"
        delete_index_row(conn, uuid, commit=False)
        counters["deleted"] += 1
        if counters["deleted"] % max(commit_every, 1) == 0:
            commit_ingest(conn)
        return

    mid = item.get("id")
    if not mid:
        return
    uuid = f"m365:{mid}"
    subject = str(item.get("subject") or "").strip()
    preview = str(item.get("bodyPreview") or "").strip()
    body = item.get("body")
    if isinstance(body, dict):
        plain = _plain_body(body, preview)
    else:
        plain = preview
    if not plain.strip():
        counters["skipped_empty"] += 1
        return

    received = item.get("receivedDateTime") or item.get("sentDateTime") or ""
    from_row = (item.get("from") or {}).get("emailAddress") or {}
    from_addr = (from_row.get("address") or "").strip()
    from_name = (from_row.get("name") or "").strip()
    from_disp = f"{from_name} <{from_addr}>" if from_name and from_addr else (from_addr or from_name or "")

    to_s = _recipient_line(item.get("toRecipients"))
    cc_s = _recipient_line(item.get("ccRecipients"))
    conv = str(item.get("conversationId") or "")
    has_att = bool(item.get("hasAttachments"))

    meta_bits = [
        "source=outlook_graph",
        f"id={mid}",
        f"conversation={conv}",
        f"from={from_disp}",
        f"has_attachments={int(has_att)}",
    ]
    if to_s:
        meta_bits.append(f"to={to_s}")
    if cc_s:
        meta_bits.append(f"cc={cc_s}")

    filename = f"mail:{_safe_mail_filename(subject)}"

    upsert_photo(
        conn,
        uuid=uuid,
        filename=filename,
        date_iso=str(received).strip() or None,
        ocr_text=plain,
        vlm_text=" ".join(meta_bits),
        image_path_used="",
        commit=False,
    )
    counters["indexed"] += 1
    if counters["indexed"] % max(commit_every, 1) == 0:
        commit_ingest(conn)


def run_outlook_graph_ingest(
    *,
    index_db_path: Path,
    client_id: str,
    tenant: str,
    token_cache_path: Path,
    delta_path: Path,
    mailbox_upn: str | None,
    folder_well_known: str | None,
    auth_mode: str,
    reset_delta: bool,
    commit_every: int,
    page_hint: int,
) -> dict[str, int | float]:
    mail_scope = _normalize_mail_scope(mailbox_upn, folder_well_known)
    state = _load_delta_state(delta_path)
    stored_scope = str(state.get("mail_scope") or "")
    delta_link = None if reset_delta else state.get("delta_link")
    if isinstance(delta_link, str):
        delta_link = delta_link.strip() or None
    if stored_scope and stored_scope != mail_scope:
        _log(f"[graph] Mailbox scope changed ({stored_scope!r} -> {mail_scope!r}); resetting delta.")
        delta_link = None

    token = _acquire_token(
        client_id=client_id,
        tenant=tenant,
        cache_path=token_cache_path,
        auth_mode=auth_mode,
    )
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Prefer": 'outlook.body-content-type="text"',
        }
    )

    mb_seg = _mailbox_segment(mailbox_upn)
    if delta_link:
        url = delta_link
        _log("[graph] Continuing incremental sync from saved delta link.")
    else:
        url = _delta_entry_url(mb_seg, folder_well_known)
        params = {"$top": str(max(1, min(page_hint, 999)))}
        url = f"{url}?{urllib.parse.urlencode(params)}"
        _log(f"[graph] Starting full sync: {url.split('?')[0]}")

    conn = connect(index_db_path)
    init_schema(conn)
    counters = {"indexed": 0, "deleted": 0, "skipped_empty": 0, "pages": 0}
    t0 = time.perf_counter()
    new_delta: str | None = None

    try:
        while url:
            payload = _graph_get(session, url)
            counters["pages"] += 1
            batch = payload.get("value")
            if isinstance(batch, list):
                for item in batch:
                    if not isinstance(item, dict):
                        continue
                    _process_message_item(conn, item, commit_every=commit_every, counters=counters)

            url = payload.get("@odata.nextLink")
            if isinstance(url, str) and url.strip():
                url = url.strip()
            else:
                url = None

            dl = payload.get("@odata.deltaLink")
            if isinstance(dl, str) and dl.strip():
                new_delta = dl.strip()

        if new_delta:
            _save_delta_state(delta_path, delta_link=new_delta, mail_scope=mail_scope)
            _log("[graph] Saved delta link for next incremental run.")
        else:
            _log("[graph warn] No @odata.deltaLink in last response; incremental sync may re-fetch broadly.")

        commit_ingest(conn)
    finally:
        conn.close()

    elapsed = time.perf_counter() - t0
    _log(
        f"[graph done] indexed={counters['indexed']} deleted={counters['deleted']} "
        f"skipped_empty={counters['skipped_empty']} pages={counters['pages']} "
        f"time={elapsed:.1f}s db={index_db_path}"
    )
    counters["elapsed"] = elapsed
    return counters


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Index Microsoft 365 mail via Graph API into photo_index SQLite/FTS.",
        epilog="Azure Entra registration (delegated Mail.Read) and env vars: see module docstring.",
    )
    p.add_argument("--db", type=str, default=str(_DEFAULT_DB), help="Target SQLite index DB path.")
    p.add_argument(
        "--client-id",
        type=str,
        default=os.environ.get("GRAPH_CLIENT_ID", ""),
        help="Entra application (client) ID (or set GRAPH_CLIENT_ID).",
    )
    p.add_argument(
        "--tenant",
        type=str,
        default=os.environ.get("GRAPH_TENANT_ID", "organizations"),
        help="Tenant segment: organizations (default), common, or tenant GUID.",
    )
    p.add_argument(
        "--token-cache",
        type=str,
        default=str(_DEFAULT_TOKEN_CACHE),
        help="Path for MSAL token cache JSON.",
    )
    p.add_argument(
        "--delta-path",
        type=str,
        default=str(_DEFAULT_DELTA_PATH),
        help="Path for delta checkpoint JSON.",
    )
    p.add_argument(
        "--mailbox",
        type=str,
        default=os.environ.get("GRAPH_MAILBOX_UPN", "") or None,
        help="Optional user principal name (sync /users/{UPN}/mailFolders/… instead of /me).",
    )
    p.add_argument(
        "--folder",
        type=str,
        default=None,
        help='Well-known folder only (e.g. "inbox"). Default: all folders reachable from /messages/delta.',
    )
    p.add_argument(
        "--auth",
        choices=("interactive", "device"),
        default="interactive",
        help="OAuth acquisition mode (first run or expired refresh token).",
    )
    p.add_argument(
        "--reset-delta",
        action="store_true",
        help="Ignore saved delta link and perform a full sync again.",
    )
    p.add_argument("--commit-every", type=int, default=50, help="SQLite commit every N indexed/deleted rows.")
    p.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="Graph $top hint per page during initial delta crawl (max 999).",
    )
    p.add_argument(
        "--no-global-ingest-lock",
        action="store_true",
        help="Disable shared content-ingest lock (not recommended).",
    )
    args = p.parse_args(argv)

    if args.commit_every < 1:
        p.error("--commit-every must be >= 1")

    cid = (args.client_id or "").strip()
    if not cid:
        p.error(
            "Missing client id: pass --client-id or set GRAPH_CLIENT_ID "
            "(Entra app registration → Application ID)."
        )

    index_db_path = Path(os.path.abspath(args.db))
    token_cache_path = Path(os.path.abspath(args.token_cache))
    delta_path = Path(os.path.abspath(args.delta_path))

    def _run() -> None:
        run_outlook_graph_ingest(
            index_db_path=index_db_path,
            client_id=cid,
            tenant=str(args.tenant or "organizations"),
            token_cache_path=token_cache_path,
            delta_path=delta_path,
            mailbox_upn=(args.mailbox.strip() if args.mailbox else None),
            folder_well_known=args.folder,
            auth_mode=str(args.auth),
            reset_delta=bool(args.reset_delta),
            commit_every=int(args.commit_every),
            page_hint=int(args.page_size),
        )

    if args.no_global_ingest_lock:
        _run()
        return

    with global_ingest_lock() as have_lock:
        if not have_lock:
            _log("[lock] Another content ingest is already running; skipping this run.")
            return
        _run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
