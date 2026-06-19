"""Ingest Microsoft 365 mail through Microsoft Graph into ``photo_index`` SQLite/FTS.

Azure Entra ID setup (one-time)
-----------------------------
1. **Entra admin center → Applications → App registrations → New registration.**
   - Name: e.g. ``personal-photo-index-mail``.
   - Redirect URI: *Mobile and desktop applications* → add ``http://localhost``.
2. **API permissions → Microsoft Graph → Delegated permissions:**
   - ``Mail.Read`` (required).
   - ``User.Read`` (recommended; aligns with ``/me`` and typical consent screens).
   - ``offline_access`` (recommended so refresh tokens are issued; keep this in the portal —
     do **not** pass it in code scopes with MSAL 1.36+, MSAL adds OIDC scopes automatically).
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

CLI scope (no Gradio involvement): ``--folder inbox`` (or other well-known folder name)
limits sync to that folder; use a separate ``--delta-path`` per folder if you index several.
``--since ISO8601`` skips older mail when indexing (Graph still enumerates the delta).
``--max-messages N`` stops after N indexed messages and does not advance the delta checkpoint.

``--sync-named-presets`` runs the built-in **custom** folders only (no Inbox).

``--sync-inbox-and-named-presets`` runs **Inbox** and **Sent** (well-known ``inbox`` and
``sentitems`` only — not the whole mailbox) **plus** those custom folders—still **no**
mailbox-wide ``/messages/delta``.

``--progress-every-pages N`` prints a short progress line every N Graph delta pages
(use ``0`` to disable).

``--test-connection`` acquires a token and performs one minimal Graph mail call (no ingest).

Runs share ``data/content_ingest.lock`` with other ingest scripts unless you pass
``--no-global-ingest-lock``.

Token cache path: ``data/graph_mail_token_cache.json`` (under repo ``data/``).
If Graph returns **401** after sign-in, delete this file and re-auth — stale tokens often
lack **Mail.Read** after permission changes. Confirm Entra **Grant admin consent** for delegated Mail.Read.
Try ``--force-token-refresh`` once after changing permissions.
If **401** persists while the logged JWT already lists ``Mail.Read`` (and ``--test-connection``
shows whether ``/me`` works but ``mailFolders`` does not), the signed-in identity may have no
**Exchange Online mailbox** or tenant policy blocks mail APIs — verify mail in **Outlook on the web``.
If ``/me`` shows a UPN containing ``#EXT#``, that is an **invited / external Azure AD identity**
in this tenant; Graph may still show ``userType=Member``, but **Exchange often does not expose
a mailbox** for Graph mail APIs — sign in with the **native mailbox account** for that org
(or set ``GRAPH_TENANT_ID`` / ``--tenant`` to the tenant where your mailbox lives).
If ``userType=Guest``, the same mailbox limitation usually applies.
Logs include Graph ``request-id`` headers when present for support tickets.

Delta checkpoint path: ``data/graph_mail_delta.json``.
Indexed UUID prefix: ``m365:{graph_message_id}``. Each row stores Graph ``webLink``
in ``photo_meta.open_url`` so the Gradio UI can open the message in Outlook on the web.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
from dataclasses import dataclass
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
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
# (well_known_segment, delta_filename) for --sync-inbox-and-named-presets (before named presets).
_WELL_KNOWN_PRESET_BUNDLE: tuple[tuple[str, str], ...] = (
    ("inbox", "graph_mail_delta_wk_inbox.json"),
    ("sentitems", "graph_mail_delta_wk_sentitems.json"),
)

_GRAPH_ROOT = "https://graph.microsoft.com/v1.0"
# MSAL 1.36+ rejects reserved OIDC scopes in this list; do not add offline_access/openid/profile here.
# Still register delegated ``offline_access`` in Entra — MSAL adds it on the wire as needed.
_SCOPES = [
    "https://graph.microsoft.com/Mail.Read",
    "https://graph.microsoft.com/User.Read",
]

_BR_HTML_RE = re.compile(r"<\s*br\s*/?>", re.I)
_TAG_HTML_RE = re.compile(r"<[^>]+>")
_FILENAME_SAFE_RE = re.compile(r'[/\\:*?"<>|\x00-\x1f]')


@dataclass(frozen=True)
class NamedFolderPreset:
    """Custom mailbox folder by exact Outlook display name + index lookback."""

    display_name: str
    lookback_days: int
    delta_slug: str


# Built-in named-folder syncs (display names must match Outlook exactly).
NAMED_FOLDER_PRESETS_DEFAULT: tuple[NamedFolderPreset, ...] = (
    NamedFolderPreset("002-Temp Save FOlder", 730, "named_002_temp_save"),  # ~2 years
    NamedFolderPreset("0-Learning", 365, "named_0_learning"),  # ~1 year
    NamedFolderPreset("0-Merch", 243, "named_0_merch"),  # ~8 months (243 ≈ 8×365/12)
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _parse_iso_datetime(s: str) -> datetime:
    """Parse Graph-style ISO-8601 datetimes (including trailing ``Z``)."""
    t = s.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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


def _delta_entry_url(
    mailbox_segment: str,
    folder_well_known: str | None,
    folder_graph_id: str | None = None,
) -> str:
    if folder_graph_id and folder_graph_id.strip():
        fid = urllib.parse.quote(folder_graph_id.strip(), safe="")
        return f"{_GRAPH_ROOT}{mailbox_segment}/mailFolders/{fid}/messages/delta"
    if folder_well_known and folder_well_known.strip():
        fk = urllib.parse.quote(folder_well_known.strip(), safe="")
        return f"{_GRAPH_ROOT}{mailbox_segment}/mailFolders/{fk}/messages/delta"
    return f"{_GRAPH_ROOT}{mailbox_segment}/messages/delta"


def _normalize_mail_scope(
    mailbox_upn: str | None,
    folder_well_known: str | None,
    folder_graph_id: str | None = None,
) -> str:
    mb = (mailbox_upn or "").strip() or "me"
    if folder_graph_id and folder_graph_id.strip():
        return f"{mb}|id:{folder_graph_id.strip()}"
    fd = (folder_well_known or "").strip() or "all"
    return f"{mb}|{fd}"


def _graph_collect_folder_pages(session: requests.Session, first_url: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    url: str | None = first_url
    while url:
        payload = _graph_get(session, url)
        batch = payload.get("value")
        if isinstance(batch, list):
            for item in batch:
                if isinstance(item, dict):
                    out.append(item)
        nl = payload.get("@odata.nextLink")
        url = nl.strip() if isinstance(nl, str) and nl.strip() else None
    return out


def find_mail_folder_ids_by_display_name(
    session: requests.Session,
    mailbox_segment: str,
    display_name: str,
) -> list[str]:
    """Resolve Graph mailFolder id(s) with exact ``displayName`` (searches nested folders)."""
    target = (display_name or "").strip()
    if not target:
        return []
    matches: list[str] = []
    seen_enqueue: set[str] = set()

    q: deque[str] = deque()
    root_url = f"{_GRAPH_ROOT}{mailbox_segment}/mailFolders?$top=200"
    for item in _graph_collect_folder_pages(session, root_url):
        fid = str(item.get("id") or "").strip()
        name = str(item.get("displayName") or "")
        if name == target and fid:
            matches.append(fid)
        if fid and fid not in seen_enqueue:
            seen_enqueue.add(fid)
            q.append(fid)

    while q:
        parent_id = q.popleft()
        parent_enc = urllib.parse.quote(parent_id, safe="")
        child_url = (
            f"{_GRAPH_ROOT}{mailbox_segment}/mailFolders/{parent_enc}/childFolders?$top=200"
        )
        for item in _graph_collect_folder_pages(session, child_url):
            cid = str(item.get("id") or "").strip()
            name = str(item.get("displayName") or "")
            if name == target and cid:
                matches.append(cid)
            if cid and cid not in seen_enqueue:
                seen_enqueue.add(cid)
                q.append(cid)

    return matches


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


_GRAPH_GRAPH_AUDIENCES = frozenset(
    {
        "https://graph.microsoft.com",
        "https://graph.microsoft.com/",
        "00000003-0000-0000-c000-000000000000",
    }
)


def _graph_log_http_error(resp: requests.Response) -> None:
    www = (resp.headers.get("WWW-Authenticate") or "").strip()
    if www:
        _log(f"[graph] HTTP {resp.status_code} WWW-Authenticate: {www[:2000]}")
    for hn in ("request-id", "client-request-id", "x-ms-ags-diagnostic", "Date"):
        hv = (resp.headers.get(hn) or "").strip()
        if hv:
            _log(f"[graph] HTTP {resp.status_code} {hn}: {hv[:500]}")
    snippet = (resp.text or "")[:4000].strip()
    if snippet:
        _log(f"[graph] HTTP {resp.status_code} response body:\n{snippet}")
    else:
        _log(f"[graph] HTTP {resp.status_code} (empty body)")


def _log_access_token_jwt_claims(access_token: str) -> None:
    """Log aud/scp from JWT payload without cryptographic verification (debug only)."""
    parts = access_token.split(".")
    if len(parts) < 2:
        _log("[graph warn] Access token is not a JWT; skipping claim decode.")
        return
    try:
        payload_b64 = parts[1]
        pad = "=" * (-len(payload_b64) % 4)
        raw = base64.urlsafe_b64decode(payload_b64 + pad)
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        _log(f"[graph warn] Could not decode JWT payload: {e}")
        return
    aud = payload.get("aud")
    scp = payload.get("scp")
    roles = payload.get("roles")
    exp = payload.get("exp")
    app_id = payload.get("appid") or payload.get("azp")
    tid = payload.get("tid")
    _log(
        f"[graph] JWT (unverified): aud={aud!r} tid={tid!r} azp/appid={app_id!r} "
        f"scp={scp!r} roles={roles!r} exp_unix={exp}"
    )
    aud_ok = aud in _GRAPH_GRAPH_AUDIENCES
    if isinstance(aud, list):
        aud_ok = bool(set(aud) & _GRAPH_GRAPH_AUDIENCES)
    if not aud_ok:
        _log(
            "[graph warn] Token audience is not Microsoft Graph — mail APIs typically return 401. "
            "Delete the token cache and sign in again, or fix app registration / tenant."
        )


def _graph_get(session: requests.Session, url: str, *, timeout: float = 120.0) -> dict[str, Any]:
    backoff = 3.0
    for attempt in range(8):
        resp = session.get(url, timeout=timeout)
        if resp.url != url:
            _log(f"[graph warn] GET redirected {url!r} → final URL {resp.url!r} (Authorization may be dropped on cross-host redirects)")
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
        if resp.status_code >= 400:
            _graph_log_http_error(resp)
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Microsoft Graph: too many 429 retries")


def _graph_identity_profile(session: requests.Session, mb_seg: str) -> dict[str, Any]:
    url = f"{_GRAPH_ROOT}{mb_seg}?$select=id,userPrincipalName,mail,userType"
    resp = session.get(url, timeout=60.0)
    if resp.url != url:
        _log(f"[graph warn] GET redirected {url!r} → final URL {resp.url!r}")
    if resp.status_code >= 400:
        _graph_log_http_error(resp)
        resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def _invite_email_hint_from_ext_upn(upn: str) -> str | None:
    """Best-effort decode of Microsoft's B2B ``{email_with_@→_}#EXT#@host`` UPN prefix.

    Example: ``jack_jackpoorman.com#EXT#@...`` → ``jack@jackpoorman.com``.
    Fails for addresses whose local-part contains ``_`` before the ``@`` (ambiguous).
    """
    u = (upn or "").strip()
    el = u.lower()
    idx = el.find("#ext#")
    if idx < 0:
        return None
    prefix = u[:idx].strip()
    if "_" not in prefix:
        return None
    local, _, rest = prefix.partition("_")
    if not local or not rest or "." not in rest:
        return None
    return f"{local}@{rest}"


def _is_external_invited_upn(me: dict[str, Any]) -> bool:
    """True if UPN uses Microsoft's #EXT# pattern (invited / external identity in directory)."""
    upn = str(me.get("userPrincipalName") or "")
    return "#EXT#" in upn.upper()


def _is_graph_guest_user(me: dict[str, Any]) -> bool:
    return str(me.get("userType") or "").strip().lower() == "guest"


def _mailbox_401_ext_identity_message(me: dict[str, Any]) -> str:
    upn = me.get("userPrincipalName")
    ut = me.get("userType")
    ext = _is_external_invited_upn(me)
    guest = _is_graph_guest_user(me)
    lines = [
        "Microsoft Graph returned 401 on mailFolders for this signed-in user.",
        f"  userPrincipalName={upn!r}",
        f"  userType={ut!r}",
        "",
    ]
    if ext:
        hint = _invite_email_hint_from_ext_upn(str(upn or ""))
        hint_line = ""
        if hint:
            hint_line = (
                f"This is still **you**: Azure encodes invited users as a technical UPN "
                f"(often matching ~**{hint}**). It is **not** the same as a native \"work\" user "
                f"object that hosts Exchange mail in this tenant.\n\n"
            )
        lines.extend(
            [
                hint_line
                + "Your UPN contains **#EXT#** — an **invited / external Azure AD identity** "
                "in this tenant. Graph may still show **userType Member**; regardless, "
                "**Exchange often does not expose a mailbox** for this identity here, so Mail.Read "
                "gets **401** on mailFolders while **/me** works.",
                "",
            ]
        )
    elif guest:
        lines.extend(
            [
                "Microsoft Graph reports **userType Guest**. Guests typically do **not** have an "
                "Exchange Online mailbox in the inviting tenant for Graph mail APIs.",
                "",
            ]
        )
    lines.extend(
        [
            "**Fix:** If this **is** your real email but Azure shows a **#EXT#** UPN, your object is "
            "**invited/external** in this tenant; Graph mail APIs need a **mailbox-homed** identity "
            "(often a native member user with an Exchange license) **or** the **tenant ID** where "
            "your mailbox actually lives (try **GRAPH_TENANT_ID** / **--tenant**). Ask your admin if "
            "unsure. Delete **data/graph_mail_token_cache.json** after any change, then "
            "**--auth device**.",
            "",
        ]
    )
    return "\n".join(lines)


def _ensure_graph_mail_access(session: requests.Session, mb_seg: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """GET profile + mailFolders?$top=1; raises RuntimeError with guidance for common #EXT#/guest 401."""
    me = _graph_identity_profile(session, mb_seg)
    upn_log = str(me.get("userPrincipalName") or "")
    hint = _invite_email_hint_from_ext_upn(upn_log)
    _log(
        f"[graph] Graph identity: userPrincipalName={me.get('userPrincipalName')!r} "
        f"userType={me.get('userType')!r} mail={me.get('mail')!r}"
    )
    if hint:
        _log(
            f"[graph] Note: #EXT# UPN above usually corresponds to invited identity ~{hint!r} "
            "(your email), not a native mailbox principal in this tenant."
        )
    probe = f"{_GRAPH_ROOT}{mb_seg}/mailFolders?$top=1"
    resp = session.get(probe, timeout=60.0)
    if resp.url != probe:
        _log(f"[graph warn] GET redirected {probe!r} → final URL {resp.url!r}")
    if resp.status_code == 401 and (_is_external_invited_upn(me) or _is_graph_guest_user(me)):
        _graph_log_http_error(resp)
        raise RuntimeError(_mailbox_401_ext_identity_message(me))
    if resp.status_code >= 400:
        _graph_log_http_error(resp)
        resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Microsoft Graph mailFolders: unexpected JSON payload")
    return me, payload


def _acquire_token(
    *,
    client_id: str,
    tenant: str,
    cache_path: Path,
    auth_mode: str,
    force_refresh: bool = False,
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
        _log(
            f"[graph] Trying MSAL silent token ({len(accounts)} account(s) in cache); "
            f"force_refresh={force_refresh}"
        )
        result = app.acquire_token_silent(
            _SCOPES,
            account=accounts[0],
            force_refresh=force_refresh,
        )
        if result and result.get("access_token"):
            _log("[graph] Token source: silent cache (or refreshed)")
            scope_s = str(result.get("scope") or "").strip()
            if scope_s:
                _log(f"[graph] Token scopes (silent): {scope_s}")
            _log_access_token_jwt_claims(str(result["access_token"]))
            if "mail.read" not in scope_s.lower():
                _log(
                    "[graph warn] Mail.Read may be missing from scope string — if Graph returns 401, "
                    "use --force-token-refresh or delete the token cache and sign in again."
                )
            cache.flush()
            return str(result["access_token"]).strip()
        if result:
            err = result.get("error") or result.get("error_description") or result
            _log(f"[graph warn] Silent token not obtained: {err!r}; falling back to {auth_mode!r}")
    else:
        _log("[graph] No MSAL accounts in cache; interactive/device sign-in required")

    auth_mode = auth_mode.strip().lower()
    if auth_mode == "device":
        flow = app.initiate_device_flow(scopes=_SCOPES)
        if "user_code" not in flow:
            raise RuntimeError(f"Device flow failed: {flow}")
        _log(flow["message"])
        _log("[graph] Token source: device flow")
        result = app.acquire_token_by_device_flow(flow)
    elif auth_mode == "interactive":
        _log("[graph] Token source: interactive browser")
        result = app.acquire_token_interactive(scopes=_SCOPES)
    else:
        raise ValueError(f"Unknown --auth mode: {auth_mode}")

    cache.flush()
    if not result or not result.get("access_token"):
        err = (result or {}).get("error_description") or (result or {}).get("error") or "unknown"
        raise RuntimeError(f"Authentication failed: {err}")
    scope_s = str((result or {}).get("scope") or "").strip()
    if scope_s:
        _log(f"[graph] Token scopes from identity platform: {scope_s}")
        if "mail.read" not in scope_s.lower():
            _log(
                "[graph warn] Mail.Read not listed in granted scopes — Graph mail calls may "
                "401/403. In Entra: API permissions → delegated Mail.Read → Grant admin consent; "
                "then delete the token cache file and sign in again."
            )
    _log_access_token_jwt_claims(str(result["access_token"]))
    return str(result["access_token"]).strip()


def _process_message_item(
    conn: sqlite3.Connection,
    item: dict[str, Any],
    *,
    commit_every: int,
    counters: dict[str, int],
    since_dt: datetime | None,
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
    if since_dt:
        raw_dt = str(received).strip()
        if raw_dt:
            try:
                msg_dt = _parse_iso_datetime(raw_dt)
                if msg_dt < since_dt:
                    counters["skipped_since"] += 1
                    return
            except ValueError:
                pass
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
    web_link = str(item.get("webLink") or "").strip()

    upsert_photo(
        conn,
        uuid=uuid,
        filename=filename,
        date_iso=str(received).strip() or None,
        ocr_text=plain,
        vlm_text=" ".join(meta_bits),
        image_path_used="",
        open_url=web_link,
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
    folder_graph_id: str | None,
    auth_mode: str,
    reset_delta: bool,
    commit_every: int,
    page_hint: int,
    since_dt: datetime | None,
    max_messages: int | None,
    progress_every_pages: int = 1,
    force_token_refresh: bool = False,
    skip_mail_access_probe: bool = False,
) -> dict[str, int | float]:
    fw = (folder_well_known or "").strip() or None
    fg = (folder_graph_id or "").strip() or None
    if fw and fg:
        raise ValueError("Pass only one of folder_well_known or folder_graph_id")

    mail_scope = _normalize_mail_scope(mailbox_upn, folder_well_known, folder_graph_id)
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
        force_refresh=force_token_refresh,
    )
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Prefer": 'outlook.body-content-type="text"',
        }
    )

    mb_seg = _mailbox_segment(mailbox_upn)
    if not skip_mail_access_probe:
        _ensure_graph_mail_access(session, mb_seg)

    if delta_link:
        url = delta_link
        _log("[graph] Continuing incremental sync from saved delta link.")
    else:
        url = _delta_entry_url(mb_seg, folder_well_known, folder_graph_id)
        params = {"$top": str(max(1, min(page_hint, 999)))}
        url = f"{url}?{urllib.parse.urlencode(params)}"
        _log(f"[graph] Starting full sync: {url.split('?')[0]}")

    conn = connect(index_db_path)
    init_schema(conn)
    counters = {"indexed": 0, "deleted": 0, "skipped_empty": 0, "skipped_since": 0, "pages": 0}
    t0 = time.perf_counter()
    new_delta: str | None = None
    stopped_early = False

    try:
        while url:
            payload = _graph_get(session, url)
            counters["pages"] += 1
            batch = payload.get("value")
            batch_n = len(batch) if isinstance(batch, list) else 0
            if isinstance(batch, list):
                for item in batch:
                    if not isinstance(item, dict):
                        continue
                    _process_message_item(
                        conn,
                        item,
                        commit_every=commit_every,
                        counters=counters,
                        since_dt=since_dt,
                    )
                    cap = max_messages if max_messages is not None else 0
                    if cap > 0 and counters["indexed"] >= cap:
                        stopped_early = True
                        url = None
                        break

            if progress_every_pages > 0 and counters["pages"] % progress_every_pages == 0:
                elapsed = time.perf_counter() - t0
                _log(
                    f"[graph {mail_scope}] page {counters['pages']} "
                    f"batch_items={batch_n} | indexed={counters['indexed']} "
                    f"deleted={counters['deleted']} skipped_empty={counters['skipped_empty']} "
                    f"skipped_since={counters['skipped_since']} | {elapsed:.1f}s elapsed"
                )

            if stopped_early:
                break

            url = payload.get("@odata.nextLink")
            if isinstance(url, str) and url.strip():
                url = url.strip()
            else:
                url = None

            dl = payload.get("@odata.deltaLink")
            if isinstance(dl, str) and dl.strip():
                new_delta = dl.strip()

        if stopped_early:
            _log(
                "[graph warn] Stopped early (--max-messages): delta checkpoint not updated; "
                "run again with a higher limit or omit --max-messages to finish initial sync."
            )
            new_delta = None
        elif new_delta:
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
        f"skipped_empty={counters['skipped_empty']} skipped_since={counters['skipped_since']} "
        f"pages={counters['pages']} time={elapsed:.1f}s db={index_db_path}"
    )
    counters["elapsed"] = elapsed
    return counters


def run_named_folder_preset_batch(
    *,
    index_db_path: Path,
    client_id: str,
    tenant: str,
    token_cache_path: Path,
    mailbox_upn: str | None,
    presets: tuple[NamedFolderPreset, ...] | list[NamedFolderPreset],
    delta_dir: Path,
    auth_mode: str,
    reset_delta: bool,
    commit_every: int,
    page_hint: int,
    max_messages: int | None,
    progress_every_pages: int = 1,
    force_token_refresh: bool = False,
) -> dict[str, int | float]:
    """Discover folders by display name and sync each with its own delta file and lookback."""
    mb_seg = _mailbox_segment(mailbox_upn)
    token = _acquire_token(
        client_id=client_id,
        tenant=tenant,
        cache_path=token_cache_path,
        auth_mode=auth_mode,
        force_refresh=force_token_refresh,
    )
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Prefer": 'outlook.body-content-type="text"',
        }
    )
    _ensure_graph_mail_access(session, mb_seg)
    totals = {"indexed": 0, "deleted": 0, "skipped_empty": 0, "skipped_since": 0, "pages": 0}
    elapsed_sum = 0.0
    try:
        for preset in presets:
            ids = find_mail_folder_ids_by_display_name(session, mb_seg, preset.display_name)
            if not ids:
                _log(f"[graph warn] No folder with displayName={preset.display_name!r}; skipping preset.")
                continue
            if len(ids) > 1:
                _log(
                    f"[graph warn] {len(ids)} folders named {preset.display_name!r}; "
                    "syncing each with its own delta file."
                )
            since_dt = datetime.now(timezone.utc) - timedelta(days=preset.lookback_days)
            _log(
                f"[graph] Preset folder {preset.display_name!r}: lookback={preset.lookback_days}d "
                f"(index on/after {since_dt.date()} UTC)"
            )
            for idx, fid in enumerate(ids):
                slug = preset.delta_slug if len(ids) == 1 else f"{preset.delta_slug}__{idx}"
                delta_path = delta_dir / f"graph_mail_delta_{slug}.json"
                ctr = run_outlook_graph_ingest(
                    index_db_path=index_db_path,
                    client_id=client_id,
                    tenant=tenant,
                    token_cache_path=token_cache_path,
                    delta_path=delta_path,
                    mailbox_upn=mailbox_upn,
                    folder_well_known=None,
                    folder_graph_id=fid,
                    auth_mode=auth_mode,
                    reset_delta=reset_delta,
                    commit_every=commit_every,
                    page_hint=page_hint,
                    since_dt=since_dt,
                    max_messages=max_messages,
                    progress_every_pages=progress_every_pages,
                    force_token_refresh=force_token_refresh,
                    skip_mail_access_probe=True,
                )
                for k in totals:
                    totals[k] += int(ctr[k])
                elapsed_sum += float(ctr["elapsed"])
    finally:
        session.close()

    totals["elapsed"] = elapsed_sum
    return totals


def run_graph_connection_test(
    *,
    client_id: str,
    tenant: str,
    token_cache_path: Path,
    mailbox_upn: str | None,
    auth_mode: str,
    force_token_refresh: bool = False,
) -> None:
    """Acquire token and GET mailFolders?$top=1 (validates Mail.Read; does not write SQLite)."""
    _log("[graph test] Acquiring token …")
    token = _acquire_token(
        client_id=client_id,
        tenant=tenant,
        cache_path=token_cache_path,
        auth_mode=auth_mode,
        force_refresh=force_token_refresh,
    )
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    mb_seg = _mailbox_segment(mailbox_upn)
    _log("[graph test] GET /me (profile) and mailFolders?$top=1 …")
    me_payload, payload = _ensure_graph_mail_access(session, mb_seg)
    upn = me_payload.get("userPrincipalName")
    mail = me_payload.get("mail")
    _log(f"[graph test] /me OK — userPrincipalName={upn!r} mail={mail!r}")
    folders = payload.get("value")
    n = len(folders) if isinstance(folders, list) else 0
    _log(f"[graph test] OK — Mail.Read works; mailFolders sample returned {n} row(s).")
    if mailbox_upn:
        _log(f"[graph test] Mailbox: delegated access to {mailbox_upn.strip()!r} (/users/{{upn}}/…).")
    else:
        _log("[graph test] Mailbox: signed-in user (/me/…).")
    if isinstance(folders, list) and folders:
        fd0 = folders[0]
        if isinstance(fd0, dict):
            _log(f"[graph test] First folder in response: {str(fd0.get('displayName') or '?')!r}")


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
        "--since",
        type=str,
        default=None,
        metavar="ISO8601",
        help="Only index messages on or after this instant (UTC), e.g. 2024-01-01 or 2024-01-01T00:00:00Z. "
        "Graph still paginates the folder; older mail is skipped locally.",
    )
    p.add_argument(
        "--max-messages",
        type=int,
        default=None,
        metavar="N",
        help="Stop after indexing N non-empty messages (testing/partial pull). "
        "Does not save a delta link; omit for a full sync.",
    )
    p.add_argument(
        "--progress-every-pages",
        type=int,
        default=1,
        metavar="N",
        help="Log a progress line every N Graph delta pages (default: 1). Use 0 to disable.",
    )
    p.add_argument(
        "--force-token-refresh",
        action="store_true",
        help="Ask MSAL to refresh cached tokens (helps after consent changes). "
        "Or delete data/graph_mail_token_cache.json and sign in again.",
    )
    p.add_argument(
        "--no-global-ingest-lock",
        action="store_true",
        help="Disable shared content-ingest lock (not recommended).",
    )
    p.add_argument(
        "--sync-named-presets",
        action="store_true",
        help="Sync built-in custom folders only (exact Outlook display names) with per-folder "
        "lookback; does not sync Inbox or mailbox-wide mail. See --list-named-presets.",
    )
    p.add_argument(
        "--sync-inbox-and-named-presets",
        action="store_true",
        help="Sync well-known Inbox and Sent Items only, then built-in named-folder presets "
        "(no mailbox-wide /messages delta). Same preset lookbacks as --list-named-presets.",
    )
    p.add_argument(
        "--list-named-presets",
        action="store_true",
        help="Print built-in named-folder presets and exit.",
    )
    p.add_argument(
        "--named-preset-delta-dir",
        type=str,
        default=str(_DEFAULT_DB.parent),
        help="Directory for named-preset delta JSON files (default: same dir as default DB).",
    )
    p.add_argument(
        "--test-connection",
        action="store_true",
        help="Acquire token and call Graph mailFolders?$top=1 only (no SQLite ingest). Exit 1 on failure.",
    )
    args = p.parse_args(argv)

    if args.list_named_presets:
        _log("Built-in named-folder presets (exact Outlook displayName match):")
        for pr in NAMED_FOLDER_PRESETS_DEFAULT:
            _log(
                f"  - {pr.display_name!r}: lookback {pr.lookback_days} days "
                f"(delta files graph_mail_delta_{pr.delta_slug}.json)"
            )
        _log("Well-known folder deltas when bundled (under --named-preset-delta-dir):")
        for wk, fname in _WELL_KNOWN_PRESET_BUNDLE:
            _log(f"  - {wk}: {fname}")
        _log("Sync Inbox + Sent + presets: --sync-inbox-and-named-presets")
        return

    if args.test_connection:
        ingest_flags = (
            args.sync_named_presets,
            args.sync_inbox_and_named_presets,
            bool(args.folder),
            bool(args.since),
            args.reset_delta,
            args.max_messages is not None,
        )
        if any(ingest_flags):
            p.error("--test-connection cannot be combined with ingest options (--folder, --since, …).")
        cid = (args.client_id or "").strip()
        if not cid:
            p.error(
                "Missing client id: pass --client-id or set GRAPH_CLIENT_ID "
                "(Entra app registration → Application ID)."
            )
        token_cache_path = Path(os.path.abspath(args.token_cache))
        try:
            run_graph_connection_test(
                client_id=cid,
                tenant=str(args.tenant or "organizations"),
                token_cache_path=token_cache_path,
                mailbox_upn=(args.mailbox.strip() if args.mailbox else None),
                auth_mode=str(args.auth),
                force_token_refresh=bool(args.force_token_refresh),
            )
        except Exception as e:
            _log(f"[graph test] FAILED: {e}")
            sys.exit(1)
        return

    if args.commit_every < 1:
        p.error("--commit-every must be >= 1")

    if args.sync_named_presets and args.folder:
        p.error("Use either --sync-named-presets or --folder, not both.")
    if args.sync_inbox_and_named_presets and args.folder:
        p.error("--sync-inbox-and-named-presets already includes Inbox and Sent; do not pass --folder.")
    if args.sync_inbox_and_named_presets and args.sync_named_presets:
        p.error("Use either --sync-named-presets or --sync-inbox-and-named-presets, not both.")
    if args.sync_named_presets and args.since:
        p.error("--since applies only to the standard sync; named presets use fixed lookbacks.")
    if args.sync_inbox_and_named_presets and args.since:
        p.error("--since is not supported with --sync-inbox-and-named-presets (presets use fixed lookbacks).")

    since_dt: datetime | None = None
    if args.since:
        try:
            since_dt = _parse_iso_datetime(args.since)
        except ValueError:
            p.error("--since must be ISO-8601 (e.g. 2024-06-01 or 2024-06-01T12:00:00Z)")

    max_messages = args.max_messages
    if max_messages is not None and max_messages < 1:
        p.error("--max-messages must be >= 1")

    progress_every_pages = int(args.progress_every_pages)
    if progress_every_pages < 0:
        p.error("--progress-every-pages must be >= 0")

    force_token_refresh = bool(args.force_token_refresh)

    cid = (args.client_id or "").strip()
    if not cid:
        p.error(
            "Missing client id: pass --client-id or set GRAPH_CLIENT_ID "
            "(Entra app registration → Application ID)."
        )

    index_db_path = Path(os.path.abspath(args.db))
    token_cache_path = Path(os.path.abspath(args.token_cache))
    delta_path = Path(os.path.abspath(args.delta_path))

    if args.sync_inbox_and_named_presets:
        preset_delta_dir = Path(os.path.abspath(args.named_preset_delta_dir))
        preset_delta_dir.mkdir(parents=True, exist_ok=True)

        def _run_well_known_bundle_then_presets() -> None:
            n_wk = len(_WELL_KNOWN_PRESET_BUNDLE)
            total_phases = n_wk + 1
            for i, (wk, delta_fname) in enumerate(_WELL_KNOWN_PRESET_BUNDLE, start=1):
                _log(
                    f"[graph] Phase {i}/{total_phases}: Well-known folder {wk!r} "
                    "(not whole mailbox) …"
                )
                ctr = run_outlook_graph_ingest(
                    index_db_path=index_db_path,
                    client_id=cid,
                    tenant=str(args.tenant or "organizations"),
                    token_cache_path=token_cache_path,
                    delta_path=preset_delta_dir / delta_fname,
                    mailbox_upn=(args.mailbox.strip() if args.mailbox else None),
                    folder_well_known=wk,
                    folder_graph_id=None,
                    auth_mode=str(args.auth),
                    reset_delta=bool(args.reset_delta),
                    commit_every=int(args.commit_every),
                    page_hint=int(args.page_size),
                    since_dt=None,
                    max_messages=max_messages,
                    progress_every_pages=progress_every_pages,
                    force_token_refresh=force_token_refresh,
                    skip_mail_access_probe=i != 1,
                )
                _log(
                    f"[graph {wk} done] indexed={ctr['indexed']} deleted={ctr['deleted']} "
                    f"skipped_empty={ctr['skipped_empty']} skipped_since={ctr['skipped_since']} "
                    f"pages={ctr['pages']} time={ctr['elapsed']:.1f}s"
                )
            _log(f"[graph] Phase {total_phases}/{total_phases}: Named-folder presets …")
            agg = run_named_folder_preset_batch(
                index_db_path=index_db_path,
                client_id=cid,
                tenant=str(args.tenant or "organizations"),
                token_cache_path=token_cache_path,
                mailbox_upn=(args.mailbox.strip() if args.mailbox else None),
                presets=NAMED_FOLDER_PRESETS_DEFAULT,
                delta_dir=preset_delta_dir,
                auth_mode=str(args.auth),
                reset_delta=bool(args.reset_delta),
                commit_every=int(args.commit_every),
                page_hint=int(args.page_size),
                max_messages=max_messages,
                progress_every_pages=progress_every_pages,
                force_token_refresh=force_token_refresh,
            )
            _log(
                f"[graph presets done] indexed={agg['indexed']} deleted={agg['deleted']} "
                f"skipped_empty={agg['skipped_empty']} skipped_since={agg['skipped_since']} "
                f"pages={agg['pages']} time={agg['elapsed']:.1f}s db={index_db_path}"
            )

        if args.no_global_ingest_lock:
            _run_well_known_bundle_then_presets()
            return

        with global_ingest_lock() as have_lock:
            if not have_lock:
                _log("[lock] Another content ingest is already running; skipping this run.")
                return
            _run_well_known_bundle_then_presets()
        return

    if args.sync_named_presets:
        preset_delta_dir = Path(os.path.abspath(args.named_preset_delta_dir))
        preset_delta_dir.mkdir(parents=True, exist_ok=True)

        def _run_presets() -> None:
            agg = run_named_folder_preset_batch(
                index_db_path=index_db_path,
                client_id=cid,
                tenant=str(args.tenant or "organizations"),
                token_cache_path=token_cache_path,
                mailbox_upn=(args.mailbox.strip() if args.mailbox else None),
                presets=NAMED_FOLDER_PRESETS_DEFAULT,
                delta_dir=preset_delta_dir,
                auth_mode=str(args.auth),
                reset_delta=bool(args.reset_delta),
                commit_every=int(args.commit_every),
                page_hint=int(args.page_size),
                max_messages=max_messages,
                progress_every_pages=progress_every_pages,
                force_token_refresh=force_token_refresh,
            )
            _log(
                f"[graph presets done] indexed={agg['indexed']} deleted={agg['deleted']} "
                f"skipped_empty={agg['skipped_empty']} skipped_since={agg['skipped_since']} "
                f"pages={agg['pages']} time={agg['elapsed']:.1f}s db={index_db_path}"
            )

        if args.no_global_ingest_lock:
            _run_presets()
            return

        with global_ingest_lock() as have_lock:
            if not have_lock:
                _log("[lock] Another content ingest is already running; skipping this run.")
                return
            _run_presets()
        return

    def _run() -> None:
        run_outlook_graph_ingest(
            index_db_path=index_db_path,
            client_id=cid,
            tenant=str(args.tenant or "organizations"),
            token_cache_path=token_cache_path,
            delta_path=delta_path,
            mailbox_upn=(args.mailbox.strip() if args.mailbox else None),
            folder_well_known=args.folder,
            folder_graph_id=None,
            auth_mode=str(args.auth),
            reset_delta=bool(args.reset_delta),
            commit_every=int(args.commit_every),
            page_hint=int(args.page_size),
            since_dt=since_dt,
            max_messages=max_messages,
            progress_every_pages=progress_every_pages,
            force_token_refresh=force_token_refresh,
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
