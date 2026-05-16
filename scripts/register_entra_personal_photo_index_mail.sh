#!/usr/bin/env bash
# Register an Entra ID (Azure AD) application named "personal-photo-index-mail"
# with delegated Microsoft Graph Mail.Read (+ offline_access when available).
#
# Prerequisites:
#   brew install azure-cli   # or https://learn.microsoft.com/cli/azure/install-azure-cli
#   az login --allow-no-subscriptions [--tenant TENANT_ID]
#       Use explicit --tenant <Entra Tenant ID> if login crashes in "Tenant and subscription selection"
#       (known azure-cli bug when there are zero subscriptions). Upgrade CLI: brew upgrade azure-cli
#   az login                             # if you have subscriptions
#
# Usage:
#   ./scripts/register_entra_personal_photo_index_mail.sh
#
# Optional:
#   SIGN_IN_AUDIENCE=AzureADandPersonalMicrosoftAccount ./scripts/register_entra_personal_photo_index_mail.sh
#     (use if you need Outlook.com / personal Microsoft accounts in addition to work/school)

set -euo pipefail

DISPLAY_NAME="${DISPLAY_NAME:-personal-photo-index-mail}"
SIGN_IN_AUDIENCE="${SIGN_IN_AUDIENCE:-AzureADMultipleOrgs}"
# Microsoft Graph application ID (resource).
GRAPH_API_ID="00000003-0000-0000-c000-000000000000"
# Delegated permission IDs on that resource (stable; do not use ``az ad sp show``
# here — tenant-only CLI logins often fail with "User was not found").
# Source: https://learn.microsoft.com/en-us/graph/permissions-reference
GRAPH_MAIL_READ_DELEGATED="570282fd-fa5c-430d-a7fd-fc8dc98a9dca"
GRAPH_OFFLINE_ACCESS_DELEGATED="7427e0e9-2fba-42fe-b0c0-848c9e6a8182"

if ! command -v az >/dev/null 2>&1; then
  for _brew_az in /opt/homebrew/bin/az /usr/local/bin/az; do
    if [[ -x "$_brew_az" ]]; then
      PATH="$(dirname "$_brew_az"):$PATH"
      export PATH
      break
    fi
  done
fi

if ! command -v az >/dev/null 2>&1; then
  echo "Azure CLI not found."
  echo "Install:  brew install azure-cli"
  echo "Then open a new terminal tab or run:  eval \"\$(brew shellenv)\""
  exit 1
fi

_logged_into_az() {
  if az ad signed-in-user show >/dev/null 2>&1; then
    return 0
  fi
  # Tenant-only login (--allow-no-subscriptions) sometimes works for Entra but
  # ``signed-in-user`` returns "User was not found" for certain account types.
  if az account show >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

if ! _logged_into_az; then
  echo "Azure CLI has no active directory login."
  echo "Signing into portal.azure.com does not log in the CLI."
  echo ""
  echo "If you see \"No subscriptions\", you still need an Entra login without a subscription:"
  echo "  az login --allow-no-subscriptions [--tenant YOUR_TENANT_ID]"
  echo ""
  echo "At \"Select a subscription\", press Enter or type 1 (do not paste the tenant GUID)."
  echo ""
  echo "Otherwise:"
  echo "  az login"
  exit 1
fi

echo "Registering application via Microsoft Graph REST (``az ad app list`` fails with tenant-only / no-subscription login)."
command -v python3 >/dev/null 2>&1 || {
  echo "python3 is required for Graph REST registration."
  exit 1
}

GRAPH_TOKEN="$(az account get-access-token --resource "https://graph.microsoft.com" --query accessToken -o tsv 2>/dev/null || true)"
if [[ -z "${GRAPH_TOKEN:-}" ]]; then
  echo "ERROR: Could not get a Microsoft Graph access token."
  echo "Try:  az login --allow-no-subscriptions --tenant YOUR_TENANT_ID"
  exit 1
fi

export GRAPH_TOKEN DISPLAY_NAME SIGN_IN_AUDIENCE GRAPH_API_ID GRAPH_MAIL_READ_DELEGATED GRAPH_OFFLINE_ACCESS_DELEGATED

set +e
_PY_ASSIGN="$(
  python3 << 'PY'
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

TOKEN = os.environ["GRAPH_TOKEN"]
DISPLAY_NAME = os.environ["DISPLAY_NAME"]
SIGN_IN_AUDIENCE = os.environ["SIGN_IN_AUDIENCE"]
GRAPH_API_ID = os.environ["GRAPH_API_ID"]
MAIL_READ = os.environ["GRAPH_MAIL_READ_DELEGATED"]
OFFLINE = os.environ["GRAPH_OFFLINE_ACCESS_DELEGATED"]


def graph_req(method: str, url: str, body=None):
    payload = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method=method,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, raw if raw.strip() else "{}"
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code} {method}\n{err}", file=sys.stderr)
        sys.exit(1)


def find_apps():
    esc = DISPLAY_NAME.replace("'", "''")
    filt = f"displayName eq '{esc}'"
    qs = urllib.parse.urlencode({"$filter": filt, "$select": "id,appId,displayName"})
    _, raw = graph_req("GET", f"https://graph.microsoft.com/v1.0/applications?{qs}")
    return json.loads(raw).get("value") or []


def perm_block():
    return {
        "resourceAppId": GRAPH_API_ID,
        "resourceAccess": [
            {"id": MAIL_READ, "type": "Scope"},
            {"id": OFFLINE, "type": "Scope"},
        ],
    }


def merge_rra(existing):
    blocks = list(existing or [])
    want = perm_block()
    idx = next((i for i, b in enumerate(blocks) if b.get("resourceAppId") == GRAPH_API_ID), None)
    if idx is None:
        blocks.append(want)
        return blocks
    ra_list = list(blocks[idx].get("resourceAccess") or [])
    by_key = {(x.get("id"), x.get("type")): x for x in ra_list}
    for w in want["resourceAccess"]:
        by_key[(w["id"], w["type"])] = w
    blocks[idx] = {**blocks[idx], "resourceAccess": list(by_key.values())}
    return blocks


def create_app():
    body = {
        "displayName": DISPLAY_NAME,
        "signInAudience": SIGN_IN_AUDIENCE,
        "isFallbackPublicClient": True,
        "publicClient": {"redirectUris": ["http://localhost"]},
        "requiredResourceAccess": [perm_block()],
    }
    _, raw = graph_req("POST", "https://graph.microsoft.com/v1.0/applications", body)
    obj = json.loads(raw)
    return obj["id"], obj["appId"]


def patch_app(obj_id):
    sel = urllib.parse.urlencode(
        {"$select": "id,appId,requiredResourceAccess,publicClient,isFallbackPublicClient"}
    )
    url = f"https://graph.microsoft.com/v1.0/applications/{urllib.parse.quote(obj_id)}?{sel}"
    _, raw = graph_req("GET", url)
    cur = json.loads(raw)
    pc = dict(cur.get("publicClient") or {})
    uris = list(pc.get("redirectUris") or [])
    if "http://localhost" not in uris:
        uris.append("http://localhost")
    pc["redirectUris"] = uris
    patch = {
        "isFallbackPublicClient": True,
        "publicClient": pc,
        "requiredResourceAccess": merge_rra(cur.get("requiredResourceAccess")),
    }
    graph_req(
        "PATCH",
        f"https://graph.microsoft.com/v1.0/applications/{urllib.parse.quote(obj_id)}",
        patch,
    )
    return cur["id"], cur["appId"]


def main():
    apps = find_apps()
    if apps:
        print(f"Found existing app '{DISPLAY_NAME}' — updating redirects and API permissions.", file=sys.stderr)
        oid, aid = patch_app(apps[0]["id"])
    else:
        print(f"Creating app '{DISPLAY_NAME}' …", file=sys.stderr)
        oid, aid = create_app()
    print(f"OBJ_ID={json.dumps(oid)}")
    print(f"APP_ID={json.dumps(aid)}")


main()
PY
)"
_PY_RC=$?
set -e

if [[ "$_PY_RC" -ne 0 ]] || [[ -z "${_PY_ASSIGN:-}" ]]; then
  echo ""
  echo "=== Graph app registration failed ==="
  echo "If you saw HTTP 403 / Authentication_Unauthorized / \"User was not found\":"
  echo "Azure CLI issued a token, but Microsoft Graph does not associate it with a USER OBJECT"
  echo "in this Entra tenant. Delegated calls to /applications are then rejected."
  echo ""
  echo "This often happens with Microsoft accounts / tenant-only \"N/A subscription\" CLI profiles."
  echo "Reliable workaround — register once in the browser (same identity you use for mail):"
  echo "  1) https://entra.microsoft.com  → App registrations → New registration"
  echo "  2) Name: ${DISPLAY_NAME}"
  echo "  3) Supported types: match how you sign into Outlook (work/school vs personal)"
  echo "  4) Redirect URI: Mobile and desktop applications → http://localhost"
  echo "  5) Register → API permissions → Microsoft Graph → Delegated → Mail.Read + offline_access"
  echo "  6) Grant admin consent (or consent on first sign-in)"
  echo "  7) Overview → copy Application (client) ID → export GRAPH_CLIENT_ID='…'"
  exit 1
fi

eval "$_PY_ASSIGN"

echo "Success: ${DISPLAY_NAME} — Mail.Read + offline_access + http://localhost redirect set via Graph API."

echo "Attempting tenant admin consent (may fail without admin rights; optional) …"
set +e
az ad app permission admin-consent --id "$OBJ_ID"
RC_AC=$?
set -e
if [[ "$RC_AC" -ne 0 ]]; then
  echo ""
  echo "Automatic admin-consent did not succeed (common with tenant-only CLI). In Entra portal:"
  echo "  App registrations → ${DISPLAY_NAME} → API permissions → Grant admin consent"
  echo "Or ignore if you only need user consent on first Outlook ingest sign-in."
fi

echo ""
echo "Done."
echo "  Application (client) ID: ${APP_ID}"
echo ""
echo "Add to your shell or launchd:"
echo "  export GRAPH_CLIENT_ID='${APP_ID}'"
echo ""
echo "Then:"
echo "  python -m photo_index.outlook_graph_ingest --auth interactive"
