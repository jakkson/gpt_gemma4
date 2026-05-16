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

echo "Using Microsoft Graph delegated permission IDs (Mail.Read, offline_access) …"
MAIL_READ_SCOPE="$GRAPH_MAIL_READ_DELEGATED"
OFFLINE_SCOPE="$GRAPH_OFFLINE_ACCESS_DELEGATED"

EXISTING_OBJ="$(az ad app list --filter "displayName eq '${DISPLAY_NAME}'" --query "[0].id" -o tsv)"
if [[ -n "${EXISTING_OBJ:-}" && "$EXISTING_OBJ" != "null" ]]; then
  OBJ_ID="$EXISTING_OBJ"
  APP_ID="$(az ad app show --id "$OBJ_ID" --query appId -o tsv)"
  echo "Found existing app registration named '${DISPLAY_NAME}' (object ${OBJ_ID})."
  echo "Updating public-client redirect URI http://localhost …"
  az ad app update --id "$OBJ_ID" \
    --public-client-redirect-uris "http://localhost"
else
  echo "Creating app registration '${DISPLAY_NAME}' …"
  OBJ_ID="$(
    az ad app create \
      --display-name "$DISPLAY_NAME" \
      --sign-in-audience "$SIGN_IN_AUDIENCE" \
      --public-client-redirect-uris "http://localhost" \
      --query id -o tsv
  )"
  APP_ID="$(az ad app show --id "$OBJ_ID" --query appId -o tsv)"
fi

echo "Adding Graph delegated permissions …"
set +e
az ad app permission add --id "$OBJ_ID" --api "$GRAPH_API_ID" \
  --api-permissions "${MAIL_READ_SCOPE}=Scope" 2>/dev/null
RC_MAIL=$?
set -e
if [[ "$RC_MAIL" -ne 0 ]]; then
  echo "(Mail.Read may already be registered on this app — continuing.)"
fi

if [[ -n "${OFFLINE_SCOPE:-}" && "$OFFLINE_SCOPE" != "null" ]]; then
  set +e
  az ad app permission add --id "$OBJ_ID" --api "$GRAPH_API_ID" \
    --api-permissions "${OFFLINE_SCOPE}=Scope" 2>/dev/null
  set -e
fi

echo "Attempting tenant admin consent (may fail without admin rights) …"
set +e
az ad app permission admin-consent --id "$OBJ_ID"
RC_AC=$?
set -e
if [[ "$RC_AC" -ne 0 ]]; then
  echo ""
  echo "Admin consent was not applied automatically. In Entra portal:"
  echo "  App registrations → ${DISPLAY_NAME} → API permissions → Grant admin consent"
  echo "(Personal tenants: your account is usually enough after clicking Grant.)"
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
