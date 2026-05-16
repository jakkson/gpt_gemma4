#!/usr/bin/env bash
# Register an Entra ID (Azure AD) application named "personal-photo-index-mail"
# with delegated Microsoft Graph Mail.Read (+ offline_access when available).
#
# Prerequisites:
#   brew install azure-cli   # or https://learn.microsoft.com/cli/azure/install-azure-cli
#   az login --allow-no-subscriptions    # if you have no Azure subscription (Entra-only)
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
GRAPH_API_ID="00000003-0000-0000-c000-000000000000"

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

if ! az ad signed-in-user show >/dev/null 2>&1; then
  echo "Azure CLI has no active directory login."
  echo "Signing into portal.azure.com does not log in the CLI."
  echo ""
  echo "If you see \"No subscriptions\", you still need an Entra login without a subscription:"
  echo "  az login --allow-no-subscriptions"
  echo ""
  echo "Otherwise:"
  echo "  az login"
  exit 1
fi

echo "Resolving Microsoft Graph delegated permission IDs …"
MAIL_READ_SCOPE="$(
  az ad sp show --id "$GRAPH_API_ID" \
    --query "oauth2PermissionScopes[?value=='Mail.Read'].id | [0]" -o tsv
)"
if [[ -z "$MAIL_READ_SCOPE" || "$MAIL_READ_SCOPE" == "null" ]]; then
  echo "Failed to resolve Mail.Read scope id for Graph. Try: az ad sp show --id $GRAPH_API_ID"
  exit 1
fi

OFFLINE_SCOPE="$(
  az ad sp show --id "$GRAPH_API_ID" \
    --query "oauth2PermissionScopes[?value=='offline_access'].id | [0]" -o tsv 2>/dev/null || true
)"

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
