#!/bin/sh
set -eu
umask 077

: "${VAULT_ADDR:?VAULT_ADDR is required}"
: "${VAULT_TOKEN:?VAULT_TOKEN is required}"
: "${IOAP_DATABASE_URL:?IOAP_DATABASE_URL is required}"
: "${IOAP_OIDC_CLIENT_SECRET:?IOAP_OIDC_CLIENT_SECRET is required}"
: "${IOAP_MINIO_ACCESS_KEY:?IOAP_MINIO_ACCESS_KEY is required}"
: "${IOAP_MINIO_SECRET_KEY:?IOAP_MINIO_SECRET_KEY is required}"
: "${IOAP_NEXTAUTH_SECRET:?IOAP_NEXTAUTH_SECRET is required}"
: "${IOAP_ENTERPRISE_TOOL_GATEWAY_TOKEN:?enterprise sandbox token is required}"
: "${IOAP_NOTIFICATION_PROVIDER_API_TOKEN:?notification provider token is required}"
: "${IOAP_NOTIFICATION_WEBHOOK_SECRET:?notification webhook secret is required}"
: "${IOAP_PROCUREMENT_PROVIDER_API_TOKEN:?procurement provider token is required}"
: "${IOAP_PROCUREMENT_WEBHOOK_SECRET:?procurement webhook secret is required}"
: "${IOAP_REFUND_PROVIDER_API_TOKEN:?refund provider token is required}"
: "${IOAP_REFUND_WEBHOOK_SECRET:?refund webhook secret is required}"
: "${IOAP_FSM_ASSIGNMENT_PROVIDER_API_TOKEN:?FSM provider token is required}"
: "${IOAP_FSM_ASSIGNMENT_WEBHOOK_SECRET:?FSM webhook secret is required}"
: "${IOAP_SERVICE_QUOTATION_PROVIDER_API_TOKEN:?quotation provider token is required}"

until vault status >/dev/null 2>&1; do
  sleep 1
done

if ! vault secrets list -format=json | grep -q '"secret/"'; then
  vault secrets enable -path=secret kv-v2 >/dev/null
fi

label_studio_api_token=${IOAP_LABEL_STUDIO_API_TOKEN:-}
model_gateway_api_key=${IOAP_MODEL_GATEWAY_API_KEY:-}
edge_pack_signing_private_key=${IOAP_EDGE_PACK_SIGNING_PRIVATE_KEY:-}
realtime_turn_shared_secret=${IOAP_REALTIME_TURN_SHARED_SECRET:-}
enterprise_tool_gateway_token=${IOAP_ENTERPRISE_TOOL_GATEWAY_TOKEN:-}
notification_provider_api_token=${IOAP_NOTIFICATION_PROVIDER_API_TOKEN:-}
notification_webhook_secret=${IOAP_NOTIFICATION_WEBHOOK_SECRET:-}
procurement_provider_api_token=${IOAP_PROCUREMENT_PROVIDER_API_TOKEN:-}
procurement_webhook_secret=${IOAP_PROCUREMENT_WEBHOOK_SECRET:-}
refund_provider_api_token=${IOAP_REFUND_PROVIDER_API_TOKEN:-}
refund_webhook_secret=${IOAP_REFUND_WEBHOOK_SECRET:-}
fsm_assignment_provider_api_token=${IOAP_FSM_ASSIGNMENT_PROVIDER_API_TOKEN:-}
fsm_assignment_webhook_secret=${IOAP_FSM_ASSIGNMENT_WEBHOOK_SECRET:-}
service_quotation_provider_api_token=${IOAP_SERVICE_QUOTATION_PROVIDER_API_TOKEN:-}
enterprise_mcp_client_secret=${IOAP_ENTERPRISE_MCP_CLIENT_SECRET:-}
supplier_a2a_client_secret=${IOAP_SUPPLIER_A2A_CLIENT_SECRET:-}
neo4j_password=${IOAP_NEO4J_PASSWORD:-}
opensearch_password=${IOAP_OPENSEARCH_PASSWORD:-}
secret_json=/tmp/m1-runtime-secrets.json
token_file=/vault/runtime/vault-token
nextauth_file=/vault/runtime/nextauth-secret
token_tmp="${token_file}.tmp.$$"
nextauth_tmp="${nextauth_file}.tmp.$$"
trap 'rm -f "$secret_json" "$token_tmp" "$nextauth_tmp"' EXIT HUP INT TERM
printf '%s\n' "{\"database_url\":\"$IOAP_DATABASE_URL\",\"oidc_client_secret\":\"$IOAP_OIDC_CLIENT_SECRET\",\"minio_access_key\":\"$IOAP_MINIO_ACCESS_KEY\",\"minio_secret_key\":\"$IOAP_MINIO_SECRET_KEY\",\"nextauth_secret\":\"$IOAP_NEXTAUTH_SECRET\",\"label_studio_api_token\":\"$label_studio_api_token\",\"model_gateway_api_key\":\"$model_gateway_api_key\",\"edge_pack_signing_private_key\":\"$edge_pack_signing_private_key\",\"realtime_turn_shared_secret\":\"$realtime_turn_shared_secret\",\"enterprise_tool_gateway_token\":\"$enterprise_tool_gateway_token\",\"notification_provider_api_token\":\"$notification_provider_api_token\",\"notification_webhook_secret\":\"$notification_webhook_secret\",\"procurement_provider_api_token\":\"$procurement_provider_api_token\",\"procurement_webhook_secret\":\"$procurement_webhook_secret\",\"refund_provider_api_token\":\"$refund_provider_api_token\",\"refund_webhook_secret\":\"$refund_webhook_secret\",\"fsm_assignment_provider_api_token\":\"$fsm_assignment_provider_api_token\",\"fsm_assignment_webhook_secret\":\"$fsm_assignment_webhook_secret\",\"service_quotation_provider_api_token\":\"$service_quotation_provider_api_token\",\"enterprise_mcp_client_secret\":\"$enterprise_mcp_client_secret\",\"supplier_a2a_client_secret\":\"$supplier_a2a_client_secret\",\"neo4j_password\":\"$neo4j_password\",\"opensearch_password\":\"$opensearch_password\"}" > "$secret_json"
vault kv put -mount=secret industrial-ops/m1 "@$secret_json" >/dev/null
rm -f "$secret_json"

vault policy write m1-api /vault/config/m1-api.hcl >/dev/null
app_token=$(vault token create -field=token -policy=m1-api -period=24h)
printf '%s' "$app_token" > "$token_tmp"
printf '%s' "$IOAP_NEXTAUTH_SECRET" > "$nextauth_tmp"
chown 10001:10001 "$token_tmp" "$nextauth_tmp"
chmod 0400 "$token_tmp" "$nextauth_tmp"
mv -f "$token_tmp" "$token_file"
mv -f "$nextauth_tmp" "$nextauth_file"
trap - EXIT HUP INT TERM
echo "Vault M1 policy and runtime injection files are ready."
