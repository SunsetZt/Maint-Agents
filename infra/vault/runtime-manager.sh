#!/bin/sh
set -u

: "${VAULT_ADDR:?VAULT_ADDR is required}"
: "${VAULT_TOKEN:?VAULT_TOKEN is required}"

token_file=/vault/runtime/vault-token
ready_file=/vault/runtime/vault-runtime-ready
bootstrap_script=/vault/config/bootstrap.sh
reconcile_seconds=${IOAP_VAULT_RECONCILE_SECONDS:-15}

case "$reconcile_seconds" in
  ''|*[!0-9]*)
    echo "Vault runtime manager: reconcile interval must be an integer" >&2
    exit 2
    ;;
esac
if [ "$reconcile_seconds" -lt 5 ]; then
  echo "Vault runtime manager: reconcile interval must be at least 5 seconds" >&2
  exit 2
fi

rm -f "$ready_file"
trap 'rm -f "$ready_file"' EXIT HUP INT TERM

vault_is_ready() {
  vault status >/dev/null 2>&1
}

runtime_token_is_valid() {
  [ -s "$token_file" ] || return 1
  runtime_token=$(cat "$token_file") || return 1
  [ -n "$runtime_token" ] || return 1
  VAULT_TOKEN="$runtime_token" vault token lookup >/dev/null 2>&1 || return 1
  VAULT_TOKEN="$runtime_token" \
    vault kv get -mount=secret -field=database_url industrial-ops/m1 >/dev/null 2>&1
}

renew_runtime_token() {
  runtime_token=$(cat "$token_file") || return 1
  VAULT_TOKEN="$runtime_token" vault token renew >/dev/null
}

sync_model_secret() {
  [ "${IOAP_MODEL_SECRET_SYNC_ENABLED:-false}" = true ] || return 0
  /bin/sh /vault/config/model-secret-sync.sh
}

reconcile() {
  if runtime_token_is_valid && renew_runtime_token; then
    if ! sync_model_secret; then
      rm -f "$ready_file"
      return 1
    fi
    : > "$ready_file"
    return 0
  fi

  rm -f "$ready_file"
  echo "Vault runtime manager: application token is missing or stale; reconciling."
  if /bin/sh "$bootstrap_script" && runtime_token_is_valid && sync_model_secret; then
    : > "$ready_file"
    echo "Vault runtime manager: runtime credentials are healthy."
    return 0
  fi
  echo "Vault runtime manager: reconciliation failed; retrying." >&2
  return 1
}

while :; do
  if vault_is_ready; then
    reconcile || true
  else
    rm -f "$ready_file"
  fi
  sleep "$reconcile_seconds" &
  wait $!
done
