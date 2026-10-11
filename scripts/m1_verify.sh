#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  python-static)
    .venv/bin/ruff check src
    .venv/bin/mypy src
    echo "M1_PYTHON_STATIC_OK"
    ;;
  tenant-isolation)
    scripts/dev_lite.sh verify-rls
    echo "M1_TENANT_ISOLATION_OK"
    ;;
  api-client)
    scripts/generate_openapi.sh --check
    scripts/generate_openapi_client.sh --check
    echo "M1_API_CLIENT_OK"
    ;;
  compose-lite)
    scripts/dev_lite.sh verify
    echo "M1_COMPOSE_LITE_OK"
    ;;
  vault-config)
    test -x infra/vault/bootstrap.sh
    test -f infra/vault/policies/m1-api.hcl
    grep -F 'path "secret/data/industrial-ops/m1"' \
      infra/vault/policies/m1-api.hcl >/dev/null
    grep -F 'capabilities = ["read"]' \
      infra/vault/policies/m1-api.hcl >/dev/null
    echo "M1_VAULT_CONFIG_OK"
    ;;
  *)
    echo "usage: scripts/m1_verify.sh <python-static|tenant-isolation|api-client|compose-lite|vault-config>" >&2
    exit 2
    ;;
esac
