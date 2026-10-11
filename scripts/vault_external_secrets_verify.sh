#!/usr/bin/env bash
set -euo pipefail

chart_verify() {
  helm version --short
  helm lint infra/helm/industrial-ops-runtime-secrets
}

case "${1:-}" in
  chart)
    chart_verify
    echo VAULT_EXTERNAL_SECRETS_CHART_OK
    ;;
  build)
    chart_verify
    helm lint infra/helm/industrial-ops-ai-platform \
      --set-string gateway.hostname=models.staging.example.com \
      --set-string gateway.tlsSecretName=industrial-models-staging-tls
    helm lint infra/helm/industrial-ops-gitops \
      -f infra/helm/industrial-ops-gitops/values.staging.example.yaml
    helm lint infra/helm/industrial-ops-gitops \
      -f infra/helm/industrial-ops-gitops/values.production.example.yaml
    echo VAULT_EXTERNAL_SECRETS_GITOPS_OK
    ;;
  *)
    echo "usage: $0 {chart|build}" >&2
    exit 2
    ;;
esac
