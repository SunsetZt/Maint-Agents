#!/usr/bin/env bash
set -euo pipefail

chart_lint() {
  local digest="sha256:4444444444444444444444444444444444444444444444444444444444444444"
  local args=(
    --set-string runtime.configMapName=industrial-ops-runtime
    --set-string runtime.secretName=industrial-ops-runtime
    --set-string gateway.enabled=true
    --set-string gateway.className=industrial-ops-envoy
    --set-string gateway.hostname=ops.staging.example.com
    --set-string gateway.tlsSecretName=industrial-ops-application-tls
  )
  local image
  for image in api web mediaWorker workflowWorker eventWorker; do
    args+=(
      --set-string "images.${image}.repository=registry.example.com/industrial/${image}"
      --set-string "images.${image}.digest=${digest}"
    )
  done
  helm lint infra/helm/industrial-ops-application "${args[@]}"
}

case "${1:-}" in
  chart)
    helm version --short
    chart_lint
    echo APPLICATION_WORKLOADS_CHART_OK
    ;;
  build)
    "$0" chart
    helm lint infra/helm/industrial-ops-gitops \
      -f infra/helm/industrial-ops-gitops/values.staging.example.yaml
    helm lint infra/helm/industrial-ops-gitops \
      -f infra/helm/industrial-ops-gitops/values.production.example.yaml
    echo APPLICATION_WORKLOADS_GITOPS_OK
    ;;
  *)
    echo "usage: $0 {chart|build}" >&2
    exit 2
    ;;
esac
