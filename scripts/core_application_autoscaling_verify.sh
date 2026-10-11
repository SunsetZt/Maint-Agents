#!/usr/bin/env bash
set -euo pipefail

chart_lint() {
  local digest="sha256:6666666666666666666666666666666666666666666666666666666666666666"
  local args=(
    --set-string runtime.configMapName=industrial-ops-runtime
    --set-string runtime.secretName=industrial-ops-runtime
    --set-string gateway.enabled=true
    --set-string gateway.className=industrial-ops-envoy
    --set-string gateway.hostname=ops.staging.example.com
    --set-string gateway.tlsSecretName=industrial-ops-application-tls
    --set-string autoscaling.enabled=true
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
    echo CORE_APPLICATION_AUTOSCALING_CHART_OK
    ;;
  build)
    "$0" chart
    helm lint infra/helm/industrial-ops-ai-platform \
      --set-string gateway.hostname=models.staging.example.com \
      --set-string gateway.tlsSecretName=industrial-models-staging-tls
    helm lint infra/helm/industrial-ops-gitops \
      -f infra/helm/industrial-ops-gitops/values.staging.example.yaml
    helm lint infra/helm/industrial-ops-gitops \
      -f infra/helm/industrial-ops-gitops/values.production.example.yaml
    echo CORE_APPLICATION_AUTOSCALING_GITOPS_OK
    ;;
  *)
    echo "usage: $0 {chart|build}" >&2
    exit 2
    ;;
esac
