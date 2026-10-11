#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  backend)
    .venv/bin/python -B -m ruff check \
      src/industrial_ops_agent/deployment/service.py \
      src/industrial_ops_agent/deployment/kserve.py \
      src/industrial_ops_agent/api/routes/model_deployments.py
    .venv/bin/python -B -m mypy --cache-dir .mypy_cache \
      src/industrial_ops_agent/deployment/service.py \
      src/industrial_ops_agent/deployment/kserve.py \
      src/industrial_ops_agent/api/routes/model_deployments.py
    echo INFERENCE_POOL_BACKEND_STATIC_OK
    ;;
  frontend)
    (
      cd web
      ./node_modules/.bin/tsc --noEmit
    )
    scripts/generate_openapi.sh --check
    scripts/generate_openapi_client.sh --check
    echo INFERENCE_POOL_FRONTEND_STATIC_OK
    ;;
  gitops)
    helm version --short
    helm lint infra/helm/industrial-ops-gitops \
      -f infra/helm/industrial-ops-gitops/values.staging.example.yaml
    helm lint infra/helm/industrial-ops-gitops \
      -f infra/helm/industrial-ops-gitops/values.production.example.yaml
    echo INFERENCE_POOL_GITOPS_OK
    ;;
  *)
    echo "usage: $0 {backend|frontend|gitops}" >&2
    exit 2
    ;;
esac
