#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  static)
    .venv/bin/python -B -m mypy --cache-dir .mypy_cache \
      src/industrial_ops_agent/config.py \
      src/industrial_ops_agent/deployment/cli.py \
      src/industrial_ops_agent/deployment/kserve.py \
      src/industrial_ops_agent/deployment/leader_election.py
    echo RELEASE_CONTROLLER_HA_STATIC_OK
    ;;
  build)
    digest="sha256:2222222222222222222222222222222222222222222222222222222222222222"
    helm version --short
    helm lint infra/helm/industrial-ops-release-controller \
      --set "image.digest=${digest}"
    helm lint infra/helm/industrial-ops-gitops \
      -f infra/helm/industrial-ops-gitops/values.staging.example.yaml
    echo RELEASE_CONTROLLER_HA_HELM_OK
    ;;
  *)
    echo "usage: $0 {static|build}" >&2
    exit 2
    ;;
esac
