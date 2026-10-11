#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  end-to-end)
    scripts/m3_integration.sh verify
    echo M3_DATA_END_TO_END_OK
    ;;
  integration-build)
    scripts/m3_integration.sh build
    echo M3_DATA_INTEGRATION_BUILD_OK
    ;;
  task-five-static)
    scripts/generate_openapi.sh --check
    scripts/generate_openapi_client.sh --check
    .venv/bin/ruff check \
      src/industrial_ops_agent/api/app.py \
      src/industrial_ops_agent/api/dependencies.py \
      src/industrial_ops_agent/api/routes/data_feedback.py \
      src/industrial_ops_agent/auth/policy.py \
      src/industrial_ops_agent/config.py \
      src/industrial_ops_agent/data_pipeline/service.py \
      src/industrial_ops_agent/data_pipeline/cli.py \
      src/industrial_ops_agent/data_pipeline/spark_curate.py \
      src/industrial_ops_agent/event_worker/handlers.py \
      src/industrial_ops_agent/knowledge/seed.py \
      src/industrial_ops_agent/labeling/service.py \
      src/industrial_ops_agent/runtime.py \
      scripts/m3_integration_probe.py
    (
      cd web
      ./node_modules/.bin/tsc --noEmit
    )
    echo M3_DATA_TASK_FIVE_STATIC_OK
    ;;
  *)
    echo "usage: $0 {end-to-end|integration-build|task-five-static}" >&2
    exit 2
    ;;
esac
