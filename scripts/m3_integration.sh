#!/usr/bin/env bash
set -euo pipefail

command=${1:-}
environment_file=${M1_RUNTIME_ENV_FILE:-.env.m1.local}
[[ -f "$environment_file" ]] || {
  echo "M1 runtime environment is missing; run scripts/dev_lite.sh init first." >&2
  exit 2
}
export M1_RUNTIME_ENV_FILE="$environment_file"

compose=(
  docker compose
  --env-file "$environment_file"
  -f compose.lite.yaml
  -f compose.integration.yaml
)

case "$command" in
  build)
    scripts/m3_prepare_build_cache.sh
    "${compose[@]}" config --quiet
    "${compose[@]}" build api web event-worker data-worker
    ;;
  verify)
    "${compose[@]}" up --detach --wait --wait-timeout 480 \
      api web event-worker data-worker spark-worker \
      label-studio-init marquez debezium-init
    "${compose[@]}" exec -T debezium \
      curl --fail --silent http://localhost:8083/connectors/industrial-ops-outbox/status \
      | grep -q 'RUNNING'
    "${compose[@]}" exec -T data-worker \
      airflow dags list --output json \
      | grep -q 'build_dataset_snapshot'
    "${compose[@]}" exec -T data-worker \
      python /opt/industrial-ops/m3_integration_probe.py \
      | grep -q 'M3_INTEGRATION_PROBE_OK'

    offline_services=(
      event-worker data-worker spark-worker spark-master
      debezium kafka label-studio marquez
    )
    "${compose[@]}" stop "${offline_services[@]}"
    "${compose[@]}" exec -T web node -e \
      "Promise.all([fetch('http://api:8000/health/live'),fetch('http://localhost:3000/login')]).then(rs=>{if(rs.some(r=>!r.ok))process.exit(1)}).catch(()=>process.exit(1))"
    "${compose[@]}" up --detach --wait --wait-timeout 480 "${offline_services[@]}"
    ;;
  *)
    echo "usage: $0 {build|verify}" >&2
    exit 2
    ;;
esac
