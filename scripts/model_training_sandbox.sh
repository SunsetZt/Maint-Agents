#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repository_root"

training_image=${IOAP_MODEL_SANDBOX_IMAGE:-industrial-ops/m4-training-worker:local}
runtime_env_file=${M1_RUNTIME_ENV_FILE:-$repository_root/.env.m1.local}
compose_project=${M1_COMPOSE_PROJECT_NAME:-industrial-ops-m1}
compose_network="${compose_project}_backend"
export M1_RUNTIME_ENV_FILE="$runtime_env_file"

fail() {
  printf 'MODEL_TRAINING_SANDBOX_FAILED reason=%s\n' "$1" >&2
  exit 3
}

run_cli_probe() {
  docker run --rm \
    --entrypoint python \
    --volume "$repository_root/src:/workspace/src:ro" \
    --env PYTHONPATH=/workspace/src \
    "$training_image" \
    -B -m industrial_ops_agent.training.reproducible_sandbox_cli --contract-probe
}

wait_for_mlflow() {
  local container_id status attempt
  for attempt in $(seq 1 60); do
    container_id=$(docker compose --env-file "$runtime_env_file" \
      -f compose.lite.yaml -f compose.integration.yaml ps -q mlflow)
    if [[ -n "$container_id" ]]; then
      status=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id")
      [[ "$status" == healthy ]] && return 0
    fi
    sleep 1
  done
  return 1
}

verify() {
  command -v docker >/dev/null || fail docker_unavailable
  docker info >/dev/null 2>&1 || fail docker_daemon_unavailable
  docker image inspect "$training_image" >/dev/null 2>&1 || fail training_image_missing
  [[ -f "$runtime_env_file" ]] || fail runtime_environment_missing

  local probe
  probe=$(run_cli_probe) || fail sandbox_cli_probe_failed
  [[ "$probe" == MODEL_TRAINING_SANDBOX_CLI_READY ]] || fail sandbox_cli_not_ready

  docker compose --env-file "$runtime_env_file" \
    -f compose.lite.yaml -f compose.integration.yaml up -d mlflow >/dev/null
  wait_for_mlflow || fail mlflow_not_healthy

  local run_id output_root output_directory image_digest
  run_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
  output_root="$repository_root/artifacts/model-training-sandbox"
  output_directory="$output_root/$run_id"
  mkdir -p "$output_directory"
  image_digest=$(docker image inspect --format '{{.Id}}' "$training_image")
  [[ "$image_digest" == sha256:* ]] || fail training_image_digest_invalid

  docker run --rm \
    --user "$(id -u):$(id -g)" \
    --network "$compose_network" \
    --entrypoint python \
    --volume "$repository_root/src:/workspace/src:ro" \
    --volume "$repository_root/datasets:/workspace/datasets:ro" \
    --volume "$output_directory:/workspace/output" \
    --env PYTHONPATH=/workspace/src \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_HUB_DISABLE_TELEMETRY=1 \
    --env IOAP_SANDBOX_IMAGE_DIGEST="$image_digest" \
    "$training_image" \
    -B -m industrial_ops_agent.training.reproducible_sandbox_cli \
    --dataset-root /workspace/datasets/model-training-sandbox/v1 \
    --output-directory /workspace/output \
    --mlflow-url http://mlflow:5000

  local summary
  summary=$(.venv/bin/python -B - "$output_directory/promotion-receipt.json" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert receipt["schema_version"] == "model-training-promotion-receipt/v1"
assert receipt["marker"] == "SIMULATED_NON_PRODUCTION"
assert receipt["decision"] == "CANDIDATE_ELIGIBLE"
assert receipt["formal_release_created"] is False
assert receipt["selected_method"] in {"LORA", "QLORA"}
assert set(receipt["mlflow_run_ids"]) == {"BASELINE", "LORA", "QLORA"}
print(
    "selected_method=" + receipt["selected_method"],
    "relative_improvement="
    + format(receipt["relative_improvements"][receipt["selected_method"]], ".6f"),
)
PY
  ) || fail promotion_receipt_invalid

  printf '%s\n' "$summary"
  printf 'artifact_directory=%s\n' "artifacts/model-training-sandbox/$run_id"
  printf 'SIMULATED_NON_PRODUCTION\n'
  printf 'MODEL_TRAINING_SANDBOX_CANDIDATE_OK\n'
}

case "${1:-}" in
  verify) verify ;;
  *)
    printf 'usage: %s verify\n' "$0" >&2
    exit 2
    ;;
esac
