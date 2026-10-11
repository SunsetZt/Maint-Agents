#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"

runtime=${M1_CONTAINER_CLI:-docker}
runtime_env=${M1_RUNTIME_ENV_FILE:-.env.m1.local}
base_compose=compose.lite.yaml
model_compose=compose.real-model.yaml
temporary_files=()
auth_config=
target_environment=
diagnosis_alias=
vlm_alias=
api_url=
access_token_file=
diagnosis_image=
vlm_image=
router_image=
gpu_probe_image=
diagnosis_model_id=
diagnosis_model_path=
vlm_model_id=
vlm_model_path=
resolved_release_id=
host_gpu_cli=

cleanup() {
  local item
  for item in "${temporary_files[@]}"; do
    [[ ! -e "$item" || -f "$item" && ! -L "$item" ]] || continue
    rm -f -- "$item"
  done
}
trap cleanup EXIT

fail() {
  local reason=$1
  local exit_code=${2:-3}
  printf 'Real-model operations failed: %s\n' "$reason" >&2
  exit "$exit_code"
}

usage() {
  echo "usage: scripts/real_model_business_loop.sh <preflight|up|bind|verify|down>" >&2
}

validate_runtime_env_path() {
  [[ "$runtime_env" != /* ]] || fail "runtime_environment_path_outside_repository"
  case "/$runtime_env/" in
    */../*) fail "runtime_environment_path_outside_repository" ;;
  esac
  [[ -f "$runtime_env" && ! -L "$runtime_env" ]] || fail "runtime_environment_unavailable"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required_command_unavailable:$1"
}

require_runtime() {
  require_command "$runtime"
  "$runtime" compose version >/dev/null 2>&1 || fail "compose_plugin_unavailable"
}

resolve_host_gpu_cli() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    host_gpu_cli=$(command -v nvidia-smi)
  elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    host_gpu_cli=/usr/lib/wsl/lib/nvidia-smi
  else
    fail "required_command_unavailable:nvidia-smi"
  fi
}

compose() {
  M1_RUNTIME_ENV_FILE="$runtime_env" \
    "$runtime" compose \
      --env-file "$runtime_env" \
      -f "$base_compose" \
      -f "$model_compose" \
      "$@"
}

load_contract() {
  local contract
  mapfile -t contract < <(
    python3 - "$runtime_env" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

values: dict[str, str] = {}
for raw_line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    values[key.strip()] = value.strip()

defaults = {
    "IOAP_MODEL_GATEWAY_REQUIRED_ENVIRONMENT": "STAGING",
    "IOAP_MODEL_GATEWAY_ALIAS": "industrial-diagnosis-staging",
    "IOAP_VLM_MODEL_ALIAS": "industrial-diagnosis-staging",
    "IOAP_REAL_MODEL_API_URL": "http://127.0.0.1:8000/api/v1",
    "IOAP_REAL_MODEL_ACCESS_TOKEN_FILE": "",
}
keys = (
    "IOAP_MODEL_GATEWAY_REQUIRED_ENVIRONMENT",
    "IOAP_MODEL_GATEWAY_ALIAS",
    "IOAP_VLM_MODEL_ALIAS",
    "IOAP_REAL_MODEL_API_URL",
    "IOAP_REAL_MODEL_ACCESS_TOKEN_FILE",
    "IOAP_DIAGNOSIS_MODEL_IMAGE",
    "IOAP_VLM_MODEL_IMAGE",
    "IOAP_MODEL_ROUTER_IMAGE",
    "IOAP_GPU_PROBE_IMAGE",
    "IOAP_DIAGNOSIS_MODEL_ID",
    "IOAP_DIAGNOSIS_MODEL_PATH",
    "IOAP_VLM_MODEL_ID",
    "IOAP_VLM_MODEL_PATH",
)
for key in keys:
    value = os.environ.get(key, values.get(key, defaults.get(key, ""))).strip()
    if "\n" in value or "\r" in value:
        raise SystemExit(2)
    print(value)
PY
  ) || fail "runtime_contract_invalid"
  [[ ${#contract[@]} -eq 13 ]] || fail "runtime_contract_invalid"
  target_environment=${contract[0]^^}
  diagnosis_alias=${contract[1]}
  vlm_alias=${contract[2]}
  api_url=${contract[3]%/}
  access_token_file=${contract[4]}
  diagnosis_image=${contract[5]}
  vlm_image=${contract[6]}
  router_image=${contract[7]}
  gpu_probe_image=${contract[8]}
  diagnosis_model_id=${contract[9]}
  diagnosis_model_path=${contract[10]}
  vlm_model_id=${contract[11]}
  vlm_model_path=${contract[12]}
}

validate_contract() {
  local expected_alias image
  case "$target_environment" in
    STAGING) expected_alias=industrial-diagnosis-staging ;;
    PRODUCTION) expected_alias=industrial-diagnosis ;;
    *) fail "target_environment_invalid" ;;
  esac
  [[ "$diagnosis_alias" == "$expected_alias" ]] || fail "diagnosis_alias_environment_mismatch"
  [[ "$vlm_alias" == "$expected_alias" ]] || fail "vlm_alias_environment_mismatch"
  [[ "$api_url" =~ ^https?:// ]] || fail "model_gateway_api_url_invalid"
  [[ "${api_url#*://}" != *@* ]] || fail "model_gateway_api_url_contains_credentials"
  if [[ "$target_environment" == PRODUCTION && "$api_url" != https://* ]]; then
    fail "production_model_gateway_requires_https"
  fi
  for image in "$diagnosis_image" "$vlm_image" "$router_image" "$gpu_probe_image"; do
    [[ "$image" =~ @sha256:[0-9a-f]{64}$ ]] || fail "model_image_not_digest_pinned"
  done
  [[ -n "$diagnosis_model_id" && -n "$vlm_model_id" ]] || fail "model_identity_missing"
  [[ "$diagnosis_model_path" == /* && "$vlm_model_path" == /* ]] || fail "model_path_invalid"
}

prepare_auth_config() {
  [[ -n "$access_token_file" ]] || fail "model_gateway_access_file_missing"
  [[ -f "$access_token_file" && ! -L "$access_token_file" ]] || fail "model_gateway_access_file_unavailable"
  local access_value permissions
  permissions=$(stat -c '%a' "$access_token_file")
  (( (8#$permissions & 077) == 0 )) || fail "model_gateway_access_file_permissions"
  access_value=$(<"$access_token_file")
  [[ "$access_value" =~ ^[A-Za-z0-9._~+/=-]+$ ]] || fail "model_gateway_access_value_invalid"
  umask 077
  auth_config=$(mktemp)
  temporary_files+=("$auth_config")
  printf 'header = "Authorization: Bearer %s"\n' "$access_value" >"$auth_config"
  unset access_value
}

api_get() {
  local path=$1
  local output=$2
  curl \
    --fail \
    --silent \
    --show-error \
    --config "$auth_config" \
    --output "$output" \
    "$api_url$path" || fail "model_gateway_api_unavailable"
}

check_host_gpu() {
  "$host_gpu_cli" \
    --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader >/dev/null 2>&1 || fail "gpu_host_unavailable"
}

check_runtime_gpu() {
  compose run --rm --no-deps --gpus all gpu-probe \
    nvidia-smi --query-gpu=name,memory.total,driver_version \
      --format=csv,noheader >/dev/null 2>&1 || fail "gpu_runtime_unavailable"
}

check_runtime_status() {
  local status_file release
  status_file=$(mktemp)
  temporary_files+=("$status_file")
  api_get "/model-gateway/runtime-status" "$status_file"
  release=$(python3 - "$status_file" "$target_environment" "$diagnosis_alias" "$vlm_alias" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    data = payload["data"]
    environment, diagnosis_alias, vlm_alias = sys.argv[2:5]
    assert data["status"] == "READY"
    assert data["target_environment"] == environment
    assert data["gateway_status"] == "READY"
    assert data["gpu_serving_required"] is True
    components = data["components"]
    assert {"ocr", "vlm", "diagnosis"} <= set(components)
    ocr = components["ocr"]
    assert ocr["binding_status"] == "READY"
    assert str(ocr["provider"]).lower() == "paddleocr"
    assert ocr["processor_version"]
    assert not str(ocr["processor_version"]).lower().startswith("development-")
    for name, alias in (("vlm", vlm_alias), ("diagnosis", diagnosis_alias)):
        component = components[name]
        assert component["binding_status"] == "READY"
        assert component["endpoint_status"] == "READY"
        assert component["alias"] == alias
        assert component["release_id"]
        assert component["deployment_id"]
        assert "GPU" in str(component["accelerator"]).upper()
    releases = {components[name]["release_id"] for name in ("ocr", "vlm", "diagnosis")}
    assert len(releases) == 1
except (AssertionError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1) from None
print(releases.pop())
PY
  ) || fail "model_runtime_unavailable"
  resolved_release_id=$release
}

check_routes() {
  local routes_file
  routes_file=$(mktemp)
  temporary_files+=("$routes_file")
  api_get "/model-gateway/routes" "$routes_file"
  python3 - \
    "$routes_file" \
    "$target_environment" \
    "$diagnosis_alias" \
    "$vlm_alias" \
    "$resolved_release_id" <<'PY' || fail "governed_model_route_unavailable"
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    environment, diagnosis_alias, vlm_alias, release_id = sys.argv[2:6]
    routes = payload["data"]
    by_alias = {route["alias"]: route for route in routes}
    for alias in {diagnosis_alias, vlm_alias}:
        route = by_alias[alias]
        assert route["target_environment"] == environment
        assert route["active_release_id"] == release_id
        assert route["deployment_id"]
        assert route["status"] == "ACTIVE"
        assert route["quota"]["enabled"] is True
except (AssertionError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1) from None
PY
}

check_inference_audits() {
  local inferences_file
  inferences_file=$(mktemp)
  temporary_files+=("$inferences_file")
  api_get "/model-gateway/inferences?limit=100" "$inferences_file"
  python3 - "$inferences_file" "$resolved_release_id" <<'PY' || fail "real_model_audit_missing"
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    release_id = sys.argv[2]
    successful = {
        record["request_class"]
        for record in payload["data"]
        if record.get("status") == "SUCCEEDED"
        and record.get("resolved_release_id") == release_id
    }
    assert {"VLM", "DIAGNOSIS"} <= successful
except (AssertionError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1) from None
PY
}

local_preflight() {
  validate_runtime_env_path
  require_command python3
  require_command curl
  resolve_host_gpu_cli
  require_runtime
  load_contract
  validate_contract
  [[ -f "$base_compose" && -f "$model_compose" ]] || fail "required_compose_overlay_missing"
  compose config >/dev/null || fail "compose_contract_invalid"
  check_host_gpu
  check_runtime_gpu
}

governance_preflight() {
  prepare_auth_config
  check_runtime_status
  check_routes
}

fixed_staging_services=(
  web
  api
  worker
  workflow-worker
  model-serving-gateway
  diagnosis-model
  vlm-model
)

case "${1:-}" in
  preflight)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    local_preflight
    governance_preflight
    echo "REAL_MODEL_PREFLIGHT_OK"
    ;;
  up)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    local_preflight
    governance_preflight
    compose up --build --detach --wait
    check_runtime_status
    check_routes
    echo "REAL_MODEL_STACK_READY"
    ;;
  bind)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    local_preflight
    governance_preflight
    echo "REAL_MODEL_BINDING_READY"
    ;;
  verify)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    local_preflight
    governance_preflight
    check_inference_audits
    compose ps --status running >/dev/null
    echo "REAL_MODEL_BUSINESS_AUDIT_OK"
    ;;
  down)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    validate_runtime_env_path
    require_command python3
    require_runtime
    load_contract
    [[ "$target_environment" == STAGING ]] || fail "staging_only_down_required"
    validate_contract
    compose stop "${fixed_staging_services[@]}"
    compose rm --force --stop "${fixed_staging_services[@]}"
    echo "REAL_MODEL_STAGING_RESOURCES_STOPPED"
    ;;
  *)
    usage
    exit 2
    ;;
esac
