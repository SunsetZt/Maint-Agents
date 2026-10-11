#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TERRAFORM_ROOT="$REPOSITORY_ROOT/infra/terraform/aws"
DERIVED_ROOT="$REPOSITORY_ROOT/.cache/terraform"
TERRAFORM_VERSION="1.15.8"
TERRAFORM_IMAGE="hashicorp/terraform:${TERRAFORM_VERSION}"
VALIDATE_WORKSPACE=""

export TF_IN_AUTOMATION=1
export AWS_EC2_METADATA_DISABLED=true
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE
unset AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE AWS_WEB_IDENTITY_TOKEN_FILE
unset AWS_ROLE_ARN AWS_ROLE_SESSION_NAME
unset AWS_CONTAINER_CREDENTIALS_FULL_URI AWS_CONTAINER_CREDENTIALS_RELATIVE_URI

terraform_engine() {
  if command -v terraform >/dev/null 2>&1; then
    local native_version
    native_version=$(terraform version | sed -n '1s/^Terraform v//p')
    if [[ "$native_version" == "$TERRAFORM_VERSION" ]]; then
      printf '%s\n' native
      return 0
    fi
  fi

  command -v docker >/dev/null 2>&1 || {
    echo "[ERR] Terraform ${TERRAFORM_VERSION} or Docker is required." >&2
    return 1
  }
  docker image inspect "$TERRAFORM_IMAGE" >/dev/null 2>&1 || {
    echo "[ERR] Missing pinned image ${TERRAFORM_IMAGE}; install it explicitly before verification." >&2
    return 1
  }
  printf '%s\n' docker
}

run_terraform() {
  local engine
  engine=$(terraform_engine)
  if [[ "$engine" == native ]]; then
    local argument
    local native_arguments=()
    for argument in "$@"; do
      case "$argument" in
        -chdir=/workspace/*)
          native_arguments+=("-chdir=$REPOSITORY_ROOT/${argument#-chdir=/workspace/}")
          ;;
        *)
          native_arguments+=("$argument")
          ;;
      esac
    done
    terraform "${native_arguments[@]}"
    return
  fi

  local proxy_name
  local docker_env=(
    --env TF_IN_AUTOMATION=1
    --env AWS_EC2_METADATA_DISABLED=true
  )
  for proxy_name in HTTP_PROXY HTTPS_PROXY NO_PROXY; do
    if [[ -n "${!proxy_name:-}" ]]; then
      docker_env+=(--env "$proxy_name")
    fi
  done

  docker run --rm --pull=never \
    --user "$(id -u):$(id -g)" \
    "${docker_env[@]}" \
    --volume "$REPOSITORY_ROOT:/workspace" \
    --workdir /workspace \
    "$TERRAFORM_IMAGE" "$@"
}

python_command() {
  if [[ -x "$REPOSITORY_ROOT/.venv/bin/python" ]]; then
    printf '%s\n' "$REPOSITORY_ROOT/.venv/bin/python"
  else
    command -v python3
  fi
}

verify_identity() {
  local engine
  engine=$(terraform_engine)
  echo "Terraform v${TERRAFORM_VERSION} (${engine})"
  echo "TERRAFORM_AWS_TOOLCHAIN_OK"
}

verify_static_contract() {
  local python
  python=$(python_command)
  "$python" -B - "$REPOSITORY_ROOT" "$TERRAFORM_VERSION" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path


repository_root = Path(sys.argv[1]).resolve()
terraform_version = sys.argv[2]
terraform_root = repository_root / "infra" / "terraform" / "aws"
terraform_files = sorted(terraform_root.glob("*.tf"))
if not terraform_files:
    raise SystemExit("AWS Terraform root has no configuration files")

terraform_text = "\n".join(path.read_text(encoding="utf-8") for path in terraform_files)
forbidden_hcl = (
    r'provider\s+"(?:kubernetes|helm|kubectl|argocd)"',
    r'resource\s+"(?:kubernetes_|helm_|kubectl_|argocd_)',
)
for pattern in forbidden_hcl:
    if re.search(pattern, terraform_text, flags=re.IGNORECASE):
        raise SystemExit(f"Terraform crossed the GitOps ownership boundary: {pattern}")

required_versions = (
    'required_version = "~> 1.15.0"',
    'version = "6.55.0"',
    'source = "https://github.com/terraform-aws-modules/terraform-aws-vpc/archive/refs/tags/v6.6.1.tar.gz//*?archive=tar.gz&checksum=sha256:97e1dcabe56e258f31195639c72166ef27a0a063f1526f2b5818e2660d7a13eb"',
    'source = "https://github.com/terraform-aws-modules/terraform-aws-eks/archive/refs/tags/v21.24.0.tar.gz//*?archive=tar.gz&checksum=sha256:7975b9bb28e853524e79657be71ea8a35b9b0c6fbddbf8543f6a54b5d1a9e17f"',
    'source = "https://github.com/terraform-aws-modules/terraform-aws-kms/archive/refs/tags/v4.2.0.tar.gz//*?archive=tar.gz&checksum=sha256:511b89c143c8f8adb4c6dc002439bc6518db2c4c320fdabece67fd8bf4623d0d"',
    'source = "https://github.com/terraform-aws-modules/terraform-aws-rds/archive/refs/tags/v7.2.0.tar.gz//*?archive=tar.gz&checksum=sha256:93f9670f6a2ac0e223589f697dd05094f0796bb0e9e4865f13550e33221eedd5"',
    'source = "https://github.com/terraform-aws-modules/terraform-aws-s3-bucket/archive/refs/tags/v5.14.1.tar.gz//*?archive=tar.gz&checksum=sha256:62c92c5b40af22cfbd1aae96b7aa72f3b005b2fa64527bb346dbc235429cb728"',
)
for fragment in required_versions:
    if fragment not in terraform_text:
        raise SystemExit(f"Unpinned or missing Terraform dependency: {fragment}")

required_log_encryption = (
    r"cloudwatch_log_group_kms_key_id\s*=\s*module\.data_kms\.key_arn",
    r"flow_log_cloudwatch_log_group_kms_key_id\s*=\s*module\.data_kms\.key_arn",
)
for pattern in required_log_encryption:
    if re.search(pattern, terraform_text) is None:
        raise SystemExit(f"Cloud audit log encryption contract is missing: {pattern}")

ci_text = (repository_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
script_text = (repository_root / "scripts" / "terraform_aws_verify.sh").read_text(encoding="utf-8")
for fragment in (
    "for proxy_name in HTTP_PROXY HTTPS_PROXY NO_PROXY",
    'docker_env+=(--env "$proxy_name")',
):
    if fragment not in script_text:
        raise SystemExit(f"Docker Terraform proxy forwarding contract is missing: {fragment}")
mutation = re.compile(
    r"^\s*(?:run_terraform\s+|terraform\s+)(?:plan|apply|destroy)\b",
    flags=re.IGNORECASE | re.MULTILINE,
)
if mutation.search(ci_text) or mutation.search(script_text):
    raise SystemExit("Repository Terraform verification attempted a cloud mutation command")
if re.search(r"\bAWS_(?:ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN)\b\s*:", ci_text):
    raise SystemExit("Terraform CI must not receive AWS credentials")
if f'terraform_version: "{terraform_version}"' not in ci_text:
    raise SystemExit("CI Terraform patch version is not pinned")
PY
  echo "TERRAFORM_AWS_STATIC_OK"
}

cleanup_validate_workspace() {
  local workspace=${1:-}
  case "$workspace" in
    "$DERIVED_ROOT"/terraform-aws-validate.*)
      rm -rf -- "$workspace"
      ;;
    "")
      ;;
    *)
      echo "[ERR] Refusing to remove an unexpected Terraform validation path." >&2
      return 1
      ;;
  esac
}

trap 'cleanup_validate_workspace "$VALIDATE_WORKSPACE"' EXIT
trap 'exit 143' HUP INT TERM

verify_configuration() {
  mkdir -p "$DERIVED_ROOT"
  mkdir -p "$DERIVED_ROOT/terraform-plugin-cache"
  export TF_PLUGIN_CACHE_DIR="$DERIVED_ROOT/terraform-plugin-cache"
  VALIDATE_WORKSPACE=$(mktemp -d "$DERIVED_ROOT/terraform-aws-validate.XXXXXX")

  cp "$TERRAFORM_ROOT"/*.tf "$VALIDATE_WORKSPACE/"
  local init_arguments=(init -backend=false -input=false -no-color)
  if [[ -f "$TERRAFORM_ROOT/.terraform.lock.hcl" ]]; then
    local required_lock_fragments=(
      'provider "registry.terraform.io/hashicorp/aws"'
      'version     = "6.55.0"'
    )
    local lock_fragment
    for lock_fragment in "${required_lock_fragments[@]}"; do
      grep -Fq -- "$lock_fragment" "$TERRAFORM_ROOT/.terraform.lock.hcl" || {
        echo "[ERR] Terraform dependency lock drift: ${lock_fragment}" >&2
        return 1
      }
    done
    cp "$TERRAFORM_ROOT/.terraform.lock.hcl" "$VALIDATE_WORKSPACE/"
    init_arguments+=(-lockfile=readonly)
  else
    echo "[ERR] Terraform dependency lock drift: .terraform.lock.hcl is missing" >&2
    return 1
  fi

  run_terraform "-chdir=/workspace/${VALIDATE_WORKSPACE#"$REPOSITORY_ROOT/"}" fmt -check -recursive
  run_terraform "-chdir=/workspace/${VALIDATE_WORKSPACE#"$REPOSITORY_ROOT/"}" "${init_arguments[@]}"
  run_terraform "-chdir=/workspace/${VALIDATE_WORKSPACE#"$REPOSITORY_ROOT/"}" validate -no-color
  echo "TERRAFORM_AWS_VALIDATE_OK"
}

generate_dependency_lock() {
  run_terraform "-chdir=/workspace/infra/terraform/aws" init \
    -backend=false -input=false -no-color
  run_terraform "-chdir=/workspace/infra/terraform/aws" providers lock \
    -platform=linux_amd64 -platform=linux_arm64
  echo "TERRAFORM_AWS_LOCK_OK"
}

case "${1:-}" in
  identity)
    verify_identity
    ;;
  static)
    verify_static_contract
    ;;
  validate)
    verify_configuration
    ;;
  lock)
    generate_dependency_lock
    ;;
  *)
    echo "usage: $0 {identity|static|validate|lock}" >&2
    exit 2
    ;;
esac
