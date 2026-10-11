#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  static-core)
    .venv/bin/python -B -m ruff check \
      src/industrial_ops_agent/supply_chain/verifier.py \
      src/industrial_ops_agent/staging_acceptance/attestation.py \
      src/industrial_ops_agent/staging_acceptance/attestation_cli.py \
      src/industrial_ops_agent/persistence/models.py \
      src/industrial_ops_agent/auth/policy.py
    .venv/bin/python -B -m mypy --strict \
      --cache-dir .mypy_cache \
      src/industrial_ops_agent/supply_chain/verifier.py \
      src/industrial_ops_agent/staging_acceptance/attestation.py \
      src/industrial_ops_agent/staging_acceptance/attestation_cli.py
    echo STAGING_ATTESTATION_CORE_STATIC_OK
    ;;
  static-gate)
    .venv/bin/python -B -m ruff check \
      src/industrial_ops_agent/staging_acceptance/attestation.py \
      src/industrial_ops_agent/assurance/service.py \
      src/industrial_ops_agent/api/routes/assurance.py
    .venv/bin/python -B -m mypy --strict \
      --cache-dir .mypy_cache \
      src/industrial_ops_agent/staging_acceptance/attestation.py \
      src/industrial_ops_agent/assurance/service.py \
      src/industrial_ops_agent/api/routes/assurance.py
    echo STAGING_ATTESTATION_GATE_STATIC_OK
    ;;
  static-old)
    scripts/generate_openapi.sh --check
    echo STAGING_ATTESTATION_OLD_STATIC_OK
    ;;
  static-replacement)
    scripts/generate_openapi.sh --check
    scripts/generate_openapi_client.sh --check
    npm --prefix web run typecheck
    echo STAGING_ATTESTATION_REPLACEMENT_STATIC_OK
    ;;
  *)
    echo "usage: $0 {static-core|static-gate|static-old|static-replacement}" >&2
    exit 2
    ;;
esac
