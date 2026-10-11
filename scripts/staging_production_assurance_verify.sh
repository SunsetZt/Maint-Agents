#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  static-backend)
    .venv/bin/python -B -m ruff check \
      src/industrial_ops_agent/staging_acceptance/dynamic_manifest.py \
      src/industrial_ops_agent/assurance/service.py
    .venv/bin/python -B -m mypy --strict \
      --cache-dir .mypy_cache \
      src/industrial_ops_agent/staging_acceptance/dynamic_manifest.py \
      src/industrial_ops_agent/assurance/service.py
    echo STAGING_PRODUCTION_ASSURANCE_BACKEND_STATIC_OK
    ;;
  static)
    .venv/bin/python -B -m ruff check \
      src/industrial_ops_agent/staging_acceptance/dynamic_manifest.py \
      src/industrial_ops_agent/assurance/service.py \
      src/industrial_ops_agent/api/routes/assurance.py
    .venv/bin/python -B -m mypy --strict \
      --cache-dir .mypy_cache \
      src/industrial_ops_agent/staging_acceptance/dynamic_manifest.py \
      src/industrial_ops_agent/assurance/service.py \
      src/industrial_ops_agent/api/routes/assurance.py
    npm --prefix web run typecheck
    echo STAGING_PRODUCTION_ASSURANCE_STATIC_OK
    ;;
  *)
    echo "usage: $0 {static-backend|static}" >&2
    exit 2
    ;;
esac
