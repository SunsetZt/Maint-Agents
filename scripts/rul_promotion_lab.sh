#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${IOAP_RUL_LAB_PYTHON:-${repo_root}/.venv/bin/python3}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="$(command -v python3)"
fi
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${repo_root}"
exec "${python_bin}" -B -m industrial_ops_agent.predictive_maintenance.rul_promotion_lab "$@"
