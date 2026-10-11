#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${IOAP_GRPO_ROLLOUT_PYTHON:-${repo_root}/.venv/bin/python3}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="$(command -v python3)"
fi
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
cd "${repo_root}"
exec "${python_bin}" -B -m industrial_ops_agent.simulation.grpo_kserve_rollout_cli "$@"
