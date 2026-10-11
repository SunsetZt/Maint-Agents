#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
[[ -z "$mode" || "$mode" == "--check" ]] || {
  echo "usage: scripts/generate_openapi.sh [--check]" >&2
  exit 2
}

mkdir -p contracts
generated=$(mktemp "${TMPDIR:-/tmp}/m1-openapi.XXXXXX")
trap 'rm -f -- "$generated"' EXIT

.venv/bin/python -B - <<'PY' >"$generated"
import json

from industrial_ops_agent.api.app import create_app
from industrial_ops_agent.config import Environment, Settings

app = create_app(Settings(environment=Environment.TEST, _env_file=None))
print(json.dumps(app.openapi(), ensure_ascii=False, indent=2, sort_keys=True))
PY

if [[ "$mode" == "--check" ]]; then
  cmp -s "$generated" contracts/openapi.json || {
    echo "OpenAPI contract drift: run scripts/generate_openapi.sh" >&2
    exit 1
  }
else
  mv -f -- "$generated" contracts/openapi.json
fi
