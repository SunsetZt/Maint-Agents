#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
[[ -z "$mode" || "$mode" == "--check" ]] || {
  echo "usage: scripts/generate_openapi_client.sh [--check]" >&2
  exit 2
}
[[ -x web/node_modules/.bin/openapi-typescript ]] || {
  echo "web dependencies are not installed" >&2
  exit 3
}

mkdir -p web/generated/api
generated=$(mktemp "${TMPDIR:-/tmp}/m1-openapi-client.XXXXXX")
trap 'rm -f -- "$generated"' EXIT
web/node_modules/.bin/openapi-typescript contracts/openapi.json -o "$generated" >/dev/null

if [[ "$mode" == "--check" ]]; then
  cmp -s "$generated" web/generated/api/schema.d.ts || {
    echo "Generated API client drift: run scripts/generate_openapi_client.sh" >&2
    exit 1
  }
else
  mv -f -- "$generated" web/generated/api/schema.d.ts
fi
