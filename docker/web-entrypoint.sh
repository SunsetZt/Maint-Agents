#!/bin/sh
set -eu

secret_file=${NEXTAUTH_SECRET_FILE:-/run/industrial-ops-secrets/nextauth-secret}
if [ ! -r "$secret_file" ]; then
  echo "web startup failed: secret source=file category=unavailable name=nextauth_secret" >&2
  exit 4
fi
NEXTAUTH_SECRET=$(tr -d '\r\n' < "$secret_file")
if [ -z "$NEXTAUTH_SECRET" ]; then
  echo "web startup failed: secret source=file category=missing name=nextauth_secret" >&2
  exit 4
fi
export NEXTAUTH_SECRET
exec "$@"
