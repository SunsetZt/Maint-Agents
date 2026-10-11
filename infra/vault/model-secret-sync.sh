#!/bin/sh
# Executed only inside the existing Vault bootstrap/runtime-manager containers.
# No host export, environment injection into models, or secret-bearing logs.
set -eu
umask 077

[ "${IOAP_MODEL_SECRET_SYNC_ENABLED:-false}" = true ] || exit 0
directory=/vault/model-serving
destination=$directory/model-gateway-api-key
ready=$directory/ready
temporary=
cleanup() {
  [ -z "$temporary" ] || rm -f -- "$temporary"
}
trap cleanup EXIT HUP INT TERM
fail() {
  rm -f -- "$ready"
  echo "Vault model secret injection unavailable." >&2
  exit 1
}
[ -d "$directory" ] && [ ! -L "$directory" ] || fail
chown 65532:65532 "$directory"
chmod 0700 "$directory"
[ ! -L "$destination" ] && [ ! -L "$ready" ] || fail
temporary=$(mktemp "$directory/.gateway.XXXXXX") || fail
if ! vault kv get -mount=secret -field=model_gateway_api_key industrial-ops/m1 >"$temporary" 2>/dev/null; then
  fail
fi
# Existing gateway keys are bounded single-line UTF-8 credentials.
[ -s "$temporary" ] || fail
[ "$(wc -c < "$temporary")" -le 4096 ] || fail
[ "$(wc -l < "$temporary")" -eq 0 ] || fail
if LC_ALL=C grep -q '[[:cntrl:]]' "$temporary"; then fail; fi
chown 65532:65532 "$temporary"
chmod 0400 "$temporary"
if [ -f "$destination" ] && cmp -s "$temporary" "$destination"; then
  chown 65532:65532 "$destination"
  chmod 0400 "$destination"
else
  mv -f -- "$temporary" "$destination"
fi
# Consumers mount the directory, not one inode, so atomic replacement is visible.
[ -f "$ready" ] || : > "$ready"
chown 65532:65532 "$ready"
chmod 0400 "$ready"
