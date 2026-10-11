#!/bin/sh
set -eu

: "${LABEL_STUDIO_URL:?LABEL_STUDIO_URL is required}"
: "${LABEL_STUDIO_USER_TOKEN:?LABEL_STUDIO_USER_TOKEN is required}"

until curl --fail --silent "$LABEL_STUDIO_URL/health" >/dev/null; do
  sleep 2
done

if curl --fail --silent \
  -H "Authorization: Token $LABEL_STUDIO_USER_TOKEN" \
  "$LABEL_STUDIO_URL/api/projects?page_size=100" \
  | grep -q 'Industrial Root Cause Review'; then
  curl --fail --silent --show-error \
    -X PATCH \
    -H "Authorization: Token $LABEL_STUDIO_USER_TOKEN" \
    -H "Content-Type: application/json" \
    --data-binary @/bootstrap/project.json \
    "$LABEL_STUDIO_URL/api/projects/1" >/dev/null
  echo "Label Studio governed project is up to date."
  exit 0
fi

curl --fail --silent --show-error \
  -X POST \
  -H "Authorization: Token $LABEL_STUDIO_USER_TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @/bootstrap/project.json \
  "$LABEL_STUDIO_URL/api/projects" >/dev/null
echo "Label Studio governed project is ready."
