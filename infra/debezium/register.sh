#!/bin/sh
set -eu

: "${DEBEZIUM_URL:?DEBEZIUM_URL is required}"

until curl --fail --silent "$DEBEZIUM_URL/connectors" >/dev/null; do
  sleep 2
done

if curl --fail --silent "$DEBEZIUM_URL/connectors/industrial-ops-outbox" >/dev/null; then
  curl --fail --silent --show-error \
    -X DELETE \
    "$DEBEZIUM_URL/connectors/industrial-ops-outbox" >/dev/null
  until ! curl --fail --silent "$DEBEZIUM_URL/connectors/industrial-ops-outbox" >/dev/null 2>&1; do
    sleep 1
  done
fi

curl --fail --silent --show-error \
  -X POST \
  -H "Content-Type: application/json" \
  --data-binary @/bootstrap/connector.json \
  "$DEBEZIUM_URL/connectors" >/dev/null
echo "Debezium industrial Outbox connector is ready."
