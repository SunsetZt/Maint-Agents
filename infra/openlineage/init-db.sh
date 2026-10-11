#!/bin/sh
set -eu

: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${MARQUEZ_USER:?MARQUEZ_USER is required}"
: "${MARQUEZ_PASSWORD:?MARQUEZ_PASSWORD is required}"
: "${MARQUEZ_DB:?MARQUEZ_DB is required}"

psql --no-psqlrc --set=ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" \
  --set=marquez_user="$MARQUEZ_USER" \
  --set=marquez_password="$MARQUEZ_PASSWORD" \
  --set=marquez_db="$MARQUEZ_DB" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'marquez_user', :'marquez_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'marquez_user') \gexec
SELECT format('ALTER ROLE %I LOGIN PASSWORD %L', :'marquez_user', :'marquez_password') \gexec
SELECT format('CREATE DATABASE %I OWNER %I', :'marquez_db', :'marquez_user')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'marquez_db') \gexec
SQL
echo "Marquez database is ready."

