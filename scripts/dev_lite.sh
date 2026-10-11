#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"

runtime=${M1_CONTAINER_CLI:-docker}
runtime_env=${M1_RUNTIME_ENV_FILE:-.env.m1.local}
compose_file=compose.lite.yaml

fail() {
  echo "M1 Lite: $*" >&2
  exit 3
}

validate_runtime_env_path() {
  [[ "$runtime_env" != /* ]] || fail "runtime environment path must stay inside the repository"
  case "/$runtime_env/" in
    */../*) fail "runtime environment path must stay inside the repository" ;;
  esac
}

require_runtime() {
  command -v "$runtime" >/dev/null 2>&1 || fail "container runtime '$runtime' is required"
  "$runtime" compose version >/dev/null 2>&1 || fail "container runtime must provide the Compose plugin"
}

generate_runtime_env() {
  validate_runtime_env_path
  if [[ -e "$runtime_env" ]]; then
    [[ -f "$runtime_env" && ! -L "$runtime_env" ]] || fail "runtime environment path is unsafe"
  fi
  umask 077
  python3 - "$runtime_env" <<'PY'
from __future__ import annotations

import secrets
import sys
from pathlib import Path
from urllib.parse import quote

target = Path(sys.argv[1])
if target.is_absolute() or ".." in target.parts:
    raise SystemExit("runtime environment path must stay inside the repository")

def value(size: int = 32) -> str:
    return secrets.token_urlsafe(size)

if target.exists():
    raw_lines = target.read_text(encoding="utf-8").splitlines()
    lines = dict(
        line.split("=", 1)
        for line in raw_lines
        if line and not line.startswith("#") and "=" in line
    )
else:
    postgres_password = value()
    lines = {
        "M1_COMPOSE_PROJECT_NAME": "industrial-ops-m1",
        "M1_WEB_PORT": "3000",
        "M1_KEYCLOAK_PORT": "8080",
        "M1_MINIO_CONSOLE_PORT": "9001",
        "POSTGRES_DB": "industrial_ops",
        "POSTGRES_USER": "m1_db_admin",
        "POSTGRES_PASSWORD": postgres_password,
        "MINIO_ROOT_USER": "m1_minio",
        "MINIO_ROOT_PASSWORD": value(),
        "KEYCLOAK_ADMIN": "m1_admin",
        "KEYCLOAK_ADMIN_PASSWORD": value(),
        "M1_DEMO_USER_PASSWORD": value(18),
        "VAULT_DEV_ROOT_TOKEN_ID": value(),
        "IOAP_OIDC_CLIENT_SECRET": value(),
        "IOAP_NEXTAUTH_SECRET": value(48),
        "IOAP_ENTERPRISE_MCP_CLIENT_SECRET": value(),
        "IOAP_SUPPLIER_A2A_CLIENT_SECRET": value(),
    }

required = ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")
if any(not lines.get(key) for key in required):
    raise SystemExit("existing runtime environment lacks PostgreSQL bootstrap values")

lines.setdefault("IOAP_SUPPLIER_A2A_CLIENT_SECRET", value())
for key in (
    "IOAP_ENTERPRISE_TOOL_GATEWAY_TOKEN",
    "IOAP_NOTIFICATION_PROVIDER_API_TOKEN",
    "IOAP_NOTIFICATION_WEBHOOK_SECRET",
    "IOAP_PROCUREMENT_PROVIDER_API_TOKEN",
    "IOAP_PROCUREMENT_WEBHOOK_SECRET",
    "IOAP_REFUND_PROVIDER_API_TOKEN",
    "IOAP_REFUND_WEBHOOK_SECRET",
    "IOAP_FSM_ASSIGNMENT_PROVIDER_API_TOKEN",
    "IOAP_FSM_ASSIGNMENT_WEBHOOK_SECRET",
    "IOAP_SERVICE_QUOTATION_PROVIDER_API_TOKEN",
    "IOAP_SANDBOX_CONTROL_TOKEN",
    "IOAP_MODEL_GATEWAY_API_KEY",
):
    lines.setdefault(key, value())

runtime_user = lines.setdefault("M1_POSTGRES_RUNTIME_USER", "m1_runtime")
runtime_password = lines.setdefault("M1_POSTGRES_RUNTIME_PASSWORD", value())
database = quote(lines["POSTGRES_DB"], safe="")
admin_user = quote(lines["POSTGRES_USER"], safe="")
admin_password = quote(lines["POSTGRES_PASSWORD"], safe="")
app_user = quote(runtime_user, safe="")
app_password = quote(runtime_password, safe="")
lines["IOAP_MIGRATION_DATABASE_URL"] = (
    f"postgresql+psycopg://{admin_user}:{admin_password}@postgres:5432/{database}"
)
lines["IOAP_DATABASE_URL"] = (
    f"postgresql+psycopg://{app_user}:{app_password}@postgres:5432/{database}"
)

target.write_text(
    "".join(f"{key}={item}\n" for key, item in lines.items()),
    encoding="utf-8",
)
target.chmod(0o600)
PY
  echo "M1 runtime environment is current: $runtime_env"
}

compose() {
  M1_RUNTIME_ENV_FILE="$runtime_env" \
    "$runtime" compose --env-file "$runtime_env" -f "$compose_file" "$@"
}

ensure_vault_runtime() {
  compose up --detach --wait vault-runtime-manager
}

sandbox_cli() {
  compose exec -T enterprise-sandbox python -m industrial_ops_agent.sandbox.cli "$@"
}

supplier_a2a_cli() {
  compose exec -T supplier-agent python -m industrial_ops_agent.supplier_sandbox.cli "$@"
}

require_sandbox_runtime() {
  require_runtime
  validate_runtime_env_path
  [[ -f "$runtime_env" ]] || fail "run init first"
}

verify_enterprise_sandbox() {
  compose exec -T enterprise-sandbox python - <<'PY'
from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

from industrial_ops_agent.sandbox.app import SandboxSettings
from industrial_ops_agent.secrets import SecretSource, SecretValue
from industrial_ops_agent.tools.contracts import ReservationOutcomeUnknown
from industrial_ops_agent.tools.customer_notifications import HttpNotificationDeliveryAdapter
from industrial_ops_agent.tools.enterprise_http import EnterpriseHttpAdapter
from industrial_ops_agent.tools.fsm_assignments import HttpFsmAssignmentAdapter
from industrial_ops_agent.tools.procurement import HttpProcurementDeliveryAdapter
from industrial_ops_agent.tools.quotations import HttpServiceQuotationCatalogAdapter
from industrial_ops_agent.tools.refunds import HttpRefundDeliveryAdapter

settings = SandboxSettings.from_environment()
base_url = os.environ.get("IOAP_SANDBOX_BASE_URL", "http://127.0.0.1:8090")
secret = lambda value: SecretValue(value, source=SecretSource.ENVIRONMENT)

def control(method: str, path: str, payload: dict[str, object] | None = None) -> object:
    raw = None if payload is None else json.dumps(payload).encode()
    request = Request(
        f"{base_url}{path}",
        data=raw,
        headers={
            "Authorization": f"Bearer {settings.control_token}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read())

enterprise = EnterpriseHttpAdapter(
    base_url, secret(settings.enterprise_token), allow_plain_http=True
)
assert enterprise.invoke("asset.get", {"asset_id": "asset-m1-pump"})["model_code"] == "PUMP-X100"
control("PUT", "/sandbox/v1/faults/wms", {"mode": "OUTCOME_UNKNOWN", "remaining": 1})
try:
    enterprise.reserve(
        operation_id="compose-sandbox-unknown", part_number="BRG-6312-C3", quantity=1
    )
except ReservationOutcomeUnknown:
    pass
else:
    raise AssertionError("WMS unknown outcome was not surfaced")
reservation = enterprise.reconcile("compose-sandbox-unknown")
assert reservation is not None
assert enterprise.reserve(
    operation_id="compose-sandbox-unknown", part_number="BRG-6312-C3", quantity=1
) == reservation

fsm = HttpFsmAssignmentAdapter(
    base_url,
    provider_id="enterprise-fsm",
    api_token=secret(settings.fsm_token),
    webhook_secret=secret(settings.fsm_webhook_secret),
    allow_plain_http=True,
)
assert fsm.list_candidates(
    tenant_id="tenant-m1-demo",
    work_order_id="compose-sandbox",
    asset_id="asset-m1-pump",
    site_id="site-m1-demo",
    service_window_start=None,
    service_window_end=None,
)
procurement = HttpProcurementDeliveryAdapter(
    base_url,
    provider_id="enterprise-erp",
    api_token=secret(settings.procurement_token),
    webhook_secret=secret(settings.procurement_webhook_secret),
    allow_plain_http=True,
)
assert procurement.resolve_profile(tenant_id="tenant-m1-demo") is not None
refund = HttpRefundDeliveryAdapter(
    base_url,
    provider_id="enterprise-finance",
    api_token=secret(settings.refund_token),
    webhook_secret=secret(settings.refund_webhook_secret),
    allow_plain_http=True,
)
assert refund.resolve_profile(
    tenant_id="tenant-m1-demo", customer_subject_id="customer-m1-demo"
) is not None
quotation = HttpServiceQuotationCatalogAdapter(
    base_url,
    provider_id="enterprise-cpq",
    api_token=secret(settings.quotation_token),
    allow_plain_http=True,
)
assert quotation.list_candidates(
    tenant_id="tenant-m1-demo",
    incident_ref="compose-sandbox",
    asset_ref="asset-m1-pump",
    diagnosis_ref="compose-sandbox",
)
notification = HttpNotificationDeliveryAdapter(
    base_url,
    provider_id="enterprise-notify",
    api_token=secret(settings.notification_token),
    webhook_secret=secret(settings.notification_webhook_secret),
    allow_plain_http=True,
)
assert notification.resolve_contacts(
    tenant_id="tenant-m1-demo", recipient_subject_id="customer-m1-demo"
)
print("ENTERPRISE_SANDBOX_ADAPTERS_OK")
PY
}

verify_postgres_rls() {
  compose run --rm --no-deps -T api python - <<'PY'
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from industrial_ops_agent.config import get_settings
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import TENANT_TABLE_NAMES
from industrial_ops_agent.secrets import SecretName, build_secret_provider

tenant_a = "tenant-m1-demo"
tenant_b = "tenant-m1-rls-probe"
asset_a = "asset-m1-pump"
asset_b = "asset-m1-rls-probe"
database_url = build_secret_provider(get_settings()).get(SecretName.DATABASE_URL)
database = Database(database_url.reveal())

try:
    with database.engine.begin() as connection:
        role = connection.execute(
            text(
                "SELECT current_user, rolsuper, rolbypassrls "
                "FROM pg_roles WHERE rolname = current_user"
            )
        ).one()
        assert role.rolsuper is False, "application database role is a superuser"
        assert role.rolbypassrls is False, "application database role can bypass RLS"

        table_rows = {
            row.table_name: row
            for row in connection.execute(
                text(
                    "SELECT c.relname AS table_name, "
                    "c.relrowsecurity AS rls_enabled, "
                    "c.relforcerowsecurity AS rls_forced "
                    "FROM pg_class AS c "
                    "JOIN pg_namespace AS n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = current_schema() "
                    "AND c.relkind IN ('r', 'p')"
                )
            ).mappings()
        }
        policy_rows = {
            (row.table_name, row.policy_name): row
            for row in connection.execute(
                text(
                    "SELECT tablename AS table_name, policyname AS policy_name, "
                    "qual, with_check "
                    "FROM pg_policies WHERE schemaname = current_schema()"
                )
            ).mappings()
        }
        for table_name in sorted(TENANT_TABLE_NAMES):
            table = table_rows.get(table_name)
            assert table is not None, f"tenant table is missing: {table_name}"
            assert table.rls_enabled is True, f"RLS is disabled: {table_name}"
            assert table.rls_forced is True, f"RLS is not forced: {table_name}"
            policy_name = f"{table_name}_tenant_isolation"[:63]
            policy = policy_rows.get((table_name, policy_name))
            assert policy is not None, f"tenant policy is missing: {policy_name}"
            assert "app.tenant_id" in (policy.qual or ""), (
                f"tenant policy USING clause is invalid: {policy_name}"
            )
            assert "app.tenant_id" in (policy.with_check or ""), (
                f"tenant policy WITH CHECK clause is invalid: {policy_name}"
            )

        connection.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": tenant_b})
        connection.execute(
            text(
                "INSERT INTO tenants (id, status) VALUES (:tenant, 'active') "
                "ON CONFLICT (id) DO UPDATE SET status = EXCLUDED.status"
            ),
            {"tenant": tenant_b},
        )
        connection.execute(text("DELETE FROM assets WHERE asset_id = :asset"), {"asset": asset_b})
        connection.execute(
            text(
                "INSERT INTO assets "
                "(asset_id, tenant_id, source_system, source_record_id, version) "
                "VALUES (:asset, :tenant, 'rls-probe', 'rls-probe', 1)"
            ),
            {"asset": asset_b, "tenant": tenant_b},
        )

    with database.engine.begin() as connection:
        connection.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": tenant_a})
        assert connection.scalar(
            text("SELECT count(*) FROM assets WHERE asset_id = :asset"), {"asset": asset_a}
        ) == 1, "same-tenant asset is unavailable"
        assert connection.scalar(
            text("SELECT count(*) FROM assets WHERE asset_id = :asset"), {"asset": asset_b}
        ) == 0, "cross-tenant asset is visible"
        assert connection.execute(
            text("UPDATE assets SET version = version + 1 WHERE asset_id = :asset"),
            {"asset": asset_b},
        ).rowcount == 0, "cross-tenant asset is mutable"
        savepoint = connection.begin_nested()
        try:
            connection.execute(
                text(
                    "INSERT INTO assets "
                    "(asset_id, tenant_id, source_system, source_record_id, version) "
                    "VALUES ('asset-m1-cross-tenant-write', :tenant, 'rls-probe', 'cross-write', 1)"
                ),
                {"tenant": tenant_b},
            )
        except DBAPIError:
            savepoint.rollback()
        else:
            savepoint.rollback()
            raise AssertionError("cross-tenant insert bypassed RLS")

    with database.engine.begin() as connection:
        connection.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": tenant_b})
        connection.execute(text("DELETE FROM assets WHERE asset_id = :asset"), {"asset": asset_b})
        connection.execute(text("DELETE FROM tenants WHERE id = :tenant"), {"tenant": tenant_b})
finally:
    database.dispose()

print("M1_POSTGRES_RLS_OK")
PY
}

case "${1:-}" in
  doctor)
    require_runtime
    command -v python3 >/dev/null 2>&1 || fail "python3 is required"
    command -v curl >/dev/null 2>&1 || fail "curl is required"
    echo "M1 Lite prerequisites are available."
    ;;
  init)
    generate_runtime_env
    ;;
  up)
    generate_runtime_env
    M1_CONTAINER_CLI="$runtime" M1_RUNTIME_ENV_FILE="$runtime_env" \
      "$root/scripts/real_model_business_loop.sh" up
    ;;
  verify)
    generate_runtime_env
    M1_CONTAINER_CLI="$runtime" M1_RUNTIME_ENV_FILE="$runtime_env" \
      "$root/scripts/real_model_business_loop.sh" verify
    ;;
  verify-rls)
    require_runtime
    generate_runtime_env
    ensure_vault_runtime
    compose up --build --detach --wait seed
    verify_postgres_rls
    ;;
  sandbox-status)
    require_sandbox_runtime
    sandbox_cli status
    ;;
  sandbox-reset)
    require_sandbox_runtime
    sandbox_cli reset
    ;;
  sandbox-fault)
    require_sandbox_runtime
    [[ $# -eq 4 ]] || fail "usage: sandbox-fault <provider> <mode> <remaining>"
    sandbox_cli fault "$2" "$3" "$4"
    ;;
  sandbox-clear-fault)
    require_sandbox_runtime
    [[ $# -eq 2 ]] || fail "usage: sandbox-clear-fault <provider>"
    sandbox_cli clear-fault "$2"
    ;;
  sandbox-complete)
    require_sandbox_runtime
    [[ $# -ge 4 ]] || fail "usage: sandbox-complete <provider> <operation-id> <status> [options]"
    sandbox_cli complete "${@:2}"
    ;;
  sandbox-fulfill)
    require_sandbox_runtime
    [[ $# -ge 4 ]] || fail "usage: sandbox-fulfill <operation-id> <status> <quantity> [options]"
    sandbox_cli fulfill "${@:2}"
    ;;
  sandbox-verify)
    require_runtime
    generate_runtime_env
    compose up --build --detach --wait keycloak vault enterprise-sandbox
    compose run --rm --no-deps vault-bootstrap
    curl --fail --silent --show-error "http://127.0.0.1:${M1_KEYCLOAK_PORT:-8080}/realms/industrial-ops/.well-known/openid-configuration" >/dev/null
    compose exec -T vault sh -ec '
      export VAULT_TOKEN="$VAULT_DEV_ROOT_TOKEN_ID"
      for key in enterprise_tool_gateway_token notification_provider_api_token notification_webhook_secret procurement_provider_api_token procurement_webhook_secret refund_provider_api_token refund_webhook_secret fsm_assignment_provider_api_token fsm_assignment_webhook_secret service_quotation_provider_api_token; do
        vault kv get -mount=secret -field="$key" industrial-ops/m1 >/dev/null
      done
    '
    evidence_dir=artifacts/enterprise-integration-sandbox
    mkdir -p "$evidence_dir"
    temporary_report=$(mktemp "$evidence_dir/.acceptance.XXXXXX")
    sandbox_cli acceptance >"$temporary_report"
    python3 - "$temporary_report" <<'PY'
from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
chain = report.pop("evidence_chain_sha256", None)
observed = sha256(
    json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
assert report["status"] == "ENTERPRISE_SANDBOX_CLOSED_LOOP_PASSED"
assert report["classification"] == "SIMULATED_NON_PRODUCTION"
assert report["production_claim"] is False
assert len(report["cases"]) == 8
assert chain == observed
PY
    mv "$temporary_report" "$evidence_dir/acceptance.json"
    echo "ENTERPRISE_SANDBOX_CLOSED_LOOP_OK"
    echo "SIMULATED_NON_PRODUCTION"
    echo "$evidence_dir/acceptance.json"
    ;;
  supplier-a2a-status)
    require_sandbox_runtime
    supplier_a2a_cli status
    ;;
  supplier-a2a-reset)
    require_sandbox_runtime
    supplier_a2a_cli reset
    ;;
  supplier-a2a-fault)
    require_sandbox_runtime
    [[ $# -eq 3 ]] || fail "usage: supplier-a2a-fault <mode> <remaining>"
    supplier_a2a_cli fault "$2" "$3"
    ;;
  supplier-a2a-clear-fault)
    require_sandbox_runtime
    supplier_a2a_cli clear-fault
    ;;
  supplier-a2a-verify)
    require_runtime
    generate_runtime_env
    compose up --build --detach --wait keycloak supplier-agent
    supplier_a2a_cli status
    echo "SUPPLIER_A2A_COMPOSE_READY"
    ;;
  supplier-a2a-lab)
    [[ -x .venv/bin/python ]] || fail "run uv sync before supplier-a2a-lab"
    PYTHONPATH=src .venv/bin/python -B \
      -m industrial_ops_agent.simulation.supplier_a2a_lab_cli run \
      --repo-root "$root"
    ;;
  supplier-a2a-lab-verify)
    [[ -x .venv/bin/python ]] || fail "run uv sync before supplier-a2a-lab-verify"
    PYTHONPATH=src .venv/bin/python -B \
      -m industrial_ops_agent.simulation.supplier_a2a_lab_cli verify \
      --repo-root "$root"
    ;;
  candidate-rollout)
    [[ -x .venv/bin/python ]] || fail "run uv sync before candidate-rollout"
    PYTHONPATH=src .venv/bin/python -B \
      -m industrial_ops_agent.simulation.enterprise_candidate_rollout_cli run \
      --repo-root "$root"
    ;;
  simulated-closure)
    "$0" sandbox-verify
    [[ -x .venv/bin/python ]] || fail "run uv sync before simulated-closure"
    "$0" project-assurance
    "$0" supplier-a2a-lab
    PYTHONPATH=src .venv/bin/python -B \
      -m industrial_ops_agent.simulation.vlm_enterprise_staging_cli adopt \
      --repo-root "$root"
    "$0" candidate-rollout
    PYTHONPATH=src .venv/bin/python -B \
      -m industrial_ops_agent.simulation.closure_cli \
      --repo-root "$root"
    "$0" enterprise-adoption
    ;;
  project-assurance)
    [[ -x .venv/bin/python ]] || fail "run uv sync before project-assurance"
    PYTHONPATH=src .venv/bin/python -B \
      -m industrial_ops_agent.simulation.project_assurance_lab_cli run \
      --repo-root "$root"
    ;;
  project-assurance-verify)
    [[ -x .venv/bin/python ]] || fail "run uv sync before project-assurance-verify"
    PYTHONPATH=src .venv/bin/python -B \
      -m industrial_ops_agent.simulation.project_assurance_lab_cli verify \
      --repo-root "$root"
    ;;
  enterprise-adoption)
    [[ -x .venv/bin/python ]] || fail "run uv sync before enterprise-adoption"
    PYTHONPATH=src .venv/bin/python -B \
      -m industrial_ops_agent.simulation.enterprise_project_adoption_cli adopt \
      --repo-root "$root"
    ;;
  status)
    require_runtime
    validate_runtime_env_path
    [[ -f "$runtime_env" ]] || fail "run init first"
    compose ps
    ;;
  logs)
    require_runtime
    validate_runtime_env_path
    [[ -f "$runtime_env" ]] || fail "run init first"
    compose logs --tail=200 "${@:2}"
    ;;
  down)
    validate_runtime_env_path
    [[ -f "$runtime_env" ]] || fail "run init first"
    M1_CONTAINER_CLI="$runtime" M1_RUNTIME_ENV_FILE="$runtime_env" \
      "$root/scripts/real_model_business_loop.sh" down
    ;;
  clean)
    require_runtime
    validate_runtime_env_path
    [[ -f "$runtime_env" ]] || fail "run init first"
    compose down --volumes --remove-orphans
    ;;
  *)
    echo "usage: scripts/dev_lite.sh <doctor|init|up|verify|verify-rls|sandbox-status|sandbox-reset|sandbox-fault|sandbox-clear-fault|sandbox-complete|sandbox-fulfill|sandbox-verify|supplier-a2a-status|supplier-a2a-reset|supplier-a2a-fault|supplier-a2a-clear-fault|supplier-a2a-verify|supplier-a2a-lab|supplier-a2a-lab-verify|candidate-rollout|simulated-closure|project-assurance|project-assurance-verify|enterprise-adoption|status|logs|down|clean>" >&2
    exit 2
    ;;
esac
