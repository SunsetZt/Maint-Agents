"""Low-resource closed-loop acceptance for the project supplier A2A peer."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from secrets import token_urlsafe
from typing import Any, Literal

import httpx
import uvicorn
from pydantic import BaseModel, ConfigDict, Field

from industrial_ops_agent.collaboration.a2a import (
    A2AClientCredentialsTokenProvider,
    A2ATaskOutcome,
    OfficialA2ASupplierAgentClient,
    SupplierA2AError,
)
from industrial_ops_agent.secrets import SecretSource, SecretValue
from industrial_ops_agent.supplier_sandbox.app import (
    LocalOAuthFixture,
    SupplierSandboxSettings,
    create_app,
)
from industrial_ops_agent.supplier_sandbox.auth import (
    StaticSupplierSandboxTokenVerifier,
)
from industrial_ops_agent.supplier_sandbox.state import SupplierSandboxState

SCHEMA_VERSION: Literal["supplier-a2a-sandbox-acceptance/v1"] = (
    "supplier-a2a-sandbox-acceptance/v1"
)
CLASSIFICATION: Literal["SIMULATED_NON_PRODUCTION"] = "SIMULATED_NON_PRODUCTION"
STATUS: Literal["SUPPLIER_A2A_SANDBOX_CLOSED_LOOP_PASSED"] = (
    "SUPPLIER_A2A_SANDBOX_CLOSED_LOOP_PASSED"
)
OUTPUT_RELATIVE = Path("artifacts/supplier-a2a-sandbox/acceptance.json")

_IMPLEMENTATION_PATHS = (
    "src/industrial_ops_agent/supplier_sandbox/app.py",
    "src/industrial_ops_agent/supplier_sandbox/auth.py",
    "src/industrial_ops_agent/supplier_sandbox/state.py",
    "src/industrial_ops_agent/supplier_sandbox/cli.py",
    "src/industrial_ops_agent/collaboration/a2a.py",
    "src/industrial_ops_agent/collaboration/service.py",
    "src/industrial_ops_agent/api/routes/collaborations.py",
    "web/app/collaboration/page.tsx",
    "infra/keycloak/m1-realm.json",
    "compose.lite.yaml",
    "scripts/dev_lite.sh",
)


class SupplierA2AAcceptanceError(RuntimeError):
    """The supplier A2A receipt is missing, stale, or violates a hard gate."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImplementationEvidence(_ClosedModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProtocolEvidence(_ClosedModel):
    official_a2a_sdk: Literal[True] = True
    protocol_binding: Literal["JSONRPC"] = "JSONRPC"
    protocol_version: Literal["1.0"] = "1.0"
    oauth_flow: Literal["client_credentials"] = "client_credentials"
    oauth_scope: Literal["supplier.diagnosis.review"] = "supplier.diagnosis.review"
    request_contract: Literal["supplier-diagnosis-request/v1"] = (
        "supplier-diagnosis-request/v1"
    )
    response_contract: Literal["supplier-diagnosis-advice/v1"] = (
        "supplier-diagnosis-advice/v1"
    )
    advisory_only: Literal[True] = True
    independent_human_review_required: Literal[True] = True


class ExecutionEvidence(_ClosedModel):
    normal_task_completed: Literal[True] = True
    oauth_service_identity_accepted: Literal[True] = True
    agent_card_identity_pinned: Literal[True] = True
    outcome_unknown_replayed_idempotently: Literal[True] = True
    malicious_artifact_blocked: Literal[True] = True
    malformed_artifact_blocked: Literal[True] = True
    raw_sensitive_request_rejected: Literal[True] = True
    wrong_tenant_rejected: Literal[True] = True
    missing_bearer_rejected: Literal[True] = True
    oversized_request_rejected: Literal[True] = True
    remote_cancellation_completed: Literal[True] = True
    provider_record_counts: dict[str, int]
    business_side_effects: dict[str, int]


class SupplierA2ASandboxAcceptanceReport(_ClosedModel):
    schema_version: Literal["supplier-a2a-sandbox-acceptance/v1"] = SCHEMA_VERSION
    classification: Literal["SIMULATED_NON_PRODUCTION"] = CLASSIFICATION
    production_claim: Literal[False] = False
    enterprise_production_data: Literal[False] = False
    generated_at: datetime
    status: Literal["SUPPLIER_A2A_SANDBOX_CLOSED_LOOP_PASSED"] = STATUS
    ready_for_project_enterprise_demo: Literal[True] = True
    ready_for_external_enterprise_production: Literal[False] = False
    protocol: ProtocolEvidence
    execution: ExecutionEvidence
    implementation: tuple[ImplementationEvidence, ...]
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


async def run_supplier_a2a_lab(repo_root: Path) -> SupplierA2ASandboxAcceptanceReport:
    """Exercise the official client and provider without Docker or external services."""

    root = repo_root.resolve(strict=True)
    state = SupplierSandboxState()
    access_token = token_urlsafe(32)
    client_secret = token_urlsafe(32)
    control_token = token_urlsafe(32)
    async with _running_supplier(
        state,
        access_token=access_token,
        client_secret=client_secret,
        control_token=control_token,
    ) as base_url:
        client = _client(base_url, client_secret=client_secret)

        normal = await _completed(client, _request("acceptance-normal"))
        if normal.advice is None or normal.guardrail_policy_version is None:
            raise SupplierA2AAcceptanceError("supplier_normal_advice_is_incomplete")

        state.configure_fault("OUTCOME_UNKNOWN", 1)
        replay_payload = _request("acceptance-outcome-unknown")
        await _expect_error(client.submit(replay_payload), "supplier_a2a_unavailable")
        replayed = await _completed(client, replay_payload)
        if replayed.status != "COMPLETED":
            raise SupplierA2AAcceptanceError("supplier_idempotent_replay_failed")

        state.configure_fault("MALICIOUS_ARTIFACT", 1)
        malicious = await client.submit(_request("acceptance-malicious"))
        await _expect_terminal_error(
            client,
            malicious.remote_task_id,
            "supplier_a2a_prompt_injection_blocked",
        )

        state.configure_fault("MALFORMED_ARTIFACT", 1)
        malformed = await client.submit(_request("acceptance-malformed"))
        await _expect_terminal_error(
            client,
            malformed.remote_task_id,
            "supplier_a2a_advice_contract_invalid",
        )

        sensitive_payload = _request("acceptance-sensitive")
        sensitive_payload["incident"]["description"] = "Call 13912345678 immediately"
        sensitive = await client.submit(sensitive_payload)
        rejected = await _terminal(client, sensitive.remote_task_id)
        if rejected.status != "REJECTED":
            raise SupplierA2AAcceptanceError("supplier_raw_sensitive_request_was_not_rejected")

        wrong_tenant = _client(
            base_url,
            client_secret=client_secret,
            remote_tenant="unauthorized-tenant",
        )
        await _expect_error(
            wrong_tenant.submit(_request("acceptance-wrong-tenant")),
            "supplier_a2a_remote_request_rejected",
        )

        state.configure_fault("WORKING", 1)
        working = await client.submit(_request("acceptance-cancel"))
        canceled = await client.cancel(working.remote_task_id)
        await asyncio.sleep(0.3)
        canceled_after_wait = await client.refresh(working.remote_task_id)
        if canceled.status != "CANCELED" or canceled_after_wait.status != "CANCELED":
            raise SupplierA2AAcceptanceError("supplier_remote_cancellation_failed")

        async with httpx.AsyncClient(base_url=base_url) as http_client:
            missing_bearer = await http_client.post("/a2a", json={})
            oversized = await http_client.post(
                "/a2a",
                content=b"x" * (128 * 1024 + 1),
                headers={"Authorization": f"Bearer {access_token}"},
            )
            control = await http_client.get(
                "/supplier-sandbox/v1/state",
                headers={"Authorization": f"Bearer {control_token}"},
            )
        if missing_bearer.status_code != 401 or oversized.status_code != 413:
            raise SupplierA2AAcceptanceError("supplier_transport_guard_failed")
        if control.status_code != 200:
            raise SupplierA2AAcceptanceError("supplier_control_snapshot_unavailable")
        snapshot = control.json()

    counts = _integer_mapping(snapshot.get("record_counts"), "supplier_record_counts_invalid")
    side_effects = _integer_mapping(
        snapshot.get("business_side_effects"),
        "supplier_side_effect_snapshot_invalid",
    )
    if counts.get("idempotent_replays", 0) < 1 or counts.get("unique_tasks", 0) < 6:
        raise SupplierA2AAcceptanceError("supplier_idempotency_evidence_is_incomplete")
    if side_effects != _zero_side_effects():
        raise SupplierA2AAcceptanceError("supplier_business_side_effect_boundary_failed")

    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "production_claim": False,
        "enterprise_production_data": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": STATUS,
        "ready_for_project_enterprise_demo": True,
        "ready_for_external_enterprise_production": False,
        "protocol": ProtocolEvidence().model_dump(mode="json"),
        "execution": {
            "normal_task_completed": True,
            "oauth_service_identity_accepted": True,
            "agent_card_identity_pinned": True,
            "outcome_unknown_replayed_idempotently": True,
            "malicious_artifact_blocked": True,
            "malformed_artifact_blocked": True,
            "raw_sensitive_request_rejected": True,
            "wrong_tenant_rejected": True,
            "missing_bearer_rejected": True,
            "oversized_request_rejected": True,
            "remote_cancellation_completed": True,
            "provider_record_counts": counts,
            "business_side_effects": side_effects,
        },
        "implementation": [
            {
                "path": relative,
                "sha256": sha256(_inside_file(root, Path(relative)).read_bytes()).hexdigest(),
            }
            for relative in _IMPLEMENTATION_PATHS
        ],
    }
    draft = SupplierA2ASandboxAcceptanceReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    normalized = draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    return draft.model_copy(update={"evidence_chain_sha256": _digest(normalized)})


def write_supplier_a2a_lab(
    report: SupplierA2ASandboxAcceptanceReport,
    output_path: Path,
) -> Path:
    target = output_path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_bytes(_canonical(report.model_dump(mode="json")) + b"\n")
    temporary.replace(target)
    return target


def verify_supplier_a2a_lab(
    repo_root: Path,
    acceptance_path: Path = OUTPUT_RELATIVE,
) -> SupplierA2ASandboxAcceptanceReport:
    root = repo_root.resolve(strict=True)
    path = _inside_file(root, acceptance_path)
    try:
        report = SupplierA2ASandboxAcceptanceReport.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise SupplierA2AAcceptanceError("supplier_a2a_acceptance_is_invalid") from exc
    normalized = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(normalized):
        raise SupplierA2AAcceptanceError("supplier_a2a_acceptance_digest_mismatch")
    if (
        report.execution.provider_record_counts.get("idempotent_replays", 0) < 1
        or report.execution.provider_record_counts.get("unique_tasks", 0) < 6
        or report.execution.business_side_effects != _zero_side_effects()
    ):
        raise SupplierA2AAcceptanceError("supplier_a2a_acceptance_hard_gate_failed")
    observed_paths = tuple(item.path for item in report.implementation)
    if observed_paths != _IMPLEMENTATION_PATHS or len(set(observed_paths)) != len(observed_paths):
        raise SupplierA2AAcceptanceError("supplier_a2a_implementation_manifest_changed")
    for item in report.implementation:
        if sha256(_inside_file(root, Path(item.path)).read_bytes()).hexdigest() != item.sha256:
            raise SupplierA2AAcceptanceError(
                f"supplier_a2a_implementation_digest_changed:{item.path}"
            )
    return report


async def _completed(
    client: OfficialA2ASupplierAgentClient,
    payload: dict[str, Any],
) -> A2ATaskOutcome:
    outcome = await client.submit(payload)
    for _ in range(100):
        if outcome.status == "COMPLETED":
            return outcome
        if outcome.status not in {"SUBMITTED", "WORKING"}:
            raise SupplierA2AAcceptanceError("supplier_task_entered_unexpected_state")
        await asyncio.sleep(0.01)
        outcome = await client.refresh(outcome.remote_task_id)
    raise SupplierA2AAcceptanceError("supplier_task_did_not_complete")


async def _terminal(
    client: OfficialA2ASupplierAgentClient,
    remote_task_id: str,
) -> A2ATaskOutcome:
    for _ in range(100):
        outcome = await client.refresh(remote_task_id)
        if outcome.status in {"COMPLETED", "FAILED", "CANCELED", "REJECTED"}:
            return outcome
        await asyncio.sleep(0.01)
    raise SupplierA2AAcceptanceError("supplier_task_did_not_reach_terminal_state")


async def _expect_terminal_error(
    client: OfficialA2ASupplierAgentClient,
    remote_task_id: str,
    expected_reason: str,
) -> None:
    for _ in range(100):
        try:
            outcome = await client.refresh(remote_task_id)
        except SupplierA2AError as exc:
            if exc.reason != expected_reason:
                raise SupplierA2AAcceptanceError(
                    f"supplier_unexpected_error:{exc.reason}"
                ) from exc
            return
        if outcome.status in {"FAILED", "CANCELED", "REJECTED"}:
            raise SupplierA2AAcceptanceError("supplier_guard_did_not_reject_artifact")
        await asyncio.sleep(0.01)
    raise SupplierA2AAcceptanceError("supplier_guard_result_was_not_observed")


async def _expect_error(awaitable: Any, expected_reason: str) -> None:
    try:
        await awaitable
    except SupplierA2AError as exc:
        if exc.reason != expected_reason:
            raise SupplierA2AAcceptanceError(
                f"supplier_unexpected_error:{exc.reason}"
            ) from exc
        return
    raise SupplierA2AAcceptanceError("supplier_expected_error_was_not_raised")


def _client(
    base_url: str,
    *,
    client_secret: str,
    remote_tenant: str = "industrial-ops",
) -> OfficialA2ASupplierAgentClient:
    provider = A2AClientCredentialsTokenProvider(
        token_url=f"{base_url}/oauth/token",
        client_id="industrial-ops-a2a-client",
        client_secret=SecretValue(client_secret, SecretSource.ENVIRONMENT),
        resource=base_url,
        scope="supplier.diagnosis.review",
        timeout_seconds=2.0,
        allow_plain_http=True,
    )
    return OfficialA2ASupplierAgentClient(
        base_url=base_url,
        agent_name="industrial-supplier-diagnosis-agent",
        agent_version="1.0.0",
        skill_id="industrial.vendor.diagnosis-review",
        remote_tenant=remote_tenant,
        timeout_seconds=2.0,
        token_provider=provider,
        allow_plain_http=True,
    )


def _request(collaboration_id: str) -> dict[str, Any]:
    digest = f"sha256:{sha256(collaboration_id.encode()).hexdigest()}"
    return {
        "contract_version": "supplier-diagnosis-request/v1",
        "collaboration_id": collaboration_id,
        "skill_id": "industrial.vendor.diagnosis-review",
        "asset": {"model_code": "PUMP-X100"},
        "incident": {
            "severity": "P2",
            "category": "MECHANICAL",
            "description": "Redacted drive-end vibration anomaly",
        },
        "question": "Review the diagnosis and provide safe advisory guidance.",
        "diagnosis": {
            "status": "COMPLETED",
            "conclusion": "Possible drive-end bearing wear",
            "contradictions": [],
            "missing_information": ["supplier service bulletin"],
            "next_checks": ["lockout/tagout before bearing inspection"],
            "confidence": 0.62,
            "manifest_digest": digest,
            "report_digest": digest,
        },
        "constraints": {
            "advisory_only": True,
            "may_execute_tools": False,
            "may_change_work_order": False,
            "may_control_equipment": False,
        },
    }


@asynccontextmanager
async def _running_supplier(
    state: SupplierSandboxState,
    *,
    access_token: str,
    client_secret: str,
    control_token: str,
) -> AsyncIterator[str]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = int(listener.getsockname()[1])
    base_url = f"http://127.0.0.1:{port}"
    settings = SupplierSandboxSettings(
        enabled=True,
        environment="test",
        base_url=base_url,
        agent_name="industrial-supplier-diagnosis-agent",
        agent_version="1.0.0",
        skill_id="industrial.vendor.diagnosis-review",
        remote_tenant="industrial-ops",
        oauth_token_url=f"{base_url}/oauth/token",
        oidc_issuer="http://localhost:8080/realms/industrial-ops",
        oidc_jwks_url="http://localhost:8080/realms/industrial-ops/certs",
        oidc_audience="supplier-agent",
        oauth_client_id="industrial-ops-a2a-client",
        oauth_scope="supplier.diagnosis.review",
        control_token=control_token,
        processing_delay_seconds=0.01,
        working_delay_seconds=0.25,
    )
    application = create_app(
        settings,
        state=state,
        token_verifier=StaticSupplierSandboxTokenVerifier(access_token),
        oauth_fixture=LocalOAuthFixture(
            client_id="industrial-ops-a2a-client",
            client_secret=client_secret,
            access_token=access_token,
        ),
    )
    server = uvicorn.Server(
        uvicorn.Config(
            application,
            log_level="error",
            access_log=False,
            lifespan="on",
        )
    )
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(200):
            if server.started:
                break
            if server_task.done():
                await server_task
            await asyncio.sleep(0.01)
        else:
            raise SupplierA2AAcceptanceError("supplier_sandbox_did_not_start")
        yield base_url
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=5)
        listener.close()


def _integer_mapping(value: Any, reason: str) -> dict[str, int]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or isinstance(item, bool)
        or not isinstance(item, int)
        or item < 0
        for key, item in value.items()
    ):
        raise SupplierA2AAcceptanceError(reason)
    return dict(value)


def _zero_side_effects() -> dict[str, int]:
    return {
        "tool_executions": 0,
        "work_order_mutations": 0,
        "equipment_controls": 0,
    }


def _inside_file(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise SupplierA2AAcceptanceError("supplier_a2a_evidence_path_is_invalid")
    return resolved


def _digest(document: dict[str, Any]) -> str:
    return sha256(_canonical(document)).hexdigest()


def _canonical(document: dict[str, Any]) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
