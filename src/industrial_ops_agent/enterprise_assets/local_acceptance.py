"""Ephemeral loopback environment for the enterprise model-import acceptance."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from sys import exc_info
from threading import Thread
from time import monotonic, sleep

import jwt
import uvicorn
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from industrial_ops_agent.api.app import create_app
from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.oidc import OidcVerifier, StaticSigningKeyProvider
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.config import Environment, Settings
from industrial_ops_agent.enterprise_assets.acceptance import (
    COMPONENTS,
    EnterpriseModelImportAcceptanceReport,
    execute_enterprise_model_import_acceptance,
)
from industrial_ops_agent.enterprise_assets.baseline import EnterpriseStagingBaselineService
from industrial_ops_agent.enterprise_assets.evidence import collect_enterprise_runtime_evidence
from industrial_ops_agent.enterprise_assets.models import EnterpriseRuntimeEvidence
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import Base, TenantRecord
from industrial_ops_agent.releases.service import ModelReleaseService
from industrial_ops_agent.security_audit import InMemorySecurityAuditSink, SecurityAuditor
from industrial_ops_agent.simulation.enterprise_project_adoption import (
    EnterpriseProjectAdoptionError,
    EnterpriseProjectAdoptionReport,
    load_enterprise_project_adoption_envelope,
)

TENANT_ID = "tenant-enterprise-model-import-acceptance"
_ISSUER = "http://localhost/enterprise-model-import-acceptance"
_AUDIENCE = "industrial-ops-enterprise-model-import-acceptance"
_SERVER_THREAD_NAME = "enterprise-model-import-loopback"


class LocalEnterpriseModelImportAcceptanceError(RuntimeError):
    """The ephemeral local staging environment could not prove the acceptance."""


@dataclass(frozen=True, slots=True)
class _ScopedModelImportAdoptionReader:
    report: EnterpriseProjectAdoptionReport
    evidence: tuple[EnterpriseRuntimeEvidence, ...]

    def snapshot(self) -> EnterpriseProjectAdoptionReport:
        return self.report

    def runtime_evidence(self) -> tuple[EnterpriseRuntimeEvidence, ...]:
        return self.evidence


def execute_local_enterprise_model_import_acceptance(
    repo_root: Path,
    *,
    timeout_seconds: float = 300.0,
) -> EnterpriseModelImportAcceptanceReport:
    """Run the real HTTP routes against a temporary approved Staging tenant."""

    root = repo_root.resolve(strict=True)
    if not 1 <= timeout_seconds <= 300:
        raise ValueError("timeout_seconds must be between 1 and 300")
    database = _database()
    try:
        auditor = SecurityAuditor(
            InMemorySecurityAuditSink(),
            hash_key=b"enterprise-model-import-local-acceptance",
        )
        authorizer = Authorizer(auditor)
        engineer = _identity(Role.MODEL_ENGINEER)
        baseline_release_id = _prepare_approved_baseline(
            database,
            authorizer,
            engineer,
        )
        adoption = _scoped_model_import_adoption_reader(root)
        verifier, token = _signed_local_identity(engineer)
        app = create_app(
            Settings(
                environment=Environment.TEST,
                log_level="WARNING",
                enterprise_project_repo_root=root,
                _env_file=None,
            ),
            oidc_verifier=verifier,
            security_auditor=auditor,
            database=database,
            authorizer=authorizer,
            enterprise_project_adoption_service=adoption,
        )
        with _serve_loopback(app, startup_timeout_seconds=min(timeout_seconds, 30.0)) as base_url:
            report = execute_enterprise_model_import_acceptance(
                base_url,
                token,
                timeout_seconds=timeout_seconds,
                execution_mode="EPHEMERAL_LOOPBACK_STAGING",
            )
        if any(
            baseline_release_id not in component.compatible_baseline_release_ids
            for component in report.components
        ):
            raise LocalEnterpriseModelImportAcceptanceError(
                "approved_local_baseline_not_bound_to_every_component"
            )
        return report
    finally:
        database.dispose()


def _scoped_model_import_adoption_reader(
    root: Path,
) -> _ScopedModelImportAdoptionReader:
    """Revalidate only the seven model sources needed by this ephemeral import."""

    try:
        report = load_enterprise_project_adoption_envelope(root)
        evidence = collect_enterprise_runtime_evidence(root, report)
    except (EnterpriseProjectAdoptionError, OSError, ValueError) as exc:
        raise LocalEnterpriseModelImportAcceptanceError(
            "enterprise_model_import_source_evidence_is_invalid"
        ) from exc
    if tuple(item.component for item in evidence) != COMPONENTS:
        raise LocalEnterpriseModelImportAcceptanceError(
            "enterprise_model_import_source_components_incomplete"
        )
    return _ScopedModelImportAdoptionReader(report=report, evidence=evidence)


def _database() -> Database:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    now = datetime.now(UTC)
    with Session(engine) as session:
        session.add(
            TenantRecord(
                id=TENANT_ID,
                status="active",
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    return Database.from_engine(engine)


def _prepare_approved_baseline(
    database: Database,
    authorizer: Authorizer,
    engineer: IdentityContext,
) -> str:
    draft = EnterpriseStagingBaselineService(database, authorizer).create_draft(
        engineer,
        active_mcp_server_versions={},
        idempotency_key="enterprise-model-import-local-baseline",
        request_id="enterprise-model-import-local-baseline",
    )
    releases = ModelReleaseService(database, authorizer)
    candidate = releases.validate(
        engineer,
        draft.release_id,
        expected_version=draft.version,
        request_id="enterprise-model-import-local-baseline-validate",
    )
    pending = releases.submit_for_approval(
        engineer,
        draft.release_id,
        expected_version=candidate.release.version,
        request_id="enterprise-model-import-local-baseline-submit",
    )
    if pending.approval is None:
        raise LocalEnterpriseModelImportAcceptanceError(
            "local_staging_baseline_approval_not_created"
        )
    approved = releases.decide_approval(
        _identity(Role.MODEL_RELEASE_APPROVER),
        draft.release_id,
        expected_version=pending.release.version,
        expected_approval_version=pending.approval.version,
        decision="APPROVED",
        reason="independent ephemeral Staging baseline review passed",
        request_id="enterprise-model-import-local-baseline-approve",
    )
    if approved.approval is None or approved.approval.status != "APPROVED":
        raise LocalEnterpriseModelImportAcceptanceError(
            "local_staging_baseline_approval_not_passed"
        )
    return draft.release_id


def _identity(role: Role) -> IdentityContext:
    now = datetime.now(UTC)
    return IdentityContext(
        subject_id=f"local-{role.value}",
        oidc_subject=f"oidc-local-{role.value}",
        tenant_id=TENANT_ID,
        roles=frozenset({role}),
        asset_ids=frozenset(),
        site_ids=frozenset(),
        issued_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(minutes=10),
    )


def _signed_local_identity(identity: IdentityContext) -> tuple[OidcVerifier, str]:
    private_key = rsa.generate_private_key(public_exponent=65_537, key_size=2_048)
    token = jwt.encode(
        {
            "iss": _ISSUER,
            "aud": _AUDIENCE,
            "sub": identity.oidc_subject,
            "subject_id": identity.subject_id,
            "tenant_id": identity.tenant_id,
            "roles": sorted(role.value for role in identity.roles),
            "asset_ids": sorted(identity.asset_ids),
            "site_ids": sorted(identity.site_ids),
            "iat": int(identity.issued_at.timestamp()),
            "exp": int(identity.expires_at.timestamp()),
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "ephemeral-local-acceptance"},
    )
    return (
        OidcVerifier(
            issuer=_ISSUER,
            audience=_AUDIENCE,
            signing_keys=StaticSigningKeyProvider(private_key.public_key()),
        ),
        token,
    )


@contextmanager
def _serve_loopback(
    app: FastAPI,
    *,
    startup_timeout_seconds: float,
) -> Iterator[str]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = int(listener.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            access_log=False,
            date_header=False,
            lifespan="off",
            limit_concurrency=8,
            log_level="warning",
            proxy_headers=False,
            server_header=False,
            timeout_graceful_shutdown=5,
            timeout_keep_alive=1,
            workers=1,
        )
    )
    server_errors: list[BaseException] = []

    def run_server() -> None:
        try:
            server.run(sockets=[listener])
        except BaseException as exc:  # pragma: no cover - propagated below
            server_errors.append(exc)

    thread = Thread(target=run_server, name=_SERVER_THREAD_NAME, daemon=True)
    thread.start()
    deadline = monotonic() + startup_timeout_seconds
    while not server.started:
        if server_errors:
            raise LocalEnterpriseModelImportAcceptanceError(
                "ephemeral_loopback_api_start_failed"
            ) from server_errors[0]
        if not thread.is_alive() or monotonic() >= deadline:
            raise LocalEnterpriseModelImportAcceptanceError("ephemeral_loopback_api_start_timeout")
        sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        failure_in_flight = exc_info()[0] is not None
        server.should_exit = True
        thread.join(timeout=10)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=2)
        listener.close()
        if thread.is_alive() and not failure_in_flight:
            raise LocalEnterpriseModelImportAcceptanceError("ephemeral_loopback_api_stop_timeout")
        if server_errors and not failure_in_flight:
            raise LocalEnterpriseModelImportAcceptanceError(
                "ephemeral_loopback_api_failed"
            ) from server_errors[0]


__all__ = [
    "LocalEnterpriseModelImportAcceptanceError",
    "execute_local_enterprise_model_import_acceptance",
]
