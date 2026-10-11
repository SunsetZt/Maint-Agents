"""Persist and expose verifier-owned supply-chain evidence."""

from __future__ import annotations

import builtins
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain.json import strict_document_digest as _manifest_hash
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    ModelAliasRecord,
    ModelDeploymentRecord,
    ModelReleaseRecord,
    ModelReleaseTransitionRecord,
    SupplyChainEvidenceRecord,
    TrainingArtifactRecord,
    TrainingExperimentRecord,
)

SCHEMA_VERSION = "industrial-ops-supply-chain/v1"
SBOM_FORMAT = "spdx-json"
ALLOWED_SEVERITIES = frozenset({"UNKNOWN", "NEGLIGIBLE", "LOW", "MEDIUM", "HIGH", "CRITICAL"})


class SupplyChainEvidenceNotVisible(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ComponentSupplyChainIdentity:
    """Pinned identity of a served component, not the main LLM or training image."""

    image_repository: str
    image_digest: str
    model_artifact_digest: str
    source_repository: str
    source_revision: str


@dataclass(frozen=True, slots=True)
class VerifiedSupplyChainProof:
    image_signature_digest: str
    model_signature_digest: str
    certificate_identity: str
    certificate_oidc_issuer: str
    verifier_version: str


@dataclass(frozen=True, slots=True)
class SupplyChainStatement:
    schema_version: str
    image_repository: str
    image_digest: str
    model_artifact_digest: str
    source_repository: str
    source_revision: str
    sbom_digest: str
    sbom_format: str
    vulnerability_report_digest: str
    vulnerability_scan_status: str
    maximum_vulnerability_severity: str
    vulnerability_scanner: str
    license_report_digest: str
    license_status: str
    provenance_digest: str


class SupplyChainEvidenceService:
    """Write only verifier results; release callers can only reference stored rows."""

    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def record_verified(
        self,
        identity: IdentityContext,
        statement: SupplyChainStatement,
        proof: VerifiedSupplyChainProof,
        *,
        request_id: str,
    ) -> SupplyChainEvidenceRecord:
        self._require(
            identity,
            Action.VERIFY_SUPPLY_CHAIN_EVIDENCE,
            "supply-chain-evidence",
            request_id,
        )
        _validate_statement(statement)
        _validate_proof(proof)
        now = datetime.now(UTC)
        verification_hash = evidence_verification_hash(statement, proof)
        with self._database.transaction(identity.tenant_context) as session:
            existing = session.scalar(
                select(SupplyChainEvidenceRecord).where(
                    SupplyChainEvidenceRecord.tenant_id == identity.tenant_id,
                    SupplyChainEvidenceRecord.verification_hash == verification_hash,
                )
            )
            if existing is not None:
                if (
                    existing.verification_status != "VERIFIED"
                    or record_verification_hash(existing) != verification_hash
                ):
                    raise ValueError("stored supply-chain evidence failed its integrity check")
                return existing
            record = SupplyChainEvidenceRecord(
                evidence_id=f"supply-chain-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                **asdict(statement),
                **asdict(proof),
                verification_status="VERIFIED",
                verification_hash=verification_hash,
                verified_by_subject_id=identity.subject_id,
                verified_at=now,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.flush()
            return record

    def list(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        release_ready_only: bool = False,
    ) -> list[SupplyChainEvidenceRecord]:
        self._require(
            identity,
            Action.READ_SUPPLY_CHAIN_EVIDENCE,
            "supply-chain-evidence",
            request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            query = select(SupplyChainEvidenceRecord).where(
                SupplyChainEvidenceRecord.tenant_id == identity.tenant_id,
                SupplyChainEvidenceRecord.verification_status == "VERIFIED",
            )
            if release_ready_only:
                query = query.where(
                    SupplyChainEvidenceRecord.vulnerability_scan_status == "PASSED",
                    SupplyChainEvidenceRecord.license_status == "APPROVED",
                )
            records = list(
                session.scalars(
                    query.order_by(
                        SupplyChainEvidenceRecord.verified_at.desc(),
                        SupplyChainEvidenceRecord.evidence_id,
                    )
                )
            )
            return [
                record
                for record in records
                if record.verification_hash == record_verification_hash(record)
            ]

    def list_for_component(
        self,
        identity: IdentityContext,
        *,
        component: str,
        model_id: str,
        request_id: str,
    ) -> tuple[builtins.list[SupplyChainEvidenceRecord], str | None]:
        self._require(
            identity, Action.READ_SUPPLY_CHAIN_EVIDENCE, "supply-chain-evidence", request_id
        )
        with self._database.transaction(identity.tenant_context) as session:
            return matching_component_evidence(session, identity.tenant_id, component, model_id)

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str,
        request_id: str,
    ) -> None:
        self._authorizer.require(
            identity,
            action,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
            request_id=request_id,
        )


def component_supply_chain_identity(
    session: Session, tenant_id: str, component: str, model_id: str
) -> ComponentSupplyChainIdentity:
    """Resolve existing registration; never infer serving image from training image.

    Older imports without runtime/source metadata need a new, verified registration.
    They remain readable, but are not silently upgraded by an evidence-list request.
    """
    kinds = {"VLM": "vlm_adapter_bundle", "ASR": "asr_adapter_bundle"}
    if component not in kinds:
        raise ValueError("COMPONENT_NOT_SUPPORTED")
    candidate = session.scalar(
        select(TrainingExperimentRecord).where(
            TrainingExperimentRecord.tenant_id == tenant_id,
            TrainingExperimentRecord.experiment_id == model_id,
        )
    )
    if candidate is None:
        raise ValueError("COMPONENT_MODEL_NOT_VISIBLE")
    artifacts = list(
        session.scalars(
            select(TrainingArtifactRecord).where(
                TrainingArtifactRecord.tenant_id == tenant_id,
                TrainingArtifactRecord.experiment_id == model_id,
                TrainingArtifactRecord.kind == kinds[component],
            )
        )
    )
    if len(artifacts) != 1:
        raise ValueError("COMPONENT_ARTIFACT_REQUIRED")
    artifact = artifacts[0]
    if artifact.size_bytes <= 0 or not re.fullmatch(
        r"(?:sha256:)?[0-9a-f]{64}", artifact.content_hash
    ):
        raise ValueError("COMPONENT_ARTIFACT_DIGEST_REQUIRED")
    metadata = artifact.metadata_json
    if not isinstance(metadata, dict):
        raise ValueError("COMPONENT_RUNTIME_IDENTITY_REQUIRED")
    runtime = metadata.get("runtime")
    source = metadata.get("source")
    if not isinstance(runtime, dict) or not isinstance(source, dict):
        raise ValueError("COMPONENT_RUNTIME_IDENTITY_REQUIRED")
    try:
        image_repository = _string(runtime.get("image_repository"), "runtime.image_repository")
        image_digest = _string(runtime.get("image_digest"), "runtime.image_digest")
        source_repository = _string(source.get("repository"), "source.repository")
        source_revision = _string(source.get("revision"), "source.revision")
        _validate_image_repository(image_repository)
        _validate_digest(image_digest, "runtime.image_digest")
        if (
            not source_repository.startswith("https://")
            or not re.fullmatch(r"[0-9a-f]{40}", source_revision)
            or source_revision != candidate.git_commit
        ):
            raise ValueError("source identity mismatch")
    except ValueError as exc:
        raise ValueError("COMPONENT_RUNTIME_IDENTITY_REQUIRED") from exc
    return ComponentSupplyChainIdentity(
        image_repository,
        image_digest,
        "sha256:" + artifact.content_hash.removeprefix("sha256:"),
        source_repository,
        source_revision,
    )


def component_evidence_matches(
    record: SupplyChainEvidenceRecord, tenant_id: str, binding: ComponentSupplyChainIdentity
) -> bool:
    """Only exact, positive verifier-owned evidence can authorize a new binding."""
    return (
        record.tenant_id == tenant_id
        and record.verification_status == "VERIFIED"
        and record.verification_hash == record_verification_hash(record)
        and record.vulnerability_scan_status == "PASSED"
        and record.license_status == "APPROVED"
        # Historical local attestations were never Cosign verification results.
        and not record.certificate_identity.startswith("project:")
        and not record.vulnerability_scanner.startswith("project-")
        and all(getattr(record, key) == value for key, value in asdict(binding).items())
    )


def matching_component_evidence(
    session: Session, tenant_id: str, component: str, model_id: str
) -> tuple[list[SupplyChainEvidenceRecord], str | None]:
    try:
        binding = component_supply_chain_identity(session, tenant_id, component, model_id)
    except ValueError as exc:
        return [], str(exc)
    records = session.scalars(
        select(SupplyChainEvidenceRecord)
        .where(
            SupplyChainEvidenceRecord.tenant_id == tenant_id,
            SupplyChainEvidenceRecord.model_artifact_digest == binding.model_artifact_digest,
        )
        .order_by(
            SupplyChainEvidenceRecord.verified_at.desc(), SupplyChainEvidenceRecord.evidence_id
        )
    )
    matches = [
        record for record in records if component_evidence_matches(record, tenant_id, binding)
    ]
    return matches, None if matches else "COMPONENT_SUPPLY_CHAIN_EVIDENCE_REQUIRED"


def statement_from_document(document: dict[str, Any]) -> SupplyChainStatement:
    if set(document) != {"schema_version", "image", "model", "source", "artifacts"}:
        raise ValueError("supply_chain_statement fields are incomplete")
    image = _exact_mapping(document["image"], {"repository", "digest"}, "image")
    model = _exact_mapping(document["model"], {"artifact_digest"}, "model")
    source = _exact_mapping(document["source"], {"repository", "revision"}, "source")
    artifacts = _exact_mapping(
        document["artifacts"],
        {"sbom", "vulnerability_scan", "license_report", "provenance"},
        "artifacts",
    )
    sbom = _exact_mapping(artifacts["sbom"], {"digest", "format"}, "artifacts.sbom")
    vulnerability = _exact_mapping(
        artifacts["vulnerability_scan"],
        {"digest", "status", "maximum_severity", "scanner"},
        "artifacts.vulnerability_scan",
    )
    license_report = _exact_mapping(
        artifacts["license_report"], {"digest", "status"}, "artifacts.license_report"
    )
    provenance = _exact_mapping(artifacts["provenance"], {"digest"}, "artifacts.provenance")
    statement = SupplyChainStatement(
        schema_version=_string(document["schema_version"], "schema_version"),
        image_repository=_string(image["repository"], "image.repository"),
        image_digest=_string(image["digest"], "image.digest"),
        model_artifact_digest=_string(model["artifact_digest"], "model.artifact_digest"),
        source_repository=_string(source["repository"], "source.repository"),
        source_revision=_string(source["revision"], "source.revision"),
        sbom_digest=_string(sbom["digest"], "artifacts.sbom.digest"),
        sbom_format=_string(sbom["format"], "artifacts.sbom.format"),
        vulnerability_report_digest=_string(
            vulnerability["digest"], "artifacts.vulnerability_scan.digest"
        ),
        vulnerability_scan_status=_string(
            vulnerability["status"], "artifacts.vulnerability_scan.status"
        ),
        maximum_vulnerability_severity=_string(
            vulnerability["maximum_severity"],
            "artifacts.vulnerability_scan.maximum_severity",
        ),
        vulnerability_scanner=_string(
            vulnerability["scanner"], "artifacts.vulnerability_scan.scanner"
        ),
        license_report_digest=_string(license_report["digest"], "artifacts.license_report.digest"),
        license_status=_string(license_report["status"], "artifacts.license_report.status"),
        provenance_digest=_string(provenance["digest"], "artifacts.provenance.digest"),
    )
    _validate_statement(statement)
    return statement


def evidence_verification_hash(
    statement: SupplyChainStatement,
    proof: VerifiedSupplyChainProof,
) -> str:
    payload = {
        "statement": asdict(statement),
        "proof": asdict(proof),
        "verification_status": "VERIFIED",
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return "sha256:" + sha256(encoded).hexdigest()


def record_verification_hash(record: SupplyChainEvidenceRecord) -> str:
    statement = SupplyChainStatement(
        **{field: getattr(record, field) for field in SupplyChainStatement.__dataclass_fields__}
    )
    proof = VerifiedSupplyChainProof(
        **{field: getattr(record, field) for field in VerifiedSupplyChainProof.__dataclass_fields__}
    )
    return evidence_verification_hash(statement, proof)


def _validate_statement(statement: SupplyChainStatement) -> None:
    if statement.schema_version != SCHEMA_VERSION:
        raise ValueError("supply_chain_statement.schema_version is unsupported")
    _validate_image_repository(statement.image_repository)
    for field in (
        "image_digest",
        "model_artifact_digest",
        "sbom_digest",
        "vulnerability_report_digest",
        "license_report_digest",
        "provenance_digest",
    ):
        _validate_digest(getattr(statement, field), field)
    if not re.fullmatch(r"[0-9a-f]{40}", statement.source_revision):
        raise ValueError("supply_chain_statement.source_revision must be a full Git commit")
    if not statement.source_repository.startswith("https://"):
        raise ValueError("supply_chain_statement.source_repository must use HTTPS")
    if statement.sbom_format != SBOM_FORMAT:
        raise ValueError("supply_chain_statement.sbom_format must be spdx-json")
    if statement.vulnerability_scan_status not in {"PASSED", "FAILED"}:
        raise ValueError("supply_chain_statement vulnerability status is invalid")
    if statement.maximum_vulnerability_severity not in ALLOWED_SEVERITIES:
        raise ValueError("supply_chain_statement vulnerability severity is invalid")
    if not statement.vulnerability_scanner or len(statement.vulnerability_scanner) > 128:
        raise ValueError("supply_chain_statement vulnerability scanner is invalid")
    if statement.license_status not in {"APPROVED", "REJECTED"}:
        raise ValueError("supply_chain_statement license status is invalid")


def _validate_proof(proof: VerifiedSupplyChainProof) -> None:
    _validate_digest(proof.image_signature_digest, "image_signature_digest")
    _validate_digest(proof.model_signature_digest, "model_signature_digest")
    if not proof.certificate_identity or len(proof.certificate_identity) > 1024:
        raise ValueError("certificate_identity is invalid")
    if not proof.certificate_oidc_issuer.startswith("https://"):
        raise ValueError("certificate_oidc_issuer must use HTTPS")
    if not re.fullmatch(r"v?[0-9]+\.[0-9]+\.[0-9]+", proof.verifier_version):
        raise ValueError("verifier_version is invalid")


def _validate_digest(value: str, field: str) -> None:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError(f"supply_chain_statement.{field} must be a sha256 digest")


def _validate_image_repository(value: str) -> None:
    pattern = re.compile(
        r"^[a-z0-9](?:[a-z0-9.-]*)(?::[0-9]{1,5})?"
        r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+$"
    )
    if len(value) > 512 or pattern.fullmatch(value) is None:
        raise ValueError("supply_chain_statement.image_repository is invalid")


def _exact_mapping(value: Any, fields: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"supply_chain_statement.{path} fields are incomplete")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ValueError(f"supply_chain_statement.{path} is invalid")
    return value


def supply_chain_manifest(evidence: SupplyChainEvidenceRecord) -> dict[str, str]:
    return {
        "evidence_id": evidence.evidence_id,
        "verification_status": evidence.verification_status,
        "verification_hash": evidence.verification_hash,
        "verified_at": evidence.verified_at.isoformat(),
        "verified_by_subject_id": evidence.verified_by_subject_id,
        "verifier_version": evidence.verifier_version,
        "certificate_identity": evidence.certificate_identity,
        "certificate_oidc_issuer": evidence.certificate_oidc_issuer,
        "source_repository": evidence.source_repository,
        "source_revision": evidence.source_revision,
        "sbom_digest": evidence.sbom_digest,
        "sbom_format": evidence.sbom_format,
        "vulnerability_report_digest": evidence.vulnerability_report_digest,
        "vulnerability_scan_status": evidence.vulnerability_scan_status,
        "maximum_vulnerability_severity": evidence.maximum_vulnerability_severity,
        "vulnerability_scanner": evidence.vulnerability_scanner,
        "license_report_digest": evidence.license_report_digest,
        "license_status": evidence.license_status,
        "provenance_digest": evidence.provenance_digest,
        "image_signature_digest": evidence.image_signature_digest,
        "model_signature_digest": evidence.model_signature_digest,
    }


def media_release_supply_chain_status(session: Session, release: ModelReleaseRecord) -> str:
    """Check only actually served media, preserving a distinct missing-legacy result."""
    manifest = release.manifest_json
    components = manifest.get("specialized_components", {})
    models = manifest.get("multimodal_model_ids", {})
    if not isinstance(components, dict) or not isinstance(models, dict):
        raise ValueError("component_supply_chain_binding_invalid")
    legacy_missing = False
    for name in ("vlm", "asr"):
        item = components.get(name)
        if item is None and name not in models:
            continue
        if (
            not isinstance(item, dict)
            or item.get("component") != name
            or item.get("runtime_model_id") != models.get(name)
            or not isinstance(item.get("training"), dict)
            or item["training"].get("experiment_id") != item.get("runtime_model_id")
            or not isinstance(item.get("artifact"), dict)
        ):
            raise ValueError(f"{name}_supply_chain_binding_invalid")
        receipt = item.get("supply_chain")
        if receipt is None and manifest.get("schema_version") == "ai-release-manifest/v4":
            legacy_missing = True
            continue
        if (
            not isinstance(receipt, dict)
            or not isinstance(item.get("runtime"), dict)
            or not isinstance(item.get("source"), dict)
        ):
            raise ValueError(f"{name}_supply_chain_binding_invalid")
        binding = component_supply_chain_identity(
            session, release.tenant_id, name.upper(), item["runtime_model_id"]
        )
        evidence = session.scalar(
            select(SupplyChainEvidenceRecord).where(
                SupplyChainEvidenceRecord.tenant_id == release.tenant_id,
                SupplyChainEvidenceRecord.evidence_id == receipt.get("evidence_id"),
            )
        )
        if evidence is None or not component_evidence_matches(evidence, release.tenant_id, binding):
            raise ValueError(f"{name}_supply_chain_evidence_mismatch")
        artifact_hash = item["artifact"].get("content_hash")
        if (
            not isinstance(artifact_hash, str)
            or "sha256:" + artifact_hash.removeprefix("sha256:") != binding.model_artifact_digest
            or item["runtime"].get("image_repository") != binding.image_repository
            or item["runtime"].get("image_digest") != binding.image_digest
            or item["source"]
            != {"repository": binding.source_repository, "revision": binding.source_revision}
            or receipt != supply_chain_manifest(evidence)
        ):
            raise ValueError(f"{name}_supply_chain_binding_changed")
    return "LEGACY_EVIDENCE_REQUIRED" if legacy_missing else "VERIFIED"


def require_media_deployment_supply_chain(
    session: Session,
    release: ModelReleaseRecord,
    deployment: ModelDeploymentRecord | None = None,
    *,
    allow_exact_recovery: bool = False,
) -> str:
    status = media_release_supply_chain_status(session, release)
    if status != "LEGACY_EVIDENCE_REQUIRED":
        return status
    if not allow_exact_recovery or deployment is None:
        raise ValueError("legacy_supply_chain_evidence_required")
    owner = release
    if deployment.release_id != release.release_id:
        stored_owner = session.scalar(
            select(ModelReleaseRecord).where(
                ModelReleaseRecord.tenant_id == release.tenant_id,
                ModelReleaseRecord.release_id == deployment.release_id,
            )
        )
        if stored_owner is None or stored_owner.rollback_release_id != release.release_id:
            raise ValueError("legacy_rollback_target_mismatch")
        owner = stored_owner
    _require_exact_media_recovery(session, owner, deployment)
    return status


def _require_exact_media_recovery(
    session: Session, release: ModelReleaseRecord, deployment: ModelDeploymentRecord
) -> None:
    _require_applied_media_identity(session, release, deployment)
    active_release = release
    if deployment.current_stage == "ROLLED_BACK":
        target_release = session.scalar(
            select(ModelReleaseRecord).where(
                ModelReleaseRecord.tenant_id == release.tenant_id,
                ModelReleaseRecord.release_id == release.rollback_release_id,
            )
        )
        if (
            target_release is None
            or target_release.target_environment != release.target_environment
            or release.manifest_json.get("rollback")
            != {
                "release_id": target_release.release_id,
                "manifest_hash": target_release.manifest_hash,
            }
        ):
            raise ValueError("legacy_rollback_target_mismatch")
        active_release = target_release
        previous = session.scalar(
            select(ModelDeploymentRecord).where(
                ModelDeploymentRecord.tenant_id == release.tenant_id,
                ModelDeploymentRecord.release_id == active_release.release_id,
            )
        )
        if (
            previous is None
            or previous.current_stage != "PRODUCTION"
            or previous.observed_traffic_percent != 100.0
            or previous.endpoint_url != deployment.endpoint_url
        ):
            raise ValueError("legacy_rollback_identity_required")
        # No alias bypass: the target's old deployment proves identity; the actual
        # existing alias below must still identify this already-applied rollback.
        _require_applied_media_identity(session, active_release, previous)
        media_release_supply_chain_status(session, active_release)
    aliases = list(
        session.scalars(
            select(ModelAliasRecord).where(
                ModelAliasRecord.tenant_id == release.tenant_id,
                ModelAliasRecord.deployment_id == deployment.deployment_id,
            )
        )
    )
    if deployment.current_stage in {"PRODUCTION", "ROLLED_BACK"} and not aliases:
        raise ValueError("legacy_existing_alias_required")
    if any(
        alias.status != "ACTIVE"
        or alias.active_release_id != active_release.release_id
        or alias.manifest_hash != active_release.manifest_hash
        or alias.endpoint_url != deployment.endpoint_url
        or alias.runtime_profile
        != active_release.manifest_json.get("runtime", {}).get("profile_id")
        for alias in aliases
    ):
        raise ValueError("legacy_existing_alias_mismatch")


def _require_applied_media_identity(
    session: Session, release: ModelReleaseRecord, deployment: ModelDeploymentRecord
) -> None:
    spec = deployment.desired_spec_json
    if (
        deployment.tenant_id != release.tenant_id
        or deployment.release_id != release.release_id
        or _manifest_hash(release.manifest_json) != release.manifest_hash
        or not isinstance(spec, dict)
        or spec.get("release_id") != release.release_id
        or spec.get("manifest_hash", spec.get("release_manifest_hash")) != release.manifest_hash
        or _manifest_hash(spec) != deployment.desired_spec_hash
        or deployment.applied_spec_hash != deployment.desired_spec_hash
        or deployment.current_stage
        not in {"SHADOW", "CANARY_5", "CANARY_25", "PRODUCTION", "ROLLED_BACK"}
        or deployment.current_stage != deployment.desired_stage
        or deployment.observed_traffic_percent != deployment.desired_traffic_percent
        or not deployment.endpoint_url
        or not deployment.endpoint_url.startswith(("https://", "http://"))
        or not deployment.provider_revision
    ):
        raise ValueError("legacy_deployment_identity_mismatch")
    transitions = session.scalars(
        select(ModelReleaseTransitionRecord).where(
            ModelReleaseTransitionRecord.tenant_id == release.tenant_id,
            ModelReleaseTransitionRecord.release_id == release.release_id,
        )
    )
    # Previously applied, hash-bound evidence is necessary; a READY flag alone is not authority.
    if not any(
        isinstance(item.evidence_json, dict)
        and _manifest_hash(item.evidence_json) == item.evidence_hash
        and item.reason_code
        == {
            "SHADOW": "shadow_route_verified",
            "CANARY_5": "canary_5_route_verified",
            "CANARY_25": "canary_25_route_verified",
            "PRODUCTION": "production_route_verified",
            "ROLLED_BACK": "rollback_route_verified",
        }[deployment.current_stage]
        and item.evidence_json.get("deployment_id") == deployment.deployment_id
        and item.evidence_json.get("desired_spec_hash") == deployment.desired_spec_hash
        and item.evidence_json.get("stage") == deployment.current_stage
        and item.evidence_json.get("observed_traffic_percent")
        == deployment.observed_traffic_percent
        for item in transitions
    ):
        raise ValueError("legacy_applied_evidence_required")
