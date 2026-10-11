"""Verifier-owned persistence for signed Staging acceptance manifests."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import StagingAcceptanceAttestationRecord
from industrial_ops_agent.staging_acceptance.dynamic_contracts import (
    StagingDynamicAcceptanceManifest,
)
from industrial_ops_agent.staging_acceptance.dynamic_manifest import (
    validate_dynamic_acceptance_manifest_identity,
)
from industrial_ops_agent.supply_chain.verifier import VerifiedBlobProof

SCHEMA_VERSION = "industrial-ops-staging-acceptance-attestation/v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COSIGN_VERSION = re.compile(r"v?[0-9]+\.[0-9]+\.[0-9]+")


class StagingAcceptanceAttestationService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def record_verified(
        self,
        identity: IdentityContext,
        manifest: StagingDynamicAcceptanceManifest,
        proof: VerifiedBlobProof,
        *,
        request_id: str,
        now: datetime | None = None,
    ) -> StagingAcceptanceAttestationRecord:
        self._authorizer.require(
            identity,
            Action.VERIFY_STAGING_ACCEPTANCE_ATTESTATION,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id="staging-acceptance-attestation",
            ),
            request_id=request_id,
        )
        validate_dynamic_acceptance_manifest_identity(manifest)
        _validate_proof(proof)
        verified_at = now or datetime.now(UTC)
        if verified_at.tzinfo is None:
            raise ValueError("Staging attestation verification time must be timezone-aware")
        verified_at = verified_at.astimezone(UTC)
        values = _record_values(manifest, proof)
        with self._database.transaction(identity.tenant_context) as session:
            existing = session.scalar(
                select(StagingAcceptanceAttestationRecord)
                .where(
                    StagingAcceptanceAttestationRecord.tenant_id
                    == identity.tenant_id,
                    *(
                        getattr(StagingAcceptanceAttestationRecord, field) == value
                        for field, value in values.items()
                    ),
                )
                .order_by(
                    StagingAcceptanceAttestationRecord.verified_at,
                    StagingAcceptanceAttestationRecord.evidence_id,
                )
            )
            if existing is not None:
                if not attestation_record_is_verified(existing):
                    raise ValueError("stored Staging attestation failed its integrity check")
                return existing
            verification_hash = attestation_verification_hash(
                {
                    **values,
                    "verified_by_subject_id": identity.subject_id,
                    "verified_at": _canonical_verified_at(verified_at),
                }
            )
            record = StagingAcceptanceAttestationRecord(
                evidence_id=f"staging-attestation-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                **values,
                verification_status="VERIFIED",
                verification_hash=verification_hash,
                verified_by_subject_id=identity.subject_id,
                verified_at=verified_at,
                created_at=verified_at,
                updated_at=verified_at,
            )
            session.add(record)
            session.flush()
            return record


def resolve_verified_attestation(
    session: Session,
    tenant_id: str,
    evidence_id: str,
) -> StagingAcceptanceAttestationRecord | None:
    record = session.scalar(
        select(StagingAcceptanceAttestationRecord).where(
            StagingAcceptanceAttestationRecord.tenant_id == tenant_id,
            StagingAcceptanceAttestationRecord.evidence_id == evidence_id,
        )
    )
    if record is None or not attestation_record_is_verified(record):
        return None
    return record


def attestation_record_is_verified(record: StagingAcceptanceAttestationRecord) -> bool:
    if record.verification_status != "VERIFIED":
        return False
    try:
        expected = attestation_verification_hash(_values_from_record(record))
    except (AttributeError, TypeError, ValueError):
        return False
    return record.verification_hash == expected


def attestation_verification_hash(values: dict[str, str]) -> str:
    expected_fields = {
        "schema_version",
        "manifest_id",
        "manifest_digest",
        "environment_id",
        "git_commit",
        "signature_bundle_digest",
        "certificate_identity",
        "certificate_oidc_issuer",
        "verifier_version",
        "verified_by_subject_id",
        "verified_at",
    }
    if set(values) != expected_fields:
        raise ValueError("Staging attestation fields are incomplete")
    payload = {**values, "verification_status": "VERIFIED"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + sha256(encoded).hexdigest()


def _record_values(
    manifest: StagingDynamicAcceptanceManifest,
    proof: VerifiedBlobProof,
) -> dict[str, str]:
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest.manifest_id,
        "manifest_digest": proof.signed_blob_digest,
        "environment_id": manifest.environment_id,
        "git_commit": manifest.git_commit,
        "signature_bundle_digest": proof.signature_bundle_digest,
        "certificate_identity": proof.certificate_identity,
        "certificate_oidc_issuer": proof.certificate_oidc_issuer,
        "verifier_version": proof.verifier_version,
    }


def _values_from_record(record: StagingAcceptanceAttestationRecord) -> dict[str, str]:
    return {
        "schema_version": record.schema_version,
        "manifest_id": record.manifest_id,
        "manifest_digest": record.manifest_digest,
        "environment_id": record.environment_id,
        "git_commit": record.git_commit,
        "signature_bundle_digest": record.signature_bundle_digest,
        "certificate_identity": record.certificate_identity,
        "certificate_oidc_issuer": record.certificate_oidc_issuer,
        "verifier_version": record.verifier_version,
        "verified_by_subject_id": record.verified_by_subject_id,
        "verified_at": _canonical_verified_at(record.verified_at),
    }


def _canonical_verified_at(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise ValueError("Staging attestation verification time is invalid")
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_proof(proof: VerifiedBlobProof) -> None:
    for value in (proof.signed_blob_digest, proof.signature_bundle_digest):
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("Staging attestation proof digest is invalid")
    if (
        not proof.certificate_identity
        or len(proof.certificate_identity) > 1024
        or any(character in proof.certificate_identity for character in "\x00\r\n")
    ):
        raise ValueError("Staging attestation certificate identity is invalid")
    if (
        not proof.certificate_oidc_issuer.startswith("https://")
        or len(proof.certificate_oidc_issuer) > 1024
    ):
        raise ValueError("Staging attestation certificate issuer is invalid")
    if _COSIGN_VERSION.fullmatch(proof.verifier_version) is None:
        raise ValueError("Staging attestation verifier version is invalid")
