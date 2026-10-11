"""Evidence-gated device-family onboarding and diagnosis routing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetRecord,
    DeviceFamilyModelBindingRecord,
    DeviceFamilyProfileRecord,
    EvaluationSuiteRecord,
    IndexReleaseRecord,
    KnowledgeDocumentVersionRecord,
    ModelEvaluationRunRecord,
)

ReviewDecision = Literal["ACCEPT", "REJECT"]
_FAMILY_CODE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_RISK_CLASSES = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})
_MUTABLE_STATUSES = frozenset({"BLOCKED", "READY"})
_SOURCE_MAX_AGE = timedelta(days=30)


class DeviceFamilyNotVisible(Exception):
    pass


class DeviceFamilyConflict(Exception):
    def __init__(self, reason: str, *, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


class DeviceFamilyRoutingBlocked(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DeviceFamilyCreate:
    family_code: str
    display_name: str
    risk_class: str
    model_codes: tuple[str, ...]
    source_system: str
    source_record_id: str
    source_as_of: datetime
    knowledge_release_id: str
    evaluation_suite_id: str
    evaluation_id: str
    minimum_gold_samples: int = 30


class DeviceFamilyProfileService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def list(
        self,
        identity: IdentityContext,
        *,
        status: str | None,
        request_id: str,
    ) -> list[DeviceFamilyProfileRecord]:
        self._require(identity, Action.READ_DEVICE_FAMILY, None, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            statement = select(DeviceFamilyProfileRecord).where(
                DeviceFamilyProfileRecord.tenant_id == identity.tenant_id
            )
            if status is not None:
                statement = statement.where(DeviceFamilyProfileRecord.status == status.upper())
            return list(
                session.scalars(
                    statement.order_by(
                        DeviceFamilyProfileRecord.family_code,
                        DeviceFamilyProfileRecord.family_version.desc(),
                    )
                )
            )

    def get(
        self,
        identity: IdentityContext,
        profile_id: str,
        *,
        request_id: str,
    ) -> DeviceFamilyProfileRecord:
        self._require(identity, Action.READ_DEVICE_FAMILY, profile_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            return _profile_or_hidden(session, identity.tenant_id, profile_id)

    def create(
        self,
        identity: IdentityContext,
        command: DeviceFamilyCreate,
        *,
        idempotency_key: str,
        request_id: str,
    ) -> tuple[DeviceFamilyProfileRecord, bool]:
        self._require(identity, Action.PROPOSE_DEVICE_FAMILY, None, request_id)
        normalized = _normalize_create(command)
        request_hash = _digest(_configuration_payload(normalized))
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            replay = session.scalar(
                select(DeviceFamilyProfileRecord).where(
                    DeviceFamilyProfileRecord.tenant_id == identity.tenant_id,
                    DeviceFamilyProfileRecord.requested_by_subject_id == identity.subject_id,
                    DeviceFamilyProfileRecord.idempotency_key == idempotency_key,
                )
            )
            if replay is not None:
                if replay.request_hash != request_hash:
                    raise DeviceFamilyConflict("idempotency_key_payload_changed")
                return replay, False

            family_version = int(
                session.scalar(
                    select(func.max(DeviceFamilyProfileRecord.family_version)).where(
                        DeviceFamilyProfileRecord.tenant_id == identity.tenant_id,
                        DeviceFamilyProfileRecord.family_code == normalized.family_code,
                    )
                )
                or 0
            ) + 1
            readiness, evidence_digest = _readiness(
                session,
                identity.tenant_id,
                normalized,
                now=now,
            )
            profile = DeviceFamilyProfileRecord(
                profile_id=f"device-family-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                family_code=normalized.family_code,
                family_version=family_version,
                display_name=normalized.display_name,
                risk_class=normalized.risk_class,
                model_codes_json=list(normalized.model_codes),
                source_system=normalized.source_system,
                source_record_id=normalized.source_record_id,
                source_as_of=normalized.source_as_of,
                knowledge_release_id=normalized.knowledge_release_id,
                evaluation_suite_id=normalized.evaluation_suite_id,
                evaluation_id=normalized.evaluation_id,
                minimum_gold_samples=normalized.minimum_gold_samples,
                readiness_json=readiness,
                evidence_digest=evidence_digest,
                configuration_digest=request_hash,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                status="READY" if not readiness["blockers"] else "BLOCKED",
                requested_by_subject_id=identity.subject_id,
                reviewed_by_subject_id=None,
                review_decision=None,
                review_reason=None,
                reviewed_at=None,
                activated_at=None,
                retired_at=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(profile)
            session.add_all(
                [
                    DeviceFamilyModelBindingRecord(
                        binding_id=f"device-family-binding-{uuid4().hex}",
                        tenant_id=identity.tenant_id,
                        profile_id=profile.profile_id,
                        family_code=profile.family_code,
                        profile_version=profile.family_version,
                        model_code=model_code,
                        is_active=False,
                        created_at=now,
                        updated_at=now,
                    )
                    for model_code in normalized.model_codes
                ]
            )
            session.flush()
            return profile, True

    def revalidate(
        self,
        identity: IdentityContext,
        profile_id: str,
        *,
        expected_version: int,
        request_id: str,
    ) -> DeviceFamilyProfileRecord:
        self._require(identity, Action.PROPOSE_DEVICE_FAMILY, profile_id, request_id)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            profile = _profile_or_hidden(session, identity.tenant_id, profile_id)
            _require_version(profile, expected_version)
            if profile.status not in _MUTABLE_STATUSES:
                raise DeviceFamilyConflict("device_family_profile_is_terminal")
            command = _command_from_record(profile)
            readiness, evidence_digest = _readiness(
                session,
                identity.tenant_id,
                command,
                now=now,
            )
            profile.readiness_json = readiness
            profile.evidence_digest = evidence_digest
            profile.status = "READY" if not readiness["blockers"] else "BLOCKED"
            profile.version += 1
            profile.updated_at = now
            session.flush()
            return profile

    def review(
        self,
        identity: IdentityContext,
        profile_id: str,
        *,
        decision: ReviewDecision,
        reason: str,
        expected_version: int,
        request_id: str,
    ) -> DeviceFamilyProfileRecord:
        self._require(identity, Action.REVIEW_DEVICE_FAMILY, profile_id, request_id)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            profile = _profile_or_hidden(session, identity.tenant_id, profile_id)
            _require_version(profile, expected_version)
            if profile.status not in _MUTABLE_STATUSES:
                raise DeviceFamilyConflict("device_family_profile_is_terminal")
            if profile.requested_by_subject_id == identity.subject_id:
                raise DeviceFamilyConflict("requester_cannot_review_device_family")

            normalized_decision = decision.upper()
            if normalized_decision == "ACCEPT":
                command = _command_from_record(profile)
                readiness, evidence_digest = _readiness(
                    session,
                    identity.tenant_id,
                    command,
                    now=now,
                )
                profile.readiness_json = readiness
                profile.evidence_digest = evidence_digest
                if readiness["blockers"]:
                    raise DeviceFamilyConflict(
                        "device_family_readiness_changed",
                        current_version=profile.version,
                    )
                _require_no_active_model_conflict(session, profile)
                current = session.scalar(
                    select(DeviceFamilyProfileRecord).where(
                        DeviceFamilyProfileRecord.tenant_id == identity.tenant_id,
                        DeviceFamilyProfileRecord.family_code == profile.family_code,
                        DeviceFamilyProfileRecord.status == "ACTIVE",
                        DeviceFamilyProfileRecord.profile_id != profile.profile_id,
                    )
                )
                if current is not None:
                    current.status = "RETIRED"
                    current.retired_at = now
                    current.version += 1
                    current.updated_at = now
                    for binding in session.scalars(
                        select(DeviceFamilyModelBindingRecord).where(
                            DeviceFamilyModelBindingRecord.tenant_id == identity.tenant_id,
                            DeviceFamilyModelBindingRecord.profile_id == current.profile_id,
                            DeviceFamilyModelBindingRecord.is_active.is_(True),
                        )
                    ):
                        binding.is_active = False
                        binding.updated_at = now
                    session.flush()
                bindings = list(
                    session.scalars(
                        select(DeviceFamilyModelBindingRecord).where(
                            DeviceFamilyModelBindingRecord.tenant_id == identity.tenant_id,
                            DeviceFamilyModelBindingRecord.profile_id == profile.profile_id,
                        )
                    )
                )
                if {item.model_code for item in bindings} != set(profile.model_codes_json):
                    raise DeviceFamilyConflict("device_family_model_bindings_changed")
                for binding in bindings:
                    binding.is_active = True
                    binding.updated_at = now
                profile.status = "ACTIVE"
                profile.activated_at = now
            elif normalized_decision == "REJECT":
                profile.status = "REJECTED"
            else:
                raise ValueError("decision must be ACCEPT or REJECT")

            profile.reviewed_by_subject_id = identity.subject_id
            profile.review_decision = normalized_decision
            profile.review_reason = reason.strip()
            profile.reviewed_at = now
            profile.version += 1
            profile.updated_at = now
            try:
                session.flush()
            except IntegrityError as exc:
                raise DeviceFamilyConflict(
                    "model_code_already_active_in_another_family"
                ) from exc
            return profile

    def _require(
        self,
        identity: IdentityContext,
        action: Action,
        resource_id: str | None,
        request_id: str,
    ) -> None:
        try:
            self._authorizer.require(
                identity,
                action,
                ResourceContext(tenant_id=identity.tenant_id, resource_id=resource_id),
                request_id=request_id,
            )
        except AuthorizationDenied as exc:
            raise DeviceFamilyNotVisible from exc


def resolve_device_family(
    session: Session,
    *,
    tenant_id: str,
    model_code: str,
    knowledge_release_id: str,
) -> str:
    """Resolve a model through an active profile, retaining legacy fallback only if unknown."""

    normalized_model = _normalize_model_code(model_code)
    bindings = list(
        session.scalars(
            select(DeviceFamilyModelBindingRecord).where(
                DeviceFamilyModelBindingRecord.tenant_id == tenant_id,
                DeviceFamilyModelBindingRecord.model_code == normalized_model,
            )
        )
    )
    active = [item for item in bindings if item.is_active]
    if len(active) > 1:
        raise DeviceFamilyRoutingBlocked("ambiguous_active_device_family")
    if active:
        profile = session.scalar(
            select(DeviceFamilyProfileRecord).where(
                DeviceFamilyProfileRecord.tenant_id == tenant_id,
                DeviceFamilyProfileRecord.profile_id == active[0].profile_id,
                DeviceFamilyProfileRecord.status == "ACTIVE",
            )
        )
        if profile is None:
            raise DeviceFamilyRoutingBlocked("active_device_family_binding_is_invalid")
        if profile.knowledge_release_id != knowledge_release_id:
            raise DeviceFamilyRoutingBlocked("device_family_knowledge_release_changed")
        return profile.family_code
    if bindings:
        raise DeviceFamilyRoutingBlocked("device_family_not_active")
    return model_code.split("-", maxsplit=1)[0].lower()


def _readiness(
    session: Session,
    tenant_id: str,
    command: DeviceFamilyCreate,
    *,
    now: datetime,
) -> tuple[dict[str, Any], str]:
    blockers: list[str] = []
    release = session.scalar(
        select(IndexReleaseRecord).where(
            IndexReleaseRecord.tenant_id == tenant_id,
            IndexReleaseRecord.release_id == command.knowledge_release_id,
        )
    )
    release_ready = bool(release and release.status == "PUBLISHED" and release.is_active)
    if not release_ready:
        blockers.append("knowledge_release_not_active")

    versions = list(
        session.scalars(
            select(KnowledgeDocumentVersionRecord).where(
                KnowledgeDocumentVersionRecord.tenant_id == tenant_id,
                KnowledgeDocumentVersionRecord.status == "PUBLISHED",
                KnowledgeDocumentVersionRecord.valid_from <= now,
            )
        )
    )
    eligible_versions = [
        item
        for item in versions
        if (item.valid_to is None or item.valid_to > now)
        and (
            command.family_code in {str(value).lower() for value in item.device_families}
            or bool(
                set(command.model_codes)
                & {_normalize_model_code(value) for value in item.device_models}
            )
        )
    ]
    if not eligible_versions:
        blockers.append("published_family_knowledge_missing")

    suite = session.scalar(
        select(EvaluationSuiteRecord).where(
            EvaluationSuiteRecord.tenant_id == tenant_id,
            EvaluationSuiteRecord.suite_id == command.evaluation_suite_id,
        )
    )
    suite_ready = bool(suite and suite.tier == "GOLD" and suite.status == "FROZEN")
    if not suite_ready:
        blockers.append("gold_evaluation_suite_not_frozen")
    slice_count = _family_slice_count(suite, command.family_code) if suite is not None else 0
    if slice_count < command.minimum_gold_samples:
        blockers.append("device_family_gold_slice_insufficient")

    evaluation = session.scalar(
        select(ModelEvaluationRunRecord).where(
            ModelEvaluationRunRecord.tenant_id == tenant_id,
            ModelEvaluationRunRecord.evaluation_id == command.evaluation_id,
        )
    )
    evaluation_ready = bool(
        evaluation
        and evaluation.suite_id == command.evaluation_suite_id
        and evaluation.status == "COMPLETED"
        and evaluation.decision == "CANDIDATE"
        and evaluation.hard_gate_results
        and all(evaluation.hard_gate_results.values())
    )
    if not evaluation_ready:
        blockers.append("device_family_evaluation_not_passed")

    asset_records = list(
        session.scalars(
            select(AssetRecord).where(AssetRecord.tenant_id == tenant_id)
        )
    )
    asset_count = sum(
        1
        for asset in asset_records
        if asset.model_code and _normalize_model_code(asset.model_code) in command.model_codes
    )
    if asset_count == 0:
        blockers.append("authoritative_asset_model_missing")

    source_as_of = _aware(command.source_as_of)
    if source_as_of > now or source_as_of < now - _SOURCE_MAX_AGE:
        blockers.append("authoritative_model_mapping_stale")

    evidence = {
        "schema_version": "device-family-readiness/v1",
        "family_code": command.family_code,
        "knowledge_release": {
            "release_id": command.knowledge_release_id,
            "ready": release_ready,
            "content_checksum": release.content_checksum if release is not None else None,
        },
        "knowledge": {
            "published_document_count": len(eligible_versions),
            "document_version_ids": sorted(item.document_version_id for item in eligible_versions),
            "content_checksums": sorted(item.content_checksum for item in eligible_versions),
        },
        "gold_evaluation": {
            "suite_id": command.evaluation_suite_id,
            "suite_ready": suite_ready,
            "manifest_hash": suite.manifest_hash if suite is not None else None,
            "slice_count": slice_count,
            "minimum_slice_count": command.minimum_gold_samples,
            "evaluation_id": command.evaluation_id,
            "evaluation_ready": evaluation_ready,
            "decision": evaluation.decision if evaluation is not None else None,
            "report_hash": evaluation.report_hash if evaluation is not None else None,
        },
        "asset_mapping": {
            "source_system": command.source_system,
            "source_record_id": command.source_record_id,
            "source_as_of": source_as_of.isoformat(),
            "matching_asset_count": asset_count,
            "model_codes": list(command.model_codes),
        },
        "blockers": sorted(set(blockers)),
    }
    return evidence, _digest(evidence)


def _require_no_active_model_conflict(
    session: Session,
    profile: DeviceFamilyProfileRecord,
) -> None:
    candidates = list(
        session.scalars(
            select(DeviceFamilyProfileRecord).where(
                DeviceFamilyProfileRecord.tenant_id == profile.tenant_id,
                DeviceFamilyProfileRecord.status == "ACTIVE",
                DeviceFamilyProfileRecord.family_code != profile.family_code,
            )
        )
    )
    requested = set(profile.model_codes_json)
    for item in candidates:
        if requested & set(item.model_codes_json):
            raise DeviceFamilyConflict("model_code_already_active_in_another_family")


def _profile_or_hidden(
    session: Session,
    tenant_id: str,
    profile_id: str,
) -> DeviceFamilyProfileRecord:
    profile = session.scalar(
        select(DeviceFamilyProfileRecord).where(
            DeviceFamilyProfileRecord.tenant_id == tenant_id,
            DeviceFamilyProfileRecord.profile_id == profile_id,
        )
    )
    if profile is None:
        raise DeviceFamilyNotVisible
    return profile


def _require_version(profile: DeviceFamilyProfileRecord, expected_version: int) -> None:
    if profile.version != expected_version:
        raise DeviceFamilyConflict(
            "device_family_profile_version_conflict",
            current_version=profile.version,
        )


def _normalize_create(command: DeviceFamilyCreate) -> DeviceFamilyCreate:
    family_code = command.family_code.strip().lower()
    if not _FAMILY_CODE.fullmatch(family_code):
        raise ValueError("invalid device family code")
    display_name = command.display_name.strip()
    if not display_name:
        raise ValueError("display_name is required")
    risk_class = command.risk_class.strip().upper()
    if risk_class not in _RISK_CLASSES:
        raise ValueError("unsupported risk_class")
    model_codes = tuple(sorted({_normalize_model_code(value) for value in command.model_codes}))
    if not model_codes:
        raise ValueError("at least one model code is required")
    if len(model_codes) > 100:
        raise ValueError("at most 100 model codes are allowed")
    source_system = command.source_system.strip()
    source_record_id = command.source_record_id.strip()
    if not source_system or not source_record_id:
        raise ValueError("authoritative source binding is required")
    if not 1 <= command.minimum_gold_samples <= 100_000:
        raise ValueError("minimum_gold_samples is out of range")
    return DeviceFamilyCreate(
        family_code=family_code,
        display_name=display_name,
        risk_class=risk_class,
        model_codes=model_codes,
        source_system=source_system,
        source_record_id=source_record_id,
        source_as_of=_aware(command.source_as_of),
        knowledge_release_id=command.knowledge_release_id.strip(),
        evaluation_suite_id=command.evaluation_suite_id.strip(),
        evaluation_id=command.evaluation_id.strip(),
        minimum_gold_samples=command.minimum_gold_samples,
    )


def _command_from_record(profile: DeviceFamilyProfileRecord) -> DeviceFamilyCreate:
    return DeviceFamilyCreate(
        family_code=profile.family_code,
        display_name=profile.display_name,
        risk_class=profile.risk_class,
        model_codes=tuple(profile.model_codes_json),
        source_system=profile.source_system,
        source_record_id=profile.source_record_id,
        source_as_of=profile.source_as_of,
        knowledge_release_id=profile.knowledge_release_id,
        evaluation_suite_id=profile.evaluation_suite_id,
        evaluation_id=profile.evaluation_id,
        minimum_gold_samples=profile.minimum_gold_samples,
    )


def _configuration_payload(command: DeviceFamilyCreate) -> dict[str, Any]:
    return {
        "family_code": command.family_code,
        "display_name": command.display_name,
        "risk_class": command.risk_class,
        "model_codes": list(command.model_codes),
        "source_system": command.source_system,
        "source_record_id": command.source_record_id,
        "source_as_of": command.source_as_of.isoformat(),
        "knowledge_release_id": command.knowledge_release_id,
        "evaluation_suite_id": command.evaluation_suite_id,
        "evaluation_id": command.evaluation_id,
        "minimum_gold_samples": command.minimum_gold_samples,
    }


def _family_slice_count(suite: EvaluationSuiteRecord, family_code: str) -> int:
    values = suite.slice_counts
    return max(
        int(values.get(f"device_family:{family_code}", 0)),
        int(values.get(family_code, 0)),
    )


def _normalize_model_code(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized or len(normalized) > 128:
        raise ValueError("invalid model code")
    return normalized


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()
