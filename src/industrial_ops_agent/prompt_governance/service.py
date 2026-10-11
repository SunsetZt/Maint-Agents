"""Prompt metadata registration, evaluation binding, review, and retirement."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    EvaluationArtifactRecord,
    ModelEvaluationRunRecord,
    ModelReleaseRecord,
    PromptBundleRecord,
    PromptBundleTransitionRecord,
)
from industrial_ops_agent.prompting import PromptBundleDefinition, default_prompt_registry
from industrial_ops_agent.prompting.registry import PromptBundleNotDeployed, PromptBundleRegistry

PromptReviewDecision = Literal["APPROVED", "REJECTED"]
PromptOrigin = Literal["BUILTIN", "REGISTERED"]

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9.-]+)?$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_VARIABLE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TASK_TYPES = frozenset({"DIAGNOSIS", "VLM", "ASR", "TTS", "RERANKING"})


class PromptBundleNotVisible(LookupError):
    pass


class PromptBundleConflict(RuntimeError):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


class PromptBundleGateDenied(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class PromptTransition:
    transition_id: str
    sequence: int
    from_status: str
    to_status: str
    actor_subject_id: str
    reason_code: str
    evidence: dict[str, object]
    evidence_hash: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class PromptBundleView:
    prompt_bundle_id: str
    origin: PromptOrigin
    name: str
    bundle_version: str
    task_type: str
    status: str
    content_hash: str
    source_commit: str
    source_path: str
    section_names: tuple[str, ...]
    required_variables: tuple[str, ...]
    output_schema_name: str
    compatible_model_aliases: tuple[str, ...]
    change_summary: str
    evaluation_id: str | None
    evaluation_report_hash: str | None
    created_by_subject_id: str
    reviewed_by_subject_id: str | None
    review_reason: str | None
    reviewed_at: datetime | None
    runtime_available: bool
    runtime_hash_matches: bool
    release_reference_count: int
    active_production_reference: bool
    version: int
    created_at: datetime | None
    updated_at: datetime | None
    legal_actions: tuple[str, ...]
    transitions: tuple[PromptTransition, ...]


class PromptGovernanceService:
    POLICY_VERSION = "prompt-bundle-gitops/v1"

    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer
        self._runtime_registry = default_prompt_registry()

    def list(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
    ) -> tuple[PromptBundleView, ...]:
        self._require(identity, Action.READ_PROMPT_BUNDLE, "prompt-bundles", request_id)
        with self._database.transaction(identity.tenant_context) as session:
            records = tuple(
                session.scalars(
                    select(PromptBundleRecord)
                    .where(PromptBundleRecord.tenant_id == identity.tenant_id)
                    .order_by(PromptBundleRecord.created_at.desc())
                )
            )
            transitions = _transitions_by_bundle(session, identity.tenant_id)
            references = _release_references(session, identity.tenant_id)
        registered = tuple(
            self._record_view(
                identity,
                item,
                transitions.get(item.prompt_bundle_id, ()),
                references,
            )
            for item in records
        )
        builtins = tuple(
            self._builtin_view(identity, item, references)
            for item in self._runtime_registry.definitions()
            if all(record.prompt_bundle_id != item.prompt_bundle_id for record in records)
        )
        return builtins + registered

    def get(
        self,
        identity: IdentityContext,
        prompt_bundle_id: str,
        *,
        request_id: str,
    ) -> PromptBundleView:
        self._require(identity, Action.READ_PROMPT_BUNDLE, prompt_bundle_id, request_id)
        with self._database.transaction(identity.tenant_context) as session:
            record = _record_or_none(session, identity.tenant_id, prompt_bundle_id)
            references = _release_references(session, identity.tenant_id)
            if record is not None:
                transitions = _transitions_for(session, identity.tenant_id, prompt_bundle_id)
                return self._record_view(identity, record, transitions, references)
        try:
            definition = self._runtime_registry.get(prompt_bundle_id)
        except PromptBundleNotDeployed as exc:
            raise PromptBundleNotVisible(prompt_bundle_id) from exc
        return self._builtin_view(identity, definition, references)

    def create(
        self,
        identity: IdentityContext,
        *,
        prompt_bundle_id: str,
        name: str,
        bundle_version: str,
        task_type: str,
        content_hash: str,
        source_commit: str,
        source_path: str,
        section_names: Sequence[str],
        required_variables: Sequence[str],
        output_schema_name: str,
        compatible_model_aliases: Sequence[str],
        change_summary: str,
        idempotency_key: str,
        request_id: str,
    ) -> PromptBundleView:
        self._require(identity, Action.MANAGE_PROMPT_BUNDLE, prompt_bundle_id, request_id)
        document = _validated_document(
            prompt_bundle_id=prompt_bundle_id,
            name=name,
            bundle_version=bundle_version,
            task_type=task_type,
            content_hash=content_hash,
            source_commit=source_commit,
            source_path=source_path,
            section_names=section_names,
            required_variables=required_variables,
            output_schema_name=output_schema_name,
            compatible_model_aliases=compatible_model_aliases,
            change_summary=change_summary,
        )
        try:
            self._runtime_registry.get(prompt_bundle_id)
        except PromptBundleNotDeployed:
            pass
        else:
            raise PromptBundleConflict("runtime_prompt_bundle_id_already_exists")
        now = datetime.now(UTC)
        request_hash = _digest({**document, "idempotency_key": idempotency_key})
        with self._database.transaction(identity.tenant_context) as session:
            existing = session.scalar(
                select(PromptBundleRecord).where(
                    PromptBundleRecord.tenant_id == identity.tenant_id,
                    PromptBundleRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if _record_document_hash(existing, idempotency_key) != request_hash:
                    raise PromptBundleConflict("prompt_bundle_idempotency_key_reused")
                return self._record_view(
                    identity,
                    existing,
                    _transitions_for(session, identity.tenant_id, existing.prompt_bundle_id),
                    _release_references(session, identity.tenant_id),
                )
            duplicate = session.scalar(
                select(PromptBundleRecord).where(
                    PromptBundleRecord.tenant_id == identity.tenant_id,
                    (
                        (PromptBundleRecord.prompt_bundle_id == prompt_bundle_id)
                        | (
                            (PromptBundleRecord.name == name)
                            & (PromptBundleRecord.bundle_version == bundle_version)
                        )
                    ),
                )
            )
            if duplicate is not None:
                raise PromptBundleConflict("prompt_bundle_version_already_registered")
            record = PromptBundleRecord(
                tenant_id=identity.tenant_id,
                idempotency_key=idempotency_key,
                status="DRAFT",
                evaluation_id=None,
                evaluation_report_hash=None,
                created_by_subject_id=identity.subject_id,
                reviewed_by_subject_id=None,
                review_reason=None,
                reviewed_at=None,
                version=1,
                created_at=now,
                updated_at=now,
                **document,
            )
            session.add(record)
            session.flush()
            _append_transition(
                session,
                record,
                from_status="NONE",
                to_status="DRAFT",
                actor_subject_id=identity.subject_id,
                reason_code="prompt_bundle_registered",
                evidence={
                    "content_hash": content_hash,
                    "source_commit": source_commit,
                    "source_path": source_path,
                },
                occurred_at=now,
            )
            session.flush()
            return self._record_view(
                identity,
                record,
                _transitions_for(session, identity.tenant_id, prompt_bundle_id),
                {},
            )

    def submit(
        self,
        identity: IdentityContext,
        prompt_bundle_id: str,
        *,
        evaluation_id: str,
        expected_version: int,
        request_id: str,
    ) -> PromptBundleView:
        self._require(identity, Action.MANAGE_PROMPT_BUNDLE, prompt_bundle_id, request_id)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = _record_or_hidden(session, identity.tenant_id, prompt_bundle_id)
            _require_version(record, expected_version)
            if record.status not in {"DRAFT", "REJECTED"}:
                raise PromptBundleConflict("prompt_bundle_not_submittable", record.version)
            evaluation = _bound_evaluation(
                session,
                identity.tenant_id,
                evaluation_id,
                record.prompt_bundle_id,
                record.content_hash,
            )
            previous = record.status
            record.status = "REVIEW_PENDING"
            record.evaluation_id = evaluation.evaluation_id
            record.evaluation_report_hash = evaluation.report_hash
            record.reviewed_by_subject_id = None
            record.review_reason = None
            record.reviewed_at = None
            record.version += 1
            record.updated_at = now
            _append_transition(
                session,
                record,
                from_status=previous,
                to_status="REVIEW_PENDING",
                actor_subject_id=identity.subject_id,
                reason_code="prompt_evaluation_bound",
                evidence={
                    "evaluation_id": evaluation.evaluation_id,
                    "evaluation_report_hash": evaluation.report_hash,
                    "content_hash": record.content_hash,
                },
                occurred_at=now,
            )
            session.flush()
            return self._record_view(
                identity,
                record,
                _transitions_for(session, identity.tenant_id, prompt_bundle_id),
                _release_references(session, identity.tenant_id),
            )

    def review(
        self,
        identity: IdentityContext,
        prompt_bundle_id: str,
        *,
        decision: PromptReviewDecision,
        reason: str,
        expected_version: int,
        request_id: str,
    ) -> PromptBundleView:
        self._require(identity, Action.REVIEW_PROMPT_BUNDLE, prompt_bundle_id, request_id)
        normalized_reason = reason.strip()
        if not 8 <= len(normalized_reason) <= 1_000:
            raise ValueError("prompt_review_reason_invalid")
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = _record_or_hidden(session, identity.tenant_id, prompt_bundle_id)
            _require_version(record, expected_version)
            if record.status != "REVIEW_PENDING":
                raise PromptBundleConflict("prompt_bundle_not_reviewable", record.version)
            if record.created_by_subject_id == identity.subject_id:
                raise PromptBundleGateDenied("prompt_bundle_self_review_forbidden")
            if record.evaluation_id is None:
                raise PromptBundleGateDenied("prompt_bundle_evaluation_missing")
            evaluation = _bound_evaluation(
                session,
                identity.tenant_id,
                record.evaluation_id,
                record.prompt_bundle_id,
                record.content_hash,
            )
            record.status = decision
            record.reviewed_by_subject_id = identity.subject_id
            record.review_reason = normalized_reason
            record.reviewed_at = now
            record.version += 1
            record.updated_at = now
            _append_transition(
                session,
                record,
                from_status="REVIEW_PENDING",
                to_status=decision,
                actor_subject_id=identity.subject_id,
                reason_code=f"prompt_bundle_{decision.lower()}",
                evidence={
                    "content_hash": record.content_hash,
                    "evaluation_id": evaluation.evaluation_id,
                    "evaluation_report_hash": evaluation.report_hash,
                },
                occurred_at=now,
            )
            session.flush()
            return self._record_view(
                identity,
                record,
                _transitions_for(session, identity.tenant_id, prompt_bundle_id),
                _release_references(session, identity.tenant_id),
            )

    def retire(
        self,
        identity: IdentityContext,
        prompt_bundle_id: str,
        *,
        reason: str,
        expected_version: int,
        request_id: str,
    ) -> PromptBundleView:
        self._require(identity, Action.REVIEW_PROMPT_BUNDLE, prompt_bundle_id, request_id)
        normalized_reason = reason.strip()
        if not 8 <= len(normalized_reason) <= 1_000:
            raise ValueError("prompt_retirement_reason_invalid")
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = _record_or_hidden(session, identity.tenant_id, prompt_bundle_id)
            _require_version(record, expected_version)
            if record.status != "APPROVED":
                raise PromptBundleConflict("prompt_bundle_not_retirable", record.version)
            references = _release_references(session, identity.tenant_id)
            if references.get(prompt_bundle_id, (0, False))[1]:
                raise PromptBundleGateDenied("production_release_still_uses_prompt_bundle")
            record.status = "RETIRED"
            record.reviewed_by_subject_id = identity.subject_id
            record.review_reason = normalized_reason
            record.reviewed_at = now
            record.version += 1
            record.updated_at = now
            _append_transition(
                session,
                record,
                from_status="APPROVED",
                to_status="RETIRED",
                actor_subject_id=identity.subject_id,
                reason_code="prompt_bundle_retired",
                evidence={"content_hash": record.content_hash},
                occurred_at=now,
            )
            session.flush()
            return self._record_view(
                identity,
                record,
                _transitions_for(session, identity.tenant_id, prompt_bundle_id),
                references,
            )

    def _record_view(
        self,
        identity: IdentityContext,
        record: PromptBundleRecord,
        transitions: tuple[PromptBundleTransitionRecord, ...],
        references: dict[str, tuple[int, bool]],
    ) -> PromptBundleView:
        runtime = _runtime_or_none(self._runtime_registry, record.prompt_bundle_id)
        reference_count, active = references.get(record.prompt_bundle_id, (0, False))
        return PromptBundleView(
            prompt_bundle_id=record.prompt_bundle_id,
            origin="REGISTERED",
            name=record.name,
            bundle_version=record.bundle_version,
            task_type=record.task_type,
            status=record.status,
            content_hash=record.content_hash,
            source_commit=record.source_commit,
            source_path=record.source_path,
            section_names=tuple(record.section_names_json),
            required_variables=tuple(record.required_variables_json),
            output_schema_name=record.output_schema_name,
            compatible_model_aliases=tuple(record.compatible_model_aliases_json),
            change_summary=record.change_summary,
            evaluation_id=record.evaluation_id,
            evaluation_report_hash=record.evaluation_report_hash,
            created_by_subject_id=record.created_by_subject_id,
            reviewed_by_subject_id=record.reviewed_by_subject_id,
            review_reason=record.review_reason,
            reviewed_at=_optional_utc(record.reviewed_at),
            runtime_available=runtime is not None,
            runtime_hash_matches=(
                runtime is not None and runtime.content_hash == record.content_hash
            ),
            release_reference_count=reference_count,
            active_production_reference=active,
            version=record.version,
            created_at=_utc(record.created_at),
            updated_at=_utc(record.updated_at),
            legal_actions=_legal_actions(identity, self._authorizer, record, active),
            transitions=tuple(_transition(item) for item in transitions),
        )

    def _builtin_view(
        self,
        identity: IdentityContext,
        definition: PromptBundleDefinition,
        references: dict[str, tuple[int, bool]],
    ) -> PromptBundleView:
        del identity
        reference_count, active = references.get(definition.prompt_bundle_id, (0, False))
        return PromptBundleView(
            prompt_bundle_id=definition.prompt_bundle_id,
            origin="BUILTIN",
            name=definition.name,
            bundle_version=definition.version,
            task_type=definition.task_type,
            status="DEPLOYED",
            content_hash=definition.content_hash,
            source_commit="runtime-image-bound",
            source_path=definition.source_path,
            section_names=definition.section_names,
            required_variables=definition.required_variables,
            output_schema_name=definition.output_schema_name,
            compatible_model_aliases=definition.compatible_model_aliases,
            change_summary="随当前服务镜像部署的内置生产 Prompt Bundle",
            evaluation_id=None,
            evaluation_report_hash=None,
            created_by_subject_id="runtime-image",
            reviewed_by_subject_id=None,
            review_reason=None,
            reviewed_at=None,
            runtime_available=True,
            runtime_hash_matches=True,
            release_reference_count=reference_count,
            active_production_reference=active,
            version=0,
            created_at=None,
            updated_at=None,
            legal_actions=(),
            transitions=(),
        )

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


def _validated_document(
    *,
    prompt_bundle_id: str,
    name: str,
    bundle_version: str,
    task_type: str,
    content_hash: str,
    source_commit: str,
    source_path: str,
    section_names: Sequence[str],
    required_variables: Sequence[str],
    output_schema_name: str,
    compatible_model_aliases: Sequence[str],
    change_summary: str,
) -> dict[str, Any]:
    aliases = list(compatible_model_aliases)
    section_names = list(section_names)
    required_variables = list(required_variables)
    change_summary = change_summary.strip()
    if not _IDENTIFIER.fullmatch(prompt_bundle_id) or not _IDENTIFIER.fullmatch(name):
        raise ValueError("prompt_bundle_identity_invalid")
    if not _VERSION.fullmatch(bundle_version):
        raise ValueError("prompt_bundle_version_invalid")
    if task_type not in _TASK_TYPES:
        raise ValueError("prompt_bundle_task_type_invalid")
    if not _SHA256.fullmatch(content_hash):
        raise ValueError("prompt_bundle_content_hash_invalid")
    if not _COMMIT.fullmatch(source_commit):
        raise ValueError("prompt_bundle_source_commit_invalid")
    if (
        not source_path
        or source_path.startswith(("/", "~"))
        or ".." in source_path.split("/")
        or len(source_path) > 512
    ):
        raise ValueError("prompt_bundle_source_path_invalid")
    if (
        len(section_names) < 2
        or len(section_names) > 16
        or len(set(section_names)) != len(section_names)
        or any(not _VARIABLE.fullmatch(item) for item in section_names)
        or not {"system_core", "output_schema"}.issubset(section_names)
    ):
        raise ValueError("prompt_bundle_sections_invalid")
    if (
        not required_variables
        or len(required_variables) > 32
        or len(set(required_variables)) != len(required_variables)
        or any(not _VARIABLE.fullmatch(item) for item in required_variables)
    ):
        raise ValueError("prompt_bundle_variables_invalid")
    if not _VARIABLE.fullmatch(output_schema_name):
        raise ValueError("prompt_bundle_output_schema_invalid")
    if (
        not aliases
        or len(aliases) > 16
        or len(set(aliases)) != len(aliases)
        or any(not _IDENTIFIER.fullmatch(item) for item in aliases)
    ):
        raise ValueError("prompt_bundle_model_aliases_invalid")
    if not 8 <= len(change_summary) <= 1_000:
        raise ValueError("prompt_bundle_change_summary_invalid")
    return {
        "prompt_bundle_id": prompt_bundle_id,
        "name": name,
        "bundle_version": bundle_version,
        "task_type": task_type,
        "content_hash": content_hash,
        "source_commit": source_commit,
        "source_path": source_path,
        "section_names_json": section_names,
        "required_variables_json": required_variables,
        "output_schema_name": output_schema_name,
        "compatible_model_aliases_json": aliases,
        "change_summary": change_summary,
    }


def _bound_evaluation(
    session: Session,
    tenant_id: str,
    evaluation_id: str,
    prompt_bundle_id: str,
    content_hash: str,
) -> ModelEvaluationRunRecord:
    evaluation = session.scalar(
        select(ModelEvaluationRunRecord).where(
            ModelEvaluationRunRecord.tenant_id == tenant_id,
            ModelEvaluationRunRecord.evaluation_id == evaluation_id,
        )
    )
    if evaluation is None:
        raise PromptBundleGateDenied("prompt_evaluation_not_visible")
    if (
        evaluation.status != "COMPLETED"
        or evaluation.decision not in {"CANDIDATE", "SMOKE_PASSED"}
        or not evaluation.hard_gate_results
        or not all(evaluation.hard_gate_results.values())
    ):
        raise PromptBundleGateDenied("prompt_evaluation_gate_not_passed")
    binding = session.scalar(
        select(EvaluationArtifactRecord.artifact_id).where(
            EvaluationArtifactRecord.tenant_id == tenant_id,
            EvaluationArtifactRecord.evaluation_id == evaluation_id,
            EvaluationArtifactRecord.kind == "case_evidence",
            EvaluationArtifactRecord.metadata_json["prompt_bundle_id"].as_string()
            == prompt_bundle_id,
            EvaluationArtifactRecord.metadata_json["prompt_content_hash"].as_string()
            == content_hash,
        )
    )
    if binding is None:
        raise PromptBundleGateDenied("prompt_evaluation_binding_missing")
    return evaluation


def _record_or_none(
    session: Session,
    tenant_id: str,
    prompt_bundle_id: str,
) -> PromptBundleRecord | None:
    return session.scalar(
        select(PromptBundleRecord).where(
            PromptBundleRecord.tenant_id == tenant_id,
            PromptBundleRecord.prompt_bundle_id == prompt_bundle_id,
        )
    )


def _record_or_hidden(
    session: Session,
    tenant_id: str,
    prompt_bundle_id: str,
) -> PromptBundleRecord:
    record = _record_or_none(session, tenant_id, prompt_bundle_id)
    if record is None:
        raise PromptBundleNotVisible(prompt_bundle_id)
    return record


def _require_version(record: PromptBundleRecord, expected_version: int) -> None:
    if record.version != expected_version:
        raise PromptBundleConflict("prompt_bundle_version_conflict", record.version)


def _runtime_or_none(
    registry: PromptBundleRegistry,
    prompt_bundle_id: str,
) -> PromptBundleDefinition | None:
    try:
        return registry.get(prompt_bundle_id)
    except PromptBundleNotDeployed:
        return None


def _release_references(session: Session, tenant_id: str) -> dict[str, tuple[int, bool]]:
    references: dict[str, tuple[int, bool]] = {}
    releases = session.scalars(
        select(ModelReleaseRecord).where(ModelReleaseRecord.tenant_id == tenant_id)
    )
    for release in releases:
        prompt_bundle_id = release.manifest_json.get("prompt_bundle_id")
        if not isinstance(prompt_bundle_id, str):
            continue
        count, active = references.get(prompt_bundle_id, (0, False))
        references[prompt_bundle_id] = (
            count + 1,
            active or release.status == "PRODUCTION",
        )
    return references


def _transitions_by_bundle(
    session: Session,
    tenant_id: str,
) -> dict[str, tuple[PromptBundleTransitionRecord, ...]]:
    grouped: dict[str, list[PromptBundleTransitionRecord]] = {}
    for item in session.scalars(
        select(PromptBundleTransitionRecord)
        .where(PromptBundleTransitionRecord.tenant_id == tenant_id)
        .order_by(
            PromptBundleTransitionRecord.prompt_bundle_id,
            PromptBundleTransitionRecord.sequence,
        )
    ):
        grouped.setdefault(item.prompt_bundle_id, []).append(item)
    return {key: tuple(value) for key, value in grouped.items()}


def _transitions_for(
    session: Session,
    tenant_id: str,
    prompt_bundle_id: str,
) -> tuple[PromptBundleTransitionRecord, ...]:
    return tuple(
        session.scalars(
            select(PromptBundleTransitionRecord)
            .where(
                PromptBundleTransitionRecord.tenant_id == tenant_id,
                PromptBundleTransitionRecord.prompt_bundle_id == prompt_bundle_id,
            )
            .order_by(PromptBundleTransitionRecord.sequence)
        )
    )


def _append_transition(
    session: Session,
    record: PromptBundleRecord,
    *,
    from_status: str,
    to_status: str,
    actor_subject_id: str,
    reason_code: str,
    evidence: dict[str, object],
    occurred_at: datetime,
) -> None:
    sequence = (
        int(
            session.scalar(
                select(PromptBundleTransitionRecord.sequence)
                .where(
                    PromptBundleTransitionRecord.tenant_id == record.tenant_id,
                    PromptBundleTransitionRecord.prompt_bundle_id == record.prompt_bundle_id,
                )
                .order_by(PromptBundleTransitionRecord.sequence.desc())
                .limit(1)
            )
            or 0
        )
        + 1
    )
    session.add(
        PromptBundleTransitionRecord(
            tenant_id=record.tenant_id,
            transition_id=f"prompt-transition-{uuid4().hex}",
            prompt_bundle_id=record.prompt_bundle_id,
            sequence=sequence,
            from_status=from_status,
            to_status=to_status,
            actor_subject_id=actor_subject_id,
            reason_code=reason_code,
            evidence_json=evidence,
            evidence_hash=_digest(evidence),
            occurred_at=occurred_at,
            created_at=occurred_at,
            updated_at=occurred_at,
        )
    )


def _transition(record: PromptBundleTransitionRecord) -> PromptTransition:
    return PromptTransition(
        transition_id=record.transition_id,
        sequence=record.sequence,
        from_status=record.from_status,
        to_status=record.to_status,
        actor_subject_id=record.actor_subject_id,
        reason_code=record.reason_code,
        evidence=dict(record.evidence_json),
        evidence_hash=record.evidence_hash,
        occurred_at=_utc(record.occurred_at),
    )


def _legal_actions(
    identity: IdentityContext,
    authorizer: Authorizer,
    record: PromptBundleRecord,
    active_production_reference: bool,
) -> tuple[str, ...]:
    resource = ResourceContext(identity.tenant_id, record.prompt_bundle_id)
    can_manage = authorizer.decide(identity, Action.MANAGE_PROMPT_BUNDLE, resource).allowed
    can_review = authorizer.decide(identity, Action.REVIEW_PROMPT_BUNDLE, resource).allowed
    actions: list[str] = []
    if can_manage and record.status in {"DRAFT", "REJECTED"}:
        actions.append("SUBMIT")
    if (
        can_review
        and record.status == "REVIEW_PENDING"
        and record.created_by_subject_id != identity.subject_id
    ):
        actions.extend(("APPROVE", "REJECT"))
    if can_review and record.status == "APPROVED" and not active_production_reference:
        actions.append("RETIRE")
    return tuple(actions)


def _record_document_hash(record: PromptBundleRecord, idempotency_key: str) -> str:
    return _digest(
        {
            "prompt_bundle_id": record.prompt_bundle_id,
            "name": record.name,
            "bundle_version": record.bundle_version,
            "task_type": record.task_type,
            "content_hash": record.content_hash,
            "source_commit": record.source_commit,
            "source_path": record.source_path,
            "section_names_json": record.section_names_json,
            "required_variables_json": record.required_variables_json,
            "output_schema_name": record.output_schema_name,
            "compatible_model_aliases_json": record.compatible_model_aliases_json,
            "change_summary": record.change_summary,
            "idempotency_key": idempotency_key,
        }
    )


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{sha256(encoded).hexdigest()}"


def _optional_utc(value: datetime | None) -> datetime | None:
    return _utc(value) if value is not None else None
