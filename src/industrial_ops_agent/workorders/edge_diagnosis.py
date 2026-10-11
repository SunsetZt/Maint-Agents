"""Server-authoritative lifecycle for disconnected FIELD-008 diagnosis candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Literal
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.domain import as_utc as _utc
from industrial_ops_agent.edge.contracts import (
    FieldEdgeAssetBinding,
    FieldEdgeAssignmentBinding,
    FieldEdgeDiagnosisCandidate,
    FieldEdgeDiagnosisPackClaims,
    FieldEdgeDiagnosisResult,
    FieldEdgeInferenceConfig,
    FieldEdgeOfflinePackBinding,
    FieldEdgePromptBinding,
    FieldEdgeReleaseBinding,
    FieldEdgeRetrievalBinding,
    FieldEdgeRuntimeEvidence,
    FieldEdgeSnapshot,
    FieldEdgeWorkOrderBinding,
    canonical_digest,
)
from industrial_ops_agent.edge.signing import Ed25519PackSigner, FieldEdgeSignatureError
from industrial_ops_agent.guardrails.prompt_injection import PromptInjectionGuard
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    IndexReleaseRecord,
    ModelReleaseRecord,
    PromptBundleRecord,
    WorkOrderEdgeDiagnosisCandidateRecord,
    WorkOrderEdgeDiagnosisPackRecord,
    WorkOrderOfflinePackRecord,
)
from industrial_ops_agent.prompting.registry import (
    DEFAULT_PROMPT_BUNDLE_ID,
    PromptBundleNotDeployed,
    default_prompt_registry,
)
from industrial_ops_agent.workorders.offline_pack import (
    OfflinePackNotAvailable,
    WorkOrderOfflinePackService,
    WorkOrderOfflinePackView,
)
from industrial_ops_agent.workorders.service import WorkOrderConflict

EDGE_PACK_TTL = timedelta(minutes=30)


class EdgeDiagnosisUnavailable(Exception):
    """FIELD-008 is disabled or lacks one unambiguous governed runtime."""

    def __init__(self, reason: str = "edge_diagnosis_unavailable") -> None:
        self.reason = reason
        super().__init__(reason)


class EdgeDiagnosisNotVisible(Exception):
    """No candidate or pack visible to this subject."""


class EdgeDiagnosisConflict(Exception):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


class EdgeDiagnosisRejected(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class _ReleaseSelection:
    release: FieldEdgeReleaseBinding
    prompt: FieldEdgePromptBinding
    retrieval: FieldEdgeRetrievalBinding


class WorkOrderEdgeDiagnosisService:
    """Sign from current authority, then distrust and revalidate every result."""

    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        signer: Ed25519PackSigner | None,
        *,
        enabled: bool,
        injection_guard: PromptInjectionGuard | None = None,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._signer = signer
        self._enabled = enabled
        self._guard = injection_guard or PromptInjectionGuard()
        self._offline_packs = WorkOrderOfflinePackService(database, authorizer)

    def issue(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        expected_work_order_version: int,
        idempotency_key: str,
        request_id: str,
    ) -> tuple[WorkOrderEdgeDiagnosisPackRecord, bool]:
        signer = self._require_signer()
        offline = self._offline_packs.current(
            identity,
            work_order_id,
            request_id=request_id,
        )
        if (
            offline.work_order_version != expected_work_order_version
            or offline.snapshot.get("work_order", {}).get("status")
            not in {"ACCEPTED", "IN_PROGRESS", "ON_HOLD"}
        ):
            raise WorkOrderConflict("version_conflict", offline.work_order_version)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            current = _current_offline_pack(session, identity, work_order_id, offline)
            selection = _select_release(session, identity.tenant_id)
            release = selection.release
            request_digest = canonical_digest(
                {
                    "work_order_id": work_order_id,
                    "work_order_version": expected_work_order_version,
                    "offline_pack_id": current.pack_id,
                    "offline_pack_content_hash": current.content_hash,
                    "release": release.model_dump(mode="json"),
                    "key_id": signer.key_id,
                }
            )
            existing = session.scalar(
                select(WorkOrderEdgeDiagnosisPackRecord).where(
                    WorkOrderEdgeDiagnosisPackRecord.tenant_id == identity.tenant_id,
                    WorkOrderEdgeDiagnosisPackRecord.subject_id == identity.subject_id,
                    WorkOrderEdgeDiagnosisPackRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if existing.request_digest != request_digest:
                    raise EdgeDiagnosisConflict(
                        "edge_diagnosis_idempotency_conflict",
                        existing.version,
                    )
                return existing, False

            expires_at = min(_utc(current.expires_at), now + EDGE_PACK_TTL)
            if expires_at <= now + timedelta(seconds=1):
                raise OfflinePackNotAvailable
            prompt = selection.prompt
            retrieval = selection.retrieval
            pack_id = f"edge-pack-{uuid4().hex}"
            epoch = int(now.timestamp())
            claims = FieldEdgeDiagnosisPackClaims(
                sub=identity.subject_id,
                tenant_id=identity.tenant_id,
                iat=epoch,
                nbf=epoch,
                exp=int(expires_at.timestamp()),
                jti=pack_id,
                work_order=FieldEdgeWorkOrderBinding(
                    work_order_id=work_order_id,
                    version=current.work_order_version,
                    status=str(current.snapshot_json["work_order"]["status"]),
                ),
                asset=FieldEdgeAssetBinding(
                    asset_id=current.asset_id,
                    version=current.asset_version,
                ),
                assignment=FieldEdgeAssignmentBinding(
                    assignment_id=current.assignment_id,
                    assigned_at=_utc(current.assignment_assigned_at),
                ),
                offline_pack=FieldEdgeOfflinePackBinding(
                    pack_id=current.pack_id,
                    content_hash=current.content_hash,
                    expires_at=_utc(current.expires_at),
                    snapshot=FieldEdgeSnapshot.model_validate(current.snapshot_json),
                ),
                release=release,
                prompt_bundle=prompt,
                retrieval=retrieval,
            )
            token = signer.sign(claims)
            for prior in session.scalars(
                select(WorkOrderEdgeDiagnosisPackRecord)
                .where(
                    WorkOrderEdgeDiagnosisPackRecord.tenant_id == identity.tenant_id,
                    WorkOrderEdgeDiagnosisPackRecord.subject_id == identity.subject_id,
                    WorkOrderEdgeDiagnosisPackRecord.work_order_id == work_order_id,
                    WorkOrderEdgeDiagnosisPackRecord.status == "ACTIVE",
                )
                .with_for_update()
            ):
                prior.status = "SUPERSEDED"
                prior.version += 1
                prior.updated_at = now
            record = WorkOrderEdgeDiagnosisPackRecord(
                pack_id=pack_id,
                tenant_id=identity.tenant_id,
                subject_id=identity.subject_id,
                work_order_id=work_order_id,
                work_order_version=current.work_order_version,
                asset_id=current.asset_id,
                asset_version=current.asset_version,
                assignment_id=current.assignment_id,
                assignment_assigned_at=_utc(current.assignment_assigned_at),
                offline_pack_id=current.pack_id,
                offline_pack_content_hash=current.content_hash,
                release_id=release.release_id,
                manifest_hash=release.manifest_hash,
                model_file=release.model_file,
                model_content_hash=release.model_content_hash,
                prompt_bundle_id=prompt.prompt_bundle_id,
                prompt_bundle_hash=prompt.content_hash,
                index_release_id=retrieval.index_release_id,
                index_content_checksum=retrieval.content_checksum,
                key_id=signer.key_id,
                claims_json=claims.model_dump(mode="json"),
                signed_token=token,
                pack_digest=_token_digest(token),
                idempotency_key=idempotency_key,
                request_digest=request_digest,
                status="ACTIVE",
                issued_at=now,
                expires_at=expires_at,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.flush()
            return record, True

    def import_result(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        client_operation_id: str,
        result: FieldEdgeDiagnosisResult,
        request_id: str,
    ) -> tuple[WorkOrderEdgeDiagnosisCandidateRecord, bool]:
        signer = self._require_signer()
        if not result.has_valid_content_hash():
            raise EdgeDiagnosisRejected("edge_diagnosis_result_hash_invalid")
        try:
            claims = signer.verify(result.pack_token)
        except FieldEdgeSignatureError as exc:
            raise EdgeDiagnosisRejected("edge_diagnosis_signature_invalid") from exc
        if (
            claims.tenant_id != identity.tenant_id
            or claims.sub != identity.subject_id
            or claims.work_order.work_order_id != work_order_id
        ):
            raise EdgeDiagnosisNotVisible
        offline = self._offline_packs.current(
            identity,
            work_order_id,
            request_id=request_id,
        )
        _validate_result_bindings(claims, result)
        allowed_citations = {
            item.citation_id for item in claims.offline_pack.snapshot.citations
        }
        if not set(result.candidate.citation_ids).issubset(allowed_citations):
            raise EdgeDiagnosisRejected("edge_diagnosis_citation_out_of_scope")
        guard = self._guard.inspect_output(result.candidate.model_dump(mode="json"))
        if guard.decision != "ALLOWED":
            raise EdgeDiagnosisRejected("edge_diagnosis_output_rejected")

        now = datetime.now(UTC)
        request_digest = canonical_digest(result.model_dump(mode="json"))
        with self._database.transaction(identity.tenant_context) as session:
            pack = session.scalar(
                select(WorkOrderEdgeDiagnosisPackRecord)
                .where(
                    WorkOrderEdgeDiagnosisPackRecord.tenant_id == identity.tenant_id,
                    WorkOrderEdgeDiagnosisPackRecord.subject_id == identity.subject_id,
                    WorkOrderEdgeDiagnosisPackRecord.work_order_id == work_order_id,
                    WorkOrderEdgeDiagnosisPackRecord.pack_id == claims.jti,
                )
                .with_for_update()
            )
            if pack is None:
                raise EdgeDiagnosisNotVisible
            _validate_pack_record(pack, claims, result.pack_token, now)
            current = _current_offline_pack(session, identity, work_order_id, offline)
            _validate_authority_bindings(pack, current)
            current_selection = _select_release(session, identity.tenant_id)
            if (
                current_selection.release != claims.release
                or current_selection.prompt != claims.prompt_bundle
                or current_selection.retrieval != claims.retrieval
            ):
                raise EdgeDiagnosisConflict("edge_diagnosis_release_binding_changed")
            existing = session.scalar(
                select(WorkOrderEdgeDiagnosisCandidateRecord).where(
                    WorkOrderEdgeDiagnosisCandidateRecord.tenant_id == identity.tenant_id,
                    WorkOrderEdgeDiagnosisCandidateRecord.subject_id == identity.subject_id,
                    WorkOrderEdgeDiagnosisCandidateRecord.client_operation_id
                    == client_operation_id,
                )
            )
            if existing is not None:
                if existing.request_digest != request_digest:
                    raise EdgeDiagnosisConflict(
                        "edge_diagnosis_idempotency_conflict",
                        existing.version,
                    )
                return existing, False
            record = WorkOrderEdgeDiagnosisCandidateRecord(
                candidate_id=f"edge-candidate-{uuid4().hex}",
                tenant_id=identity.tenant_id,
                subject_id=identity.subject_id,
                work_order_id=work_order_id,
                pack_id=pack.pack_id,
                client_operation_id=client_operation_id,
                request_digest=request_digest,
                result_content_hash=result.content_hash,
                candidate_json=result.candidate.model_dump(mode="json"),
                runtime_evidence_json=result.runtime_evidence.model_dump(mode="json"),
                security_policy_version=guard.policy_version,
                status="PENDING_REVIEW",
                reviewed_by_subject_id=None,
                review_reason=None,
                reviewed_at=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.flush()
            return record, True

    def list_candidates(
        self,
        identity: IdentityContext,
        work_order_id: str,
        *,
        request_id: str,
    ) -> list[WorkOrderEdgeDiagnosisCandidateRecord]:
        self._require_signer()
        self._offline_packs.current(identity, work_order_id, request_id=request_id)
        with self._database.transaction(identity.tenant_context) as session:
            records = session.scalars(
                select(WorkOrderEdgeDiagnosisCandidateRecord)
                .where(
                    WorkOrderEdgeDiagnosisCandidateRecord.tenant_id == identity.tenant_id,
                    WorkOrderEdgeDiagnosisCandidateRecord.subject_id == identity.subject_id,
                    WorkOrderEdgeDiagnosisCandidateRecord.work_order_id == work_order_id,
                )
                .order_by(
                    WorkOrderEdgeDiagnosisCandidateRecord.created_at.desc(),
                    WorkOrderEdgeDiagnosisCandidateRecord.candidate_id.desc(),
                )
            )
            return list(records)

    def decide(
        self,
        identity: IdentityContext,
        work_order_id: str,
        candidate_id: str,
        *,
        expected_version: int,
        decision: Literal["ACCEPTED", "REJECTED"],
        reason: str,
        request_id: str,
    ) -> WorkOrderEdgeDiagnosisCandidateRecord:
        self._require_signer()
        self._offline_packs.current(identity, work_order_id, request_id=request_id)
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            record = session.scalar(
                select(WorkOrderEdgeDiagnosisCandidateRecord)
                .where(
                    WorkOrderEdgeDiagnosisCandidateRecord.tenant_id == identity.tenant_id,
                    WorkOrderEdgeDiagnosisCandidateRecord.subject_id == identity.subject_id,
                    WorkOrderEdgeDiagnosisCandidateRecord.work_order_id == work_order_id,
                    WorkOrderEdgeDiagnosisCandidateRecord.candidate_id == candidate_id,
                )
                .with_for_update()
            )
            if record is None:
                raise EdgeDiagnosisNotVisible
            if record.status != "PENDING_REVIEW":
                if (
                    record.status == decision
                    and record.reviewed_by_subject_id == identity.subject_id
                    and record.review_reason == reason
                ):
                    return record
                raise EdgeDiagnosisConflict("edge_diagnosis_candidate_already_decided", record.version)
            if record.version != expected_version:
                raise EdgeDiagnosisConflict("edge_diagnosis_version_conflict", record.version)
            record.status = decision
            record.reviewed_by_subject_id = identity.subject_id
            record.review_reason = reason
            record.reviewed_at = now
            record.version += 1
            record.updated_at = now
            session.flush()
            return record

    def _require_signer(self) -> Ed25519PackSigner:
        if not self._enabled or self._signer is None:
            raise EdgeDiagnosisUnavailable
        return self._signer


def _select_release(session: Session, tenant_id: str) -> _ReleaseSelection:
    selections: list[_ReleaseSelection] = []
    records = session.scalars(
        select(ModelReleaseRecord)
        .where(
            ModelReleaseRecord.tenant_id == tenant_id,
            ModelReleaseRecord.status == "PRODUCTION",
            ModelReleaseRecord.target_environment == "PRODUCTION",
        )
        .order_by(ModelReleaseRecord.created_at.desc(), ModelReleaseRecord.release_id.desc())
    )
    for record in records:
        try:
            if canonical_digest(record.manifest_json) != record.manifest_hash:
                continue
            manifest = record.manifest_json
            runtime = manifest["runtime"]
            config = runtime["inference_config"]
            quantization = manifest["quantization"]
            metadata = quantization["metadata"]
            model_runtime = metadata["runtime"]
            if (
                config.get("engine") != "llama.cpp"
                or metadata.get("serialization") != "gguf"
                or model_runtime.get("target_runtime") != "llama.cpp"
                or manifest.get("target_environment", "PRODUCTION") != "PRODUCTION"
            ):
                continue
            model_hash = _sha256_digest(str(quantization["content_hash"]))
            inference = FieldEdgeInferenceConfig.model_validate(
                {
                    **config,
                    "output_tokens": 512,
                    "timeout_seconds": 120,
                    "seed": 42,
                    "temperature": 0,
                }
            )
            binding = FieldEdgeReleaseBinding(
                release_id=record.release_id,
                manifest_hash=record.manifest_hash,
                runtime_profile_id=str(runtime["profile_id"]),
                model_file=str(model_runtime["model_file"]),
                model_content_hash=model_hash,
                inference_config=inference,
            )
            prompt = _prompt_binding_from_manifest(manifest)
            retrieval = _retrieval_binding_from_manifest(manifest)
            index = session.scalar(
                select(IndexReleaseRecord).where(
                    IndexReleaseRecord.tenant_id == tenant_id,
                    IndexReleaseRecord.release_id == retrieval.index_release_id,
                )
            )
            if (
                index is None
                or index.status not in {"PUBLISHED", "ACTIVE"}
                or not index.is_active
                or index.content_checksum != retrieval.content_checksum
            ):
                continue
            definition = default_prompt_registry().get(prompt.prompt_bundle_id)
            if definition.content_hash != prompt.content_hash or not _prompt_is_current(
                session,
                tenant_id,
                prompt,
                manifest,
            ):
                continue
            selections.append(
                _ReleaseSelection(
                    release=binding,
                    prompt=prompt,
                    retrieval=retrieval,
                )
            )
        except (KeyError, TypeError, ValueError, PromptBundleNotDeployed):
            continue
    if len(selections) != 1:
        reason = (
            "edge_diagnosis_release_ambiguous"
            if len(selections) > 1
            else "edge_diagnosis_release_unavailable"
        )
        raise EdgeDiagnosisUnavailable(reason)
    return selections[0]


def _prompt_binding_from_manifest(manifest: dict[str, object]) -> FieldEdgePromptBinding:
    raw = manifest["prompt_bundle"]
    if not isinstance(raw, dict):
        raise ValueError("field_edge_prompt_binding_invalid")
    return FieldEdgePromptBinding(
        prompt_bundle_id=str(raw["id"]),
        content_hash=str(raw["content_hash"]),
    )


def _prompt_is_current(
    session: Session,
    tenant_id: str,
    binding: FieldEdgePromptBinding,
    manifest: dict[str, object],
) -> bool:
    raw = manifest.get("prompt_bundle")
    if not isinstance(raw, dict):
        return False
    record = session.scalar(
        select(PromptBundleRecord).where(
            PromptBundleRecord.tenant_id == tenant_id,
            PromptBundleRecord.prompt_bundle_id == binding.prompt_bundle_id,
        )
    )
    if record is None:
        return bool(
            binding.prompt_bundle_id == DEFAULT_PROMPT_BUNDLE_ID
            and raw.get("governance_status") == "LEGACY_BUILTIN"
        )
    return bool(
        record.status == "APPROVED"
        and record.content_hash == binding.content_hash
        and raw.get("governance_status") == "APPROVED"
    )


def _retrieval_binding_from_manifest(manifest: dict[str, object]) -> FieldEdgeRetrievalBinding:
    raw = manifest["retrieval"]
    if not isinstance(raw, dict):
        raise ValueError("field_edge_retrieval_binding_invalid")
    return FieldEdgeRetrievalBinding(
        index_release_id=str(raw["index_release_id"]),
        content_checksum=str(raw["index_content_checksum"]),
    )


def _current_offline_pack(
    session: Session,
    identity: IdentityContext,
    work_order_id: str,
    view: WorkOrderOfflinePackView,
) -> WorkOrderOfflinePackRecord:
    record = session.scalar(
        select(WorkOrderOfflinePackRecord)
        .where(
            WorkOrderOfflinePackRecord.tenant_id == identity.tenant_id,
            WorkOrderOfflinePackRecord.subject_id == identity.subject_id,
            WorkOrderOfflinePackRecord.work_order_id == work_order_id,
            WorkOrderOfflinePackRecord.pack_id == view.pack_id,
            WorkOrderOfflinePackRecord.status == "ACTIVE",
        )
        .with_for_update()
    )
    if record is None or _utc(record.expires_at) <= datetime.now(UTC):
        raise OfflinePackNotAvailable
    if record.content_hash != canonical_digest(record.snapshot_json):
        raise EdgeDiagnosisConflict("offline_pack_integrity_failed", record.version)
    return record


def _validate_result_bindings(
    claims: FieldEdgeDiagnosisPackClaims,
    result: FieldEdgeDiagnosisResult,
) -> None:
    evidence = result.runtime_evidence
    if (
        evidence.engine != "llama.cpp"
        or evidence.runtime_attestation != "UNATTESTED"
        or evidence.model_file != claims.release.model_file
        or evidence.model_content_hash != claims.release.model_content_hash
        or evidence.prompt_bundle_hash != claims.prompt_bundle.content_hash
    ):
        raise EdgeDiagnosisRejected("edge_diagnosis_runtime_binding_invalid")
    if int(evidence.completed_at.timestamp()) > claims.exp:
        raise EdgeDiagnosisRejected("edge_diagnosis_result_outside_pack_window")


def _validate_pack_record(
    record: WorkOrderEdgeDiagnosisPackRecord,
    claims: FieldEdgeDiagnosisPackClaims,
    token: str,
    now: datetime,
) -> None:
    if (
        record.status != "ACTIVE"
        or _utc(record.expires_at) <= now
        or record.pack_digest != _token_digest(token)
        or record.signed_token != token
        or record.claims_json != claims.model_dump(mode="json")
    ):
        raise EdgeDiagnosisConflict("edge_diagnosis_pack_not_current", record.version)


def _validate_authority_bindings(
    pack: WorkOrderEdgeDiagnosisPackRecord,
    offline: WorkOrderOfflinePackRecord,
) -> None:
    if (
        pack.work_order_version != offline.work_order_version
        or pack.asset_id != offline.asset_id
        or pack.asset_version != offline.asset_version
        or pack.assignment_id != offline.assignment_id
        or _utc(pack.assignment_assigned_at) != _utc(offline.assignment_assigned_at)
        or pack.offline_pack_id != offline.pack_id
        or pack.offline_pack_content_hash != offline.content_hash
    ):
        raise EdgeDiagnosisConflict("edge_diagnosis_authority_binding_changed")


def _sha256_digest(value: str) -> str:
    raw = value.removeprefix("sha256:")
    if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
        raise ValueError("field_edge_model_digest_invalid")
    return f"sha256:{raw}"


def _token_digest(token: str) -> str:
    return f"sha256:{sha256(token.encode()).hexdigest()}"
