"""Knowledge publication and citation authorization use cases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.domain import as_utc as _as_utc
from industrial_ops_agent.knowledge.models import CitationView
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    AssetRecord,
    CitationAnchorRecord,
    IdempotencyRecord,
    IndexReleaseRecord,
    KnowledgeChunkRecord,
    KnowledgeDocumentRecord,
    KnowledgeDocumentVersionRecord,
    KnowledgeGraphReleaseRecord,
    KnowledgeIndexActivationRecord,
    KnowledgeIndexEvaluationRecord,
    KnowledgeSearchProfileRecord,
)


class KnowledgeNotVisible(Exception):
    """Knowledge is absent or filtered by current authorization/applicability."""


class InvalidIndexRelease(Exception):
    """A candidate release did not satisfy publication invariants."""


@dataclass(frozen=True, slots=True)
class KnowledgeIndexActivationView:
    activation_id: str
    name: str
    action: str
    from_release_id: str | None
    to_release_id: str
    from_content_checksum: str | None
    to_content_checksum: str
    actor_subject_id: str
    reason: str
    retired_search_profile_ids: tuple[str, ...]
    revoked_graph_release_ids: tuple[str, ...]
    occurred_at: datetime


class KnowledgePublicationService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def publish(
        self,
        identity: IdentityContext,
        release_id: str,
        *,
        request_id: str,
    ) -> None:
        self._authorizer.require(
            identity,
            Action.PUBLISH_KNOWLEDGE,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=release_id),
            request_id=request_id,
        )
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            release = session.scalar(
                select(IndexReleaseRecord)
                .where(
                    IndexReleaseRecord.tenant_id == identity.tenant_id,
                    IndexReleaseRecord.release_id == release_id,
                )
                .with_for_update()
            )
            if release is not None and release.status == "PUBLISHED" and release.is_active:
                return
            if release is None or release.status != "CANDIDATE":
                raise InvalidIndexRelease("release_not_candidate")
            evaluation = session.scalar(
                select(KnowledgeIndexEvaluationRecord)
                .where(
                    KnowledgeIndexEvaluationRecord.tenant_id == identity.tenant_id,
                    KnowledgeIndexEvaluationRecord.release_id == release_id,
                )
                .order_by(
                    KnowledgeIndexEvaluationRecord.created_at.desc(),
                    KnowledgeIndexEvaluationRecord.evaluation_id.desc(),
                )
                .limit(1)
            )
            if evaluation is None or evaluation.status != "PASSED":
                raise InvalidIndexRelease("release_evaluation_not_passed")
            chunk_count = session.scalar(
                select(func.count())
                .select_from(KnowledgeChunkRecord)
                .where(
                    KnowledgeChunkRecord.tenant_id == identity.tenant_id,
                    KnowledgeChunkRecord.release_id == release_id,
                )
            )
            anchored_count = session.scalar(
                select(func.count(func.distinct(KnowledgeChunkRecord.chunk_id)))
                .select_from(KnowledgeChunkRecord)
                .join(
                    CitationAnchorRecord,
                    CitationAnchorRecord.chunk_id == KnowledgeChunkRecord.chunk_id,
                )
                .where(
                    KnowledgeChunkRecord.tenant_id == identity.tenant_id,
                    KnowledgeChunkRecord.release_id == release_id,
                    CitationAnchorRecord.tenant_id == identity.tenant_id,
                )
            )
            if not chunk_count or chunk_count != anchored_count:
                raise InvalidIndexRelease("release_quality_gate_failed")
            version_ids = set(
                session.scalars(
                    select(KnowledgeChunkRecord.document_version_id).where(
                        KnowledgeChunkRecord.tenant_id == identity.tenant_id,
                        KnowledgeChunkRecord.release_id == release_id,
                    )
                )
            )
            versions = list(
                session.scalars(
                    select(KnowledgeDocumentVersionRecord).where(
                        KnowledgeDocumentVersionRecord.tenant_id == identity.tenant_id,
                        KnowledgeDocumentVersionRecord.document_version_id.in_(version_ids),
                    )
                )
            )
            if len(versions) != len(version_ids) or any(
                version.status not in {"REVIEWED", "PUBLISHED"} for version in versions
            ):
                raise InvalidIndexRelease("release_version_gate_failed")
            separated_subjects = {
                subject
                for subject in (
                    release.created_by_subject_id,
                    *(version.created_by_subject_id for version in versions),
                )
                if subject is not None
            }
            if identity.subject_id in separated_subjects:
                raise InvalidIndexRelease("release_publication_separation_required")
            active = session.scalar(
                select(IndexReleaseRecord)
                .where(
                    IndexReleaseRecord.tenant_id == identity.tenant_id,
                    IndexReleaseRecord.name == release.name,
                    IndexReleaseRecord.is_active.is_(True),
                )
                .with_for_update()
            )
            session.execute(
                update(IndexReleaseRecord)
                .where(
                    IndexReleaseRecord.tenant_id == identity.tenant_id,
                    IndexReleaseRecord.name == release.name,
                    IndexReleaseRecord.is_active.is_(True),
                )
                .values(
                    is_active=False,
                    activation_version=IndexReleaseRecord.activation_version + 1,
                    updated_at=now,
                )
            )
            session.flush()
            release.status = "PUBLISHED"
            release.is_active = True
            release.activation_version += 1
            release.published_by = identity.subject_id
            release.published_at = now
            release.updated_at = now
            for version in versions:
                if version.status == "REVIEWED":
                    version.status = "PUBLISHED"
                    version.state_version += 1
                    version.updated_at = now
            session.add(
                KnowledgeIndexActivationRecord(
                    tenant_id=identity.tenant_id,
                    activation_id=f"knowledge-index-activation-{uuid4().hex}",
                    name=release.name,
                    action="PUBLISH",
                    from_release_id=active.release_id if active is not None else None,
                    to_release_id=release.release_id,
                    from_content_checksum=(
                        active.content_checksum if active is not None else None
                    ),
                    to_content_checksum=release.content_checksum,
                    actor_subject_id=identity.subject_id,
                    reason="candidate_passed_publication_gates",
                    retired_search_profile_ids=[],
                    revoked_graph_release_ids=[],
                    occurred_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )

    def rollback(
        self,
        identity: IdentityContext,
        release_id: str,
        *,
        expected_activation_version: int,
        expected_active_release_id: str,
        reason: str,
        idempotency_key: str,
        request_id: str,
    ) -> KnowledgeIndexActivationView:
        self._authorizer.require(
            identity,
            Action.PUBLISH_KNOWLEDGE,
            ResourceContext(tenant_id=identity.tenant_id, resource_id=release_id),
            request_id=request_id,
        )
        normalized_reason = " ".join(reason.split())
        if (
            expected_activation_version < 1
            or not 1 <= len(expected_active_release_id) <= 128
            or not 10 <= len(normalized_reason) <= 1000
            or not 8 <= len(idempotency_key) <= 200
        ):
            raise InvalidIndexRelease("release_rollback_input_invalid")
        fingerprint = sha256(
            (
                f"{release_id}\0{expected_activation_version}\0"
                f"{expected_active_release_id}\0{normalized_reason}"
            ).encode()
        ).hexdigest()
        storage_key = f"knowledge-index-rollback:{idempotency_key}"
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            replay = session.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.tenant_id == identity.tenant_id,
                    IdempotencyRecord.subject_id == identity.subject_id,
                    IdempotencyRecord.key == storage_key,
                )
            )
            if replay is not None:
                if replay.request_hash != fingerprint:
                    raise InvalidIndexRelease("release_rollback_idempotency_conflict")
                activation = session.scalar(
                    select(KnowledgeIndexActivationRecord).where(
                        KnowledgeIndexActivationRecord.tenant_id == identity.tenant_id,
                        KnowledgeIndexActivationRecord.activation_id == replay.result_ref,
                    )
                )
                if activation is None:
                    raise InvalidIndexRelease("release_rollback_replay_invalid")
                return _activation_view(activation)

            target = session.scalar(
                select(IndexReleaseRecord)
                .where(
                    IndexReleaseRecord.tenant_id == identity.tenant_id,
                    IndexReleaseRecord.release_id == release_id,
                )
                .with_for_update()
            )
            if target is None or target.status != "PUBLISHED":
                raise InvalidIndexRelease("release_rollback_target_not_published")
            if target.activation_version != expected_activation_version:
                raise InvalidIndexRelease("release_activation_state_changed")
            if target.is_active:
                raise InvalidIndexRelease("release_rollback_target_already_active")
            active = session.scalar(
                select(IndexReleaseRecord)
                .where(
                    IndexReleaseRecord.tenant_id == identity.tenant_id,
                    IndexReleaseRecord.name == target.name,
                    IndexReleaseRecord.is_active.is_(True),
                )
                .with_for_update()
            )
            if active is None or active.release_id == target.release_id:
                raise InvalidIndexRelease("release_active_alias_not_found")
            if active.release_id != expected_active_release_id:
                raise InvalidIndexRelease("release_active_alias_changed")
            if target.version >= active.version:
                raise InvalidIndexRelease("release_rollback_target_not_older")
            versions = _require_rollback_integrity(session, target)
            separated_subjects = {
                subject
                for subject in (
                    target.created_by_subject_id,
                    *(version.created_by_subject_id for version in versions),
                )
                if subject is not None
            }
            if identity.subject_id in separated_subjects:
                raise InvalidIndexRelease("release_rollback_separation_required")

            profiles = list(
                session.scalars(
                    select(KnowledgeSearchProfileRecord)
                    .where(
                        KnowledgeSearchProfileRecord.tenant_id == identity.tenant_id,
                        KnowledgeSearchProfileRecord.source_index_release_id
                        == active.release_id,
                        KnowledgeSearchProfileRecord.is_active_shadow.is_(True),
                    )
                    .with_for_update()
                )
            )
            graphs = list(
                session.scalars(
                    select(KnowledgeGraphReleaseRecord)
                    .where(
                        KnowledgeGraphReleaseRecord.tenant_id == identity.tenant_id,
                        KnowledgeGraphReleaseRecord.source_index_release_id
                        == active.release_id,
                        KnowledgeGraphReleaseRecord.is_active.is_(True),
                    )
                    .with_for_update()
                )
            )
            for profile in profiles:
                profile.status = "RETIRED"
                profile.is_active_shadow = False
                profile.state_version += 1
                profile.updated_at = now
            for graph in graphs:
                graph.status = "REVOKED"
                graph.is_active = False
                graph.state_version += 1
                graph.updated_at = now

            active.is_active = False
            active.activation_version += 1
            active.updated_at = now
            session.flush()
            target.is_active = True
            target.activation_version += 1
            target.updated_at = now
            activation = KnowledgeIndexActivationRecord(
                tenant_id=identity.tenant_id,
                activation_id=f"knowledge-index-activation-{uuid4().hex}",
                name=target.name,
                action="ROLLBACK",
                from_release_id=active.release_id,
                to_release_id=target.release_id,
                from_content_checksum=active.content_checksum,
                to_content_checksum=target.content_checksum,
                actor_subject_id=identity.subject_id,
                reason=normalized_reason,
                retired_search_profile_ids=sorted(
                    profile.search_profile_id for profile in profiles
                ),
                revoked_graph_release_ids=sorted(
                    graph.graph_release_id for graph in graphs
                ),
                occurred_at=now,
                created_at=now,
                updated_at=now,
            )
            session.add(activation)
            session.add(
                IdempotencyRecord(
                    tenant_id=identity.tenant_id,
                    record_id=f"idem-{uuid4().hex}",
                    subject_id=identity.subject_id,
                    key=storage_key,
                    request_hash=fingerprint,
                    result_ref=activation.activation_id,
                    expires_at=now + timedelta(days=90),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.flush()
            return _activation_view(activation)

    def list_activations(
        self,
        identity: IdentityContext,
        *,
        name: str | None,
        limit: int,
        request_id: str,
    ) -> tuple[KnowledgeIndexActivationView, ...]:
        self._authorizer.require(
            identity,
            Action.PUBLISH_KNOWLEDGE,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id="knowledge-index-activations",
            ),
            request_id=request_id,
        )
        normalized_name = name.strip() if name is not None else None
        if not 1 <= limit <= 100 or normalized_name == "" or (
            normalized_name is not None and len(normalized_name) > 128
        ):
            raise InvalidIndexRelease("release_activation_query_invalid")
        with self._database.transaction(identity.tenant_context) as session:
            filters = [KnowledgeIndexActivationRecord.tenant_id == identity.tenant_id]
            if normalized_name is not None:
                filters.append(KnowledgeIndexActivationRecord.name == normalized_name)
            records = session.scalars(
                select(KnowledgeIndexActivationRecord)
                .where(*filters)
                .order_by(
                    KnowledgeIndexActivationRecord.occurred_at.desc(),
                    KnowledgeIndexActivationRecord.activation_id.desc(),
                )
                .limit(limit)
            )
            return tuple(_activation_view(record) for record in records)


def _require_rollback_integrity(
    session: Session,
    release: IndexReleaseRecord,
) -> list[KnowledgeDocumentVersionRecord]:
    evaluation = session.scalar(
        select(KnowledgeIndexEvaluationRecord)
        .where(
            KnowledgeIndexEvaluationRecord.tenant_id == release.tenant_id,
            KnowledgeIndexEvaluationRecord.release_id == release.release_id,
        )
        .order_by(
            KnowledgeIndexEvaluationRecord.created_at.desc(),
            KnowledgeIndexEvaluationRecord.evaluation_id.desc(),
        )
        .limit(1)
    )
    if evaluation is None or evaluation.status != "PASSED":
        raise InvalidIndexRelease("release_rollback_evaluation_not_passed")
    chunk_count = int(
        session.scalar(
            select(func.count())
            .select_from(KnowledgeChunkRecord)
            .where(
                KnowledgeChunkRecord.tenant_id == release.tenant_id,
                KnowledgeChunkRecord.release_id == release.release_id,
            )
        )
        or 0
    )
    anchored_count = int(
        session.scalar(
            select(func.count(func.distinct(KnowledgeChunkRecord.chunk_id)))
            .select_from(KnowledgeChunkRecord)
            .join(
                CitationAnchorRecord,
                CitationAnchorRecord.chunk_id == KnowledgeChunkRecord.chunk_id,
            )
            .where(
                KnowledgeChunkRecord.tenant_id == release.tenant_id,
                KnowledgeChunkRecord.release_id == release.release_id,
                CitationAnchorRecord.tenant_id == release.tenant_id,
            )
        )
        or 0
    )
    if chunk_count == 0 or anchored_count != chunk_count:
        raise InvalidIndexRelease("release_rollback_integrity_failed")
    version_ids = set(
        session.scalars(
            select(KnowledgeChunkRecord.document_version_id).where(
                KnowledgeChunkRecord.tenant_id == release.tenant_id,
                KnowledgeChunkRecord.release_id == release.release_id,
            )
        )
    )
    versions = list(
        session.scalars(
            select(KnowledgeDocumentVersionRecord).where(
                KnowledgeDocumentVersionRecord.tenant_id == release.tenant_id,
                KnowledgeDocumentVersionRecord.document_version_id.in_(version_ids),
            )
        )
    )
    if len(versions) != len(version_ids) or any(
        version.status != "PUBLISHED" for version in versions
    ):
        raise InvalidIndexRelease("release_rollback_document_state_invalid")
    return versions


def _activation_view(record: KnowledgeIndexActivationRecord) -> KnowledgeIndexActivationView:
    return KnowledgeIndexActivationView(
        activation_id=record.activation_id,
        name=record.name,
        action=record.action,
        from_release_id=record.from_release_id,
        to_release_id=record.to_release_id,
        from_content_checksum=record.from_content_checksum,
        to_content_checksum=record.to_content_checksum,
        actor_subject_id=record.actor_subject_id,
        reason=record.reason,
        retired_search_profile_ids=tuple(record.retired_search_profile_ids),
        revoked_graph_release_ids=tuple(record.revoked_graph_release_ids),
        occurred_at=_as_utc(record.occurred_at),
    )


class CitationService:
    def __init__(self, database: Database, authorizer: Authorizer) -> None:
        self._database = database
        self._authorizer = authorizer

    def get(
        self,
        identity: IdentityContext,
        citation_id: str,
        *,
        request_id: str,
    ) -> CitationView:
        try:
            self._authorizer.require(
                identity,
                Action.READ_CITATION,
                ResourceContext(tenant_id=identity.tenant_id, resource_id=citation_id),
                request_id=request_id,
            )
        except AuthorizationDenied as exc:
            raise KnowledgeNotVisible from exc
        now = datetime.now(UTC)
        with self._database.transaction(identity.tenant_context) as session:
            row = session.execute(
                select(
                    CitationAnchorRecord,
                    KnowledgeChunkRecord,
                    KnowledgeDocumentVersionRecord,
                    KnowledgeDocumentRecord,
                )
                .join(
                    KnowledgeChunkRecord,
                    KnowledgeChunkRecord.chunk_id == CitationAnchorRecord.chunk_id,
                )
                .join(
                    KnowledgeDocumentVersionRecord,
                    KnowledgeDocumentVersionRecord.document_version_id
                    == CitationAnchorRecord.document_version_id,
                )
                .join(
                    KnowledgeDocumentRecord,
                    KnowledgeDocumentRecord.document_id
                    == KnowledgeDocumentVersionRecord.document_id,
                )
                .join(
                    IndexReleaseRecord,
                    IndexReleaseRecord.release_id == KnowledgeChunkRecord.release_id,
                )
                .where(
                    CitationAnchorRecord.tenant_id == identity.tenant_id,
                    CitationAnchorRecord.citation_id == citation_id,
                    KnowledgeChunkRecord.tenant_id == identity.tenant_id,
                    KnowledgeDocumentVersionRecord.tenant_id == identity.tenant_id,
                    KnowledgeDocumentVersionRecord.status == "PUBLISHED",
                    KnowledgeDocumentRecord.tenant_id == identity.tenant_id,
                    IndexReleaseRecord.tenant_id == identity.tenant_id,
                    IndexReleaseRecord.status == "PUBLISHED",
                )
            ).one_or_none()
            if row is None:
                raise KnowledgeNotVisible
            anchor, chunk, version, document = row
            roles = {role.value for role in identity.roles}
            acl_allowed = (
                not version.acl_subject_ids
                and not version.acl_roles
                or identity.subject_id in version.acl_subject_ids
                or bool(roles.intersection(version.acl_roles))
            )
            visible_models = set(
                session.scalars(
                    select(AssetRecord.model_code).where(
                        AssetRecord.tenant_id == identity.tenant_id,
                        AssetRecord.asset_id.in_(identity.asset_ids),
                    )
                )
            )
            model_allowed = not version.device_models or bool(
                visible_models.intersection(version.device_models)
            )
            valid_from = _as_utc(version.valid_from)
            valid_to = _as_utc(version.valid_to) if version.valid_to is not None else None
            if (
                not acl_allowed
                or not model_allowed
                or valid_from > now
                or valid_to is not None
                and valid_to <= now
                or document.classification == "restricted"
                and not roles.intersection(version.acl_roles)
            ):
                raise KnowledgeNotVisible
            return CitationView(
                citation_id=anchor.citation_id,
                document_id=document.document_id,
                document_version_id=version.document_version_id,
                document_version=version.version,
                chunk_id=chunk.chunk_id,
                title=document.title,
                source_uri=document.source_uri,
                source_checksum=version.source_checksum,
                content_checksum=chunk.content_checksum,
                content_excerpt=anchor.excerpt,
                page_number=anchor.page_number,
                bounding_box=anchor.bounding_box,
                start_offset=anchor.start_offset,
                end_offset=anchor.end_offset,
            )


__all__ = [
    "AuthorizationDenied",
    "CitationService",
    "InvalidIndexRelease",
    "KnowledgeNotVisible",
    "KnowledgePublicationService",
]
