"""Resumable batch registration of enterprise LLM, VLM, and RUL candidates."""

from __future__ import annotations

from hashlib import sha256

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.deployment.service import DeploymentPlan, ModelDeploymentService
from industrial_ops_agent.enterprise_assets.models import (
    EnterpriseCandidateReleaseBatch,
    EnterpriseCandidateReleaseBatchProgress,
    EnterpriseCandidateReleaseProgress,
    EnterpriseDeploymentAction,
    EnterpriseDeploymentNextAction,
    EnterpriseModelComponent,
    EnterpriseReleaseDraftPreview,
    EnterpriseReleaseOnboarding,
)
from industrial_ops_agent.enterprise_assets.onboarding import (
    EnterpriseReleaseOnboardingService,
)
from industrial_ops_agent.enterprise_assets.service import EnterpriseProjectAdoptionReader
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    EnterpriseModelReleaseOnboardingRecord,
    ModelDeploymentRecord,
    ModelReleaseApprovalRecord,
    ModelReleaseObservationRecord,
    ModelReleaseRecord,
)
from industrial_ops_agent.releases.service import (
    ModelReleaseService,
    ReleaseNotVisible,
)

_COMPONENTS: tuple[EnterpriseModelComponent, ...] = ("LLM", "VLM", "RUL")
_BATCH_PREFIX = "enterprise-release-batch:"
_ROLLOUT_STATUSES = frozenset({"SHADOW", "CANARY", "PRODUCTION"})
_REMEDIATION_APPROVAL_STATUSES = frozenset({"REJECTED", "EXPIRED"})


class EnterpriseCandidateReleaseBatchConflict(Exception):
    def __init__(self, reason: str, current_version: int | None = None) -> None:
        self.reason = reason
        self.current_version = current_version
        super().__init__(reason)


class EnterpriseCandidateReleaseBatchService:
    """Create three independent governed drafts with one approved baseline."""

    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        adoption_reader: EnterpriseProjectAdoptionReader,
    ) -> None:
        self._database = database
        self._authorizer = authorizer
        self._onboarding = EnterpriseReleaseOnboardingService(
            database,
            authorizer,
            adoption_reader,
        )

    def create_drafts(
        self,
        identity: IdentityContext,
        *,
        active_mcp_server_versions: dict[str, str],
        idempotency_key: str,
        request_id: str,
    ) -> EnterpriseCandidateReleaseBatch:
        self._authorizer.require(
            identity,
            Action.CREATE_MODEL_RELEASE,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id="model-releases",
            ),
            request_id=request_id,
        )
        batch_digest = sha256(idempotency_key.encode()).hexdigest()
        component_keys = {
            component: _component_idempotency_key(batch_digest, component)
            for component in _COMPONENTS
        }
        existing = self._existing_records(identity, component_keys)
        baseline_release_id = _existing_baseline(existing)

        releases: dict[EnterpriseModelComponent, EnterpriseReleaseOnboarding] = {}
        if baseline_release_id is not None:
            for component in _COMPONENTS:
                if component not in existing:
                    continue
                releases[component] = self._onboarding.create_draft(
                    identity,
                    component,
                    baseline_release_id=baseline_release_id,
                    auto_shadow_enabled=False,
                    deployment_plan=None,
                    active_mcp_server_versions=active_mcp_server_versions,
                    idempotency_key=component_keys[component],
                    request_id=f"{request_id}:{component.lower()}:replay",
                )

        missing = tuple(component for component in _COMPONENTS if component not in existing)
        previews = {
            component: self._onboarding.preview(
                identity,
                component,
                request_id=f"{request_id}:{component.lower()}:preflight",
            )
            for component in missing
        }
        for preview in previews.values():
            _require_eligible(preview)

        if baseline_release_id is None:
            baseline_release_id = _newest_common_baseline(previews)
        else:
            for preview in previews.values():
                if baseline_release_id not in {item.release_id for item in preview.baselines}:
                    raise EnterpriseCandidateReleaseBatchConflict(
                        "COMPATIBLE_BASELINE_RELEASE_REQUIRED"
                    )

        for component in missing:
            releases[component] = self._onboarding.create_draft(
                identity,
                component,
                baseline_release_id=baseline_release_id,
                auto_shadow_enabled=False,
                deployment_plan=None,
                active_mcp_server_versions=active_mcp_server_versions,
                idempotency_key=component_keys[component],
                request_id=f"{request_id}:{component.lower()}:create",
            )

        ordered = tuple(releases[component] for component in _COMPONENTS)
        return EnterpriseCandidateReleaseBatch(
            batch_key_sha256=batch_digest,
            baseline_release_id=baseline_release_id,
            target_environment="STAGING",
            operational_classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
            status="BATCH_REGISTERED",
            releases=ordered,
            created_count=len(missing),
            replayed_count=len(existing),
            next_action="VALIDATE_AND_SUBMIT_INDEPENDENT_APPROVAL",
        )

    def list_progress(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        limit: int = 20,
    ) -> tuple[EnterpriseCandidateReleaseBatchProgress, ...]:
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        self._authorizer.require(
            identity,
            Action.READ_MODEL_RELEASE,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id="model-releases",
            ),
            request_id=request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            records = tuple(
                session.scalars(
                    select(EnterpriseModelReleaseOnboardingRecord)
                    .where(
                        EnterpriseModelReleaseOnboardingRecord.tenant_id == identity.tenant_id,
                        EnterpriseModelReleaseOnboardingRecord.requested_by_subject_id
                        == identity.subject_id,
                        EnterpriseModelReleaseOnboardingRecord.idempotency_key.like(
                            f"{_BATCH_PREFIX}%"
                        ),
                    )
                    .order_by(EnterpriseModelReleaseOnboardingRecord.created_at.desc())
                    .limit(limit * len(_COMPONENTS) + len(_COMPONENTS) - 1)
                )
            )
            grouped: dict[str, list[EnterpriseModelReleaseOnboardingRecord]] = {}
            for record in records:
                parsed = _parse_component_idempotency_key(record.idempotency_key)
                if parsed is None:
                    continue
                digest, expected_component = parsed
                if record.component != expected_component:
                    raise EnterpriseCandidateReleaseBatchConflict(
                        "idempotency_key_reused",
                        record.version,
                    )
                grouped.setdefault(digest, []).append(record)
            progress = tuple(
                _progress_projection(
                    session,
                    identity,
                    self._authorizer,
                    digest,
                    tuple(items),
                )
                for digest, items in grouped.items()
            )
        return tuple(sorted(progress, key=lambda item: item.created_at, reverse=True)[:limit])

    def list_review_queue(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        limit: int = 20,
    ) -> tuple[EnterpriseCandidateReleaseBatchProgress, ...]:
        """List tenant batches that still need one or more independent decisions."""

        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        self._authorizer.require(
            identity,
            Action.DECIDE_MODEL_RELEASE_APPROVAL,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id="model-releases",
            ),
            request_id=request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            pending_records = tuple(
                session.scalars(
                    select(EnterpriseModelReleaseOnboardingRecord)
                    .join(
                        ModelReleaseApprovalRecord,
                        and_(
                            ModelReleaseApprovalRecord.tenant_id
                            == EnterpriseModelReleaseOnboardingRecord.tenant_id,
                            ModelReleaseApprovalRecord.release_id
                            == EnterpriseModelReleaseOnboardingRecord.release_id,
                        ),
                    )
                    .where(
                        EnterpriseModelReleaseOnboardingRecord.tenant_id == identity.tenant_id,
                        EnterpriseModelReleaseOnboardingRecord.idempotency_key.like(
                            f"{_BATCH_PREFIX}%"
                        ),
                        ModelReleaseApprovalRecord.status == "PENDING",
                    )
                    .order_by(EnterpriseModelReleaseOnboardingRecord.created_at.desc())
                    .limit(limit * len(_COMPONENTS))
                )
            )
            group_keys: list[tuple[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for record in pending_records:
                parsed = _parse_component_idempotency_key(record.idempotency_key)
                if parsed is None or record.component != parsed[1]:
                    raise EnterpriseCandidateReleaseBatchConflict(
                        "idempotency_key_reused",
                        record.version,
                    )
                key = (record.requested_by_subject_id, parsed[0])
                if key in seen:
                    continue
                seen.add(key)
                group_keys.append(key)
                if len(group_keys) == limit:
                    break
            if not group_keys:
                return ()

            batch_records = tuple(
                session.scalars(
                    select(EnterpriseModelReleaseOnboardingRecord).where(
                        EnterpriseModelReleaseOnboardingRecord.tenant_id == identity.tenant_id,
                        or_(
                            *(
                                and_(
                                    EnterpriseModelReleaseOnboardingRecord.requested_by_subject_id
                                    == requested_by_subject_id,
                                    EnterpriseModelReleaseOnboardingRecord.idempotency_key.in_(
                                        tuple(
                                            _component_idempotency_key(digest, component)
                                            for component in _COMPONENTS
                                        )
                                    ),
                                )
                                for requested_by_subject_id, digest in group_keys
                            )
                        ),
                    )
                )
            )
            grouped: dict[
                tuple[str, str],
                list[EnterpriseModelReleaseOnboardingRecord],
            ] = {}
            for record in batch_records:
                parsed = _parse_component_idempotency_key(record.idempotency_key)
                if parsed is None or record.component != parsed[1]:
                    raise EnterpriseCandidateReleaseBatchConflict(
                        "idempotency_key_reused",
                        record.version,
                    )
                key = (record.requested_by_subject_id, parsed[0])
                if key in seen:
                    grouped.setdefault(key, []).append(record)
            return tuple(
                _progress_projection(
                    session,
                    identity,
                    self._authorizer,
                    digest,
                    tuple(grouped[(requested_by_subject_id, digest)]),
                )
                for requested_by_subject_id, digest in group_keys
            )

    def list_deployment_queue(
        self,
        identity: IdentityContext,
        *,
        request_id: str,
        limit: int = 20,
    ) -> tuple[EnterpriseCandidateReleaseBatchProgress, ...]:
        """List complete approved batches and their active component deployments."""

        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        self._authorizer.require(
            identity,
            Action.REQUEST_MODEL_DEPLOYMENT,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id="model-deployments",
            ),
            request_id=request_id,
        )
        with self._database.transaction(identity.tenant_context) as session:
            eligible_records = tuple(
                session.scalars(
                    select(EnterpriseModelReleaseOnboardingRecord)
                    .join(
                        ModelReleaseApprovalRecord,
                        and_(
                            ModelReleaseApprovalRecord.tenant_id
                            == EnterpriseModelReleaseOnboardingRecord.tenant_id,
                            ModelReleaseApprovalRecord.release_id
                            == EnterpriseModelReleaseOnboardingRecord.release_id,
                        ),
                    )
                    .outerjoin(
                        ModelDeploymentRecord,
                        and_(
                            ModelDeploymentRecord.tenant_id
                            == EnterpriseModelReleaseOnboardingRecord.tenant_id,
                            ModelDeploymentRecord.release_id
                            == EnterpriseModelReleaseOnboardingRecord.release_id,
                        ),
                    )
                    .where(
                        EnterpriseModelReleaseOnboardingRecord.tenant_id == identity.tenant_id,
                        EnterpriseModelReleaseOnboardingRecord.idempotency_key.like(
                            f"{_BATCH_PREFIX}%"
                        ),
                        or_(
                            ModelReleaseApprovalRecord.status == "APPROVED",
                            ModelDeploymentRecord.deployment_id.is_not(None),
                        ),
                    )
                    .order_by(EnterpriseModelReleaseOnboardingRecord.created_at.desc())
                    .limit(limit * len(_COMPONENTS) * 4)
                )
            )
            group_keys: list[tuple[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for record in eligible_records:
                parsed = _parse_component_idempotency_key(record.idempotency_key)
                if parsed is None or record.component != parsed[1]:
                    raise EnterpriseCandidateReleaseBatchConflict(
                        "idempotency_key_reused",
                        record.version,
                    )
                key = (record.requested_by_subject_id, parsed[0])
                if key in seen:
                    continue
                seen.add(key)
                group_keys.append(key)
                if len(group_keys) == limit:
                    break
            if not group_keys:
                return ()

            grouped = _records_for_group_keys(
                session,
                identity.tenant_id,
                tuple(group_keys),
            )
            projected = tuple(
                _progress_projection(
                    session,
                    identity,
                    self._authorizer,
                    digest,
                    tuple(grouped.get((requested_by_subject_id, digest), ())),
                )
                for requested_by_subject_id, digest in group_keys
            )
            return tuple(
                item
                for item in projected
                if item.registered_count == len(_COMPONENTS)
                and (item.approved_count == len(_COMPONENTS) or item.deployment_requested_count)
            )

    def request_component_shadow(
        self,
        identity: IdentityContext,
        batch_key_sha256: str,
        component: EnterpriseModelComponent,
        plan: DeploymentPlan,
        *,
        requested_by_subject_id: str,
        request_id: str,
    ) -> EnterpriseCandidateReleaseBatchProgress:
        """Ensure one independently governed component has one Shadow deployment."""

        if not _valid_digest(batch_key_sha256):
            raise ValueError("batch_key_sha256 must be a SHA-256 digest")
        if component not in _COMPONENTS:
            raise ValueError("component must be LLM, VLM, or RUL")
        if not requested_by_subject_id.strip():
            raise ValueError("requested_by_subject_id is required")
        self._authorizer.require(
            identity,
            Action.REQUEST_MODEL_DEPLOYMENT,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id="model-deployments",
            ),
            request_id=request_id,
        )
        progress = self._progress_for_requester(
            identity,
            batch_key_sha256,
            requested_by_subject_id=requested_by_subject_id,
        )
        if progress.registered_count != len(_COMPONENTS):
            raise EnterpriseCandidateReleaseBatchConflict(
                "BATCH_REGISTRATION_INCOMPLETE"
            )
        if progress.approved_count != len(_COMPONENTS):
            raise EnterpriseCandidateReleaseBatchConflict(
                "BATCH_INDEPENDENT_APPROVAL_REQUIRED"
            )
        target = next(
            (item for item in progress.components if item.component == component),
            None,
        )
        if target is None:
            raise ReleaseNotVisible
        ModelDeploymentService(self._database, self._authorizer).ensure_shadow(
            identity,
            target.release_id,
            plan,
            expected_release_version=target.release_version,
            request_id=f"{request_id}:{component.lower()}",
        )
        return self._progress_for_requester(
            identity,
            batch_key_sha256,
            requested_by_subject_id=requested_by_subject_id,
        )

    def _progress_for_requester(
        self,
        identity: IdentityContext,
        batch_key_sha256: str,
        *,
        requested_by_subject_id: str,
    ) -> EnterpriseCandidateReleaseBatchProgress:
        keys = tuple(
            _component_idempotency_key(batch_key_sha256, component)
            for component in _COMPONENTS
        )
        with self._database.transaction(identity.tenant_context) as session:
            records = tuple(
                session.scalars(
                    select(EnterpriseModelReleaseOnboardingRecord).where(
                        EnterpriseModelReleaseOnboardingRecord.tenant_id
                        == identity.tenant_id,
                        EnterpriseModelReleaseOnboardingRecord.requested_by_subject_id
                        == requested_by_subject_id,
                        EnterpriseModelReleaseOnboardingRecord.idempotency_key.in_(keys),
                    )
                )
            )
            if not records:
                raise ReleaseNotVisible
            return _progress_projection(
                session,
                identity,
                self._authorizer,
                batch_key_sha256,
                records,
            )

    def get_progress(
        self,
        identity: IdentityContext,
        batch_key_sha256: str,
        *,
        request_id: str,
    ) -> EnterpriseCandidateReleaseBatchProgress:
        self._authorizer.require(
            identity,
            Action.READ_MODEL_RELEASE,
            ResourceContext(
                tenant_id=identity.tenant_id,
                resource_id="model-releases",
            ),
            request_id=request_id,
        )
        if not _valid_digest(batch_key_sha256):
            raise ValueError("batch_key_sha256 must be a SHA-256 digest")
        keys = tuple(
            _component_idempotency_key(batch_key_sha256, component) for component in _COMPONENTS
        )
        with self._database.transaction(identity.tenant_context) as session:
            records = tuple(
                session.scalars(
                    select(EnterpriseModelReleaseOnboardingRecord).where(
                        EnterpriseModelReleaseOnboardingRecord.tenant_id == identity.tenant_id,
                        EnterpriseModelReleaseOnboardingRecord.requested_by_subject_id
                        == identity.subject_id,
                        EnterpriseModelReleaseOnboardingRecord.idempotency_key.in_(keys),
                    )
                )
            )
            if not records:
                raise ReleaseNotVisible
            return _progress_projection(
                session,
                identity,
                self._authorizer,
                batch_key_sha256,
                records,
            )

    def advance_to_approval(
        self,
        identity: IdentityContext,
        batch_key_sha256: str,
        *,
        request_id: str,
    ) -> EnterpriseCandidateReleaseBatchProgress:
        progress = self.get_progress(
            identity,
            batch_key_sha256,
            request_id=f"{request_id}:preflight",
        )
        if progress.next_action in {
            "INDEPENDENT_APPROVER_DECISION",
            "REQUEST_SHADOW_IN_RELEASE_CENTER",
            "MONITOR_ROLLOUT",
        }:
            return progress
        if not progress.can_advance_to_approval:
            raise EnterpriseCandidateReleaseBatchConflict(
                f"BATCH_NOT_APPROVAL_SUBMITTABLE:{progress.status}"
            )

        releases = ModelReleaseService(self._database, self._authorizer)
        for component in progress.components:
            aggregate = releases.get(
                identity,
                component.release_id,
                request_id=f"{request_id}:{component.component.lower()}:read",
            )
            if aggregate.release.status == "DRAFT":
                aggregate = releases.validate(
                    identity,
                    component.release_id,
                    expected_version=aggregate.release.version,
                    request_id=f"{request_id}:{component.component.lower()}:validate",
                )
                if aggregate.release.status != "CANDIDATE":
                    raise EnterpriseCandidateReleaseBatchConflict(
                        f"BATCH_COMPONENT_VALIDATION_FAILED:{component.component}",
                        aggregate.release.version,
                    )
            if aggregate.release.status == "CANDIDATE":
                releases.submit_for_approval(
                    identity,
                    component.release_id,
                    expected_version=aggregate.release.version,
                    request_id=f"{request_id}:{component.component.lower()}:submit",
                )
                continue
            if _already_submitted_or_beyond(aggregate.release, aggregate.approval):
                continue
            raise EnterpriseCandidateReleaseBatchConflict(
                f"BATCH_COMPONENT_NOT_APPROVAL_SUBMITTABLE:{component.component}",
                aggregate.release.version,
            )

        return self.get_progress(
            identity,
            batch_key_sha256,
            request_id=f"{request_id}:result",
        )

    def _existing_records(
        self,
        identity: IdentityContext,
        component_keys: dict[EnterpriseModelComponent, str],
    ) -> dict[EnterpriseModelComponent, EnterpriseModelReleaseOnboardingRecord]:
        reverse_keys = {value: component for component, value in component_keys.items()}
        with self._database.transaction(identity.tenant_context) as session:
            records = tuple(
                session.scalars(
                    select(EnterpriseModelReleaseOnboardingRecord).where(
                        EnterpriseModelReleaseOnboardingRecord.tenant_id == identity.tenant_id,
                        EnterpriseModelReleaseOnboardingRecord.requested_by_subject_id
                        == identity.subject_id,
                        EnterpriseModelReleaseOnboardingRecord.idempotency_key.in_(
                            tuple(reverse_keys)
                        ),
                    )
                )
            )
        existing: dict[
            EnterpriseModelComponent,
            EnterpriseModelReleaseOnboardingRecord,
        ] = {}
        for record in records:
            expected_component = reverse_keys[record.idempotency_key]
            if record.component != expected_component:
                raise EnterpriseCandidateReleaseBatchConflict(
                    "idempotency_key_reused",
                    record.version,
                )
            existing[expected_component] = record
        return existing


def _component_idempotency_key(
    batch_digest: str,
    component: EnterpriseModelComponent,
) -> str:
    return f"enterprise-release-batch:{batch_digest}:{component.lower()}"


def _parse_component_idempotency_key(
    value: str,
) -> tuple[str, EnterpriseModelComponent] | None:
    parts = value.split(":")
    if len(parts) != 3 or parts[0] != "enterprise-release-batch":
        return None
    digest, component_value = parts[1], parts[2].upper()
    if not _valid_digest(digest) or component_value not in _COMPONENTS:
        return None
    return digest, component_value


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _existing_baseline(
    existing: dict[EnterpriseModelComponent, EnterpriseModelReleaseOnboardingRecord],
) -> str | None:
    baseline_ids = {record.baseline_release_id for record in existing.values()}
    if len(baseline_ids) > 1:
        raise EnterpriseCandidateReleaseBatchConflict("BATCH_BASELINE_DIVERGED")
    return next(iter(baseline_ids), None)


def _require_eligible(preview: EnterpriseReleaseDraftPreview) -> None:
    if preview.blockers:
        raise EnterpriseCandidateReleaseBatchConflict(preview.blockers[0])
    if not preview.eligible:
        raise EnterpriseCandidateReleaseBatchConflict("CANDIDATE_RELEASE_PREFLIGHT_FAILED")


def _newest_common_baseline(
    previews: dict[EnterpriseModelComponent, EnterpriseReleaseDraftPreview],
) -> str:
    if set(previews) != set(_COMPONENTS):
        raise EnterpriseCandidateReleaseBatchConflict("BATCH_RESUME_BASELINE_REQUIRED")
    common = set.intersection(
        *({baseline.release_id for baseline in preview.baselines} for preview in previews.values())
    )
    llm_baselines = previews["LLM"].baselines
    baseline = next(
        (item.release_id for item in llm_baselines if item.release_id in common),
        None,
    )
    if baseline is None:
        raise EnterpriseCandidateReleaseBatchConflict("COMPATIBLE_BASELINE_RELEASE_REQUIRED")
    return baseline


def _records_for_group_keys(
    session: Session,
    tenant_id: str,
    group_keys: tuple[tuple[str, str], ...],
) -> dict[tuple[str, str], list[EnterpriseModelReleaseOnboardingRecord]]:
    records = tuple(
        session.scalars(
            select(EnterpriseModelReleaseOnboardingRecord).where(
                EnterpriseModelReleaseOnboardingRecord.tenant_id == tenant_id,
                or_(
                    *(
                        and_(
                            EnterpriseModelReleaseOnboardingRecord.requested_by_subject_id
                            == requested_by_subject_id,
                            EnterpriseModelReleaseOnboardingRecord.idempotency_key.in_(
                                tuple(
                                    _component_idempotency_key(digest, component)
                                    for component in _COMPONENTS
                                )
                            ),
                        )
                        for requested_by_subject_id, digest in group_keys
                    )
                ),
            )
        )
    )
    expected = set(group_keys)
    grouped: dict[
        tuple[str, str],
        list[EnterpriseModelReleaseOnboardingRecord],
    ] = {}
    for record in records:
        parsed = _parse_component_idempotency_key(record.idempotency_key)
        if parsed is None or record.component != parsed[1]:
            raise EnterpriseCandidateReleaseBatchConflict(
                "idempotency_key_reused",
                record.version,
            )
        key = (record.requested_by_subject_id, parsed[0])
        if key in expected:
            grouped.setdefault(key, []).append(record)
    return grouped


def _progress_projection(
    session: Session,
    identity: IdentityContext,
    authorizer: Authorizer,
    batch_digest: str,
    records: tuple[EnterpriseModelReleaseOnboardingRecord, ...],
) -> EnterpriseCandidateReleaseBatchProgress:
    baseline_ids = {record.baseline_release_id for record in records}
    if len(baseline_ids) != 1:
        raise EnterpriseCandidateReleaseBatchConflict("BATCH_BASELINE_DIVERGED")
    requested_by_subject_ids = {record.requested_by_subject_id for record in records}
    if len(requested_by_subject_ids) != 1:
        raise EnterpriseCandidateReleaseBatchConflict("BATCH_REQUESTER_DIVERGED")
    requested_by_subject_id = next(iter(requested_by_subject_ids))
    record_by_component: dict[
        EnterpriseModelComponent,
        EnterpriseModelReleaseOnboardingRecord,
    ] = {}
    for record in records:
        parsed = _parse_component_idempotency_key(record.idempotency_key)
        if parsed is None or parsed[0] != batch_digest or record.component != parsed[1]:
            raise EnterpriseCandidateReleaseBatchConflict(
                "idempotency_key_reused",
                record.version,
            )
        record_by_component[parsed[1]] = record

    release_ids = tuple(record.release_id for record in records)
    releases = {
        release.release_id: release
        for release in session.scalars(
            select(ModelReleaseRecord).where(
                ModelReleaseRecord.tenant_id == identity.tenant_id,
                ModelReleaseRecord.release_id.in_(release_ids),
            )
        )
    }
    approvals = {
        approval.release_id: approval
        for approval in session.scalars(
            select(ModelReleaseApprovalRecord).where(
                ModelReleaseApprovalRecord.tenant_id == identity.tenant_id,
                ModelReleaseApprovalRecord.release_id.in_(release_ids),
            )
        )
    }
    deployments = {
        deployment.release_id: deployment
        for deployment in session.scalars(
            select(ModelDeploymentRecord).where(
                ModelDeploymentRecord.tenant_id == identity.tenant_id,
                ModelDeploymentRecord.release_id.in_(release_ids),
            )
        )
    }
    latest_observations: dict[str, ModelReleaseObservationRecord] = {}
    for observation in session.scalars(
        select(ModelReleaseObservationRecord)
        .where(
            ModelReleaseObservationRecord.tenant_id == identity.tenant_id,
            ModelReleaseObservationRecord.release_id.in_(release_ids),
        )
        .order_by(
            ModelReleaseObservationRecord.created_at.desc(),
            ModelReleaseObservationRecord.sequence.desc(),
        )
    ):
        latest_observations.setdefault(observation.release_id, observation)
    components: list[EnterpriseCandidateReleaseProgress] = []
    for component in _COMPONENTS:
        component_record = record_by_component.get(component)
        if component_record is None:
            continue
        release = releases.get(component_record.release_id)
        if release is None:
            raise ReleaseNotVisible
        approval = approvals.get(component_record.release_id)
        deployment = deployments.get(component_record.release_id)
        latest_observation = latest_observations.get(component_record.release_id)
        if approval is not None and approval.requested_by_subject_id != requested_by_subject_id:
            raise EnterpriseCandidateReleaseBatchConflict(
                "BATCH_APPROVAL_REQUESTER_DIVERGED",
                release.version,
            )
        components.append(
            EnterpriseCandidateReleaseProgress(
                component=component,
                onboarding_id=component_record.onboarding_id,
                release_id=release.release_id,
                release_status=release.status,
                release_version=release.version,
                approval_id=(approval.approval_id if approval is not None else None),
                approval_status=(approval.status if approval is not None else None),
                approval_version=(approval.version if approval is not None else None),
                approval_requested_by_subject_id=(
                    approval.requested_by_subject_id if approval is not None else None
                ),
                can_decide_approval=(
                    approval is not None
                    and approval.status == "PENDING"
                    and approval.requested_by_subject_id != identity.subject_id
                    and _allowed(
                        identity,
                        authorizer,
                        Action.DECIDE_MODEL_RELEASE_APPROVAL,
                        release.release_id,
                    )
                ),
                automation_status=component_record.automation_status,
                failure_reason=release.failure_reason,
                deployment_id=(
                    deployment.deployment_id if deployment is not None else None
                ),
                deployment_status=(deployment.status if deployment is not None else None),
                deployment_version=(deployment.version if deployment is not None else None),
                current_stage=(deployment.current_stage if deployment is not None else None),
                desired_stage=(deployment.desired_stage if deployment is not None else None),
                observed_traffic_percent=(
                    deployment.observed_traffic_percent if deployment is not None else None
                ),
                latest_observation_decision=(
                    latest_observation.decision if latest_observation is not None else None
                ),
                deployment_failure_reason=(
                    deployment.failure_reason if deployment is not None else None
                ),
                deployment_legal_actions=_deployment_legal_actions(
                    identity,
                    authorizer,
                    release,
                    approval,
                    deployment,
                    latest_observation,
                ),
                deployment_next_action=_deployment_next_action(
                    approval,
                    deployment,
                    latest_observation,
                ),
                next_action=_component_next_action(release, approval),
                created_at=component_record.created_at,
            )
        )
    status, next_action = _batch_state(tuple(components))
    return EnterpriseCandidateReleaseBatchProgress(
        batch_key_sha256=batch_digest,
        baseline_release_id=next(iter(baseline_ids)),
        requested_by_subject_id=requested_by_subject_id,
        target_environment="STAGING",
        operational_classification="LOCAL_STAGING_PROJECT_AUTHORIZED",
        status=status,
        components=tuple(components),
        registered_count=len(components),
        approval_pending_count=sum(
            component.approval_status == "PENDING" for component in components
        ),
        approved_count=sum(component.approval_status == "APPROVED" for component in components),
        decidable_count=sum(component.can_decide_approval for component in components),
        rollout_count=sum(
            component.release_status in _ROLLOUT_STATUSES for component in components
        ),
        deployment_requested_count=sum(
            component.deployment_id is not None for component in components
        ),
        shadow_ready_count=sum(
            component.deployment_status == "READY" and component.current_stage == "SHADOW"
            for component in components
        ),
        canary_count=sum(
            component.current_stage in {"CANARY_5", "CANARY_25"}
            for component in components
        ),
        production_count=sum(
            component.current_stage == "PRODUCTION" for component in components
        ),
        rolled_back_count=sum(
            component.current_stage == "ROLLED_BACK" for component in components
        ),
        can_advance_to_approval=status in {"DRAFTS_REGISTERED", "APPROVAL_SUBMISSION_IN_PROGRESS"},
        next_action=next_action,
        created_at=min(component.created_at for component in components),
    )


def _component_next_action(
    release: ModelReleaseRecord,
    approval: ModelReleaseApprovalRecord | None,
) -> str:
    if release.status == "DRAFT":
        return "VALIDATE_RELEASE"
    if release.status == "CANDIDATE":
        return "SUBMIT_APPROVAL"
    if release.status == "ROLLED_BACK":
        return "REVIEW_ROLLBACK"
    if release.status == "REJECTED" or (
        approval is not None and approval.status in _REMEDIATION_APPROVAL_STATUSES
    ):
        return "REMEDIATE_RELEASE"
    if release.status in _ROLLOUT_STATUSES:
        return "MONITOR_ROLLOUT"
    if approval is not None and approval.status == "APPROVED":
        return "REQUEST_SHADOW_IN_RELEASE_CENTER"
    if approval is not None and approval.status == "PENDING":
        return "WAIT_INDEPENDENT_APPROVAL"
    return "REMEDIATE_RELEASE"


def _deployment_legal_actions(
    identity: IdentityContext,
    authorizer: Authorizer,
    release: ModelReleaseRecord,
    approval: ModelReleaseApprovalRecord | None,
    deployment: ModelDeploymentRecord | None,
    latest_observation: ModelReleaseObservationRecord | None,
) -> tuple[EnterpriseDeploymentAction, ...]:
    if deployment is None:
        if (
            approval is not None
            and approval.status == "APPROVED"
            and _allowed(
                identity,
                authorizer,
                Action.REQUEST_MODEL_DEPLOYMENT,
                release.release_id,
            )
        ):
            return ("REQUEST_MODEL_DEPLOYMENT",)
        return ()

    actions: list[EnterpriseDeploymentAction] = []
    ready_for_promotion = (
        deployment.status == "READY"
        and deployment.current_stage == deployment.desired_stage
        and deployment.current_stage in {"SHADOW", "CANARY_5", "CANARY_25"}
        and latest_observation is not None
        and latest_observation.stage == deployment.current_stage
        and latest_observation.decision == "PASS"
    )
    if ready_for_promotion and _allowed(
        identity,
        authorizer,
        Action.PROMOTE_MODEL_RELEASE,
        release.release_id,
    ):
        actions.append("PROMOTE_MODEL_RELEASE")
    if deployment.current_stage not in {"NONE", "ROLLED_BACK"} and _allowed(
        identity,
        authorizer,
        Action.ROLLBACK_MODEL_RELEASE,
        release.release_id,
    ):
        actions.append("ROLLBACK_MODEL_RELEASE")
    return tuple(actions)


def _deployment_next_action(
    approval: ModelReleaseApprovalRecord | None,
    deployment: ModelDeploymentRecord | None,
    latest_observation: ModelReleaseObservationRecord | None,
) -> EnterpriseDeploymentNextAction:
    if deployment is None:
        return (
            "REQUEST_SHADOW"
            if approval is not None and approval.status == "APPROVED"
            else "WAIT_FOR_APPROVAL"
        )
    if deployment.current_stage == "ROLLED_BACK":
        return "REVIEW_ROLLBACK"
    if deployment.status == "FAILED":
        return "REMEDIATE_DEPLOYMENT"
    if deployment.status != "READY" or deployment.current_stage != deployment.desired_stage:
        return "WAIT_FOR_CONTROLLER"
    if deployment.current_stage == "PRODUCTION":
        return "OPERATE_PRODUCTION"
    if (
        latest_observation is not None
        and latest_observation.stage == deployment.current_stage
        and latest_observation.decision == "PASS"
    ):
        return "PROMOTE_NEXT_STAGE"
    return "COLLECT_STAGE_OBSERVATION"


def _allowed(
    identity: IdentityContext,
    authorizer: Authorizer,
    action: Action,
    resource_id: str,
) -> bool:
    return authorizer.decide(
        identity,
        action,
        ResourceContext(
            tenant_id=identity.tenant_id,
            resource_id=resource_id,
        ),
    ).allowed


def _batch_state(
    components: tuple[EnterpriseCandidateReleaseProgress, ...],
) -> tuple[str, str]:
    if len(components) < len(_COMPONENTS):
        return "DRAFT_REGISTRATION_INCOMPLETE", "RESUME_DRAFT_REGISTRATION"
    if any(
        component.release_status == "ROLLED_BACK"
        or component.current_stage == "ROLLED_BACK"
        for component in components
    ):
        return "ROLLED_BACK", "REVIEW_ROLLBACK"
    if any(
        component.next_action == "REMEDIATE_RELEASE"
        or component.release_status == "REJECTED"
        or component.approval_status in _REMEDIATION_APPROVAL_STATUSES
        for component in components
    ):
        return "REMEDIATION_REQUIRED", "REMEDIATE_RELEASE"
    rollout_count = sum(component.release_status in _ROLLOUT_STATUSES for component in components)
    deployment_count = sum(component.deployment_id is not None for component in components)
    if rollout_count or deployment_count:
        return "ROLLOUT_IN_PROGRESS", "MONITOR_ROLLOUT"
    approved_count = sum(component.approval_status == "APPROVED" for component in components)
    if approved_count == len(_COMPONENTS):
        return "APPROVED", "REQUEST_SHADOW_IN_RELEASE_CENTER"
    pending_count = sum(component.approval_status == "PENDING" for component in components)
    if pending_count + approved_count == len(_COMPONENTS):
        return "AWAITING_INDEPENDENT_APPROVAL", "INDEPENDENT_APPROVER_DECISION"
    if pending_count or approved_count:
        return (
            "APPROVAL_SUBMISSION_IN_PROGRESS",
            "CONTINUE_VALIDATE_AND_SUBMIT_APPROVAL",
        )
    return "DRAFTS_REGISTERED", "VALIDATE_AND_SUBMIT_APPROVAL"


def _already_submitted_or_beyond(
    release: ModelReleaseRecord,
    approval: ModelReleaseApprovalRecord | None,
) -> bool:
    return release.status in _ROLLOUT_STATUSES or (
        release.status == "APPROVAL_PENDING"
        and approval is not None
        and approval.status in {"PENDING", "APPROVED"}
    )
