"""Reuse the governed ModelRelease FSM with actual KServe traffic evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from industrial_ops_agent.auth.identity import Role
from industrial_ops_agent.deployment.service import (
    DeploymentAggregate,
    DeploymentPlan,
    DeploymentProvider,
    ModelDeploymentService,
    ObservationInput,
)
from industrial_ops_agent.persistence.models import (
    ModelAliasRecord,
    ModelReleaseApprovalRecord,
    ModelReleaseRecord,
    ModelReleaseTransitionRecord,
)
from industrial_ops_agent.simulation.enterprise_candidate_kserve_acceptance import (
    ModelReleaseEvidence,
    StageTrafficEvidence,
)
from industrial_ops_agent.simulation.enterprise_candidate_rollout import (
    SourceBinding,
    _authorizer,
    _database,
    _identity,
    _release_id,
    _VerifiedSource,
    _verify_sources,
)

Component = Literal["LLM", "TTS", "EMBEDDING"]
ObservedStage = Literal["SHADOW", "CANARY_5", "CANARY_25"]
RolloutStage = Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"]
STAGES: tuple[ObservedStage, ObservedStage, ObservedStage] = (
    "SHADOW",
    "CANARY_5",
    "CANARY_25",
)


class EnterpriseCandidateKServeFsmError(RuntimeError):
    """The actual provider or traffic evidence cannot advance the release FSM."""


@dataclass(frozen=True, slots=True)
class CollectedStage:
    traffic: StageTrafficEvidence
    observation: ObservationInput | None


class StageTrafficCollector(Protocol):
    def collect(self, stage: RolloutStage) -> CollectedStage: ...


@dataclass(frozen=True, slots=True)
class FsmComponentOutcome:
    component: Component
    source: SourceBinding
    stages: tuple[StageTrafficEvidence, ...]
    model_release: ModelReleaseEvidence


class EnterpriseCandidateKServeFsm:
    """Own a short-lived release database while three GPU workers run sequentially."""

    def __init__(self, repo_root: Path, *, now: datetime | None = None) -> None:
        self.root = repo_root.resolve(strict=True)
        self.now = now or datetime.now(UTC)
        self.sources = _verify_sources(self.root)
        self.database = _database(self.now, self.sources)
        self.service = ModelDeploymentService(self.database, _authorizer())
        self.operator = _identity(
            "actual-kserve-release-operator",
            Role.MODEL_RELEASE_OPERATOR,
            self.now,
        )
        self.controller = _identity(
            "actual-kserve-deployment-controller",
            Role.MODEL_DEPLOYMENT_CONTROLLER,
            self.now,
        )
        self._completed: set[Component] = set()
        self._closed = False

    def source(self, component: Component) -> _VerifiedSource:
        matches = tuple(source for source in self.sources if source.component == component)
        if len(matches) != 1:
            raise EnterpriseCandidateKServeFsmError("candidate_release_source_is_ambiguous")
        return matches[0]

    def run_component(
        self,
        component: Component,
        *,
        plan: DeploymentPlan,
        provider: DeploymentProvider,
        collector: StageTrafficCollector,
        reconcile_attempts: int = 80,
        on_reconcile_wait: Callable[[int, str | None], None] | None = None,
    ) -> FsmComponentOutcome:
        if self._closed:
            raise EnterpriseCandidateKServeFsmError("candidate_release_fsm_is_closed")
        if component in self._completed:
            raise EnterpriseCandidateKServeFsmError(
                "candidate_release_component_already_completed"
            )
        if reconcile_attempts < 1:
            raise ValueError("reconcile_attempts must be positive")
        source = self.source(component)
        release_id = _release_id(source)
        with Session(self.database.engine) as session:
            release = session.get(ModelReleaseRecord, release_id)
            if release is None:
                raise EnterpriseCandidateKServeFsmError("candidate_release_snapshot_missing")
            release_version = int(release.version)
        aggregate = self.service.request_shadow(
            self.operator,
            release_id,
            plan,
            expected_release_version=release_version,
            request_id=f"actual-kserve-{source.method.lower()}-shadow-request",
        )
        stages: list[StageTrafficEvidence] = []
        current = aggregate
        for index, stage in enumerate(STAGES):
            current = self._reconcile(
                current,
                provider,
                source.method.lower(),
                stage,
                attempts=reconcile_attempts,
                on_wait=on_reconcile_wait,
            )
            collected = collector.collect(stage)
            if collected.traffic.stage != stage or collected.observation is None:
                raise EnterpriseCandidateKServeFsmError(
                    "candidate_release_stage_evidence_is_incomplete"
                )
            observation = collected.observation
            if observation.stage != stage:
                raise EnterpriseCandidateKServeFsmError(
                    "candidate_release_observation_stage_mismatch"
                )
            current = self.service.record_observation(
                self.controller,
                current.deployment.deployment_id,
                observation,
                expected_deployment_version=current.deployment.version,
                request_id=f"actual-kserve-{source.method.lower()}-{stage.lower()}-observe",
            )
            if not current.observations or current.observations[-1].decision != "PASS":
                raise EnterpriseCandidateKServeFsmError(
                    "candidate_release_observation_did_not_pass"
                )
            stages.append(collected.traffic)
            if index < len(STAGES) - 1:
                current = self.service.promote(
                    self.operator,
                    release_id,
                    expected_deployment_version=current.deployment.version,
                    request_id=(
                        f"actual-kserve-{source.method.lower()}-{stage.lower()}-promote"
                    ),
                )

        current = self.service.request_rollback(
            self.operator,
            release_id,
            expected_deployment_version=current.deployment.version,
            reason_code="actual_local_kserve_rollback_acceptance",
            request_id=f"actual-kserve-{source.method.lower()}-rollback-request",
        )
        current = self._reconcile(
            current,
            provider,
            source.method.lower(),
            "ROLLED_BACK",
            attempts=reconcile_attempts,
            on_wait=on_reconcile_wait,
        )
        rollback = collector.collect("ROLLED_BACK")
        if rollback.traffic.stage != "ROLLED_BACK" or rollback.observation is not None:
            raise EnterpriseCandidateKServeFsmError(
                "candidate_release_rollback_evidence_is_incomplete"
            )
        stages.append(rollback.traffic)
        outcome = FsmComponentOutcome(
            component=component,
            source=source.binding(self.root),
            stages=tuple(stages),
            model_release=self._release_evidence(current),
        )
        self._completed.add(component)
        return outcome

    def close(self) -> None:
        if not self._closed:
            self.database.dispose()
            self._closed = True

    def __enter__(self) -> EnterpriseCandidateKServeFsm:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _reconcile(
        self,
        aggregate: DeploymentAggregate,
        provider: DeploymentProvider,
        method: str,
        stage: RolloutStage,
        *,
        attempts: int,
        on_wait: Callable[[int, str | None], None] | None,
    ) -> DeploymentAggregate:
        current = aggregate
        for attempt in range(1, attempts + 1):
            current = self.service.reconcile(
                self.controller,
                current.deployment.deployment_id,
                provider,
                expected_version=current.deployment.version,
                request_id=f"actual-kserve-{method}-{stage.lower()}-reconcile-{attempt}",
            )
            if (
                current.deployment.status == "READY"
                and current.deployment.current_stage == stage
            ):
                return current
            if current.deployment.status not in {"APPLYING", "FAILED"}:
                raise EnterpriseCandidateKServeFsmError(
                    "candidate_release_deployment_is_not_retriable"
                )
            if on_wait is not None:
                on_wait(attempt, current.deployment.failure_reason)
        raise EnterpriseCandidateKServeFsmError(
            f"candidate_release_{stage.lower()}_reconcile_timed_out"
        )

    def _release_evidence(self, final: DeploymentAggregate) -> ModelReleaseEvidence:
        release_id = final.release.release_id
        with Session(self.database.engine) as session:
            approval = session.scalar(
                select(ModelReleaseApprovalRecord).where(
                    ModelReleaseApprovalRecord.release_id == release_id
                )
            )
            transitions = tuple(
                session.scalars(
                    select(ModelReleaseTransitionRecord)
                    .where(ModelReleaseTransitionRecord.release_id == release_id)
                    .order_by(ModelReleaseTransitionRecord.sequence)
                )
            )
            aliases = tuple(
                session.scalars(
                    select(ModelAliasRecord).where(
                        ModelAliasRecord.active_release_id == release_id
                    )
                )
            )
        if (
            approval is None
            or approval.decided_by_subject_id is None
            or approval.requested_by_subject_id == approval.decided_by_subject_id
            or final.release.status != "ROLLED_BACK"
            or final.deployment.status != "READY"
            or final.deployment.current_stage != "ROLLED_BACK"
            or final.deployment.observed_traffic_percent != 0.0
            or aliases
        ):
            raise EnterpriseCandidateKServeFsmError(
                "candidate_release_final_state_is_invalid"
            )
        return ModelReleaseEvidence(
            release_id=release_id,
            manifest_hash=final.release.manifest_hash,
            approval_id=approval.approval_id,
            approval_requested_by_subject_id=approval.requested_by_subject_id,
            approval_decided_by_subject_id=approval.decided_by_subject_id,
            independent_approval=True,
            deployment_id=final.deployment.deployment_id,
            transition_reason_codes=tuple(item.reason_code for item in transitions),
            observation_stages=cast(
                tuple[ObservedStage, ObservedStage, ObservedStage],
                tuple(item.stage for item in final.observations),
            ),
            final_release_status="ROLLED_BACK",
            final_deployment_status="READY",
            final_deployment_stage="ROLLED_BACK",
            final_traffic_percent=0.0,
            active_production_alias_created=False,
        )


def plans_by_component(
    namespace: str,
    gateway_name: str,
    hostname: str,
    coordinates: Mapping[Component, tuple[str, str]],
) -> dict[Component, DeploymentPlan]:
    """Build plans whose route and stable service match the observed provider."""

    return {
        component: DeploymentPlan(
            namespace=namespace,
            gateway_name=gateway_name,
            hostname=hostname,
            route_name=route_name,
            stable_service_name=stable_name,
            service_account_name="ioap-model-runtime",
            serving_runtime_name="ioap-observed-local-runtime",
            artifact_uri_prefix="s3://project-enterprise-models/tenant-project-enterprise",
        )
        for component, (route_name, stable_name) in coordinates.items()
    }
