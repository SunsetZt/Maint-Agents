"""Shared release observation steps; algorithms and release policies stay in callers."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar, cast

import httpx

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.deployment.service import (
    DeploymentAggregate,
    ModelDeploymentService,
    ObservationInput,
    ProviderResult,
    ProviderTarget,
)
from industrial_ops_agent.model_evidence.kserve_routes import resource_ready
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.releases.service import ModelReleaseService
from industrial_ops_agent.simulation import local_kserve as kserve


class StageEvidence(Protocol):
    @property
    def p95_latency_ms(self) -> float: ...

    @property
    def request_count(self) -> int: ...

    @property
    def route_document_sha256(self) -> str: ...

    @property
    def stable_request_delta(self) -> int: ...

    @property
    def candidate_request_delta(self) -> int: ...

    @property
    def output_sha256s(self) -> tuple[str, ...]: ...



EvidenceT = TypeVar("EvidenceT", bound=StageEvidence)


@dataclass(slots=True)
class ReleaseState:
    database: Database
    engineer: IdentityContext
    approver: IdentityContext
    operator: IdentityContext
    controller: IdentityContext
    release_service: ModelReleaseService
    deployment_service: ModelDeploymentService
    baseline_release_id: str
    source_candidate_experiment_id: str
    imported_candidate_experiment_id: str
    evaluation_id: str
    approval_id: str
    approval_requested_by_subject_id: str
    approval_decided_by_subject_id: str
    deployment: DeploymentAggregate


@dataclass(frozen=True, slots=True)
class RolloutObservation(Generic[EvidenceT]):
    local: kserve.RolloutOperations
    stage_factory: Callable[..., EvidenceT]
    hostname: str
    route_path_prefix: str
    route_document: Callable[[Any], dict[str, Any]]
    route_rules_equal: Callable[[dict[str, Any], dict[str, Any]], bool]
    document_sha256: Callable[[object], str]
    stage_windows: dict[str, int]
    request_prefix: str
    variant_header: str

    def record_stage_observation(
        self,
        state: ReleaseState,
        *,
        stage: Literal["SHADOW", "CANARY_5", "CANARY_25"],
        stage_report: EvidenceT,
        runtime_metrics: dict[str, Any],
    ) -> DeploymentAggregate:
        end = datetime.now(UTC) - timedelta(seconds=1)
        peak = self.local.positive_integer(runtime_metrics, "gpu_peak_memory_reserved_bytes")
        total = self.local.positive_integer(runtime_metrics, "gpu_total_memory_bytes")
        metrics = {
            "cross_tenant_leak_count": 0.0,
            "unauthorized_side_effect_count": 0.0,
            "high_risk_miss_rate": 0.0,
            "task_success_delta": 0.0,
            "error_rate": 0.0,
            "p95_latency_ms": stage_report.p95_latency_ms,
            "human_edit_rate_delta": 0.0,
            "wrong_part_rate": 0.0,
            "gpu_xid_error_count": 0.0,
            "gpu_memory_utilization": peak / total,
            "queue_age_seconds": 0.0,
        }
        source_document = {
            "stage": stage,
            "route_document_sha256": stage_report.route_document_sha256,
            "request_count": stage_report.request_count,
            "stable_request_delta": stage_report.stable_request_delta,
            "candidate_request_delta": stage_report.candidate_request_delta,
            "output_sha256s": list(stage_report.output_sha256s),
        }
        return state.deployment_service.record_observation(
            state.controller,
            state.deployment.deployment.deployment_id,
            ObservationInput(
                stage=stage,
                window_start=end - timedelta(seconds=self.stage_windows[stage]),
                window_end=end,
                request_count=stage_report.request_count,
                critical_case_count=stage_report.request_count,
                metrics=metrics,
                source_refs={
                    "collector_version": f"{self.request_prefix}-kserve-rollout/v1",
                    "prometheus_endpoint_id": f"{self.request_prefix}-agent-worker-metrics",
                    "prometheus_query_bundle_hash": "sha256:"
                    + self.document_sha256({"metrics": metrics, "stage": stage}),
                    "trace_query_hash": "sha256:" + self.document_sha256(source_document),
                },
            ),
            expected_deployment_version=state.deployment.deployment.version,
            request_id=f"observe-{self.request_prefix}-{stage.lower()}",
        )

    def release_snapshot(self, state: ReleaseState) -> dict[str, Any]:
        release = state.release_service.get(
            state.engineer,
            state.deployment.release.release_id,
            request_id=f"snapshot-{self.request_prefix}-release",
        )
        deployment = state.deployment_service.get(
            state.controller,
            state.deployment.release.release_id,
            request_id=f"snapshot-{self.request_prefix}-deployment",
        )
        if (
            release.release.status != "ROLLED_BACK"
            or deployment.deployment.status != "READY"
            or deployment.deployment.current_stage != "ROLLED_BACK"
            or (deployment.deployment.observed_traffic_percent != 0.0)
            or (release.approval is None)
            or (release.approval.status != "APPROVED")
        ):
            raise self.local.error_type("final ModelRelease state is invalid")
        observation_stages = tuple(item.stage for item in deployment.observations)
        if observation_stages != ("SHADOW", "CANARY_5", "CANARY_25") or any(
            item.decision != "PASS" for item in deployment.observations
        ):
            raise self.local.error_type("ModelRelease observation chain is invalid")
        return {
            "release_id": release.release.release_id,
            "manifest": release.release.manifest_json,
            "manifest_hash": release.release.manifest_hash,
            "transition_targets": tuple(item.to_status for item in release.transitions),
            "observation_stages": observation_stages,
            "deployment_id": deployment.deployment.deployment_id,
            "release_status": release.release.status,
            "deployment_status": deployment.deployment.status,
            "deployment_stage": deployment.deployment.current_stage,
            "traffic_percent": deployment.deployment.observed_traffic_percent,
        }

    def send_gateway_traffic(
        self, url: str, payload: dict[str, Any], *, stage: str, request_count: int
    ) -> dict[str, Any]:
        responses: Counter[str] = Counter()
        output_sha256s: set[str] = set()
        latencies: list[float] = []
        timeout = httpx.Timeout(120)
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            for index in range(request_count):
                started = time.perf_counter()
                response = client.post(
                    url,
                    headers={"Host": self.hostname},
                    json={
                        **payload,
                        "request_id": f"{self.request_prefix}-{stage.lower()}-{index:05d}",
                    },
                )
                latency_ms = max((time.perf_counter() - started) * 1000, 0.001)
                response.raise_for_status()
                variant = response.headers.get(self.variant_header, "")
                if variant not in {"stable", "candidate"}:
                    raise self.local.error_type(f"gateway {self.local.label} variant is missing")
                document = response.json()
                if not isinstance(document, dict) or document.get("variant") != variant:
                    raise self.local.error_type(f"gateway {self.local.label} response is invalid")
                output_sha = document.get("output_sha256")
                if not isinstance(output_sha, str) or len(output_sha) != 64:
                    raise self.local.error_type(
                        f"gateway {self.local.label} output digest is invalid"
                    )
                responses[variant] += 1
                output_sha256s.add(output_sha)
                latencies.append(latency_ms)
        return {"responses": responses, "output_sha256s": output_sha256s, "latencies": latencies}

    def stage_evidence(
        self,
        stage: Literal["SHADOW", "CANARY_5", "CANARY_25", "ROLLED_BACK"],
        *,
        request_count: int,
        before: dict[str, Any],
        after: dict[str, Any],
        responses: Counter[str],
        output_sha256s: set[str],
        latencies: list[float],
        route: dict[str, Any],
    ) -> EvidenceT:
        stable_delta = self.local.variant_count(
            after, "request_counts", "stable"
        ) - self.local.variant_count(before, "request_counts", "stable")
        candidate_delta = self.local.variant_count(
            after, "request_counts", "candidate"
        ) - self.local.variant_count(before, "request_counts", "candidate")
        ratio = responses["candidate"] / request_count
        if stage == "SHADOW":
            passed = bool(
                responses == {"stable": request_count}
                and stable_delta == request_count
                and (candidate_delta >= request_count)
            )
            mirrored = candidate_delta >= request_count
        elif stage in {"CANARY_5", "CANARY_25"}:
            lower, upper = (0.01, 0.15) if stage == "CANARY_5" else (0.12, 0.4)
            passed = bool(
                responses["stable"] >= 1
                and responses["candidate"] >= 1
                and (lower <= ratio <= upper)
                and (stable_delta == responses["stable"])
                and (candidate_delta == responses["candidate"])
            )
            mirrored = False
        else:
            passed = bool(
                responses == {"stable": request_count}
                and stable_delta == request_count
                and (candidate_delta == 0)
            )
            mirrored = False
        if not passed or not output_sha256s or (not latencies):
            raise self.local.error_type(f"{self.local.label} {stage} traffic evidence failed")
        return self.stage_factory(
            stage=stage,
            passed=True,
            request_count=request_count,
            stable_request_delta=stable_delta,
            candidate_request_delta=candidate_delta,
            stable_response_count=responses["stable"],
            candidate_response_count=responses["candidate"],
            candidate_response_ratio=ratio,
            candidate_mirror_observed=mirrored,
            p95_latency_ms=kserve.percentile(latencies, 0.95),
            error_count=0,
            route_document_sha256=self.document_sha256(route),
            output_sha256s=tuple(sorted(output_sha256s)),
            model_release_observation_decision="NOT_REQUIRED_AFTER_ROLLBACK"
            if stage == "ROLLED_BACK"
            else "PASS",
        )


class ObservedLocalKServeProvider:
    """Bridge verified local KServe state into the existing deployment FSM."""

    def __init__(self, root: Path, stage: str, configuration: RolloutObservation[Any]) -> None:
        self._root = root
        self._stage = stage
        self._configuration = configuration

    def reconcile(self, target: ProviderTarget) -> ProviderResult:
        route = self._configuration.local.kubectl_json(
            self._root, "httproute", self._configuration.local.route_name
        )
        stable = self._configuration.local.kubectl_json(
            self._root, "inferenceservice", self._configuration.local.stable_service
        )
        candidate = self._configuration.local.kubectl_json(
            self._root, "inferenceservice", self._configuration.local.candidate_service
        )
        expected_route = self._configuration.route_document(cast(Any, self._stage))
        ready = bool(
            target.desired_stage == self._stage
            and target.route_name == self._configuration.local.route_name
            and (target.stable_service_name == self._configuration.local.stable_service)
            and resource_ready(stable)
            and resource_ready(candidate)
            and self._configuration.local.route_condition(route, "Accepted")
            and self._configuration.local.route_condition(route, "ResolvedRefs")
            and (
                route.get("metadata", {}).get("annotations", {}).get("ioap.openai.com/stage")
                == self._stage
            )
            and self._configuration.route_rules_equal(route, expected_route)
        )
        revision = (
            "local-kserve-"
            + self._configuration.document_sha256(
                {
                    "stage": self._stage,
                    "route_resource_version": route.get("metadata", {}).get("resourceVersion"),
                    "stable_resource_version": stable.get("metadata", {}).get("resourceVersion"),
                    "candidate_resource_version": candidate.get("metadata", {}).get(
                        "resourceVersion"
                    ),
                    "desired_spec_hash": target.desired_spec_hash,
                }
            )[:24]
        )
        return ProviderResult(
            ready=ready,
            observed_stage=self._stage,
            observed_traffic_percent=target.desired_traffic_percent,
            applied_spec_hash=target.desired_spec_hash if ready else None,
            provider_revision=revision,
            endpoint_url=f"http://{self._configuration.hostname}{self._configuration.route_path_prefix}/decision"
            if ready
            else None,
            reason_code=None if ready else "local_kserve_observation_mismatch",
        )
