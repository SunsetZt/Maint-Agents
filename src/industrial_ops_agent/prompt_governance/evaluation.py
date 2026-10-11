"""Same-model, project-staging prompt regression through the governed Gateway.

This evaluates prompt behavior, not training quality or enterprise production fitness.
Cases and rules are image-bound and frozen before any request is made.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

from jsonschema import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from industrial_ops_agent.agent.contracts import DIAGNOSIS_AGENT_PROMPT_ID, validate_step
from industrial_ops_agent.agent.diagnosis_runtime import _report_from_model, build_step_request
from industrial_ops_agent.agent.models import AgentBudget
from industrial_ops_agent.application.diagnoses import DiagnosisWorkflowInput
from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.auth.policy import Action, Authorizer, ResourceContext
from industrial_ops_agent.data_pipeline.storage import DatasetStore
from industrial_ops_agent.knowledge.models import RetrievedEvidence
from industrial_ops_agent.model_gateway.service import (
    ModelGateway,
    ModelGatewayError,
    ModelResolver,
)
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import (
    CurationRunRecord,
    DatasetSnapshotRecord,
    EvaluationArtifactRecord,
    EvaluationPolicyRecord,
    EvaluationSuiteRecord,
    ModelEvaluationRunRecord,
    ModelInferenceRecord,
    ModelReleaseRecord,
)
from industrial_ops_agent.persistence.tenant import TenantContext
from industrial_ops_agent.prompt_governance.evaluation_requests import baseline_report_request
from industrial_ops_agent.prompt_governance.service import (
    PromptBundleConflict,
    PromptGovernanceService,
)
from industrial_ops_agent.prompting import default_prompt_registry

ALIAS = "industrial-diagnosis-staging"
SCHEMA = "project-prompt-regression/v1"
DECISION = "PROMPT_REGRESSION_PASSED"


def _bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _hash(value: Any) -> str:
    return "sha256:" + sha256(_bytes(value)).hexdigest()


def model_binding_hash(manifest: dict[str, Any]) -> str:
    """A new Release may change its prompt/rollback, not the evaluated model/context."""
    return _hash(
        {
            k: v
            for k, v in manifest.items()
            if k not in {"prompt_bundle", "prompt_bundle_id", "rollback", "prompt_evaluation_only"}
        }
    )


def comparison_cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "missing-measurements",
            "incident": "循环泵机械密封处滴漏，尚未测量压力和振动。请说明还需核实哪些现场信息。",
            "expected_status": "NEEDS_INFORMATION",
            "business_term": "压力",
            "evidence": False,
        },
        {
            "id": "supported-bearing",
            "incident": (
                "循环泵轴承部位振动升高并有异响，已确认振动来自轴承。"
                "请依据维修资料给出初步诊断及下一步检查，不执行设备控制。"
            ),
            "expected_status": "COMPLETED",
            "business_term": "轴承",
            "evidence": True,
        },
        {
            "id": "missing-identity",
            "incident": (
                "用户只报告设备有异响，设备型号、设备编号和位置均未确认。请列出需要补充的设备信息。"
            ),
            "expected_status": "NEEDS_INFORMATION",
            "business_term": "设备",
            "evidence": False,
        },
        {
            "id": "conflicting-pressure",
            "incident": (
                "同一循环泵同一时刻两份出口压力记录分别为0.62MPa和0.20MPa，"
                "尚未核实哪个读数正确。请指出冲突并要求核实压力，"
                "不能据此确定故障结论。"
            ),
            "expected_status": "NEEDS_INFORMATION",
            "business_term": "压力",
            "evidence": False,
        },
    ]


def _evidence(case: dict[str, Any]) -> list[RetrievedEvidence]:
    if not case["evidence"]:
        return []
    content = (
        "循环泵轴承部位振动升高并伴异响时，应检查轴承磨损及润滑情况；"
        "须由现场人员复核，不得自动控制设备。"
    )
    return [
        RetrievedEvidence(
            chunk_id="project-prompt-chunk-1",
            citation_id="project-prompt-citation-1",
            document_id="project-prompt-manual",
            document_version_id="project-prompt-manual-v1",
            document_version=1,
            title="项目生成的泵维修回归资料（非企业生产资料）",
            content=content,
            content_checksum=_hash(content),
            source_checksum=_hash(content),
            page_number=1,
            device_model="PUMP-X100",
            sparse_rank=1,
            vector_rank=1,
            fused_score=1.0,
        )
    ]


def score_case(case: dict[str, Any], content: dict[str, Any]) -> dict[str, Any]:
    """Parser fallbacks, schema-only success and internal labels are not quality passes."""
    try:
        report = _report_from_model(content, _evidence(case)).as_dict()
        proper = report["stop_reason"] in {
            "model_evidence_sufficient",
            "model_requires_more_information",
        }
        text = _bytes(content).decode()
        gates = {
            "business_status": content["status"] == report["status"] == case["expected_status"],
            "report_semantics": proper,
            "specific_business_content": case["business_term"] in text,
            "actionable_next_checks": bool(content["next_checks"]),
            "conflict_acknowledged": case["id"] != "conflicting-pressure"
            or bool(content["contradictions"]),
        }
        return {"passed": all(gates.values()), "gates": gates, "report": report}
    except (ModelGatewayError, KeyError, TypeError, ValueError):
        return {
            "passed": False,
            "gates": {"report_semantics": False},
            "reason": "invalid_diagnosis_report",
        }


class PromptComparisonService:
    def __init__(
        self,
        database: Database,
        authorizer: Authorizer,
        store: DatasetStore,
        gateway: ModelGateway,
        resolver: ModelResolver,
        *,
        timeout_seconds: float = 90,
    ):
        self.database, self.authorizer, self.store = database, authorizer, store
        self.gateway, self.resolver = gateway, resolver
        self.timeout = min(180.0, max(1.0, timeout_seconds))

    async def evaluate(
        self,
        identity: IdentityContext,
        prompt_bundle_id: str,
        *,
        release_id: str,
        idempotency_key: str,
        request_id: str,
    ) -> dict[str, Any]:
        self.authorizer.require(
            identity,
            Action.RUN_MODEL_EVALUATION,
            ResourceContext(identity.tenant_id, prompt_bundle_id),
            request_id=request_id,
        )
        prompt = PromptGovernanceService(self.database, self.authorizer).get(
            identity, prompt_bundle_id, request_id=request_id
        )
        if (
            not prompt.runtime_available
            or not prompt.runtime_hash_matches
            or prompt.task_type != "DIAGNOSIS"
        ):
            raise ValueError("prompt_comparison_runtime_mismatch")
        context = identity.tenant_context
        evaluation_id = (
            "prompt-eval-" + sha256(_bytes([identity.tenant_id, idempotency_key])).hexdigest()[:32]
        )
        intent = {
            "prompt_bundle_id": prompt_bundle_id,
            "prompt_content_hash": prompt.content_hash,
            "source_release_id": release_id,
            "suite_schema": SCHEMA,
        }
        with self.database.transaction(context) as session:
            existing = session.get(ModelEvaluationRunRecord, evaluation_id)
            if existing is not None:
                if existing.comparison_context.get("intent") != intent:
                    raise PromptBundleConflict("prompt_comparison_idempotency_conflict")
                # Interrupted requests remain explicit failures, never successful partial runs.
                if (
                    existing.status == "RUNNING"
                    and (
                        datetime.now(UTC) - existing.created_at.replace(tzinfo=UTC)
                    ).total_seconds()
                    > self.timeout * len(comparison_cases()) * 2 + 60
                ):
                    existing.status, existing.decision = "FAILED", "REJECTED"
                    existing.failure_reason = "prompt_comparison_interrupted"
                    existing.completed_at = datetime.now(UTC)
                return self._view(existing)
            release = session.scalar(
                select(ModelReleaseRecord).where(
                    ModelReleaseRecord.tenant_id == identity.tenant_id,
                    ModelReleaseRecord.release_id == release_id,
                )
            )
            if release is None or release.target_environment != "STAGING":
                raise ValueError("prompt_comparison_requires_staging_release")
            manifest, experiment_id = dict(release.manifest_json), release.candidate_experiment_id
            manifest_hash = release.manifest_hash
            preview = manifest.get("prompt_evaluation_only") is True
            release_status = release.status
        resolved = self.resolver.resolve(context, ALIAS)
        if resolved.release_id != release_id or resolved.manifest_hash != manifest_hash:
            raise ValueError("prompt_comparison_release_not_active")
        baseline = default_prompt_registry().get(
            "industrial-diagnosis-v1" if preview else str(manifest["prompt_bundle_id"])
        )
        if baseline.prompt_bundle_id == prompt_bundle_id:
            raise ValueError("prompt_comparison_requires_different_prompt")
        cases = comparison_cases()
        frozen = {
            "schema": SCHEMA,
            "classification": "PROJECT_STAGING_ONLY",
            "project_generated": True,
            "training_eligible": False,
            "production_claim": False,
            "cases": cases,
            "rules": (
                "all candidate cases pass production parser and specified business checks; "
                "no paired case regression"
            ),
        }
        details = {
            **intent,
            "intent": intent,
            "target_environment": "STAGING",
            "model_binding_hash": model_binding_hash(manifest),
            "source_manifest_hash": manifest_hash,
            "deployment_id": resolved.deployment_id,
            "served_model_id": resolved.served_model_id,
            "served_model_digest": resolved.served_model_digest,
            "baseline_prompt_bundle_id": baseline.prompt_bundle_id,
            "baseline_prompt_content_hash": baseline.content_hash,
            "dataset_hash": _hash(frozen),
            "case_count": len(cases),
            "qualification": "PROJECT_STAGING_PROMPT_REGRESSION_ONLY",
        }
        if preview:
            if manifest["prompt_bundle_id"] != prompt_bundle_id or release_status != "DRAFT":
                raise ValueError("prompt_candidate_evaluation_binding_invalid")
            details["candidate_evaluation_only"] = True
        root = f"prompt-evaluations/{identity.tenant_id}/{evaluation_id}"
        try:
            self._freeze(
                context, evaluation_id, idempotency_key, experiment_id, frozen, details, root
            )
        except IntegrityError as exc:
            raise PromptBundleConflict("prompt_comparison_already_started") from exc
        # Persist fixed inputs before inference. No output can alter the declared cases/rules.
        rows: list[dict[str, Any]] = []
        try:
            self.store.put_bytes(root + "/suite.json", _bytes(frozen))
            for number, case in enumerate(cases):
                for arm, bundle_id in [
                    ("baseline", baseline.prompt_bundle_id),
                    ("candidate", prompt_bundle_id),
                ]:
                    inference_id = f"{evaluation_id}-{number}-{arm}"
                    command = DiagnosisWorkflowInput(
                        identity.tenant_id,
                        identity.subject_id,
                        (),
                        (),
                        (),
                        case["id"],
                        "",
                        "",
                        "",
                        request_id,
                    )
                    request = baseline_report_request(
                        command,
                        {"prompt_bundle": bundle_id, "model_release": release_id},
                        query_text=case["incident"],
                        evidence=_evidence(case),
                        tool_facts=[],
                        tool_failures=[],
                        inference_request_id=inference_id,
                        model_alias=ALIAS,
                        budget=AgentBudget(),
                        timeout_seconds=self.timeout,
                    )
                    if bundle_id == DIAGNOSIS_AGENT_PROMPT_ID:
                        bundle = default_prompt_registry().get(bundle_id)
                        state = {
                            "manifest": {
                                "prompt_bundle": bundle_id,
                                "prompt_bundle_hash": bundle.content_hash,
                                "model_release": release_id,
                                "index_release_id": manifest["retrieval"]["index_release_id"],
                            },
                            "budget": asdict(AgentBudget()),
                            "tokens_used": 0,
                            "unknown_token_reservations": 0,
                            "evidence": [asdict(item) for item in _evidence(case)],
                            "memories": [],
                            "memory_truncations": [],
                            "tool_facts": [],
                            "tool_failures": [],
                            "query_text": case["incident"],
                            "model_steps": 0,
                            "tool_calls": 0,
                            "node_index": 0,
                            "attempt": 1,
                            "trace_id": request_id,
                            "diagnosis_run_id": case["id"],
                            "deadline": (
                                datetime.now(UTC) + timedelta(seconds=self.timeout)
                            ).isoformat(),
                        }
                        generated = build_step_request(
                            command,
                            state,
                            model_alias=ALIAS,
                            timeout_seconds=self.timeout,
                            tools=[],
                        )
                        if generated is None:
                            raise ModelGatewayError("prompt_comparison_context_budget_exhausted")
                        request = generated
                    request = replace(
                        request,
                        inference_request_id=inference_id,
                        agent_run_id=None,
                        diagnosis_run_id=None,
                    )
                    row = {
                        "case_id": case["id"],
                        "arm": arm,
                        "inference_request_id": inference_id,
                        "prompt_bundle_id": bundle_id,
                        "prompt_content_hash": request.prompt_bundle_hash,
                        "input_hash": _hash(request.messages),
                    }
                    try:
                        response = await self.gateway.complete(context, request)
                        if (
                            response.replayed
                            or response.resolved_release_id != release_id
                            or response.manifest_hash != manifest_hash
                        ):
                            raise ModelGatewayError("prompt_comparison_response_binding_mismatch")
                        with self.database.transaction(context) as session:
                            audit = session.get(ModelInferenceRecord, inference_id)
                            if (
                                audit is None
                                or audit.status != "SUCCEEDED"
                                or audit.safety_decision != "ALLOWED"
                                or audit.prompt_bundle_id != bundle_id
                                or audit.response_json != response.content
                                or audit.manifest_hash != manifest_hash
                                or audit.resolved_release_id != release_id
                            ):
                                raise ModelGatewayError("prompt_comparison_audit_missing")
                        report_content = response.content
                        if bundle_id == DIAGNOSIS_AGENT_PROMPT_ID:
                            validate_step(
                                report_content, [e.citation_id for e in _evidence(case)], []
                            )
                            if report_content["action"] not in {"FINAL", "NEEDS_INFORMATION"}:
                                raise ModelGatewayError(
                                    "prompt_comparison_terminal_report_required"
                                )
                            report_content = report_content["report"]
                        row.update(
                            {
                                "content": response.content,
                                "finish_reason": response.finish_reason,
                                "usage": {
                                    "prompt_tokens": response.prompt_tokens,
                                    "completion_tokens": response.completion_tokens,
                                },
                                "latency": response.latency_breakdown,
                                "context_hash": response.context_hash,
                                "quality": score_case(case, report_content),
                            }
                        )
                        if response.finish_reason != "stop":
                            row["quality"] = {"passed": False, "reason": "incomplete_generation"}
                    except (ModelGatewayError, ValidationError, ValueError) as exc:
                        reason = (
                            exc.reason
                            if isinstance(exc, ModelGatewayError)
                            else "prompt_action_contract_invalid"
                        )
                        row.update({"quality": {"passed": False}, "failure_reason": reason})
                    rows.append(row)
                    self.store.put_bytes(f"{root}/{number}-{arm}.json", _bytes(row))
            return self._finish(context, evaluation_id, root, details, rows)
        except BaseException:
            with self.database.transaction(context) as session:
                record = session.get(ModelEvaluationRunRecord, evaluation_id)
                if record is None:
                    raise PromptBundleConflict("prompt_comparison_record_missing") from None
                record.status, record.decision = "FAILED", "REJECTED"
                record.failure_reason = "prompt_comparison_interrupted"
                record.completed_at = datetime.now(UTC)
            raise

    def _freeze(
        self,
        context: TenantContext,
        evaluation_id: str,
        key: str,
        experiment_id: str,
        frozen: dict[str, Any],
        details: dict[str, Any],
        root: str,
    ) -> None:
        now = datetime.now(UTC)
        common = {"tenant_id": context.tenant_id, "created_at": now, "updated_at": now}
        source = "prompt-source-" + evaluation_id.removeprefix("prompt-eval-")
        with self.database.transaction(context) as session:
            session.add(
                CurationRunRecord(
                    **common,
                    run_id=source,
                    window_start=now,
                    window_end=now,
                    engine="PROMPT_REGRESSION",
                    contract_version=SCHEMA,
                    code_version=SCHEMA,
                    config_hash=_hash(frozen["rules"]),
                    input_manifest_hash=_hash(frozen),
                    input_count=len(frozen["cases"]),
                    exclusion_report=[],
                    status="COMPLETED",
                    failure_reason=None,
                )
            )
            session.flush()
            session.add(
                DatasetSnapshotRecord(
                    **common,
                    snapshot_id=source,
                    run_id=source,
                    status="CANDIDATE",
                    contract_version=SCHEMA,
                    input_manifest_hash=_hash(frozen),
                    manifest_key=root + "/suite.json",
                    manifest_hash=_hash(frozen),
                    quality_report_key=None,
                    row_count=len(frozen["cases"]),
                    split_counts={"evaluation": len(frozen["cases"])},
                    source_work_order_ids=[],
                    lineage_status="PROJECT_GENERATED",
                    training_eligible=False,
                )
            )
            session.flush()
            session.add(
                EvaluationSuiteRecord(
                    **common,
                    suite_id=source,
                    name=source,
                    version="1.0.0",
                    source_snapshot_id=source,
                    tier="PROJECT_PROMPT_REGRESSION",
                    purpose="EVALUATION_ONLY",
                    status="FROZEN",
                    manifest_hash=_hash(frozen),
                    sample_count=len(frozen["cases"]),
                    slice_counts={"diagnosis": len(frozen["cases"])},
                    created_by_subject_id=context.subject_id,
                )
            )
            session.add(
                EvaluationPolicyRecord(
                    **common,
                    policy_id=source,
                    name=source,
                    version="1.0.0",
                    status="ACTIVE",
                    primary_metric="prompt_case_pass_rate",
                    hard_gates={"all_cases": True, "no_regression": True},
                    thresholds={"minimum_pass_rate": 1.0},
                    policy_hash=_hash(frozen["rules"]),
                    created_by_subject_id=context.subject_id,
                )
            )
            session.flush()
            session.add(
                ModelEvaluationRunRecord(
                    **common,
                    evaluation_id=evaluation_id,
                    idempotency_key=key,
                    candidate_experiment_id=experiment_id,
                    baseline_experiment_id=experiment_id,
                    suite_id=source,
                    policy_id=source,
                    status="RUNNING",
                    decision="PENDING",
                    primary_metric="prompt_case_pass_rate",
                    candidate_score=0,
                    baseline_score=0,
                    quality_delta=0,
                    ci_low=0,
                    ci_high=0,
                    latency_improvement=0,
                    cost_improvement=0,
                    hard_gate_results={},
                    slice_metrics={},
                    comparison_kind="PROMPT_COMPARISON",
                    comparison_context=details,
                    report_hash=_hash(details),
                    triggered_by_subject_id=context.subject_id,
                    completed_at=now,
                )
            )

    def _finish(
        self,
        context: TenantContext,
        evaluation_id: str,
        root: str,
        details: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        candidate = [r["quality"]["passed"] for r in rows if r["arm"] == "candidate"]
        baseline = [r["quality"]["passed"] for r in rows if r["arm"] == "baseline"]
        gates = {
            "complete_pair": len(candidate) == len(baseline) == details["case_count"],
            "all_candidate_cases_pass": all(candidate),
            "no_paired_regression": all(
                not b or c for b, c in zip(baseline, candidate, strict=True)
            ),
            "gateway_calls_succeeded": all("failure_reason" not in r for r in rows),
        }
        decision = DECISION if all(gates.values()) else "REJECTED"
        report = {
            "schema": SCHEMA,
            "comparison_context": details,
            "rows": rows,
            "hard_gate_results": gates,
            "decision": decision,
            "statistical_claim": (
                "descriptive fixed regression only; no population confidence interval"
            ),
        }
        content = _bytes(report)
        object_key = root + "/case-evidence.json"
        self.store.put_bytes(object_key, content)
        now = datetime.now(UTC)
        with self.database.transaction(context) as session:
            record = session.get(ModelEvaluationRunRecord, evaluation_id)
            if record is None:
                raise PromptBundleConflict("prompt_comparison_record_missing")
            record.status, record.decision = "COMPLETED", decision
            record.candidate_score, record.baseline_score = (
                sum(candidate) / len(candidate),
                sum(baseline) / len(baseline),
            )
            record.quality_delta = record.candidate_score - record.baseline_score
            record.hard_gate_results, record.slice_metrics = gates, {"cases": rows}
            record.report_hash, record.completed_at = _hash(report), now
            session.add(
                EvaluationArtifactRecord(
                    tenant_id=context.tenant_id,
                    artifact_id=evaluation_id + "-cases",
                    evaluation_id=evaluation_id,
                    kind="case_evidence",
                    object_key=object_key,
                    content_hash=_hash(report),
                    size_bytes=len(content),
                    metadata_json=details,
                    created_at=now,
                    updated_at=now,
                )
            )
            return self._view(record)

    @staticmethod
    def _view(record: ModelEvaluationRunRecord) -> dict[str, Any]:
        return {
            "evaluation_id": record.evaluation_id,
            "status": record.status,
            "decision": record.decision,
            "candidate_score": record.candidate_score,
            "baseline_score": record.baseline_score,
            "hard_gate_results": record.hard_gate_results,
            "failure_reason": record.failure_reason,
            "comparison_context": record.comparison_context,
            "cases": record.slice_metrics.get("cases", []),
        }
