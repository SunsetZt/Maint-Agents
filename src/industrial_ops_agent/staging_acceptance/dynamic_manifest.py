from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from industrial_ops_agent.staging_acceptance.collector import _canonical, _digest
from industrial_ops_agent.staging_acceptance.contracts import (
    StagingAcceptanceReport,
    UnverifiedScenario,
)
from industrial_ops_agent.staging_acceptance.dynamic_contracts import (
    REQUIRED_ASSERTIONS,
    DynamicAcceptanceResults,
    DynamicScenarioManifestEntry,
    DynamicScenarioResult,
    StagingDynamicAcceptanceManifest,
    normalized_utc,
)

AGGREGATOR_VERSION = "staging-dynamic-acceptance-aggregator/v1"
LIMITATION = (
    "Dynamic Staging evidence is complete only for production review; "
    "production sign-offs remain required."
)


def build_dynamic_acceptance_manifest(
    preflight: StagingAcceptanceReport,
    results: DynamicAcceptanceResults,
    *,
    preflight_artifact_digest: str,
    clock: Callable[[], datetime] | None = None,
) -> StagingDynamicAcceptanceManifest:
    generated_at = normalized_utc((clock or _utc_now)())
    global_blockers: list[str] = []
    if not _healthy_static_preflight(preflight):
        global_blockers.append("static_preflight_not_healthy")
    if not _bindings_match(preflight, results):
        global_blockers.append("dynamic_binding_mismatch")

    by_scenario = {item.scenario: item for item in results.scenarios}
    entries: list[DynamicScenarioManifestEntry] = []
    blockers = list(global_blockers)
    for scenario in UnverifiedScenario:
        result = by_scenario.get(scenario)
        entry, blocker = _manifest_entry(
            scenario,
            result,
            preflight=preflight,
            generated_at=generated_at,
            global_blockers=global_blockers,
        )
        entries.append(entry)
        if blocker is not None:
            blockers.append(blocker)

    blocker_codes = tuple(sorted(set(blockers)))
    summary = {
        "passed": sum(item.status == "PASSED" for item in entries),
        "failed": sum(item.status == "FAILED" for item in entries),
        "blocked": sum(item.status == "BLOCKED" for item in entries),
        "missing": sum(item.status == "MISSING" for item in entries),
    }
    status = (
        "BLOCKED"
        if blocker_codes
        else "STAGING_ACCEPTANCE_COMPLETE_PRODUCTION_REVIEW_REQUIRED"
    )
    dynamic_results_sha256 = _digest(
        _canonical(results.model_dump(mode="json"))
    )
    manifest = StagingDynamicAcceptanceManifest(
        schema_version=1,
        manifest_id="sha256:" + "0" * 64,
        aggregator_version=AGGREGATOR_VERSION,
        generated_at=generated_at,
        status=status,
        environment="STAGING",
        environment_id=results.environment_id,
        kube_context=results.kube_context,
        git_commit=results.git_commit,
        static_plan_sha256=results.static_plan_sha256,
        runner_version=results.runner_version,
        static_preflight_report_id=preflight.report_id,
        static_preflight_artifact_digest=preflight_artifact_digest,
        dynamic_results_sha256=dynamic_results_sha256,
        summary=summary,
        blocker_codes=blocker_codes,
        scenarios=tuple(entries),
        limitation=LIMITATION,
    )
    return manifest.model_copy(
        update={"manifest_id": dynamic_acceptance_manifest_identity(manifest)}
    )


def dynamic_acceptance_manifest_identity(
    manifest: StagingDynamicAcceptanceManifest,
) -> str:
    identity = {
        "aggregator_version": AGGREGATOR_VERSION,
        "generated_at": manifest.generated_at.isoformat(),
        "status": manifest.status,
        "environment_id": manifest.environment_id,
        "kube_context": manifest.kube_context,
        "git_commit": manifest.git_commit,
        "static_plan_sha256": manifest.static_plan_sha256,
        "runner_version": manifest.runner_version,
        "static_preflight_report_id": manifest.static_preflight_report_id,
        "static_preflight_artifact_digest": manifest.static_preflight_artifact_digest,
        "dynamic_results_sha256": manifest.dynamic_results_sha256,
        "summary": manifest.summary,
        "blocker_codes": manifest.blocker_codes,
        "scenarios": [item.model_dump(mode="json") for item in manifest.scenarios],
    }
    return _digest(_canonical(identity))


def validate_dynamic_acceptance_manifest_identity(
    manifest: StagingDynamicAcceptanceManifest,
) -> None:
    if manifest.manifest_id != dynamic_acceptance_manifest_identity(manifest):
        raise ValueError("dynamic acceptance manifest identity mismatch")


def _healthy_static_preflight(report: StagingAcceptanceReport) -> bool:
    checks_healthy = bool(report.checks) and all(
        item.status == "PASS" for item in report.checks
    )
    summary_healthy = report.summary == {
        "passed": len(report.checks),
        "blocked": 0,
    }
    unverified = [item.scenario for item in report.unverified_scenarios]
    scenario_contract_healthy = (
        len(unverified) == len(UnverifiedScenario)
        and len(set(unverified)) == len(unverified)
        and set(unverified) == set(UnverifiedScenario)
    )
    return (
        report.status == "PREFLIGHT_PASSED_ACCEPTANCE_INCOMPLETE"
        and checks_healthy
        and summary_healthy
        and scenario_contract_healthy
    )


def _bindings_match(
    report: StagingAcceptanceReport,
    results: DynamicAcceptanceResults,
) -> bool:
    return (
        report.environment == results.environment
        and report.environment_id == results.environment_id
        and report.kube_context == results.kube_context
        and report.git_commit == results.git_commit
        and report.plan_sha256 == results.static_plan_sha256
    )


def _manifest_entry(
    scenario: UnverifiedScenario,
    result: DynamicScenarioResult | None,
    *,
    preflight: StagingAcceptanceReport,
    generated_at: datetime,
    global_blockers: list[str],
) -> tuple[DynamicScenarioManifestEntry, str | None]:
    if result is None:
        return _missing_entry(scenario), "dynamic_scenario_missing"

    actual_assertions = {item.assertion_id for item in result.assertions}
    if actual_assertions != set(REQUIRED_ASSERTIONS[scenario]):
        return (
            _entry(result, "BLOCKED", "dynamic_assertion_contract_mismatch"),
            "dynamic_assertion_contract_mismatch",
        )
    if result.status == "FAILED":
        return (
            _entry(result, "FAILED", "dynamic_scenario_failed"),
            "dynamic_scenario_failed",
        )
    if result.status == "BLOCKED":
        return (
            _entry(result, "BLOCKED", "dynamic_scenario_blocked"),
            "dynamic_scenario_blocked",
        )
    if not all(item.passed for item in result.assertions):
        return (
            _entry(result, "BLOCKED", "dynamic_assertion_failed"),
            "dynamic_assertion_failed",
        )
    if result.started_at < preflight.generated_at or result.completed_at > generated_at:
        return (
            _entry(result, "BLOCKED", "dynamic_execution_window_invalid"),
            "dynamic_execution_window_invalid",
        )
    if global_blockers:
        reason = global_blockers[0]
        if reason == "static_preflight_not_healthy":
            return _entry(result, "BLOCKED", reason), None
        return _entry(result, "BLOCKED", "dynamic_binding_mismatch"), None
    return _entry(result, "PASSED", "verified"), None


def _entry(
    result: DynamicScenarioResult,
    status: str,
    reason: str,
) -> DynamicScenarioManifestEntry:
    return DynamicScenarioManifestEntry.model_validate(
        {
            "scenario": result.scenario,
            "status": status,
            "reason": reason,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
            "evidence_ref": result.evidence_ref,
            "artifact_digest": result.artifact_digest,
            "assertions": result.assertions,
        }
    )


def _missing_entry(scenario: UnverifiedScenario) -> DynamicScenarioManifestEntry:
    return DynamicScenarioManifestEntry(
        scenario=scenario,
        status="MISSING",
        reason="dynamic_scenario_missing",
        started_at=None,
        completed_at=None,
        evidence_ref=None,
        artifact_digest=None,
        assertions=(),
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)
