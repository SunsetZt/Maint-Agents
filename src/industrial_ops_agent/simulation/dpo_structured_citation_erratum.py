"""Additive DPO v3 citation-fixture erratum with real tenant-safe verification."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.evaluation.runtime_backend import DatabaseCitationVerifier
from industrial_ops_agent.knowledge.seed import install_synthetic_knowledge
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import AssetRecord, Base, TenantRecord
from industrial_ops_agent.security_audit import InMemorySecurityAuditSink, SecurityAuditor
from industrial_ops_agent.simulation import dpo_recovery_value_lab as runtime_lab
from industrial_ops_agent.simulation import dpo_structured_value_lab as lab

SCHEMA_VERSION: Literal["enterprise-dpo-structured-citation-erratum/v1"] = (
    "enterprise-dpo-structured-citation-erratum/v1"
)
STATUS: Literal[
    "DPO_STRUCTURED_ENTERPRISE_VALUE_ACCEPTED_AFTER_CITATION_FIXTURE_ERRATUM"
] = "DPO_STRUCTURED_ENTERPRISE_VALUE_ACCEPTED_AFTER_CITATION_FIXTURE_ERRATUM"
AUTHORITATIVE_LATEST_RELATIVE = lab.OUTPUT_RELATIVE / "authoritative-latest.json"
KNOWLEDGE_SEED_SOURCE = Path("src/industrial_ops_agent/knowledge/seed.py")
CITATION_BACKEND_SOURCE = Path(
    "src/industrial_ops_agent/evaluation/runtime_backend.py"
)


class DpoStructuredCitationErratumError(RuntimeError):
    """The immutable DPO v3 outcome or corrected citation proof is inconsistent."""


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileBinding(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CorrectedCitationBoundary(_ClosedModel):
    case_count: Literal[8] = 8
    original_citation_service_pass_count: Literal[0] = 0
    reproduced_wrong_model_failure_count: Literal[8] = 8
    corrected_model_pass_count: Literal[8] = 8
    original_asset_model_code: Literal["HX-710"] = "HX-710"
    eligible_citation_device_model: Literal["PUMP-X100"] = "PUMP-X100"
    corrected_asset_model_code: Literal["PUMP-X100"] = "PUMP-X100"
    real_tenant_safe_citation_service: Literal[True] = True
    isolated_in_memory_database: Literal[True] = True
    candidate_citations_unchanged: Literal[True] = True
    cases: tuple[dict[str, Any], ...] = Field(min_length=8, max_length=8)


class DpoStructuredCitationErratumReport(_ClosedModel):
    schema_version: Literal[
        "enterprise-dpo-structured-citation-erratum/v1"
    ] = SCHEMA_VERSION
    classification: Literal["LOCAL_STAGING_PROJECT_AUTHORIZED"] = lab.CLASSIFICATION
    enterprise_scope: Literal["PROJECT_INTERNAL"] = "PROJECT_INTERNAL"
    production_claim: Literal[False] = False
    external_enterprise_production_claim: Literal[False] = False
    generated_at: datetime
    status: Literal[
        "DPO_STRUCTURED_ENTERPRISE_VALUE_ACCEPTED_AFTER_CITATION_FIXTURE_ERRATUM"
    ] = STATUS
    decision: Literal["ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"] = (
        "ELIGIBLE_FOR_MODEL_RELEASE_DRAFT"
    )
    run_id: str = Field(pattern=r"^dpo-structured-[0-9a-f]{20}$")
    defect_id: Literal["CITATION_FIXTURE_ASSET_MODEL_MISMATCH"] = (
        "CITATION_FIXTURE_ASSET_MODEL_MISMATCH"
    )
    defect_scope: Literal["RUNTIME_CITATION_FIXTURE_ONLY"] = (
        "RUNTIME_CITATION_FIXTURE_ONLY"
    )
    correction_method: Literal["PAIRED_REAL_CITATION_SERVICE_MODEL_SCOPE_CHECK"] = (
        "PAIRED_REAL_CITATION_SERVICE_MODEL_SCOPE_CHECK"
    )
    source: FileBinding
    immutable_original_source: FileBinding
    immutable_runtime_helper_source: FileBinding
    knowledge_seed_source: FileBinding
    citation_backend_source: FileBinding
    original_outcome: FileBinding
    original_latest_pointer: FileBinding
    original_observations: FileBinding
    original_gold_retirement: FileBinding
    original_evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corrected_citation_boundary: CorrectedCitationBoundary
    corrected_hard_gates: dict[str, bool]
    corrected_failed_hard_gates: tuple[str, ...]
    candidate_accepted: Literal[True] = True
    release_draft_eligible: Literal[True] = True
    formal_model_release_created: Literal[False] = False
    runtime_eligible: Literal[False] = False
    same_gold_reuse_permitted: Literal[False] = False
    original_outcome_mutated: Literal[False] = False
    adapter_mutated: Literal[False] = False
    model_output_mutated: Literal[False] = False
    quality_metrics_changed: Literal[False] = False
    formal_gold_model_evaluation_replayed: Literal[False] = False
    runtime_citation_boundary_rechecked: Literal[True] = True
    original_gold_retirement_preserved: Literal[True] = True
    evidence_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_dpo_structured_citation_erratum(repo_root: Path) -> Path:
    """Correct only the mismatched citation fixture without replaying model evaluation."""

    root = repo_root.resolve(strict=True)
    original_path = _original_outcome_path(root)
    original = lab.verify_dpo_structured_value(root, original_path)
    _require_known_defect(original)
    final_directory = root / lab.OUTPUT_RELATIVE / "errata" / original.run_id
    final_path = final_directory / "acceptance.json"
    if final_path.is_file():
        verify_dpo_structured_citation_erratum(root, final_path)
        return final_path
    if final_directory.exists():
        raise DpoStructuredCitationErratumError("immutable erratum directory already exists")

    observations_path = _inside_file(root, Path(original.observations.path))
    retirement_path = _inside_file(root, Path(original.gold_retirement_path))
    gold_path = _inside_file(root, Path(original.gold_manifest.path))
    observations = _load_object(observations_path)
    gold = _load_object(gold_path)
    correction = _paired_citation_projection(
        _require_rows(gold, "rows"),
        _require_rows(observations, "rows"),
        _require_rows(observations, "runtime_boundary_cases"),
    )
    corrected_gates = corrected_hard_gates(
        dict(original.hard_gates),
        original.failed_hard_gates,
        dict(original.runtime_boundaries),
        correction,
    )
    corrected_failed = tuple(
        name for name in lab._HARD_GATES if not corrected_gates[name]
    )
    if corrected_failed:
        raise DpoStructuredCitationErratumError("corrected DPO v3 gates still fail")

    source_path = Path(__file__).resolve(strict=True)
    original_source = _inside_file(root, Path(original.source.path))
    runtime_helper_source = _inside_file(root, Path(original.runtime_helper_source.path))
    latest_pointer = _inside_file(root, lab.LATEST_RELATIVE)
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": lab.CLASSIFICATION,
        "enterprise_scope": "PROJECT_INTERNAL",
        "production_claim": False,
        "external_enterprise_production_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": STATUS,
        "decision": "ELIGIBLE_FOR_MODEL_RELEASE_DRAFT",
        "run_id": original.run_id,
        "defect_id": "CITATION_FIXTURE_ASSET_MODEL_MISMATCH",
        "defect_scope": "RUNTIME_CITATION_FIXTURE_ONLY",
        "correction_method": "PAIRED_REAL_CITATION_SERVICE_MODEL_SCOPE_CHECK",
        "source": _file_binding(root, source_path),
        "immutable_original_source": _file_binding(root, original_source),
        "immutable_runtime_helper_source": _file_binding(root, runtime_helper_source),
        "knowledge_seed_source": _file_binding(root, root / KNOWLEDGE_SEED_SOURCE),
        "citation_backend_source": _file_binding(root, root / CITATION_BACKEND_SOURCE),
        "original_outcome": _file_binding(root, original_path),
        "original_latest_pointer": _file_binding(root, latest_pointer),
        "original_observations": _file_binding(root, observations_path),
        "original_gold_retirement": _file_binding(root, retirement_path),
        "original_evidence_chain_sha256": original.evidence_chain_sha256,
        "corrected_citation_boundary": correction,
        "corrected_hard_gates": corrected_gates,
        "corrected_failed_hard_gates": corrected_failed,
        "candidate_accepted": True,
        "release_draft_eligible": True,
        "formal_model_release_created": False,
        "runtime_eligible": False,
        "same_gold_reuse_permitted": False,
        "original_outcome_mutated": False,
        "adapter_mutated": False,
        "model_output_mutated": False,
        "quality_metrics_changed": False,
        "formal_gold_model_evaluation_replayed": False,
        "runtime_citation_boundary_rechecked": True,
        "original_gold_retirement_preserved": True,
    }
    draft = DpoStructuredCitationErratumReport.model_validate(
        {**unsigned, "evidence_chain_sha256": "0" * 64}
    )
    report = draft.model_copy(
        update={
            "evidence_chain_sha256": _digest(
                draft.model_dump(mode="json", exclude={"evidence_chain_sha256"})
            )
        }
    )
    temporary = final_directory.with_name(f".{final_directory.name}.{uuid4().hex}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        _write_json(temporary / "acceptance.json", report.model_dump(mode="json"))
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(final_directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    _write_authoritative_latest(root, final_path, report)
    verify_dpo_structured_citation_erratum(root, final_path)
    return final_path


def verify_dpo_structured_citation_erratum(
    repo_root: Path,
    erratum_path: Path | None = None,
) -> DpoStructuredCitationErratumReport:
    root = repo_root.resolve(strict=True)
    path = (
        _authoritative_outcome_path(root)
        if erratum_path is None
        else _inside_file(root, erratum_path)
    )
    try:
        report = DpoStructuredCitationErratumReport.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise DpoStructuredCitationErratumError("DPO citation erratum is invalid") from exc
    unsigned = report.model_dump(mode="json", exclude={"evidence_chain_sha256"})
    if report.evidence_chain_sha256 != _digest(unsigned):
        raise DpoStructuredCitationErratumError("DPO citation erratum chain changed")
    bindings = (
        report.source,
        report.immutable_original_source,
        report.immutable_runtime_helper_source,
        report.knowledge_seed_source,
        report.citation_backend_source,
        report.original_outcome,
        report.original_latest_pointer,
        report.original_observations,
        report.original_gold_retirement,
    )
    for binding in bindings:
        if _file_binding(root, _inside_file(root, Path(binding.path))) != binding.model_dump(
            mode="json"
        ):
            raise DpoStructuredCitationErratumError("DPO citation erratum input changed")

    original_path = _inside_file(root, Path(report.original_outcome.path))
    original = lab.verify_dpo_structured_value(root, original_path)
    _require_known_defect(original)
    if (
        original.run_id != report.run_id
        or original.evidence_chain_sha256 != report.original_evidence_chain_sha256
        or Path(original.source.path) != Path(report.immutable_original_source.path)
        or Path(original.runtime_helper_source.path)
        != Path(report.immutable_runtime_helper_source.path)
        or Path(original.observations.path) != Path(report.original_observations.path)
        or Path(original.gold_retirement_path)
        != Path(report.original_gold_retirement.path)
    ):
        raise DpoStructuredCitationErratumError("DPO citation original binding changed")
    observations = _load_object(_inside_file(root, Path(report.original_observations.path)))
    gold = _load_object(_inside_file(root, Path(original.gold_manifest.path)))
    correction = _paired_citation_projection(
        _require_rows(gold, "rows"),
        _require_rows(observations, "rows"),
        _require_rows(observations, "runtime_boundary_cases"),
    )
    corrected_gates = corrected_hard_gates(
        dict(original.hard_gates),
        original.failed_hard_gates,
        dict(original.runtime_boundaries),
        correction,
    )
    corrected_failed = tuple(
        name for name in lab._HARD_GATES if not corrected_gates[name]
    )
    if (
        correction != report.corrected_citation_boundary.model_dump(mode="json")
        or corrected_gates != report.corrected_hard_gates
        or corrected_failed != report.corrected_failed_hard_gates
        or corrected_failed
    ):
        raise DpoStructuredCitationErratumError("DPO citation correction changed")
    return report


def corrected_hard_gates(
    original_gates: dict[str, bool],
    original_failed: tuple[str, ...],
    original_boundaries: dict[str, Any],
    corrected_citation: dict[str, Any],
) -> dict[str, bool]:
    """Replace only the citation-fixture-dependent runtime boundary result."""

    if original_failed != ("runtime_security_boundaries",):
        raise DpoStructuredCitationErratumError("original failure is outside erratum scope")
    if any(
        not passed
        for name, passed in original_gates.items()
        if name != "runtime_security_boundaries"
    ):
        raise DpoStructuredCitationErratumError("an unrelated original gate failed")
    case_count = int(original_boundaries.get("probe_case_count", 0))
    runtime_gates = original_boundaries.get("gate_results")
    corrected = dict(original_gates)
    corrected["runtime_security_boundaries"] = (
        case_count == 8
        and isinstance(runtime_gates, dict)
        and all(bool(value) for value in runtime_gates.values())
        and original_boundaries.get("business_side_effect_count") == 0
        and original_boundaries.get("citation_service_pass_count") == 0
        and corrected_citation.get("case_count") == case_count
        and corrected_citation.get("reproduced_wrong_model_failure_count") == case_count
        and corrected_citation.get("corrected_model_pass_count") == case_count
        and corrected_citation.get("candidate_citations_unchanged") is True
    )
    return corrected


def _paired_citation_projection(
    gold_rows: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    original_boundary_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    if not (
        len(gold_rows) == len(observations) == len(original_boundary_cases)
        and gold_rows
    ):
        raise DpoStructuredCitationErratumError("DPO citation evidence is incomplete")
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    now = datetime.now(UTC)
    with Session(engine) as session:
        session.add(TenantRecord(id="dpo-structured-erratum-tenant", status="active"))
        session.flush()
        session.add_all(
            [
                AssetRecord(
                    tenant_id="dpo-structured-erratum-tenant",
                    asset_id="device-hx-control",
                    source_system="dpo-structured-citation-erratum",
                    source_record_id="device-hx-control",
                    as_of=now,
                    model_code="HX-710",
                    version=1,
                ),
                AssetRecord(
                    tenant_id="dpo-structured-erratum-tenant",
                    asset_id="device-pump-corrected",
                    source_system="dpo-structured-citation-erratum",
                    source_record_id="device-pump-corrected",
                    as_of=now,
                    model_code="PUMP-X100",
                    version=1,
                ),
            ]
        )
        install_synthetic_knowledge(
            session,
            tenant_id="dpo-structured-erratum-tenant",
            now=now,
        )
        session.commit()
    database = Database.from_engine(engine)
    verifier = DatabaseCitationVerifier(
        database,
        Authorizer(
            SecurityAuditor(
                InMemorySecurityAuditSink(),
                hash_key=b"dpo-structured-citation-erratum",
            )
        ),
        tenant_id="dpo-structured-erratum-tenant",
        subject_id="dpo-structured-citation-erratum-evaluator",
    )
    control_failures = 0
    corrected_passes = 0
    case_evidence: list[dict[str, Any]] = []
    try:
        for row, observation, original_case in zip(
            gold_rows, observations, original_boundary_cases, strict=True
        ):
            case = runtime_lab._evaluation_case(row)
            citations = tuple(
                value.lower()
                for value in re.findall(
                    r"citation-[a-z0-9-]+",
                    str(observation["candidate"]["text"]),
                    re.I,
                )
            )
            if citations != (str(row["allowed_citation"]),):
                raise DpoStructuredCitationErratumError(
                    "candidate citation differs from frozen observation"
                )
            original_evidence = original_case.get("citation_service_evidence")
            if (
                original_case.get("case_id") != case.case_id
                or original_case.get("citation_service_passed") is not False
                or not isinstance(original_evidence, dict)
                or original_evidence.get("visible_count") != 0
                or original_evidence.get("hidden_count") != 1
            ):
                raise DpoStructuredCitationErratumError(
                    "original citation defect evidence changed"
                )
            control_case = replace(
                case,
                slices={**case.slices, "device_id": "device-hx-control"},
            )
            corrected_case = replace(
                case,
                slices={**case.slices, "device_id": "device-pump-corrected"},
            )
            control_passed, control_evidence = verifier.verify(control_case, citations)
            corrected_passed, corrected_evidence = verifier.verify(
                corrected_case, citations
            )
            control_failures += int(not control_passed)
            corrected_passes += int(corrected_passed)
            case_evidence.append(
                {
                    "case_id": case.case_id,
                    "candidate_citations": list(citations),
                    "original_recorded_evidence": original_evidence,
                    "wrong_model_control_passed": control_passed,
                    "wrong_model_control_evidence": control_evidence,
                    "corrected_model_passed": corrected_passed,
                    "corrected_model_evidence": corrected_evidence,
                }
            )
    finally:
        database.dispose()
    return {
        "case_count": len(gold_rows),
        "original_citation_service_pass_count": 0,
        "reproduced_wrong_model_failure_count": control_failures,
        "corrected_model_pass_count": corrected_passes,
        "original_asset_model_code": "HX-710",
        "eligible_citation_device_model": "PUMP-X100",
        "corrected_asset_model_code": "PUMP-X100",
        "real_tenant_safe_citation_service": True,
        "isolated_in_memory_database": True,
        "candidate_citations_unchanged": True,
        "cases": case_evidence,
    }


def _require_known_defect(original: lab.DpoStructuredValueReport) -> None:
    if (
        original.status != "DPO_STRUCTURED_AGENT_RUNTIME_CANDIDATE_REJECTED"
        or original.candidate_accepted is not False
        or original.failed_hard_gates != ("runtime_security_boundaries",)
        or original.evaluation.get("candidate", {}).get("average_semantic_score") != 1.0
        or original.evaluation.get("candidate", {}).get("safe_action_rate") != 1.0
        or original.runtime_boundaries.get("citation_service_pass_count") != 0
        or not all(original.runtime_boundaries.get("gate_results", {}).values())
        or original.runtime_boundaries.get("business_side_effect_count") != 0
    ):
        raise DpoStructuredCitationErratumError(
            "original outcome does not contain the isolated citation fixture defect"
        )


def _original_outcome_path(root: Path) -> Path:
    pointer = _load_object(_inside_file(root, lab.LATEST_RELATIVE))
    value = pointer.get("outcome")
    if not isinstance(value, str) or not value:
        raise DpoStructuredCitationErratumError("DPO v3 latest pointer is invalid")
    path = _inside_file(root, Path(value))
    if pointer.get("outcome_sha256") != _file_sha256(path):
        raise DpoStructuredCitationErratumError("DPO v3 latest outcome changed")
    return path


def _authoritative_outcome_path(root: Path) -> Path:
    pointer = _load_object(_inside_file(root, AUTHORITATIVE_LATEST_RELATIVE))
    value = pointer.get("outcome")
    if not isinstance(value, str) or not value:
        raise DpoStructuredCitationErratumError("DPO erratum pointer is invalid")
    path = _inside_file(root, Path(value))
    if (
        pointer.get("outcome_sha256") != _file_sha256(path)
        or pointer.get("evidence_chain_sha256")
        != _load_object(path).get("evidence_chain_sha256")
    ):
        raise DpoStructuredCitationErratumError("DPO erratum pointer changed")
    return path


def _write_authoritative_latest(
    root: Path,
    outcome_path: Path,
    report: DpoStructuredCitationErratumReport,
) -> None:
    _write_json(
        root / AUTHORITATIVE_LATEST_RELATIVE,
        {
            "schema_version": "enterprise-dpo-structured-authoritative-latest/v1",
            "classification": lab.CLASSIFICATION,
            "status": report.status,
            "run_id": report.run_id,
            "outcome": outcome_path.relative_to(root).as_posix(),
            "outcome_sha256": _file_sha256(outcome_path),
            "evidence_chain_sha256": report.evidence_chain_sha256,
        },
    )


def _require_rows(document: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = document.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise DpoStructuredCitationErratumError(f"DPO citation {key} is invalid")
    return value


def _inside_file(root: Path, path: Path) -> Path:
    resolved = (
        (root / path).resolve(strict=True)
        if not path.is_absolute()
        else path.resolve(strict=True)
    )
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise DpoStructuredCitationErratumError("DPO citation path escapes repository")
    return resolved


def _file_binding(root: Path, path: Path) -> dict[str, Any]:
    resolved = _inside_file(root, path)
    return {
        "path": resolved.relative_to(root).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DpoStructuredCitationErratumError("DPO citation JSON is invalid") from exc
    if not isinstance(value, dict):
        raise DpoStructuredCitationErratumError("DPO citation JSON is not an object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
