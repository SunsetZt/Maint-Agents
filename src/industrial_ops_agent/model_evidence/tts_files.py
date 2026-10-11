"""Shared file contracts for calibrated TTS evidence, independent of experiment services."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from industrial_ops_agent.model_evidence.report_files import file_sha256
from industrial_ops_agent.model_evidence.tts_contracts import (
    DirectoryBinding,
    FileBinding,
    PredecessorEvidence,
    PriorLatencyEvidence,
)

if TYPE_CHECKING:
    from industrial_ops_agent.model_evidence.tts_final import (
        TtsCalibratedValueReport as TtsFinalValueReport,
    )
    from industrial_ops_agent.model_evidence.tts_recovery import TtsRecoveryRejectionReport

EvidenceModel = TypeVar("EvidenceModel", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class TtsEvidenceFiles:
    error_type: type[RuntimeError]
    word_units: Callable[[str], tuple[str, ...]]

    def directory_binding(self, root: Path, path: Path) -> dict[str, Any]:
        return self.directory_binding_at(path.relative_to(root), path)

    def directory_binding_at(self, relative: Path, path: Path) -> dict[str, Any]:
        entries = self.directory_manifest(path)
        return {
            "path": relative.as_posix(),
            "file_count": len(entries),
            "size_bytes": sum(int(item["size_bytes"]) for item in entries),
            "manifest_sha256": digest(entries),
        }

    def directory_manifest(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_dir():
            raise self.error_type("TTS evidence directory is missing")
        entries: list[dict[str, Any]] = []
        for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
            entries.append(
                {
                    "path": file_path.relative_to(path).as_posix(),
                    "size_bytes": file_path.stat().st_size,
                    "sha256": file_sha256(file_path),
                }
            )
        if not entries:
            raise self.error_type("TTS evidence directory is empty")
        return entries

    def file_binding(self, root: Path, path: Path) -> dict[str, Any]:
        return self.file_binding_at(path.relative_to(root), path)

    def file_binding_at(self, relative: Path, path: Path) -> dict[str, Any]:
        if not path.is_file() or path.stat().st_size <= 0:
            raise self.error_type("TTS evidence file is missing")
        return {
            "path": relative.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    def verify_file_binding(self, root: Path, binding: FileBinding) -> Path:
        path = self.inside_file(root, Path(binding.path))
        expected = self.file_binding(root, path)
        if expected != binding.model_dump(mode="json"):
            raise self.error_type("TTS evidence file changed")
        return path

    def inside_file(self, root: Path, path: Path) -> Path:
        value = path if path.is_absolute() else root / path
        resolved = value.resolve(strict=True)
        self.require_inside(root, resolved)
        if not resolved.is_file():
            raise self.error_type("TTS evidence file is invalid")
        return resolved

    def inside_directory(self, root: Path, path: Path) -> Path:
        value = path if path.is_absolute() else root / path
        resolved = value.resolve(strict=True)
        self.require_inside(root, resolved)
        if not resolved.is_dir():
            raise self.error_type("TTS evidence directory is invalid")
        return resolved

    def require_inside(self, root: Path, path: Path) -> None:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise self.error_type("TTS evidence escaped the repository") from exc

    def load_object(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise self.error_type(f"invalid TTS JSON: {path}") from exc
        if not isinstance(value, dict):
            raise self.error_type("TTS JSON root must be an object")
        return {str(key): item for key, item in value.items()}

    def predecessor_evidence(
        self,
        root: Path,
        *,
        rejection_relative: Path,
        candidate_relative: Path,
        load_rejection: Callable[[Path, Path], TtsRecoveryRejectionReport],
    ) -> PredecessorEvidence:
        rejection_path = self.inside_file(root, rejection_relative)
        report = load_rejection(root, rejection_path)
        if (
            report.run_id != "tts-recovery-a2f61df7fa877cf300c5"
            or report.runtime.actual_gpu_execution is not True
            or report.runtime.optimizer_steps != 24
            or report.candidate_accepted is not False
            or report.same_gold_reuse_permitted is not False
        ):
            raise self.error_type("TTS v2 predecessor contract changed")
        candidate_path = self.inside_directory(root, candidate_relative)
        if report.candidate.path != candidate_relative.as_posix():
            raise self.error_type("TTS v2 predecessor candidate path changed")
        return PredecessorEvidence(
            rejection=FileBinding.model_validate(self.file_binding(root, rejection_path)),
            candidate=DirectoryBinding.model_validate(self.directory_binding(root, candidate_path)),
            run_id=report.run_id,
            status=report.status,
            actual_gpu_training=True,
            optimizer_steps=24,
            failed_hard_gates=(
                "gold_voice_similarity_improved",
                "tts_safety_warning_completeness",
            ),
            same_gold_reuse_permitted=False,
        )

    def prior_attempt_evidence(
        self,
        root: Path,
        manifests: tuple[tuple[int, Path, Path], ...],
        evidence_model: type[EvidenceModel],
    ) -> EvidenceModel:
        # Resolve all inputs before reading any retirement, as each version did.
        paths = [
            (version, self.inside_file(root, gold), self.inside_file(root, retirement))
            for version, gold, retirement in manifests
        ]
        for version, gold, retirement_path in paths:
            retirement = self.load_object(retirement_path)
            if (
                retirement.get("schema_version")
                != f"industrial-safety-tts-gold-retirement/v{version}"
                or retirement.get("status") != "RETIRED_AFTER_INTERRUPTED_FINAL_EVALUATION"
                or retirement.get("gold_dataset_id")
                != f"industrial-safety-tts-final-gold-v{version}"
                or retirement.get("gold_evaluation_pass_count") != 1
                or retirement.get("reuse_permitted") is not False
                or retirement.get("outcome_path") is not None
                or retirement.get("gold_manifest_sha256") != file_sha256(gold)
            ):
                raise self.error_type(f"TTS interrupted v{version} evidence changed")
        values: dict[str, Any] = {}
        for version, gold, retirement_path in paths:
            values[f"gold_v{version}_manifest"] = FileBinding.model_validate(
                self.file_binding(root, gold)
            )
            values[f"gold_v{version}_retirement"] = FileBinding.model_validate(
                self.file_binding(root, retirement_path)
            )
        return evidence_model(
            **values,
            statuses=tuple("RETIRED_AFTER_INTERRUPTED_FINAL_EVALUATION" for _ in paths),
            gold_dataset_ids=tuple(f"industrial-safety-tts-final-gold-v{v}" for v, _, _ in paths),
            gold_evaluation_pass_counts=tuple(1 for _ in paths),
            reuse_permitted=tuple(False for _ in paths),
        )


    def rows(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        value = document.get("rows")
        if not isinstance(value, list) or not value:
            raise self.error_type("TTS dataset rows are missing")
        rows: list[dict[str, Any]] = []
        for row in value:
            if not isinstance(row, dict):
                raise self.error_type("TTS dataset row is invalid")
            rows.append({str(key): item for key, item in row.items()})
        return rows

    def normalized_text(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return " ".join(self.word_units(normalized))

    def ids_and_texts(self, document: dict[str, Any]) -> tuple[set[str], set[str]]:
        ids: set[str] = set()
        texts: set[str] = set()
        for row in self.rows(document):
            case_id = row.get("case_id")
            text = row.get("text")
            if not isinstance(case_id, str) or not isinstance(text, str):
                raise self.error_type("TTS dataset row is invalid")
            ids.add(case_id)
            texts.add(self.normalized_text(text))
        return ids, texts

    def require_json(self, path: Path, value: Any) -> None:
        if self.load_object(path) != canonical_json(value):
            raise self.error_type("immutable TTS input changed")

    def latest_outcome_path(self, root: Path, latest_relative: Path) -> Path:
        latest = self.load_object(self.inside_file(root, latest_relative))
        value = latest.get("outcome_path")
        if not isinstance(value, str):
            raise self.error_type("TTS latest outcome pointer is invalid")
        path = self.inside_file(root, Path(value))
        if latest.get("outcome_sha256") != file_sha256(path):
            raise self.error_type("TTS latest outcome pointer changed")
        return path

    def verify_gold_retirement(
        self, root: Path, path: Path, *, gold_relative: Path, version: int
    ) -> dict[str, Any]:
        retirement = self.load_object(self.inside_file(root, path))
        if (
            retirement.get("schema_version") != f"industrial-safety-tts-gold-retirement/v{version}"
            or retirement.get("gold_dataset_id") != f"industrial-safety-tts-final-gold-v{version}"
            or retirement.get("gold_evaluation_pass_count") != 1
            or retirement.get("reuse_permitted") is not False
        ):
            raise self.error_type(f"TTS Gold v{version} retirement changed")
        gold_path = self.inside_file(root, gold_relative)
        if retirement.get("gold_manifest_sha256") != file_sha256(gold_path):
            raise self.error_type(f"TTS Gold v{version} retirement manifest changed")
        outcome_value = retirement.get("outcome_path")
        if isinstance(outcome_value, str):
            outcome = self.inside_file(root, Path(outcome_value))
            if retirement.get("outcome_sha256") != file_sha256(outcome):
                raise self.error_type(f"TTS Gold v{version} retirement outcome changed")
        return retirement

    def prior_latency_evidence(
        self, root: Path, *, rejection_relative: Path, gold_relative: Path,
        retirement_relative: Path,
        load_final_report: Callable[[Path, Path], TtsFinalValueReport],
    ) -> PriorLatencyEvidence:
        rejection_path = self.inside_file(root, rejection_relative)
        report = load_final_report(root, rejection_path)
        gold_path = self.inside_file(root, gold_relative)
        retirement_path = self.inside_file(root, retirement_relative)
        retirement = self.load_object(retirement_path)
        quality_gates = {
            name: value
            for name, value in report.hard_gates.items()
            if name != "resource_profile_respected"
        }
        if (
            report.run_id != "tts-final-db7b7259e10c4f3fa6ba"
            or report.status != "TTS_FINAL_ACTUAL_GPU_CANDIDATE_REJECTED"
            or report.failed_hard_gates != ("resource_profile_respected",)
            or report.candidate_accepted is not False
            or report.evaluation.candidate_pool_size != 29
            or report.evaluation.candidate_latency_ratio <= 30.0
            or not all(quality_gates.values())
            or retirement.get("status") != "RETIRED_AFTER_REJECTED_FINAL_EVALUATION"
            or retirement.get("run_id") != report.run_id
            or retirement.get("reuse_permitted") is not False
            or retirement.get("gold_manifest_sha256") != file_sha256(gold_path)
            or retirement.get("outcome_sha256") != file_sha256(rejection_path)
        ):
            raise self.error_type("TTS v7 latency rejection evidence changed")
        return PriorLatencyEvidence(
            rejection=FileBinding.model_validate(self.file_binding(root, rejection_path)),
            gold_manifest=FileBinding.model_validate(self.file_binding(root, gold_path)),
            gold_retirement=FileBinding.model_validate(self.file_binding(root, retirement_path)),
            run_id=report.run_id,
            status=report.status,
            failed_hard_gates=("resource_profile_respected",),
            candidate_pool_size=29,
            gold_candidate_latency_ratio=report.evaluation.candidate_latency_ratio,
            quality_gates_passed=True,
            same_gold_reuse_permitted=False,
        )

def canonical_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False))


def digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
