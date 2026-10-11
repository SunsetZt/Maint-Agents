"""Ragas/DeepEval dataset adapters; judge scores never override deterministic gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from industrial_ops_agent.evaluation.dataset import EvaluationCase
from industrial_ops_agent.evaluation.metrics import ModelObservation


class FrameworkUnavailable(RuntimeError):
    pass


class EvaluationFrameworkBridge(Protocol):
    def summarize(
        self,
        cases: tuple[EvaluationCase, ...],
        observations: tuple[ModelObservation, ...],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class DeterministicFrameworkBridge:
    def summarize(
        self,
        cases: tuple[EvaluationCase, ...],
        observations: tuple[ModelObservation, ...],
    ) -> dict[str, Any]:
        return {
            "mode": "DETERMINISTIC",
            "case_count": len(cases),
            "judge_metrics_used_for_hard_gates": False,
        }


class RagasDeepEvalBridge:
    """Materialize both framework-native datasets without invoking an unapproved judge."""

    def summarize(
        self,
        cases: tuple[EvaluationCase, ...],
        observations: tuple[ModelObservation, ...],
    ) -> dict[str, Any]:
        try:
            from deepeval.dataset import (
                EvaluationDataset as DeepEvalDataset,
            )
            from deepeval.test_case import LLMTestCase
            from ragas import EvaluationDataset as RagasDataset
        except ImportError as exc:  # pragma: no cover - optional evaluation image
            raise FrameworkUnavailable("ragas_or_deepeval_is_not_installed") from exc
        by_id = {item.case_id: item for item in observations}
        deepeval_dataset = DeepEvalDataset()
        ragas_rows: list[dict[str, Any]] = []
        for case in cases:
            observation = by_id[case.case_id]
            prompt = "\n".join(item["content"] for item in case.prompt)
            expected = _json_text(case.expected_output)
            deepeval_dataset.add_test_case(
                LLMTestCase(
                    input=prompt,
                    actual_output=observation.output_text,
                    expected_output=expected,
                    retrieval_context=list(case.contexts) or None,
                )
            )
            ragas_rows.append(
                {
                    "user_input": prompt,
                    "response": observation.output_text,
                    "reference": expected,
                    "retrieved_contexts": list(case.contexts),
                }
            )
        RagasDataset.from_list(ragas_rows)
        return {
            "mode": "RAGAS_DEEPEVAL",
            "deepeval_case_count": len(deepeval_dataset.test_cases),
            "ragas_case_count": len(ragas_rows),
            "judge_metrics_status": "NOT_CONFIGURED",
            "judge_metrics_used_for_hard_gates": False,
            "reason": "LLM judge requires a separately approved evaluator model and policy",
        }


def _json_text(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
