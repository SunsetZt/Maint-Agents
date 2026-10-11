"""Shared fidelity gate for a local edge-runtime stable reference."""

from __future__ import annotations

import math

from industrial_ops_agent.experiments.service import MANDATORY_HARD_GATES

QUALITY_TOLERANCE = 0.01
EFFICIENCY_IMPROVEMENT_MIN = 0.15

_STABLE_REQUIRED_HARD_GATES = set(MANDATORY_HARD_GATES) - {
    "no_blocking_regressions",
    "valid_citations",
}


def stable_edge_evaluation_preserves_source(
    *,
    decision: object,
    candidate_score: object,
    baseline_score: object,
    quality_delta: object,
    cost_improvement: object,
    hard_gate_results: object,
    slice_metrics: object,
) -> bool:
    """Accept a stable reference only when edge conversion preserves its source."""

    if (
        decision not in {"CANDIDATE", "REJECTED"}
        or not all(
            _finite(value)
            for value in (
                candidate_score,
                baseline_score,
                quality_delta,
                cost_improvement,
            )
        )
        or not isinstance(hard_gate_results, dict)
        or set(hard_gate_results) != set(MANDATORY_HARD_GATES)
        or not all(isinstance(value, bool) for value in hard_gate_results.values())
        or any(
            hard_gate_results.get(gate) is not True
            for gate in _STABLE_REQUIRED_HARD_GATES
        )
        or float(candidate_score) + QUALITY_TOLERANCE < float(baseline_score)
        or float(quality_delta) < -QUALITY_TOLERANCE
        or abs(
            (float(candidate_score) - float(baseline_score))
            - float(quality_delta)
        )
        > 1e-9
        or float(cost_improvement) < EFFICIENCY_IMPROVEMENT_MIN
        or not isinstance(slice_metrics, dict)
    ):
        return False
    business = slice_metrics.get("business")
    if not isinstance(business, dict) or not business:
        return False
    for value in business.values():
        if not isinstance(value, dict):
            return False
        slice_candidate = value.get("candidate_score")
        slice_baseline = value.get("baseline_score")
        slice_delta = value.get("delta")
        if (
            not _positive_int(value.get("sample_count"))
            or not all(
                _finite(item)
                for item in (slice_candidate, slice_baseline, slice_delta)
            )
            or float(slice_candidate) + QUALITY_TOLERANCE < float(slice_baseline)
            or abs(
                (float(slice_candidate) - float(slice_baseline))
                - float(slice_delta)
            )
            > 1e-9
        ):
            return False
    return True


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
