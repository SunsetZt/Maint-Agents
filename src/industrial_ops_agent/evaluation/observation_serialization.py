"""Pure payload conversion for VLM observations and paired evaluation reports."""

from __future__ import annotations

from typing import Any

from industrial_ops_agent.evaluation.metrics import ModelObservation, PairedMetricReport


def observations_payload(observations: tuple[ModelObservation, ...]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": item.case_id,
            "output_text": item.output_text,
            "latency_ms": item.latency_ms,
            "cost_usd": item.cost_usd,
            "input_tokens": item.input_tokens,
            "output_tokens": item.output_tokens,
            "capabilities": sorted(item.capabilities),
            "structured_output_valid": item.structured_output_valid,
            "visual_findings": [
                {
                    "label": finding.label,
                    "region": {
                        "x": finding.region.x,
                        "y": finding.region.y,
                        "width": finding.region.width,
                        "height": finding.region.height,
                    },
                }
                for finding in item.visual_findings
            ],
            "runtime_evidence": item.runtime_evidence,
        }
        for item in observations
    ]


def paired_report_payload(report: PairedMetricReport) -> dict[str, Any]:
    return {
        "aggregate_metrics": report.aggregate_metrics,
        "gate_coverage": report.gate_coverage,
        "hard_gate_results": report.evidence.hard_gate_results,
        "candidate_outcomes": report.evidence.candidate_outcomes,
        "baseline_outcomes": report.evidence.baseline_outcomes,
        "candidate_p95_latency_ms": report.evidence.candidate_p95_latency_ms,
        "baseline_p95_latency_ms": report.evidence.baseline_p95_latency_ms,
        "candidate_cost_per_case": report.evidence.candidate_cost_per_case,
        "baseline_cost_per_case": report.evidence.baseline_cost_per_case,
    }
