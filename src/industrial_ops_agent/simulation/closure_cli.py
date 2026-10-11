"""CLI for the simulated enterprise data and model closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from industrial_ops_agent.enterprise_assets.acceptance import CANONICAL_RECEIPT_PATH
from industrial_ops_agent.simulation.closure import (
    build_simulated_enterprise_closure,
    write_simulated_enterprise_closure,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-simulated-enterprise-closure")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--enterprise-report",
        type=Path,
        default=Path("artifacts/enterprise-integration-sandbox/acceptance.json"),
    )
    parser.add_argument(
        "--supplier-a2a",
        type=Path,
        default=Path("artifacts/supplier-a2a-sandbox/acceptance.json"),
    )
    parser.add_argument(
        "--maintenance-planning-value",
        type=Path,
        default=Path("artifacts/m7-maintenance-planning-value-lab/acceptance.json"),
    )
    parser.add_argument(
        "--gpu-verification",
        type=Path,
        default=Path("artifacts/gpu-model-promotion-lab/evidence-verification.json"),
    )
    parser.add_argument(
        "--rul-latest",
        type=Path,
        default=Path("artifacts/rul-promotion-lab/latest.json"),
    )
    parser.add_argument(
        "--vlm-adoption",
        type=Path,
        default=Path("artifacts/m7-vlm-enterprise-staging/acceptance.json"),
    )
    parser.add_argument(
        "--vlm-rollout",
        type=Path,
        default=Path("artifacts/m7-vlm-kserve-shadow/acceptance.json"),
    )
    parser.add_argument(
        "--dpo-latest",
        type=Path,
        default=Path("artifacts/m7-dpo-post-training-lab/latest.json"),
    )
    parser.add_argument(
        "--dpo-runtime-outcome",
        type=Path,
        default=Path("artifacts/m7-dpo-agent-runtime-value-lab/latest.json"),
    )
    parser.add_argument(
        "--dpo-runtime-erratum",
        type=Path,
        default=Path("artifacts/m7-dpo-agent-runtime-value-lab/authoritative-latest.json"),
    )
    parser.add_argument(
        "--dpo-recovery-outcome",
        type=Path,
        default=Path("artifacts/m7-dpo-recovery-value-lab/latest.json"),
    )
    parser.add_argument(
        "--dpo-structured-acceptance",
        type=Path,
        default=Path(
            "artifacts/m7-dpo-structured-value-lab/errata/"
            "dpo-structured-36292ed691c0e57623a1/acceptance.json"
        ),
    )
    parser.add_argument(
        "--grpo-latest",
        type=Path,
        default=Path("artifacts/m7-grpo-post-training-lab/latest.json"),
    )
    parser.add_argument(
        "--grpo-runtime-outcome",
        type=Path,
        default=Path("artifacts/m7-grpo-agent-runtime-value-lab/latest.json"),
    )
    parser.add_argument(
        "--grpo-rollout",
        type=Path,
        default=Path("artifacts/m7-grpo-kserve-rollout/acceptance.json"),
    )
    parser.add_argument(
        "--ppo-latest",
        type=Path,
        default=Path("artifacts/m7-ppo-research-safety-lab/latest.json"),
    )
    parser.add_argument(
        "--ppo-runtime-outcome",
        type=Path,
        default=Path("artifacts/m7-ppo-agent-runtime-value-lab/latest.json"),
    )
    parser.add_argument(
        "--ppo-rollout",
        type=Path,
        default=Path("artifacts/m7-ppo-kserve-rollout/acceptance.json"),
    )
    parser.add_argument(
        "--asr-latest",
        type=Path,
        default=Path("artifacts/m7-asr-enterprise-value-lab/latest.json"),
    )
    parser.add_argument(
        "--asr-rollout",
        type=Path,
        default=Path("artifacts/m7-asr-kserve-rollout/actual/acceptance.json"),
    )
    parser.add_argument(
        "--model-import-acceptance",
        type=Path,
        default=CANONICAL_RECEIPT_PATH,
    )
    parser.add_argument(
        "--model-outcomes",
        type=Path,
        default=Path("artifacts/m7-model-experiment-outcomes/acceptance.json"),
    )
    parser.add_argument(
        "--reranker-rollout",
        type=Path,
        default=Path("artifacts/m7-reranker-kserve-acceptance/acceptance.json"),
    )
    parser.add_argument(
        "--candidate-rollout",
        type=Path,
        default=Path(
            "artifacts/m7-enterprise-candidate-kserve-acceptance/acceptance.json"
        ),
    )
    parser.add_argument(
        "--project-assurance",
        type=Path,
        default=Path("artifacts/m6-project-assurance-lab/acceptance.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/simulated-enterprise-closure/acceptance.json"),
    )
    return parser


def run(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    report = build_simulated_enterprise_closure(
        root,
        enterprise_report_path=args.enterprise_report,
        supplier_a2a_path=args.supplier_a2a,
        maintenance_planning_value_path=args.maintenance_planning_value,
        gpu_verification_path=args.gpu_verification,
        rul_latest_path=args.rul_latest,
        vlm_adoption_path=args.vlm_adoption,
        vlm_rollout_path=args.vlm_rollout,
        dpo_latest_path=args.dpo_latest,
        dpo_runtime_path=_latest_outcome(root, args.dpo_runtime_outcome),
        dpo_runtime_erratum_path=_latest_outcome(root, args.dpo_runtime_erratum),
        dpo_structured_acceptance_path=args.dpo_structured_acceptance,
        grpo_latest_path=args.grpo_latest,
        grpo_runtime_path=_latest_outcome(root, args.grpo_runtime_outcome),
        grpo_rollout_path=args.grpo_rollout,
        ppo_latest_path=args.ppo_latest,
        ppo_runtime_path=_latest_outcome(root, args.ppo_runtime_outcome),
        ppo_rollout_path=args.ppo_rollout,
        asr_latest_path=args.asr_latest,
        asr_rollout_path=args.asr_rollout,
        model_import_acceptance_path=args.model_import_acceptance,
        model_outcomes_path=args.model_outcomes,
        reranker_rollout_path=args.reranker_rollout,
        candidate_rollout_path=args.candidate_rollout,
        project_assurance_path=args.project_assurance,
        dpo_recovery_path=_latest_outcome(root, args.dpo_recovery_outcome),
    )
    output = args.output if args.output.is_absolute() else root / args.output
    path = write_simulated_enterprise_closure(report, output)
    print(report.status)
    print(report.classification)
    print(path)


def _latest_outcome(root: Path, pointer: Path) -> Path:
    path = pointer if pointer.is_absolute() else root / pointer
    document = json.loads(path.read_text(encoding="utf-8"))
    value = document.get("outcome")
    if not isinstance(value, str) or not value:
        value = document.get("report")
    if not isinstance(value, str) or not value:
        raise ValueError("Agent Runtime latest pointer is invalid")
    return Path(value)


if __name__ == "__main__":
    run()
