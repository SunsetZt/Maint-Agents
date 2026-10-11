"""Shared execution steps for initial and continued VLM experiments."""

from __future__ import annotations

import gc
from importlib.metadata import version

from industrial_ops_agent.evaluation.backend import EvaluationRuntimeConfig
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.persistence.models import TrainingExperimentRecord
from industrial_ops_agent.persistence.tenant import TenantContext


def vlm_evaluation_runtime() -> EvaluationRuntimeConfig:
    return EvaluationRuntimeConfig(
        max_new_tokens=128,
        precision="bfloat16",
        gpu_hourly_cost_usd=0.50,
        max_image_pixels=307_200,
    )


def collect_runtime_versions() -> dict[str, str]:
    return {
        name: version(name)
        for name in (
            "torch",
            "transformers",
            "peft",
            "bitsandbytes",
            "accelerate",
            "datasets",
            "trl",
            "safetensors",
        )
    }


def release_gpu() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def persist_experiment(
    database: Database,
    context: TenantContext,
    experiment: TrainingExperimentRecord,
) -> None:
    with database.transaction(context) as session:
        session.add(experiment)
        session.flush()
