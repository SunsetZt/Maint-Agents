"""Governed SFT and post-training execution boundaries."""

from typing import TYPE_CHECKING, Any

from industrial_ops_agent.training.config import (
    CompiledDpoConfig,
    CompiledEmbeddingConfig,
    CompiledGrpoConfig,
    CompiledPpoConfig,
    CompiledRerankerConfig,
    CompiledSftConfig,
    CompiledTimeseriesTransformerConfig,
    CompiledTrainingConfig,
    compile_sft_config,
    compile_training_config,
)
from industrial_ops_agent.training.dataset import GovernedSftDatasetLoader, TrainingDatasetBundle

if TYPE_CHECKING:
    from industrial_ops_agent.training.runner import TrainingRunner, TrainingRunResult

__all__ = [
    "CompiledDpoConfig",
    "CompiledEmbeddingConfig",
    "CompiledGrpoConfig",
    "CompiledPpoConfig",
    "CompiledRerankerConfig",
    "CompiledSftConfig",
    "CompiledTimeseriesTransformerConfig",
    "CompiledTrainingConfig",
    "GovernedSftDatasetLoader",
    "TrainingDatasetBundle",
    "TrainingRunResult",
    "TrainingRunner",
    "compile_sft_config",
    "compile_training_config",
]


def __getattr__(name: str) -> Any:
    """Keep runner exports without importing the experiment registry during package load."""

    if name in {"TrainingRunner", "TrainingRunResult"}:
        from industrial_ops_agent.training.runner import (
            TrainingRunner,
            TrainingRunResult,
        )

        return {"TrainingRunner": TrainingRunner, "TrainingRunResult": TrainingRunResult}[name]
    raise AttributeError(name)
