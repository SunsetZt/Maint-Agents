"""Read-only evaluation gate contracts shared by evidence readers and governance."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TTS_HARD_GATES = frozenset(
    {
        "data_governance",
        "tts_voice_usage_authorization",
        "tts_audio_integrity",
        "tts_safety_warning_completeness",
        "tts_terminology_accuracy",
        "tts_intelligibility",
        "no_blocking_regressions",
    }
)


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

class FileBinding(_ClosedModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

class DirectoryBinding(_ClosedModel):
    path: str = Field(min_length=1)
    file_count: int = Field(gt=0)
    size_bytes: int = Field(gt=0)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

class PredecessorEvidence(_ClosedModel):
    rejection: FileBinding
    candidate: DirectoryBinding
    run_id: Literal["tts-recovery-a2f61df7fa877cf300c5"]
    status: Literal["TTS_RECOVERY_ACTUAL_GPU_CANDIDATE_REJECTED"]
    actual_gpu_training: Literal[True] = True
    optimizer_steps: Literal[24] = 24
    failed_hard_gates: tuple[
        Literal["gold_voice_similarity_improved"],
        Literal["tts_safety_warning_completeness"],
    ]
    same_gold_reuse_permitted: Literal[False] = False


class PriorLatencyEvidence(_ClosedModel):
    rejection: FileBinding
    gold_manifest: FileBinding
    gold_retirement: FileBinding
    run_id: Literal["tts-final-db7b7259e10c4f3fa6ba"]
    status: Literal["TTS_FINAL_ACTUAL_GPU_CANDIDATE_REJECTED"]
    failed_hard_gates: tuple[Literal["resource_profile_respected"]]
    candidate_pool_size: Literal[29] = 29
    gold_candidate_latency_ratio: float = Field(gt=30.0)
    quality_gates_passed: Literal[True] = True
    same_gold_reuse_permitted: Literal[False] = False
