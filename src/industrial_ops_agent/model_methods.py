"""Shared model-method capability sets for training, evaluation and release gates."""

from __future__ import annotations

PEFT_ADAPTER_METHODS = frozenset({"LORA", "QLORA", "DPO", "GRPO"})
QUANTIZATION_METHODS = frozenset({"QUANTIZATION"})
EVALUATION_CANDIDATE_METHODS = PEFT_ADAPTER_METHODS | QUANTIZATION_METHODS
RETRIEVAL_EVALUATION_CANDIDATE_METHODS = frozenset({"EMBEDDING", "RERANKER"})
VLM_EVALUATION_CANDIDATE_METHODS = frozenset({"VLM"})
ASR_EVALUATION_CANDIDATE_METHODS = frozenset({"ASR"})
TTS_EVALUATION_CANDIDATE_METHODS = frozenset({"TTS"})
TIMESERIES_EVALUATION_CANDIDATE_METHODS = frozenset({"TIMESERIES_TRANSFORMER"})
TIMESERIES_BASELINE_METHODS = frozenset({"TIMESERIES_RULE_BASELINE"})
RUL_EVALUATION_CANDIDATE_METHODS = frozenset({"RUL_TRANSFORMER"})
RUL_BASELINE_METHODS = frozenset({"RUL_EMPIRICAL_BASELINE"})
PPO_RESEARCH_EVALUATION_CANDIDATE_METHODS = frozenset({"PPO"})
# PPO keeps its dedicated research-safety evaluation profile. A candidate only
# becomes release-capable after the separate agent-runtime value gate and the
# enterprise import, approval, staged rollout, and rollback controls succeed.
PPO_GOVERNED_RELEASE_METHODS = frozenset({"PPO"})
PRODUCTION_RELEASE_METHODS = (
    PEFT_ADAPTER_METHODS | QUANTIZATION_METHODS | PPO_GOVERNED_RELEASE_METHODS
)
RESEARCH_ONLY_METHODS: frozenset[str] = frozenset()
