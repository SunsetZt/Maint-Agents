"""Deterministic single-GPU Transformers/PEFT inference for frozen evaluation cases."""

from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from industrial_ops_agent.evaluation.dataset import (
    EvaluationCase,
    NormalizedRegion,
    RetrievalContract,
)
from industrial_ops_agent.evaluation.metrics import ModelObservation, VlmObservationFinding
from industrial_ops_agent.multimodal.vlm_output import build_vlm_findings_instruction
from industrial_ops_agent.persistence.models import TrainingExperimentRecord
from industrial_ops_agent.training.model_contract import (
    verify_encoder_model_contract,
    verify_model_contract,
)


class EvaluationBackendError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeConfig:
    max_new_tokens: int | None
    precision: str
    gpu_hourly_cost_usd: float
    top_k: int | None = None
    max_image_pixels: int | None = None
    sampling_rate: int | None = None
    language: str | None = None
    max_audio_seconds: float | None = None
    anomaly_threshold: float | None = None
    rule_threshold: float | None = None
    cpu_hourly_cost_usd: float | None = None
    threads: int | None = None
    context_size: int | None = None
    asr_verifier_model_id: str | None = None
    asr_verifier_revision: str | None = None
    asr_verifier_digest: str | None = None


@dataclass(frozen=True, slots=True)
class _GeneratedTtsCase:
    case: EvaluationCase
    audio: Any
    latency_ms: float
    input_tokens: int
    runtime_evidence: dict[str, Any]


class ModelEvaluationBackend(Protocol):
    target_profile: str

    def evaluate(
        self,
        *,
        experiment: TrainingExperimentRecord,
        cases: tuple[EvaluationCase, ...],
        config: EvaluationRuntimeConfig,
        adapter_directory: Path | None,
    ) -> tuple[ModelObservation, ...]: ...


class _RewardScorer(Protocol):
    def score(
        self, texts: tuple[str, ...], *, precision: str
    ) -> tuple[tuple[float, ...], float]: ...


class TransformersModelEvaluationBackend:
    target_profile = "MODEL_COMPONENT"

    def evaluate(
        self,
        *,
        experiment: TrainingExperimentRecord,
        cases: tuple[EvaluationCase, ...],
        config: EvaluationRuntimeConfig,
        adapter_directory: Path | None,
    ) -> tuple[ModelObservation, ...]:
        if config.max_new_tokens is None:
            raise EvaluationBackendError("max_new_tokens_is_missing")
        try:
            import torch  # type: ignore[import-not-found]
            from peft import PeftModel  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - GPU image only
            raise EvaluationBackendError("evaluation_dependencies_are_not_installed") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise EvaluationBackendError("exactly_one_cuda_gpu_is_required")
        revision = experiment.training_config.get("base_model_revision")
        if not isinstance(revision, str):
            raise EvaluationBackendError("base_model_revision_is_missing")
        quantized = experiment.method == "QUANTIZATION"
        if quantized:
            if adapter_directory is None:
                raise EvaluationBackendError("quantized_model_bundle_is_missing")
            if experiment.training_config.get("target_runtime") != "VLLM":
                raise EvaluationBackendError("quantized_runtime_requires_a_dedicated_evaluator")
            _verify_quantized_bundle_manifest(adapter_directory, experiment)
            tokenizer = AutoTokenizer.from_pretrained(
                str(adapter_directory), trust_remote_code=False, local_files_only=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                experiment.base_model_id,
                revision=revision,
                trust_remote_code=False,
            )
            verify_model_contract(
                tokenizer=tokenizer,
                revision=revision,
                base_model_digest=experiment.base_model_digest,
                tokenizer_digest=experiment.tokenizer_digest,
                chat_template_digest=experiment.chat_template_digest,
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
        model_source = str(adapter_directory) if quantized else experiment.base_model_id
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": False,
            "device_map": {"": 0},
        }
        if quantized:
            model_kwargs["local_files_only"] = True
        else:
            model_kwargs["revision"] = revision
        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
        if adapter_directory is not None and not quantized:
            model = PeftModel.from_pretrained(model, str(adapter_directory), is_trainable=False)
        model.eval()
        observations: list[ModelObservation] = []
        try:
            for case in cases:
                output, latency_ms, input_tokens, output_tokens = _generate(
                    model, tokenizer, case, config.max_new_tokens, torch
                )
                replay_output = None
                if "replay_reproducibility" in case.required_gates:
                    replay_output, _, _, _ = _generate(
                        model, tokenizer, case, config.max_new_tokens, torch
                    )
                parsed = _output_contract(output)
                cost = max(
                    latency_ms / 3_600_000 * config.gpu_hourly_cost_usd,
                    1e-12,
                )
                observations.append(
                    ModelObservation(
                        case_id=case.case_id,
                        output_text=output,
                        latency_ms=latency_ms,
                        cost_usd=cost,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        capabilities=frozenset({"valid_citations", "replay_reproducibility"}),
                        citations=parsed["citations"],
                        proposed_tools=parsed["proposed_tools"],
                        decision=parsed["decision"],
                        replay_output_text=replay_output,
                    )
                )
        finally:
            del model
            torch.cuda.empty_cache()
        return tuple(observations)


class PpoResearchSafetyEvaluationBackend:
    """Generate policy responses and score all PPO safety pairs with one reward identity."""

    target_profile = "PPO_RESEARCH_SAFETY"

    def __init__(
        self,
        candidate_training_config: dict[str, Any],
        *,
        policy_backend: ModelEvaluationBackend | None = None,
        reward_scorer: _RewardScorer | None = None,
    ) -> None:
        (
            self._reward_model_id,
            self._reward_model_revision,
            self._reward_model_digest,
        ) = _ppo_reward_model_binding(candidate_training_config)
        self._policy_backend = policy_backend or TransformersModelEvaluationBackend()
        self._reward_scorer = reward_scorer or _TransformersRewardScorer(
            self._reward_model_id,
            self._reward_model_revision,
        )

    def evaluate(
        self,
        *,
        experiment: TrainingExperimentRecord,
        cases: tuple[EvaluationCase, ...],
        config: EvaluationRuntimeConfig,
        adapter_directory: Path | None,
    ) -> tuple[ModelObservation, ...]:
        if experiment.method not in {"BASELINE", "PPO"}:
            raise EvaluationBackendError("ppo_safety_backend_method_is_unsupported")
        if experiment.method == "PPO" and adapter_directory is None:
            raise EvaluationBackendError("ppo_policy_adapter_bundle_is_missing")
        if config.max_new_tokens is None:
            raise EvaluationBackendError("max_new_tokens_is_missing")
        policy_observations = self._policy_backend.evaluate(
            experiment=experiment,
            cases=cases,
            config=config,
            adapter_directory=adapter_directory,
        )
        by_id = {observation.case_id: observation for observation in policy_observations}
        if len(by_id) != len(policy_observations) or set(by_id) != {case.case_id for case in cases}:
            raise EvaluationBackendError("ppo_policy_observations_do_not_match_cases")
        reward_texts: list[str] = []
        for case in cases:
            contract = case.ppo_safety
            if contract is None:
                raise EvaluationBackendError("ppo_safety_gold_contract_is_missing")
            observation = by_id[case.case_id]
            reward_texts.extend(
                (
                    _ppo_reward_text(case, contract.preferred_response),
                    _ppo_reward_text(case, contract.adversarial_response),
                    _ppo_reward_text(case, observation.output_text),
                )
            )
        rewards, reward_latency_ms = self._reward_scorer.score(
            tuple(reward_texts), precision=config.precision
        )
        if (
            len(rewards) != len(cases) * 3
            or not math.isfinite(reward_latency_ms)
            or reward_latency_ms <= 0
            or any(not math.isfinite(value) for value in rewards)
        ):
            raise EvaluationBackendError("ppo_reward_model_returned_invalid_scores")
        latency_share = reward_latency_ms / len(cases)
        reward_cost_share = max(latency_share / 3_600_000 * config.gpu_hourly_cost_usd, 1e-12)
        observations: list[ModelObservation] = []
        for index, case in enumerate(cases):
            contract = case.ppo_safety
            assert contract is not None
            observation = by_id[case.case_id]
            preferred, adversarial, policy = rewards[index * 3 : index * 3 + 3]
            hits = tuple(
                pattern
                for pattern in contract.forbidden_patterns
                if pattern in observation.output_text
            )
            observations.append(
                replace(
                    observation,
                    latency_ms=observation.latency_ms + latency_share,
                    cost_usd=observation.cost_usd + reward_cost_share,
                    runtime_evidence={
                        **observation.runtime_evidence,
                        "ppo_safety": {
                            "reward_model_id": self._reward_model_id,
                            "reward_model_revision": self._reward_model_revision,
                            "reward_model_digest": self._reward_model_digest,
                            "preferred_reward": float(preferred),
                            "adversarial_reward": float(adversarial),
                            "policy_reward": float(policy),
                            "forbidden_pattern_hits": list(hits),
                        },
                    },
                )
            )
        return tuple(observations)


class _TransformersRewardScorer:
    def __init__(self, model_id: str, revision: str) -> None:
        self._model_id = model_id
        self._revision = revision

    def score(self, texts: tuple[str, ...], *, precision: str) -> tuple[tuple[float, ...], float]:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - GPU image only
            raise EvaluationBackendError("evaluation_dependencies_are_not_installed") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise EvaluationBackendError("exactly_one_cuda_gpu_is_required")
        dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            revision=self._revision,
            trust_remote_code=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_id,
            revision=self._revision,
            dtype=dtype,
            trust_remote_code=False,
            num_labels=1,
            device_map={"": 0},
        )
        resolved_revision = getattr(model.config, "_commit_hash", None)
        if resolved_revision and resolved_revision != self._revision:
            raise EvaluationBackendError("resolved_reward_model_revision_changed")
        model.eval()
        scores: list[float] = []
        started = time.perf_counter()
        try:
            for offset in range(0, len(texts), 8):
                encoded = tokenizer(
                    list(texts[offset : offset + 8]),
                    padding=True,
                    truncation=True,
                    max_length=4096,
                    return_tensors="pt",
                )
                encoded = {name: value.to(0) for name, value in encoded.items()}
                with torch.no_grad():
                    logits = model(**encoded).logits
                if logits.ndim != 2 or logits.shape[1] != 1:
                    raise EvaluationBackendError("reward_model_output_contract_changed")
                scores.extend(float(value) for value in logits[:, 0].float().cpu().tolist())
        finally:
            del model
            torch.cuda.empty_cache()
        latency_ms = (time.perf_counter() - started) * 1000
        return tuple(scores), latency_ms


def _ppo_reward_model_binding(config: dict[str, Any]) -> tuple[str, str, str]:
    model_id = config.get("reward_model_id")
    revision = config.get("reward_model_revision")
    digest = config.get("reward_model_digest")
    if (
        not isinstance(model_id, str)
        or not model_id.strip()
        or len(model_id) > 255
        or not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or digest != f"hf-revision:{revision}"
    ):
        raise EvaluationBackendError("ppo_reward_model_binding_is_invalid")
    return model_id.strip(), revision, str(digest)


def _ppo_reward_text(case: EvaluationCase, response: str) -> str:
    prompt = "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}" for message in case.prompt
    ).strip()
    if not prompt or not response.strip():
        raise EvaluationBackendError("ppo_reward_input_is_empty")
    return f"{prompt}\nassistant: {response.strip()}"


class LlamaCppModelEvaluationBackend:
    """Run a reviewed GGUF candidate with the pinned llama.cpp CLI."""

    target_profile = "EDGE_MODEL_COMPONENT"

    def __init__(self, binary: str | None = None, *, timeout_seconds: int = 600) -> None:
        self._binary: str = (
            binary or os.getenv("IOAP_LLAMA_CPP_CLI_BINARY") or "/opt/llama.cpp/build/bin/llama-cli"
        )
        self._timeout_seconds = timeout_seconds

    def evaluate(
        self,
        *,
        experiment: TrainingExperimentRecord,
        cases: tuple[EvaluationCase, ...],
        config: EvaluationRuntimeConfig,
        adapter_directory: Path | None,
    ) -> tuple[ModelObservation, ...]:
        if experiment.method != "QUANTIZATION":
            raise EvaluationBackendError("llama_cpp_requires_a_quantization_candidate")
        if experiment.training_config.get("target_runtime") != "LLAMA_CPP":
            raise EvaluationBackendError("llama_cpp_runtime_binding_changed")
        if adapter_directory is None:
            raise EvaluationBackendError("quantized_model_bundle_is_missing")
        if config.max_new_tokens is None:
            raise EvaluationBackendError("max_new_tokens_is_missing")
        if config.cpu_hourly_cost_usd is None:
            raise EvaluationBackendError("cpu_hourly_cost_usd_is_missing")
        if config.threads is None or config.context_size is None:
            raise EvaluationBackendError("llama_cpp_runtime_configuration_is_missing")
        binary = Path(self._binary)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise EvaluationBackendError("llama_cpp_cli_is_not_installed")
        _verify_quantized_bundle_manifest(adapter_directory, experiment)
        model_path = _gguf_model_path(adapter_directory)
        model_content_hash = _file_sha256(model_path)
        observations: list[ModelObservation] = []
        for case in cases:
            output, latency_ms = self._generate(
                binary=binary,
                model_path=model_path,
                case=case,
                config=config,
            )
            replay_output = None
            if "replay_reproducibility" in case.required_gates:
                replay_output, _ = self._generate(
                    binary=binary,
                    model_path=model_path,
                    case=case,
                    config=config,
                )
            parsed = _output_contract(output)
            prompt_text = _llama_cpp_prompt(case)
            observations.append(
                ModelObservation(
                    case_id=case.case_id,
                    output_text=output,
                    latency_ms=latency_ms,
                    cost_usd=max(
                        latency_ms / 3_600_000 * config.cpu_hourly_cost_usd,
                        1e-12,
                    ),
                    input_tokens=_estimated_token_count(prompt_text),
                    output_tokens=_estimated_token_count(output),
                    capabilities=frozenset({"valid_citations", "replay_reproducibility"}),
                    citations=parsed["citations"],
                    proposed_tools=parsed["proposed_tools"],
                    decision=parsed["decision"],
                    replay_output_text=replay_output,
                    runtime_evidence={
                        "engine": "llama.cpp",
                        "model_serialization": "gguf",
                        "model_file": model_path.name,
                        "model_content_hash": model_content_hash,
                        "threads": config.threads,
                        "context_size": config.context_size,
                        "seed": 42,
                        "temperature": 0,
                        "token_accounting": "UTF8_BYTE_ESTIMATE",
                    },
                )
            )
        return tuple(observations)

    def _generate(
        self,
        *,
        binary: Path,
        model_path: Path,
        case: EvaluationCase,
        config: EvaluationRuntimeConfig,
    ) -> tuple[str, float]:
        assert config.max_new_tokens is not None
        assert config.threads is not None
        assert config.context_size is not None
        system_prompt, prompt = _llama_cpp_messages(case)
        argv = [
            str(binary),
            "--model",
            str(model_path),
            "--prompt",
            prompt,
            "--n-predict",
            str(config.max_new_tokens),
            "--ctx-size",
            str(config.context_size),
            "--threads",
            str(config.threads),
            "--seed",
            "42",
            "--temp",
            "0",
            "--conversation",
            "--single-turn",
            "--no-display-prompt",
        ]
        if system_prompt:
            argv.extend(["--system-prompt", system_prompt])
        started = time.perf_counter()
        try:
            result = subprocess.run(
                argv,
                check=True,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                env={"LC_ALL": "C.UTF-8", "NO_COLOR": "1"},
            )
        except subprocess.TimeoutExpired as exc:
            raise EvaluationBackendError("llama_cpp_generation_timed_out") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise EvaluationBackendError("llama_cpp_generation_failed") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        raw_output = result.stdout
        if len(raw_output.encode("utf-8")) > 4 * 1024 * 1024:
            raise EvaluationBackendError("llama_cpp_returned_invalid_observation")
        output = _llama_cpp_assistant_output(raw_output, prompt=prompt)
        if (
            not output
            or not math.isfinite(latency_ms)
            or latency_ms <= 0
        ):
            raise EvaluationBackendError("llama_cpp_returned_invalid_observation")
        return output, latency_ms


class EdgeQuantizationEvaluationBackend:
    """Use Transformers for the source PEFT baseline and llama.cpp for GGUF."""

    target_profile = "EDGE_MODEL_COMPONENT"

    def __init__(
        self,
        baseline_backend: ModelEvaluationBackend | None = None,
        candidate_backend: ModelEvaluationBackend | None = None,
    ) -> None:
        self._baseline_backend = baseline_backend or TransformersModelEvaluationBackend()
        self._candidate_backend = candidate_backend or LlamaCppModelEvaluationBackend()

    def evaluate(
        self,
        *,
        experiment: TrainingExperimentRecord,
        cases: tuple[EvaluationCase, ...],
        config: EvaluationRuntimeConfig,
        adapter_directory: Path | None,
    ) -> tuple[ModelObservation, ...]:
        backend = (
            self._candidate_backend
            if experiment.method == "QUANTIZATION"
            else self._baseline_backend
        )
        return backend.evaluate(
            experiment=experiment,
            cases=cases,
            config=config,
            adapter_directory=adapter_directory,
        )


def _verify_quantized_bundle_manifest(
    directory: Path, experiment: TrainingExperimentRecord
) -> None:
    path = directory / "industrial-ops-quantization-manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationBackendError("quantization_manifest_is_missing_or_invalid") from exc
    expected = {
        "schema_version": "industrial-ops-quantized-model/v1",
        "experiment_id": experiment.experiment_id,
        "base_model_id": experiment.base_model_id,
        "base_model_digest": experiment.base_model_digest,
        "tokenizer_digest": experiment.tokenizer_digest,
        "chat_template_digest": experiment.chat_template_digest,
        "dataset_snapshot_id": experiment.dataset_snapshot_id,
        "dataset_manifest_hash": experiment.dataset_manifest_hash,
        "approval_inheritance": False,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise EvaluationBackendError("quantization_manifest_binding_changed")
    configuration = manifest.get("configuration")
    if (
        not isinstance(configuration, dict)
        or configuration.get("quantization_profile_id")
        != experiment.training_config.get("quantization_profile_id")
        or configuration.get("source_artifact_hash")
        != str(experiment.training_config.get("source_artifact_hash", "")).removeprefix("sha256:")
    ):
        raise EvaluationBackendError("quantization_profile_binding_changed")
    claimed_hash = manifest.pop("manifest_sha256", None)
    actual_hash = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if claimed_hash != actual_hash:
        raise EvaluationBackendError("quantization_manifest_hash_mismatch")


def _gguf_model_path(directory: Path) -> Path:
    root = directory.resolve()
    candidates = [
        path
        for path in directory.rglob("*.gguf")
        if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
    ]
    if len(candidates) != 1:
        raise EvaluationBackendError("quantized_bundle_must_contain_one_GGUF_model")
    model = candidates[0]
    try:
        with model.open("rb") as stream:
            magic = stream.read(4)
    except OSError as exc:
        raise EvaluationBackendError("GGUF_model_is_not_readable") from exc
    if magic != b"GGUF":
        raise EvaluationBackendError("GGUF_model_has_invalid_magic")
    return model


def _llama_cpp_messages(case: EvaluationCase) -> tuple[str, str]:
    system = "\n\n".join(item["content"] for item in case.prompt if item.get("role") == "system")
    messages = [item for item in case.prompt if item.get("role") != "system"]
    if len(messages) == 1 and messages[0].get("role") == "user":
        prompt = messages[0]["content"]
    else:
        prompt = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages
        )
    if not prompt.strip():
        raise EvaluationBackendError("llama_cpp_prompt_is_empty")
    return system, prompt


def _llama_cpp_prompt(case: EvaluationCase) -> str:
    system, prompt = _llama_cpp_messages(case)
    return f"{system}\n{prompt}" if system else prompt


def _estimated_token_count(value: str) -> int:
    return max(1, math.ceil(len(value.encode("utf-8")) / 4))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise EvaluationBackendError("GGUF_model_is_not_readable") from exc
    return digest.hexdigest()


class RetrievalComponentEvaluationBackend:
    """Evaluate a frozen encoder/reranker bundle against reviewed Retrieval Gold cases."""

    target_profile = "RETRIEVAL_COMPONENT"

    def __init__(self, method: str) -> None:
        if method not in {"EMBEDDING", "RERANKER"}:
            raise EvaluationBackendError("retrieval_evaluation_method_is_unsupported")
        self._method = method

    def evaluate(
        self,
        *,
        experiment: TrainingExperimentRecord,
        cases: tuple[EvaluationCase, ...],
        config: EvaluationRuntimeConfig,
        adapter_directory: Path | None,
    ) -> tuple[ModelObservation, ...]:
        if config.top_k is None:
            raise EvaluationBackendError("retrieval_top_k_is_missing")
        try:
            import torch
            from sentence_transformers import (  # type: ignore[import-not-found]
                CrossEncoder,
                SentenceTransformer,
            )
        except ImportError as exc:  # pragma: no cover - GPU image only
            raise EvaluationBackendError(
                "retrieval_evaluation_dependencies_are_not_installed"
            ) from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise EvaluationBackendError("exactly_one_cuda_gpu_is_required")
        revision = experiment.training_config.get("base_model_revision")
        if not isinstance(revision, str):
            raise EvaluationBackendError("base_model_revision_is_missing")
        dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
        model_path = (
            str(adapter_directory) if adapter_directory is not None else experiment.base_model_id
        )
        if self._method == "EMBEDDING":
            kwargs: dict[str, Any] = {
                "trust_remote_code": False,
                "device": "cuda:0",
                "model_kwargs": {"torch_dtype": dtype},
            }
            if adapter_directory is None:
                kwargs["revision"] = revision
            model: Any = SentenceTransformer(model_path, **kwargs)
        else:
            kwargs = {
                "trust_remote_code": False,
                "device": "cuda:0",
                "num_labels": 1,
                "model_kwargs": {"torch_dtype": dtype},
            }
            if adapter_directory is None:
                kwargs["revision"] = revision
            model = CrossEncoder(model_path, **kwargs)
        if adapter_directory is None:
            verify_encoder_model_contract(
                tokenizer=model.tokenizer,
                revision=revision,
                base_model_digest=experiment.base_model_digest,
                tokenizer_digest=experiment.tokenizer_digest,
                chat_template_digest=experiment.chat_template_digest,
            )
        observations: list[ModelObservation] = []
        try:
            for case in cases:
                if case.retrieval is None:
                    raise EvaluationBackendError("retrieval_gold_contract_is_missing")
                observations.append(
                    self._evaluate_case(
                        model=model,
                        contract=case.retrieval,
                        case_id=case.case_id,
                        top_k=config.top_k,
                        gpu_hourly_cost_usd=config.gpu_hourly_cost_usd,
                        torch=torch,
                    )
                )
        finally:
            del model
            torch.cuda.empty_cache()
        return tuple(observations)

    def _evaluate_case(
        self,
        *,
        model: Any,
        contract: RetrievalContract,
        case_id: str,
        top_k: int,
        gpu_hourly_cost_usd: float,
        torch: Any,
    ) -> ModelObservation:
        allowed_documents = tuple(
            document for document in contract.documents if document.access == "ALLOWED"
        )
        if not allowed_documents:
            raise EvaluationBackendError("retrieval_gold_has_no_allowed_documents")
        torch.cuda.synchronize()
        started = time.perf_counter()
        if self._method == "EMBEDDING":
            query_embedding = model.encode(
                [contract.query], convert_to_tensor=True, normalize_embeddings=True
            )
            document_embeddings = model.encode(
                [document.text for document in allowed_documents],
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            raw_scores = model.similarity(query_embedding, document_embeddings)[0].tolist()
        else:
            raw_scores = model.predict(
                [(contract.query, document.text) for document in allowed_documents],
                convert_to_numpy=True,
                show_progress_bar=False,
            ).tolist()
        torch.cuda.synchronize()
        latency_ms = max((time.perf_counter() - started) * 1000, 1e-9)
        scores = [float(value) for value in raw_scores]
        if len(scores) != len(allowed_documents) or any(
            not math.isfinite(value) for value in scores
        ):
            raise EvaluationBackendError("retrieval_scores_are_invalid")
        ranked = sorted(
            zip(allowed_documents, scores, strict=True),
            key=lambda item: (-item[1], item[0].document_id),
        )[: min(top_k, len(allowed_documents))]
        ranked_document_ids = tuple(document.document_id for document, _ in ranked)
        ranked_scores = tuple(score for _, score in ranked)
        output = json.dumps(
            {"ranked_document_ids": ranked_document_ids},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        input_tokens = sum(
            len(model.tokenizer.encode(value, add_special_tokens=True))
            for value in (contract.query, *(document.text for document in allowed_documents))
        )
        return ModelObservation(
            case_id=case_id,
            output_text=output,
            latency_ms=latency_ms,
            cost_usd=max(latency_ms / 3_600_000 * gpu_hourly_cost_usd, 1e-12),
            input_tokens=input_tokens,
            output_tokens=0,
            capabilities=frozenset(
                {
                    "cross_tenant_isolation",
                    "data_governance",
                    "retrieval_index_compatibility",
                }
            ),
            ranked_document_ids=ranked_document_ids,
            retrieval_scores=ranked_scores,
            runtime_evidence={
                "allowed_document_count": len(allowed_documents),
                "forbidden_document_count": len(contract.forbidden_document_ids),
                "retrieval_method": self._method,
            },
        )


class VlmComponentEvaluationBackend:
    """Run deterministic image-grounded baseline/candidate inference on Multimodal Gold."""

    target_profile = "VLM_COMPONENT"

    def evaluate(
        self,
        *,
        experiment: TrainingExperimentRecord,
        cases: tuple[EvaluationCase, ...],
        config: EvaluationRuntimeConfig,
        adapter_directory: Path | None,
    ) -> tuple[ModelObservation, ...]:
        if config.max_new_tokens is None or config.max_image_pixels is None:
            raise EvaluationBackendError("VLM_evaluation_configuration_is_incomplete")
        try:
            import torch
            from peft import PeftModel
            from PIL import Image
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
            )
        except ImportError as exc:  # pragma: no cover - GPU image only
            raise EvaluationBackendError("VLM_evaluation_dependencies_are_not_installed") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise EvaluationBackendError("exactly_one_cuda_gpu_is_required")
        revision = experiment.training_config.get("base_model_revision")
        if not isinstance(revision, str):
            raise EvaluationBackendError("base_model_revision_is_missing")
        processor = AutoProcessor.from_pretrained(
            experiment.base_model_id,
            revision=revision,
            trust_remote_code=False,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise EvaluationBackendError("VLM_processor_has_no_tokenizer")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        verify_model_contract(
            tokenizer=tokenizer,
            revision=revision,
            base_model_digest=experiment.base_model_digest,
            tokenizer_digest=experiment.tokenizer_digest,
            chat_template_digest=experiment.chat_template_digest,
        )
        dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
        model = AutoModelForImageTextToText.from_pretrained(
            experiment.base_model_id,
            revision=revision,
            dtype=dtype,
            trust_remote_code=False,
            device_map={"": 0},
        )
        if adapter_directory is not None:
            model = PeftModel.from_pretrained(model, str(adapter_directory), is_trainable=False)
        model.eval()
        observations: list[ModelObservation] = []
        try:
            for case in cases:
                if (
                    case.vlm is None
                    or case.media_content is None
                    or case.media_mime_type not in {"image/png", "image/jpeg"}
                ):
                    raise EvaluationBackendError("VLM_gold_contract_is_missing")
                try:
                    source_image = Image.open(BytesIO(case.media_content))
                    source_image.load()
                    image = source_image.convert("RGB")
                except (OSError, TypeError, ValueError) as exc:
                    raise EvaluationBackendError("VLM_image_decode_failed") from exc
                if image.width * image.height > config.max_image_pixels:
                    raise EvaluationBackendError("VLM_image_exceeds_registered_pixel_limit")
                observations.append(
                    _evaluate_vlm_case(
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        image=image,
                        case=case,
                        config=config,
                        torch=torch,
                    )
                )
        finally:
            del model
            torch.cuda.empty_cache()
        return tuple(observations)


class AsrComponentEvaluationBackend:
    """Run deterministic Whisper-style baseline/candidate inference on ASR Gold."""

    target_profile = "ASR_COMPONENT"

    def evaluate(
        self,
        *,
        experiment: TrainingExperimentRecord,
        cases: tuple[EvaluationCase, ...],
        config: EvaluationRuntimeConfig,
        adapter_directory: Path | None,
    ) -> tuple[ModelObservation, ...]:
        if (
            config.max_new_tokens is None
            or config.sampling_rate != 16_000
            or config.language is None
            or config.max_audio_seconds is None
        ):
            raise EvaluationBackendError("ASR_evaluation_configuration_is_incomplete")
        try:
            import soundfile  # type: ignore[import-not-found]
            import torch
            from peft import PeftModel
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
            )
        except ImportError as exc:  # pragma: no cover - GPU image only
            raise EvaluationBackendError("ASR_evaluation_dependencies_are_not_installed") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise EvaluationBackendError("exactly_one_cuda_gpu_is_required")
        revision = experiment.training_config.get("base_model_revision")
        if not isinstance(revision, str):
            raise EvaluationBackendError("base_model_revision_is_missing")
        processor = AutoProcessor.from_pretrained(
            experiment.base_model_id,
            revision=revision,
            language=config.language,
            task="transcribe",
            trust_remote_code=False,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise EvaluationBackendError("ASR_processor_has_no_tokenizer")
        verify_encoder_model_contract(
            tokenizer=tokenizer,
            revision=revision,
            base_model_digest=experiment.base_model_digest,
            tokenizer_digest=experiment.tokenizer_digest,
            chat_template_digest=experiment.chat_template_digest,
        )
        dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            experiment.base_model_id,
            revision=revision,
            dtype=dtype,
            trust_remote_code=False,
            device_map={"": 0},
        )
        if adapter_directory is not None:
            model = PeftModel.from_pretrained(model, str(adapter_directory), is_trainable=False)
        model.eval()
        observations: list[ModelObservation] = []
        try:
            for case in cases:
                if (
                    case.asr is None
                    or case.media_content is None
                    or case.media_mime_type not in {"audio/wav", "audio/flac"}
                    or case.asr.language != config.language
                ):
                    raise EvaluationBackendError("ASR_gold_contract_is_missing_or_incompatible")
                try:
                    audio, sampling_rate = soundfile.read(
                        BytesIO(case.media_content),
                        dtype="float32",
                        always_2d=False,
                    )
                except (RuntimeError, TypeError, ValueError) as exc:
                    raise EvaluationBackendError("ASR_audio_decode_failed") from exc
                if int(sampling_rate) != config.sampling_rate:
                    raise EvaluationBackendError("ASR_audio_sampling_rate_mismatch")
                if getattr(audio, "ndim", 1) == 2:
                    audio = audio.mean(axis=1)
                duration_seconds = len(audio) / float(config.sampling_rate)
                if duration_seconds <= 0 or duration_seconds > config.max_audio_seconds:
                    raise EvaluationBackendError("ASR_audio_duration_is_out_of_range")
                observations.append(
                    _evaluate_asr_case(
                        model=model,
                        processor=processor,
                        audio=audio,
                        duration_seconds=duration_seconds,
                        case=case,
                        config=config,
                        torch=torch,
                    )
                )
        finally:
            del model
            torch.cuda.empty_cache()
        return tuple(observations)


class TtsComponentEvaluationBackend:
    """Generate deterministic SpeechT5 audio and verify its content with frozen ASR."""

    target_profile = "TTS_COMPONENT"
    _VOCODER_MODEL_ID = "microsoft/speecht5_hifigan"
    _VOCODER_REVISION = "e8b38625359976b675c3b3e2a41351058f5ba377"

    def __init__(self, runner_config: dict[str, Any]) -> None:
        self._verifier_id = str(runner_config.get("asr_verifier_model_id", ""))
        self._verifier_revision = str(runner_config.get("asr_verifier_revision", ""))
        self._verifier_digest = str(runner_config.get("asr_verifier_digest", ""))
        if (
            not self._verifier_id
            or len(self._verifier_revision) != 40
            or self._verifier_digest != f"hf-revision:{self._verifier_revision}"
        ):
            raise EvaluationBackendError("tts_asr_verifier_binding_is_invalid")

    def evaluate(
        self,
        *,
        experiment: TrainingExperimentRecord,
        cases: tuple[EvaluationCase, ...],
        config: EvaluationRuntimeConfig,
        adapter_directory: Path | None,
    ) -> tuple[ModelObservation, ...]:
        if (
            experiment.method not in {"BASELINE", "TTS"}
            or config.sampling_rate != 16_000
            or config.language is None
            or config.max_audio_seconds is None
            or config.asr_verifier_model_id != self._verifier_id
            or config.asr_verifier_revision != self._verifier_revision
            or config.asr_verifier_digest != self._verifier_digest
            or adapter_directory is None
        ):
            raise EvaluationBackendError("TTS_evaluation_configuration_is_incomplete")
        dependencies = _tts_evaluation_dependencies()
        torch = dependencies["torch"]
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise EvaluationBackendError("exactly_one_cuda_gpu_is_required")
        profile, speaker_embedding = _load_tts_voice_profile(adapter_directory, experiment)
        if any(
            case.tts is None
            or case.tts.voice_profile_id != profile["voice_profile_id"]
            or case.tts.language != profile["language"]
            for case in cases
        ):
            raise EvaluationBackendError("TTS_gold_voice_profile_is_incompatible")
        revision = str(experiment.training_config.get("base_model_revision", ""))
        local = experiment.method == "TTS"
        model_source = str(adapter_directory) if local else experiment.base_model_id
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": False,
            **({"local_files_only": True} if local else {"revision": revision}),
        }
        processor = dependencies["SpeechT5Processor"].from_pretrained(model_source, **load_kwargs)
        model = dependencies["SpeechT5ForTextToSpeech"].from_pretrained(
            model_source,
            dtype=(torch.bfloat16 if config.precision == "bfloat16" else torch.float16),
            **load_kwargs,
        )
        model.config.use_cache = True
        vocoder = dependencies["SpeechT5HifiGan"].from_pretrained(
            self._VOCODER_MODEL_ID,
            revision=self._VOCODER_REVISION,
            dtype=torch.float32,
            trust_remote_code=False,
        )
        for item in (model, vocoder):
            item.eval()
        verifier = {
            "model_id": self._verifier_id,
            "revision": self._verifier_revision,
            "digest": self._verifier_digest,
        }
        generated_cases: list[_GeneratedTtsCase] = []
        try:
            for case in cases:
                generated_cases.append(
                    _generate_tts_case(
                        model=model,
                        processor=processor,
                        vocoder=vocoder,
                        speaker_embedding=speaker_embedding,
                        case=case,
                        config=config,
                        verifier=verifier,
                        torch=torch,
                        numpy=dependencies["numpy"],
                    )
                )
        finally:
            del model, vocoder, processor
            gc.collect()
            torch.cuda.empty_cache()

        asr_processor = dependencies["AutoProcessor"].from_pretrained(
            self._verifier_id,
            revision=self._verifier_revision,
            language=config.language,
            task="transcribe",
            trust_remote_code=False,
        )
        asr_model = dependencies["AutoModelForSpeechSeq2Seq"].from_pretrained(
            self._verifier_id,
            revision=self._verifier_revision,
            dtype=(torch.bfloat16 if config.precision == "bfloat16" else torch.float16),
            trust_remote_code=False,
            device_map={"": 0},
        )
        asr_model.eval()
        observations: list[ModelObservation] = []
        try:
            for generated_case in generated_cases:
                observations.append(
                    _transcribe_tts_case(
                        generated=generated_case,
                        asr_model=asr_model,
                        asr_processor=asr_processor,
                        config=config,
                        torch=torch,
                    )
                )
        finally:
            del asr_model, asr_processor
            gc.collect()
            torch.cuda.empty_cache()
        return tuple(observations)


def _tts_evaluation_dependencies() -> dict[str, Any]:
    try:
        import numpy
        import torch
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Processor,
        )
    except ImportError as exc:  # pragma: no cover - GPU image only
        raise EvaluationBackendError("TTS_evaluation_dependencies_are_not_installed") from exc
    return {
        "torch": torch,
        "numpy": numpy,
        "SpeechT5Processor": SpeechT5Processor,
        "SpeechT5ForTextToSpeech": SpeechT5ForTextToSpeech,
        "SpeechT5HifiGan": SpeechT5HifiGan,
        "AutoProcessor": AutoProcessor,
        "AutoModelForSpeechSeq2Seq": AutoModelForSpeechSeq2Seq,
    }


def _load_tts_voice_profile(
    directory: Path, experiment: TrainingExperimentRecord
) -> tuple[dict[str, Any], tuple[float, ...]]:
    try:
        profile = json.loads(
            (directory / "industrial-ops-tts-voice-profile.json").read_text(encoding="utf-8")
        )
        embedding_name = str(profile["speaker_embedding_file"])
        embedding_path = directory / embedding_name
        if (
            Path(embedding_name).name != embedding_name
            or profile["speaker_embedding_sha256"]
            != "sha256:" + sha256(embedding_path.read_bytes()).hexdigest()
        ):
            raise ValueError
        numpy = __import__("numpy")
        raw = numpy.load(embedding_path, allow_pickle=False)
        embedding = tuple(float(value) for value in raw.tolist())
    except (KeyError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise EvaluationBackendError("tts_voice_profile_bundle_is_invalid") from exc
    expected = {
        "schema_version": "industrial-ops-tts-voice-profile/v1",
        "voice_profile_id": experiment.training_config.get("voice_profile_id"),
        "language": experiment.training_config.get("language"),
        "sampling_rate": experiment.training_config.get("sampling_rate"),
        "speaker_embedding_dimension": experiment.training_config.get(
            "speaker_embedding_dimension"
        ),
        "speech_contract_version": experiment.training_config.get("speech_contract_version"),
        "base_model_revision": experiment.training_config.get("base_model_revision"),
    }
    if (
        not isinstance(profile, dict)
        or any(profile.get(key) != value for key, value in expected.items())
        or len(embedding) != int(profile["speaker_embedding_dimension"])
        or any(not math.isfinite(value) for value in embedding)
    ):
        raise EvaluationBackendError("tts_voice_profile_bundle_binding_changed")
    return profile, embedding


def _tts_speaker_embedding_tensor(
    *, speaker_embedding: tuple[float, ...], model: Any, torch: Any
) -> Any:
    return torch.tensor(
        [speaker_embedding],
        dtype=getattr(model, "dtype", torch.float32),
        device="cuda:0",
    )


def _generate_tts_case(
    *,
    model: Any,
    processor: Any,
    vocoder: Any,
    speaker_embedding: tuple[float, ...],
    case: EvaluationCase,
    config: EvaluationRuntimeConfig,
    verifier: dict[str, str],
    torch: Any,
    numpy: Any,
) -> _GeneratedTtsCase:
    contract = case.tts
    if contract is None or config.sampling_rate is None or config.max_audio_seconds is None:
        raise EvaluationBackendError("TTS_gold_contract_is_missing")
    inputs = processor(text=contract.target_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda:0")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda:0")
    embedding = _tts_speaker_embedding_tensor(
        speaker_embedding=speaker_embedding,
        model=model,
        torch=torch,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        waveform = model.generate_speech(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speaker_embeddings=embedding,
            vocoder=vocoder,
            threshold=0.5,
            minlenratio=0.0,
            maxlenratio=20.0,
        )
    torch.cuda.synchronize()
    latency_ms = max((time.perf_counter() - started) * 1000, 1e-9)
    audio = numpy.asarray(waveform.detach().cpu().numpy(), dtype="float32").reshape(-1)
    duration = len(audio) / float(config.sampling_rate)
    finite = bool(numpy.isfinite(audio).all())
    clipping_ratio = float(numpy.mean(numpy.abs(audio) >= 0.999)) if len(audio) else 1.0
    silence_ratio = float(numpy.mean(numpy.abs(audio) <= 0.0001)) if len(audio) else 1.0
    integrity = (
        finite
        and 0 < duration <= config.max_audio_seconds
        and clipping_ratio <= 0.01
        and silence_ratio < 0.98
    )
    if not integrity:
        raise EvaluationBackendError("TTS_generated_audio_integrity_failed")
    audio_sha256 = sha256(audio.tobytes(order="C")).hexdigest()
    return _GeneratedTtsCase(
        case=case,
        audio=audio,
        latency_ms=latency_ms,
        input_tokens=int(input_ids.shape[-1]),
        runtime_evidence={
            "tts": {
                "audio_sha256": "sha256:" + audio_sha256,
                "sampling_rate": config.sampling_rate,
                "duration_seconds": duration,
                "silence_ratio": silence_ratio,
                "clipping_ratio": clipping_ratio,
                "voice_profile_id": contract.voice_profile_id,
                "asr_verifier": verifier,
                "vocoder": {
                    "model_id": TtsComponentEvaluationBackend._VOCODER_MODEL_ID,
                    "revision": TtsComponentEvaluationBackend._VOCODER_REVISION,
                },
            }
        },
    )


def _transcribe_tts_case(
    *,
    generated: _GeneratedTtsCase,
    asr_model: Any,
    asr_processor: Any,
    config: EvaluationRuntimeConfig,
    torch: Any,
) -> ModelObservation:
    contract = generated.case.tts
    if contract is None or config.sampling_rate is None:
        raise EvaluationBackendError("TTS_gold_contract_is_missing")
    features = asr_processor.feature_extractor(
        generated.audio, sampling_rate=config.sampling_rate, return_tensors="pt"
    ).input_features.to("cuda:0")
    with torch.inference_mode():
        generated_tokens = asr_model.generate(
            input_features=features,
            do_sample=False,
            language=config.language,
            task="transcribe",
        )
    transcript = asr_processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
    return ModelObservation(
        case_id=generated.case.case_id,
        output_text=transcript,
        latency_ms=generated.latency_ms,
        cost_usd=max(
            generated.latency_ms / 3_600_000 * config.gpu_hourly_cost_usd,
            1e-12,
        ),
        input_tokens=generated.input_tokens,
        output_tokens=int(generated_tokens.shape[-1]),
        capabilities=frozenset({"tts_voice_usage_authorization", "tts_audio_integrity"}),
        structured_output_valid=bool(transcript),
        transcript_text=transcript,
        runtime_evidence=generated.runtime_evidence,
    )


def _evaluate_asr_case(
    *,
    model: Any,
    processor: Any,
    audio: Any,
    duration_seconds: float,
    case: EvaluationCase,
    config: EvaluationRuntimeConfig,
    torch: Any,
) -> ModelObservation:
    assert case.asr is not None
    assert config.max_new_tokens is not None
    assert config.sampling_rate is not None
    inputs = processor.feature_extractor(
        audio,
        sampling_rate=config.sampling_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to("cuda:0")
    input_frames = int(input_features.shape[-1])
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_features=input_features,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            language=config.language,
            task="transcribe",
        )
    torch.cuda.synchronize()
    latency_ms = max((time.perf_counter() - started) * 1000, 1e-9)
    transcript = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    output_tokens = int(generated.shape[-1])
    return ModelObservation(
        case_id=case.case_id,
        output_text=transcript,
        latency_ms=latency_ms,
        cost_usd=max(latency_ms / 3_600_000 * config.gpu_hourly_cost_usd, 1e-12),
        input_tokens=input_frames,
        output_tokens=output_tokens,
        capabilities=frozenset({"asr_media_integrity"}),
        structured_output_valid=bool(transcript),
        transcript_text=transcript,
        runtime_evidence={
            "media_id": case.media_id,
            "media_mime_type": case.media_mime_type,
            "media_size_bytes": len(case.media_content or b""),
            "sampling_rate": config.sampling_rate,
            "duration_seconds": duration_seconds,
            "language": case.asr.language,
            "noise_condition": case.asr.noise_condition,
        },
    )


def _evaluate_vlm_case(
    *,
    model: Any,
    processor: Any,
    tokenizer: Any,
    image: Any,
    case: EvaluationCase,
    config: EvaluationRuntimeConfig,
    torch: Any,
) -> ModelObservation:
    assert case.vlm is not None
    assert config.max_new_tokens is not None
    instruction = build_vlm_findings_instruction(case.vlm.instruction)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[-1])
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    latency_ms = max((time.perf_counter() - started) * 1000, 1e-9)
    output_ids = generated[0][input_tokens:]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    findings, valid, repair = _vlm_output_contract_with_evidence(output)
    return ModelObservation(
        case_id=case.case_id,
        output_text=output,
        latency_ms=latency_ms,
        cost_usd=max(latency_ms / 3_600_000 * config.gpu_hourly_cost_usd, 1e-12),
        input_tokens=input_tokens,
        output_tokens=int(output_ids.shape[-1]),
        capabilities=frozenset({"vlm_media_integrity", "vlm_region_grounding"}),
        visual_findings=findings,
        structured_output_valid=valid,
        runtime_evidence={
            "media_id": case.media_id,
            "media_mime_type": case.media_mime_type,
            "media_size_bytes": len(case.media_content or b""),
            "structured_output_repair": repair,
        },
    )


def _vlm_output_contract(
    output: str,
) -> tuple[tuple[VlmObservationFinding, ...], bool]:
    findings, valid, _ = _vlm_output_contract_with_evidence(output)
    return findings, valid


def _vlm_output_contract_with_evidence(
    output: str,
) -> tuple[tuple[VlmObservationFinding, ...], bool, str]:
    candidate = output.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1])
            if candidate.lstrip().startswith("json"):
                candidate = candidate.lstrip()[4:].lstrip()
    repair = "NONE"
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        completed = _complete_json_closing_delimiters(candidate)
        if completed is None:
            return (), False, repair
        try:
            value = json.loads(completed)
        except json.JSONDecodeError:
            return (), False, repair
        repair = "BALANCED_CLOSING_DELIMITERS"
    if not isinstance(value, dict) or set(value) != {"findings"}:
        return (), False, repair
    raw_findings = value["findings"]
    if not isinstance(raw_findings, list) or len(raw_findings) > 100:
        return (), False, repair
    findings: list[VlmObservationFinding] = []
    for item in raw_findings:
        if not isinstance(item, dict) or set(item) != {"label", "region"}:
            return (), False, repair
        label = item["label"]
        region_raw = item["region"]
        if (
            not isinstance(label, str)
            or not label.strip()
            or len(label) > 128
            or not isinstance(region_raw, dict)
            or set(region_raw) != {"x", "y", "width", "height"}
        ):
            return (), False, repair
        coordinates: list[float] = []
        for key in ("x", "y", "width", "height"):
            coordinate = region_raw[key]
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                return (), False, repair
            coordinates.append(float(coordinate))
        region = NormalizedRegion(*coordinates)
        if (
            not 0 <= region.x < 1
            or not 0 <= region.y < 1
            or not 0 < region.width <= 1
            or not 0 < region.height <= 1
            or region.x + region.width > 1
            or region.y + region.height > 1
        ):
            return (), False, repair
        findings.append(VlmObservationFinding(label=label.strip(), region=region))
    return tuple(findings), True, repair


def _complete_json_closing_delimiters(candidate: str) -> str | None:
    if not candidate or len(candidate) > 65_536:
        return None
    expected_closers: list[str] = []
    in_string = False
    escaped = False
    for character in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            expected_closers.append("}")
        elif character == "[":
            expected_closers.append("]")
        elif character in {"}", "]"} and (
            not expected_closers or expected_closers.pop() != character
        ):
            return None
    if in_string or escaped or not expected_closers or len(expected_closers) > 8:
        return None
    return candidate + "".join(reversed(expected_closers))


def _generate(
    model: Any,
    tokenizer: Any,
    case: EvaluationCase,
    max_new_tokens: int,
    torch: Any,
) -> tuple[str, float, int, int]:
    prompt = tokenizer.apply_chat_template(
        list(case.prompt), tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
    input_tokens = int(encoded["input_ids"].shape[-1])
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000
    output_ids = generated[0][input_tokens:]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    if not output or not math.isfinite(latency_ms) or latency_ms <= 0:
        raise EvaluationBackendError("model_generation_returned_invalid_observation")
    return output, latency_ms, input_tokens, int(output_ids.shape[-1])


def _llama_cpp_assistant_output(stdout: str, *, prompt: str) -> str:
    """Remove llama-cli UI/prompt echo and return its final complete JSON object."""

    prompt_offset = stdout.rfind(prompt) if prompt else -1
    assistant_region = (
        stdout[prompt_offset + len(prompt) :] if prompt_offset >= 0 else stdout
    )
    json_objects = _top_level_json_objects(assistant_region)
    return json_objects[-1] if json_objects else assistant_region.strip()


def _top_level_json_objects(text: str) -> tuple[str, ...]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if start is None:
            if character == "{":
                start = index
                depth = 1
                in_string = False
                escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    value = json.loads(candidate)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(value, dict):
                        objects.append(candidate)
                start = None
    return tuple(objects)


def _output_contract(output: str) -> dict[str, Any]:
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        value = {}
    if not isinstance(value, dict):
        value = {}
    citations = value.get("citations", [])
    proposed_tools = value.get("tool_calls", [])
    return {
        "citations": _citation_ids(citations),
        "proposed_tools": (
            tuple(
                str(item.get("tool_id", "")) if isinstance(item, dict) else str(item)
                for item in proposed_tools
            )
            if isinstance(proposed_tools, list)
            else ()
        ),
        "decision": str(value["decision"]).upper() if "decision" in value else None,
    }


def _citation_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item:
            result.append(item)
        elif isinstance(item, dict) and isinstance(item.get("citation_id"), str):
            result.append(str(item["citation_id"]))
    return tuple(dict.fromkeys(result))
