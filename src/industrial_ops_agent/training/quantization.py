"""Validated quantization profiles and production AWQ/GPTQ/FP8/GGUF execution."""

from __future__ import annotations

import ctypes
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

from industrial_ops_agent.persistence.models import TrainingExperimentRecord
from industrial_ops_agent.training.dataset import TrainingDatasetBundle
from industrial_ops_agent.training.model_contract import verify_model_contract


class QuantizationError(RuntimeError):
    pass


_PROFILES: dict[tuple[str, str], tuple[str, bool]] = {
    ("AWQ", "W4A16_ASYM"): ("VLLM", True),
    ("GPTQ", "W4A16"): ("VLLM", True),
    ("FP8", "FP8_DYNAMIC"): ("VLLM", False),
    ("GGUF", "Q4_K_M"): ("LLAMA_CPP", False),
    ("GGUF", "Q5_K_M"): ("LLAMA_CPP", False),
    ("GGUF", "Q8_0"): ("LLAMA_CPP", False),
}


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    source_experiment_id: str
    source_artifact_id: str
    source_artifact_hash: str
    quantization_profile_id: str
    algorithm: str
    scheme: str
    target_runtime: str
    target_hardware_profile: str
    tool_version: str
    calibration_sample_count: int
    max_sequence_length: int
    incompatibilities: tuple[str, ...]
    approval_inheritance: bool
    gpu_hourly_cost_usd: float


@dataclass(frozen=True, slots=True)
class QuantizationOutcome:
    output_directory: Path
    metrics: dict[str, float]
    cost_summary: dict[str, Any]
    runtime_metadata: dict[str, Any]
    manifest: dict[str, Any]


class QuantizationBackend(Protocol):
    def quantize(
        self,
        *,
        experiment: TrainingExperimentRecord,
        config: QuantizationConfig,
        dataset: TrainingDatasetBundle,
        source_adapter_directory: Path,
        output_directory: Path,
    ) -> QuantizationOutcome: ...


def compile_quantization_config(raw: dict[str, Any]) -> QuantizationConfig:
    """Compile a closed profile; unsupported algorithm/runtime pairs fail before work starts."""

    required_strings = (
        "source_experiment_id",
        "source_artifact_id",
        "source_artifact_hash",
        "quantization_profile_id",
        "algorithm",
        "scheme",
        "target_runtime",
        "target_hardware_profile",
        "tool_version",
    )
    values: dict[str, str] = {}
    for key in required_strings:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 255:
            raise ValueError(f"training_config.{key} is invalid")
        values[key] = value.strip()
    source_hash = values["source_artifact_hash"].removeprefix("sha256:")
    if len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash):
        raise ValueError("training_config.source_artifact_hash must be sha256")

    algorithm = values["algorithm"].upper()
    scheme = values["scheme"].upper()
    profile = _PROFILES.get((algorithm, scheme))
    if profile is None:
        raise ValueError("quantization algorithm and scheme are not supported")
    expected_runtime, needs_calibration = profile
    target_runtime = values["target_runtime"].upper()
    if target_runtime != expected_runtime:
        raise ValueError("quantization target runtime is incompatible with the profile")
    expected_profile_id = f"{algorithm.lower()}-{scheme.lower()}-{target_runtime.lower()}"
    if values["quantization_profile_id"] != expected_profile_id:
        raise ValueError(
            f"quantization_profile_id must be {expected_profile_id} for this profile"
        )

    calibration_count = _integer(raw.get("calibration_sample_count"), "calibration_sample_count")
    max_sequence_length = _integer(raw.get("max_sequence_length"), "max_sequence_length")
    if needs_calibration:
        if not 32 <= calibration_count <= 2048:
            raise ValueError("calibration_sample_count must be between 32 and 2048")
        if not 128 <= max_sequence_length <= 8192:
            raise ValueError("max_sequence_length must be between 128 and 8192")
    elif calibration_count != 0:
        raise ValueError("data-free or GGUF profiles require calibration_sample_count=0")
    elif not 128 <= max_sequence_length <= 131072:
        raise ValueError("max_sequence_length must be between 128 and 131072")

    incompatibilities = raw.get("incompatibilities")
    if (
        not isinstance(incompatibilities, list)
        or not all(isinstance(value, str) and value.strip() for value in incompatibilities)
        or len(set(incompatibilities)) != len(incompatibilities)
    ):
        raise ValueError("training_config.incompatibilities must be a unique string list")
    if raw.get("approval_inheritance") is not False:
        raise ValueError("quantized artifacts cannot inherit source model approval")
    hourly_cost = raw.get("gpu_hourly_cost_usd", 0.0)
    if isinstance(hourly_cost, bool) or not isinstance(hourly_cost, (int, float)):
        raise ValueError("training_config.gpu_hourly_cost_usd is invalid")
    if float(hourly_cost) < 0:
        raise ValueError("training_config.gpu_hourly_cost_usd is invalid")

    return QuantizationConfig(
        source_experiment_id=values["source_experiment_id"],
        source_artifact_id=values["source_artifact_id"],
        source_artifact_hash=source_hash,
        quantization_profile_id=values["quantization_profile_id"],
        algorithm=algorithm,
        scheme=scheme,
        target_runtime=target_runtime,
        target_hardware_profile=values["target_hardware_profile"],
        tool_version=values["tool_version"],
        calibration_sample_count=calibration_count,
        max_sequence_length=max_sequence_length,
        incompatibilities=tuple(value.strip() for value in incompatibilities),
        approval_inheritance=False,
        gpu_hourly_cost_usd=float(hourly_cost),
    )


class ProductionQuantizationBackend:
    """Run LLM Compressor profiles or a pinned llama.cpp conversion without shell expansion."""

    def __init__(
        self,
        *,
        llama_cpp_convert_script: str | None = None,
        llama_cpp_quantize_binary: str | None = None,
    ) -> None:
        self._llama_cpp_convert_script: str = llama_cpp_convert_script or os.environ.get(
            "IOAP_LLAMA_CPP_CONVERT_SCRIPT"
        ) or "/opt/llama.cpp/convert_hf_to_gguf.py"
        self._llama_cpp_quantize_binary: str = (
            llama_cpp_quantize_binary
            or os.environ.get("IOAP_LLAMA_CPP_QUANTIZE_BINARY")
            or "/opt/llama.cpp/build/bin/llama-quantize"
        )

    def quantize(
        self,
        *,
        experiment: TrainingExperimentRecord,
        config: QuantizationConfig,
        dataset: TrainingDatasetBundle,
        source_adapter_directory: Path,
        output_directory: Path,
    ) -> QuantizationOutcome:
        started = time.perf_counter()
        output_directory.mkdir(parents=True, exist_ok=False)
        if config.algorithm == "GGUF":
            runtime = self._quantize_gguf(
                experiment, config, source_adapter_directory, output_directory
            )
        else:
            runtime = self._quantize_llmcompressor(
                experiment, config, dataset, source_adapter_directory, output_directory
            )
        duration = max(time.perf_counter() - started, 1e-9)
        output_size = sum(
            path.stat().st_size
            for path in output_directory.rglob("*")
            if path.is_file()
        )
        manifest = {
            "schema_version": "industrial-ops-quantized-model/v1",
            "experiment_id": experiment.experiment_id,
            "base_model_id": experiment.base_model_id,
            "base_model_revision": experiment.training_config["base_model_revision"],
            "base_model_digest": experiment.base_model_digest,
            "tokenizer_digest": experiment.tokenizer_digest,
            "chat_template_digest": experiment.chat_template_digest,
            "dataset_snapshot_id": dataset.snapshot_id,
            "dataset_manifest_hash": dataset.manifest_hash,
            "configuration": asdict(config),
            "runtime": runtime,
            "output_size_bytes": output_size,
            "approval_inheritance": False,
            "required_evaluation": [
                "quality",
                "security",
                "long_context",
                "tool_execution",
                "latency",
                "cost",
            ],
        }
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest["manifest_sha256"] = sha256(canonical).hexdigest()
        (output_directory / "industrial-ops-quantization-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        return QuantizationOutcome(
            output_directory=output_directory,
            metrics={
                "quantization_duration_seconds": duration,
                "quantized_output_size_bytes": float(output_size),
            },
            cost_summary={
                "duration_seconds": duration,
                "gpu_hourly_cost_usd": config.gpu_hourly_cost_usd,
                "estimated_cost_usd": duration / 3600 * config.gpu_hourly_cost_usd,
            },
            runtime_metadata=runtime,
            manifest=manifest,
        )

    def _load_merged_model(
        self, experiment: TrainingExperimentRecord, source_adapter_directory: Path
    ) -> tuple[Any, Any]:
        try:
            import torch  # type: ignore[import-not-found]
            from peft import PeftModel  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - quantization image only
            raise QuantizationError("quantization_model_dependencies_are_not_installed") from exc
        revision = str(experiment.training_config["base_model_revision"])
        tokenizer = AutoTokenizer.from_pretrained(
            experiment.base_model_id, revision=revision, trust_remote_code=False
        )
        verify_model_contract(
            tokenizer=tokenizer,
            revision=revision,
            base_model_digest=experiment.base_model_digest,
            tokenizer_digest=experiment.tokenizer_digest,
            chat_template_digest=experiment.chat_template_digest,
        )
        base = AutoModelForCausalLM.from_pretrained(
            experiment.base_model_id,
            revision=revision,
            dtype=torch.bfloat16,
            trust_remote_code=False,
            device_map="auto",
        )
        adapter = PeftModel.from_pretrained(base, str(source_adapter_directory), is_trainable=False)
        return adapter.merge_and_unload(), tokenizer

    def _quantize_llmcompressor(
        self,
        experiment: TrainingExperimentRecord,
        config: QuantizationConfig,
        dataset: TrainingDatasetBundle,
        source_adapter_directory: Path,
        output_directory: Path,
    ) -> dict[str, Any]:
        try:
            from datasets import Dataset  # type: ignore[import-not-found]
            from llmcompressor import oneshot  # type: ignore[import-not-found]
            from llmcompressor.modifiers.quantization import (  # type: ignore[import-not-found]
                GPTQModifier,
                QuantizationModifier,
            )
            from llmcompressor.modifiers.transform import (  # type: ignore[import-not-found]
                AWQModifier,
            )
        except ImportError as exc:  # pragma: no cover - quantization image only
            raise QuantizationError("llmcompressor_dependencies_are_not_installed") from exc
        model, tokenizer = self._load_merged_model(experiment, source_adapter_directory)
        kwargs: dict[str, Any] = {"model": model, "tokenizer": tokenizer}
        if config.algorithm == "AWQ":
            recipe: list[Any] = [
                AWQModifier(duo_scaling="both"),
                QuantizationModifier(
                    targets=["Linear"], scheme=config.scheme, ignore=["lm_head"]
                ),
            ]
        elif config.algorithm == "GPTQ":
            recipe = [
                GPTQModifier(targets=["Linear"], scheme=config.scheme, ignore=["lm_head"])
            ]
        else:
            recipe = [
                QuantizationModifier(
                    targets=["Linear"], scheme=config.scheme, ignore=["lm_head"]
                )
            ]
        kwargs["recipe"] = recipe
        if config.calibration_sample_count:
            texts = _calibration_texts(dataset, tokenizer, config.calibration_sample_count)
            kwargs.update(
                dataset=Dataset.from_dict({"text": texts}),
                max_seq_length=config.max_sequence_length,
                num_calibration_samples=config.calibration_sample_count,
            )
        oneshot(**kwargs)
        model.save_pretrained(output_directory, save_compressed=True, safe_serialization=True)
        tokenizer.save_pretrained(output_directory)
        return {
            "engine": "llmcompressor",
            "version": config.tool_version,
            "target_runtime": "vllm",
            "calibration_samples": config.calibration_sample_count,
        }

    def _quantize_gguf(
        self,
        experiment: TrainingExperimentRecord,
        config: QuantizationConfig,
        source_adapter_directory: Path,
        output_directory: Path,
    ) -> dict[str, Any]:
        convert = Path(self._llama_cpp_convert_script)
        quantize = Path(self._llama_cpp_quantize_binary)
        if not convert.is_file() or not quantize.is_file():
            raise QuantizationError("llama_cpp_tools_are_not_installed")
        model, tokenizer = self._load_merged_model(experiment, source_adapter_directory)
        with TemporaryDirectory(prefix="industrial-ops-merged-hf-") as temporary:
            merged = Path(temporary) / "model"
            merged.mkdir()
            model.save_pretrained(merged, safe_serialization=True)
            tokenizer.save_pretrained(merged)
            del model, tokenizer
            _release_model_memory()
            intermediate = Path(temporary) / "model-f16.gguf"
            target = output_directory / f"model-{config.scheme.lower()}.gguf"
            _run_checked(
                [
                    sys.executable,
                    str(convert),
                    str(merged),
                    "--outfile",
                    str(intermediate),
                    "--outtype",
                    "f16",
                ],
                "llama_cpp_conversion_failed",
            )
            _run_checked(
                [str(quantize), str(intermediate), str(target), config.scheme],
                "llama_cpp_quantization_failed",
            )
            if not target.is_file() or target.stat().st_size <= 4:
                raise QuantizationError("GGUF_quantized_artifact_is_missing")
            if target.read_bytes()[:4] != b"GGUF":
                raise QuantizationError("GGUF_quantized_artifact_has_invalid_magic")
        return {
            "engine": "llama.cpp",
            "version": config.tool_version,
            "target_runtime": "llama.cpp",
            "gguf_type": config.scheme,
            "model_file": target.name,
        }


def _calibration_texts(dataset: TrainingDatasetBundle, tokenizer: Any, count: int) -> list[str]:
    if len(dataset.train) < count:
        raise QuantizationError("calibration_snapshot_has_insufficient_samples")
    texts: list[str] = []
    for sample in dataset.train[:count]:
        messages = [*sample.prompt, *sample.completion]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if not isinstance(rendered, str) or not rendered.strip():
            raise QuantizationError("calibration_prompt_rendering_failed")
        texts.append(rendered)
    return texts


def _run_checked(argv: list[str], reason: str) -> None:
    try:
        subprocess.run(
            argv,
            check=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=7200,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise QuantizationError(reason) from exc


def _release_model_memory() -> None:
    """Return merged-model memory before starting llama.cpp subprocesses."""

    gc.collect()
    torch_module = sys.modules.get("torch")
    cuda = getattr(torch_module, "cuda", None)
    is_initialized = getattr(cuda, "is_initialized", None)
    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(is_initialized) and is_initialized() and callable(empty_cache):
        empty_cache()

    try:
        allocator = ctypes.CDLL(None)
    except OSError:  # pragma: no cover - non-glibc runtime
        return
    malloc_trim = getattr(allocator, "malloc_trim", None)
    if callable(malloc_trim):
        malloc_trim(0)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"training_config.{name} must be an integer")
    return int(value)
