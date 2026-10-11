"""Offline, non-production LoRA/QLoRA training and evaluation sandbox."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol

from industrial_ops_agent.experiments.mlflow import ExperimentTracker

MARKER = "SIMULATED_NON_PRODUCTION"
DATASET_SCHEMA = "model-training-sandbox-dataset/v1"
CASE_SCHEMA = "model-training-sandbox-case/v1"
ADAPTER_MANIFEST_SCHEMA = "model-training-adapter-manifest/v1"
REPORT_SCHEMA = "model-training-sandbox-report/v1"
PROMOTION_SCHEMA = "model-training-promotion-receipt/v1"
METHODS = ("LORA", "QLORA")
SPLITS = ("train", "validation", "gold")
CASE_FIELDS = frozenset({"schema_version", "case_id", "split", "prompt", "completion", "risk_level"})
RISK_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH"})


class SandboxDataError(ValueError):
    pass

class SandboxTrainingError(RuntimeError):
    pass

class SandboxArtifactError(RuntimeError):
    pass

class SandboxEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DatasetCase:
    schema_version: str
    case_id: str
    split: Literal["train", "validation", "gold"]
    prompt: str
    completion: str
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]


@dataclass(frozen=True, slots=True)
class FrozenDataset:
    version: str
    marker: str
    train: tuple[DatasetCase, ...]
    validation: tuple[DatasetCase, ...]
    gold: tuple[DatasetCase, ...]
    file_digests: Mapping[str, str]
    dataset_digest: str


_BASE_CONFIG: dict[str, Any] = {
    "architecture": "GPT2LMHeadModel",
    "vocab_size": 128,
    "n_positions": 64,
    "n_embd": 64,
    "n_layer": 2,
    "n_head": 2,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
}


@dataclass(frozen=True, slots=True)
class SandboxPlan:
    comparison_group: str
    seed: int
    max_steps: int
    batch_size: int
    learning_rate: float
    max_sequence_length: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    minimum_relative_improvement: float
    replay_tolerance: float
    dataset_digest: str
    base_config: Mapping[str, Any]

    @classmethod
    def default(cls, dataset: FrozenDataset) -> SandboxPlan:
        return cls(
            comparison_group="industrial-maintenance-tiny-v1",
            seed=20260817,
            max_steps=40,
            batch_size=2,
            learning_rate=0.02,
            max_sequence_length=48,
            lora_rank=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=("c_attn", "c_proj"),
            minimum_relative_improvement=0.02,
            replay_tolerance=1e-6,
            dataset_digest=dataset.dataset_digest,
            base_config=dict(_BASE_CONFIG),
        )

    @property
    def base_config_digest(self) -> str:
        return sha256_json(self.base_config)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class BaseModelIdentity:
    directory: Path
    config_digest: str
    weights_digest: str


@dataclass(frozen=True, slots=True)
class RawTrainingEvidence:
    method: str
    optimizer_steps: int
    train_loss: float
    validation_loss: float
    trainable_parameters: int
    total_parameters: int
    quantized_modules: int
    peft_only: bool
    visited_case_ids: tuple[str, ...]
    runtime_versions: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AdapterManifest:
    directory: Path
    method: str
    base_config_digest: str
    base_weights_digest: str
    dataset_digest: str
    plan_digest: str
    files: tuple[ArtifactFile, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class TrainingExperimentEvidence:
    method: str
    optimizer_steps: int
    train_loss: float
    validation_loss: float
    trainable_parameters: int
    total_parameters: int
    quantized_modules: int
    peft_only: bool
    visited_case_ids_digest: str
    runtime_versions: Mapping[str, str]
    adapter_manifest: AdapterManifest


@dataclass(frozen=True, slots=True)
class TrainingStage:
    base: BaseModelIdentity
    experiments: Mapping[str, TrainingExperimentEvidence]


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    gold_loss: float
    gold_token_accuracy: float
    evaluated_cases: int
    evaluated_completion_tokens: int


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    image_digest: str


@dataclass(frozen=True, slots=True)
class SandboxReport:
    marker: str
    decision: Literal["CANDIDATE_ELIGIBLE", "NO_GAIN", "REJECTED"]
    selected_method: str | None
    formal_release_created: bool
    comparison_group: str
    dataset_digest: str
    base_config_digest: str
    base_weights_digest: str
    mlflow_run_ids: Mapping[str, str]
    metrics: Mapping[str, EvaluationMetrics]
    relative_improvements: Mapping[str, float]
    gates: Mapping[str, bool]


class SandboxBackend(Protocol):
    def prepare_base(
        self, plan: SandboxPlan, output_directory: Path
    ) -> BaseModelIdentity: ...

    def train(
        self,
        *,
        method: str,
        plan: SandboxPlan,
        train_cases: tuple[DatasetCase, ...],
        validation_cases: tuple[DatasetCase, ...],
        base: BaseModelIdentity,
        output_directory: Path,
    ) -> RawTrainingEvidence: ...

    def evaluate(
        self,
        *,
        method: str,
        plan: SandboxPlan,
        gold_cases: tuple[DatasetCase, ...],
        base: BaseModelIdentity,
        adapter_directory: Path | None,
    ) -> EvaluationMetrics: ...


class TransformersPeftSandboxBackend:
    """Tiny CPU backend loaded only inside the existing training image."""

    def prepare_base(
        self, plan: SandboxPlan, output_directory: Path
    ) -> BaseModelIdentity:
        dependencies = self._dependencies()
        torch = dependencies["torch"]
        config_type = dependencies["GPT2Config"]
        model_type = dependencies["GPT2LMHeadModel"]
        torch.manual_seed(plan.seed)
        config_values = {
            key: value for key, value in plan.base_config.items() if key != "architecture"
        }
        model = model_type(config_type(**config_values))
        output_directory.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(output_directory, safe_serialization=True)
        weights = tuple(output_directory.glob("*.safetensors"))
        if len(weights) != 1:
            raise SandboxTrainingError("base_model_weights_missing")
        identity = BaseModelIdentity(
            directory=output_directory,
            config_digest=plan.base_config_digest,
            weights_digest=sha256_file(weights[0]),
        )
        del model
        return identity

    def train(
        self,
        *,
        method: str,
        plan: SandboxPlan,
        train_cases: tuple[DatasetCase, ...],
        validation_cases: tuple[DatasetCase, ...],
        base: BaseModelIdentity,
        output_directory: Path,
    ) -> RawTrainingEvidence:
        dependencies = self._dependencies()
        torch = dependencies["torch"]
        torch.manual_seed(plan.seed)
        model = self._load_model(method, plan, base, dependencies)
        peft_config = dependencies["LoraConfig"](
            task_type="CAUSAL_LM",
            r=plan.lora_rank,
            lora_alpha=plan.lora_alpha,
            lora_dropout=plan.lora_dropout,
            bias="none",
            target_modules=list(plan.target_modules),
        )
        model = dependencies["get_peft_model"](model, peft_config)
        model.config.use_cache = False
        named_parameters = tuple(model.named_parameters())
        trainable = tuple((name, value) for name, value in named_parameters if value.requires_grad)
        peft_only = bool(trainable) and all("lora_" in name for name, _ in trainable)
        optimizer = torch.optim.AdamW(
            [value for _, value in trainable], lr=plan.learning_rate, weight_decay=0.0
        )
        batches = tuple(
            train_cases[index : index + plan.batch_size]
            for index in range(0, len(train_cases), plan.batch_size)
        )
        losses: list[float] = []
        model.train()
        for step in range(plan.max_steps):
            inputs = self._batch(batches[step % len(batches)], plan, torch)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs).loss
            if loss is None or not torch.isfinite(loss):
                raise SandboxTrainingError("training_loss_non_finite")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_loss, _, _ = self._score(model, validation_cases, plan, torch)
        output_directory.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(output_directory, safe_serialization=True)
        quantized_modules = sum(
            isinstance(module, dependencies["Linear4bit"]) for module in model.modules()
        )
        evidence = RawTrainingEvidence(
            method=method,
            optimizer_steps=plan.max_steps,
            train_loss=sum(losses) / len(losses),
            validation_loss=validation_loss,
            trainable_parameters=sum(int(value.numel()) for _, value in trainable),
            total_parameters=sum(int(value.numel()) for _, value in named_parameters),
            quantized_modules=quantized_modules,
            peft_only=peft_only,
            visited_case_ids=tuple(case.case_id for case in train_cases),
            runtime_versions=self._runtime_versions(),
        )
        del model, optimizer
        return evidence

    def evaluate(
        self,
        *,
        method: str,
        plan: SandboxPlan,
        gold_cases: tuple[DatasetCase, ...],
        base: BaseModelIdentity,
        adapter_directory: Path | None,
    ) -> EvaluationMetrics:
        dependencies = self._dependencies()
        torch = dependencies["torch"]
        model = self._load_model(method, plan, base, dependencies)
        if method != "BASELINE":
            if adapter_directory is None:
                raise SandboxArtifactError("evaluation_adapter_missing")
            model = dependencies["PeftModel"].from_pretrained(
                model, adapter_directory, is_trainable=False, local_files_only=True
            )
        loss, accuracy, tokens = self._score(model, gold_cases, plan, torch)
        del model
        return EvaluationMetrics(
            gold_loss=loss,
            gold_token_accuracy=accuracy,
            evaluated_cases=len(gold_cases),
            evaluated_completion_tokens=tokens,
        )

    @staticmethod
    def _dependencies() -> dict[str, Any]:
        try:
            import importlib
            import importlib.metadata

            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            peft = importlib.import_module("peft")
            bitsandbytes = importlib.import_module("bitsandbytes")
        except ImportError as exc:
            raise SandboxTrainingError("training_dependencies_are_not_installed") from exc
        return {
            "torch": torch,
            "GPT2Config": transformers.GPT2Config,
            "GPT2LMHeadModel": transformers.GPT2LMHeadModel,
            "AutoModelForCausalLM": transformers.AutoModelForCausalLM,
            "BitsAndBytesConfig": transformers.BitsAndBytesConfig,
            "LoraConfig": peft.LoraConfig,
            "PeftModel": peft.PeftModel,
            "get_peft_model": peft.get_peft_model,
            "prepare_model_for_kbit_training": peft.prepare_model_for_kbit_training,
            "Linear4bit": bitsandbytes.nn.Linear4bit,
        }

    @staticmethod
    def _load_model(
        method: str,
        plan: SandboxPlan,
        base: BaseModelIdentity,
        dependencies: Mapping[str, Any],
    ) -> Any:
        torch = dependencies["torch"]
        arguments: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": torch.float32,
        }
        if method == "QLORA":
            arguments.update(
                {
                    "device_map": {"": "cpu"},
                    "quantization_config": dependencies["BitsAndBytesConfig"](
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float32,
                    ),
                }
            )
        model = dependencies["AutoModelForCausalLM"].from_pretrained(
            base.directory, **arguments
        )
        if method == "QLORA":
            model = dependencies["prepare_model_for_kbit_training"](model)
        return model

    @classmethod
    def _score(
        cls,
        model: Any,
        cases: tuple[DatasetCase, ...],
        plan: SandboxPlan,
        torch: Any,
    ) -> tuple[float, float, int]:
        model.eval()
        inputs = cls._batch(cases, plan, torch)
        with torch.no_grad():
            result = model(**inputs)
        loss = result.loss
        if loss is None or not torch.isfinite(loss):
            raise SandboxEvaluationError("evaluation_loss_non_finite")
        shifted_logits = result.logits[:, :-1, :]
        shifted_labels = inputs["labels"][:, 1:]
        mask = shifted_labels.ne(-100)
        tokens = int(mask.sum().item())
        correct = int((shifted_logits.argmax(dim=-1).eq(shifted_labels) & mask).sum().item())
        return float(loss.cpu()), correct / tokens, tokens

    @staticmethod
    def _batch(
        cases: tuple[DatasetCase, ...], plan: SandboxPlan, torch: Any
    ) -> dict[str, Any]:
        encoded: list[tuple[list[int], list[int]]] = []
        for case in cases:
            prompt = [1, *(_token_id(token) for token in case.prompt.split()), 3]
            completion = [*(_token_id(token) for token in case.completion.split()), 2]
            input_ids = (prompt + completion)[: plan.max_sequence_length]
            labels = ([-100] * len(prompt) + completion)[: len(input_ids)]
            encoded.append((input_ids, labels))
        length = max(len(item[0]) for item in encoded)
        input_tensor = torch.zeros((len(encoded), length), dtype=torch.long)
        attention = torch.zeros((len(encoded), length), dtype=torch.long)
        label_tensor = torch.full((len(encoded), length), -100, dtype=torch.long)
        for index, (input_ids, labels) in enumerate(encoded):
            size = len(input_ids)
            input_tensor[index, :size] = torch.tensor(input_ids, dtype=torch.long)
            attention[index, :size] = 1
            label_tensor[index, :size] = torch.tensor(labels, dtype=torch.long)
        return {"input_ids": input_tensor, "attention_mask": attention, "labels": label_tensor}

    @staticmethod
    def _runtime_versions() -> dict[str, str]:
        import importlib.metadata

        return {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "bitsandbytes")
        }


def load_frozen_dataset(root: Path) -> FrozenDataset:
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SandboxDataError("dataset_manifest_invalid") from exc
    _require_keys(
        manifest,
        {"schema_version", "version", "marker", "case_schema", "required_fields", "files"},
        "dataset_manifest_schema_invalid",
    )
    if (
        manifest["schema_version"] != DATASET_SCHEMA
        or manifest["version"] != "v1"
        or manifest["marker"] != MARKER
        or manifest["case_schema"] != CASE_SCHEMA
        or manifest["required_fields"] != [
            "schema_version",
            "case_id",
            "split",
            "prompt",
            "completion",
            "risk_level",
        ]
        or not isinstance(manifest["files"], list)
        or len(manifest["files"]) != 3
    ):
        raise SandboxDataError("dataset_manifest_contract_invalid")

    cases_by_split: dict[str, tuple[DatasetCase, ...]] = {}
    file_digests: dict[str, str] = {}
    all_ids: set[str] = set()
    expected_paths = {"train": "train.jsonl", "validation": "validation.jsonl", "gold": "gold.jsonl"}
    expected_purposes = {
        "train": "optimization",
        "validation": "training_validation",
        "gold": "evaluation_only",
    }
    for entry in manifest["files"]:
        _require_keys(
            entry,
            {"path", "split", "sha256", "case_count", "purpose"},
            "dataset_file_manifest_invalid",
        )
        split = entry["split"]
        if (
            split not in SPLITS
            or entry["path"] != expected_paths[split]
            or entry["purpose"] != expected_purposes[split]
            or not _is_digest(entry["sha256"])
            or not isinstance(entry["case_count"], int)
            or entry["case_count"] <= 0
            or split in cases_by_split
        ):
            raise SandboxDataError("dataset_file_manifest_invalid")
        path = root / entry["path"]
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise SandboxDataError("dataset_file_unavailable") from exc
        digest = sha256(content).hexdigest()
        if digest != entry["sha256"]:
            raise SandboxDataError("dataset_file_digest_mismatch")
        cases = _parse_cases(content, split)
        if len(cases) != entry["case_count"]:
            raise SandboxDataError("dataset_case_count_mismatch")
        for case in cases:
            if case.case_id in all_ids:
                raise SandboxDataError("dataset_case_id_reused")
            all_ids.add(case.case_id)
        cases_by_split[split] = cases
        file_digests[split] = digest
    if set(cases_by_split) != set(SPLITS):
        raise SandboxDataError("dataset_split_set_invalid")
    dataset_digest = sha256_json(
        {
            "version": manifest["version"],
            "file_digests": file_digests,
            "cases": {
                split: [asdict(case) for case in cases_by_split[split]] for split in SPLITS
            },
        }
    )
    return FrozenDataset(
        version=manifest["version"],
        marker=manifest["marker"],
        train=cases_by_split["train"],
        validation=cases_by_split["validation"],
        gold=cases_by_split["gold"],
        file_digests=file_digests,
        dataset_digest=dataset_digest,
    )


def run_training_stage(
    *,
    dataset: FrozenDataset,
    plan: SandboxPlan,
    backend: SandboxBackend,
    output_directory: Path,
) -> TrainingStage:
    if plan.dataset_digest != dataset.dataset_digest:
        raise SandboxTrainingError("training_plan_dataset_digest_mismatch")
    output_directory.mkdir(parents=True, exist_ok=True)
    base = backend.prepare_base(plan, output_directory / "base-model")
    if base.config_digest != plan.base_config_digest or not _is_digest(base.weights_digest):
        raise SandboxTrainingError("base_model_identity_invalid")
    experiments: dict[str, TrainingExperimentEvidence] = {}
    for method in METHODS:
        adapter_directory = output_directory / method.lower() / "adapter"
        raw = backend.train(
            method=method,
            plan=plan,
            train_cases=dataset.train,
            validation_cases=dataset.validation,
            base=base,
            output_directory=adapter_directory,
        )
        _validate_training_evidence(raw, method, dataset)
        manifest = build_adapter_manifest(
            adapter_directory,
            method=method,
            base=base,
            dataset_digest=dataset.dataset_digest,
            plan_digest=plan.digest,
        )
        experiments[method] = TrainingExperimentEvidence(
            method=method,
            optimizer_steps=raw.optimizer_steps,
            train_loss=raw.train_loss,
            validation_loss=raw.validation_loss,
            trainable_parameters=raw.trainable_parameters,
            total_parameters=raw.total_parameters,
            quantized_modules=raw.quantized_modules,
            peft_only=raw.peft_only,
            visited_case_ids_digest=sha256_json(sorted(raw.visited_case_ids)),
            runtime_versions=dict(raw.runtime_versions),
            adapter_manifest=manifest,
        )
    return TrainingStage(base=base, experiments=experiments)


def build_adapter_manifest(
    directory: Path,
    *,
    method: str,
    base: BaseModelIdentity,
    dataset_digest: str,
    plan_digest: str,
) -> AdapterManifest:
    files = _artifact_files(directory)
    if not files:
        raise SandboxArtifactError("adapter_artifact_empty")
    payload = {
        "schema_version": ADAPTER_MANIFEST_SCHEMA,
        "marker": MARKER,
        "method": method,
        "base_config_digest": base.config_digest,
        "base_weights_digest": base.weights_digest,
        "dataset_digest": dataset_digest,
        "plan_digest": plan_digest,
        "files": [asdict(item) for item in files],
    }
    digest = sha256_json(payload)
    _write_json(directory / "adapter-manifest.json", {**payload, "manifest_digest": digest})
    return AdapterManifest(
        directory=directory,
        method=method,
        base_config_digest=base.config_digest,
        base_weights_digest=base.weights_digest,
        dataset_digest=dataset_digest,
        plan_digest=plan_digest,
        files=files,
        digest=digest,
    )


def verify_adapter_manifest(manifest: AdapterManifest) -> None:
    actual = _artifact_files(manifest.directory)
    if actual != manifest.files:
        raise SandboxArtifactError("adapter_artifact_digest_mismatch")
    payload = {
        "schema_version": ADAPTER_MANIFEST_SCHEMA,
        "marker": MARKER,
        "method": manifest.method,
        "base_config_digest": manifest.base_config_digest,
        "base_weights_digest": manifest.base_weights_digest,
        "dataset_digest": manifest.dataset_digest,
        "plan_digest": manifest.plan_digest,
        "files": [asdict(item) for item in manifest.files],
    }
    if sha256_json(payload) != manifest.digest:
        raise SandboxArtifactError("adapter_manifest_digest_mismatch")


def run_sandbox(
    *,
    dataset_root: Path,
    output_directory: Path,
    backend: SandboxBackend,
    tracker: ExperimentTracker,
    runtime: RuntimeIdentity,
) -> SandboxReport:
    if not runtime.image_digest.startswith("sha256:") or not _is_digest(
        runtime.image_digest.removeprefix("sha256:")
    ):
        raise SandboxTrainingError("runtime_image_digest_invalid")
    dataset = load_frozen_dataset(dataset_root)
    plan = SandboxPlan.default(dataset)
    output_directory.mkdir(parents=True, exist_ok=True)
    runs: dict[str, str] = {}
    try:
        for method in ("BASELINE", *METHODS):
            run = tracker.start_run(
                experiment_name="industrial-ops-model-training-sandbox",
                run_name=method,
                params=_tracking_params(method, dataset, plan, runtime),
                tags={"ioap.environment": MARKER, "ioap.comparison_group": plan.comparison_group},
            )
            runs[method] = run.run_id
        stage = run_training_stage(
            dataset=dataset,
            plan=plan,
            backend=backend,
            output_directory=output_directory,
        )
        metrics, replay_gate = _evaluate_reloaded(dataset, plan, stage, backend)
        baseline_loss = metrics["BASELINE"].gold_loss
        improvements = {
            method: (baseline_loss - metrics[method].gold_loss) / baseline_loss for method in METHODS
        }
        gates = {
            "dataset_integrity": True,
            "gold_isolation": True,
            "base_identity": True,
            "adapter_integrity": True,
            "real_optimizer_steps": all(
                item.optimizer_steps > 0 for item in stage.experiments.values()
            ),
            "peft_only": all(item.peft_only for item in stage.experiments.values()),
            "qlora_nf4": stage.experiments["QLORA"].quantized_modules > 0,
            "replay_reproducibility": replay_gate,
        }
        selected: str | None = None
        if not all(gates.values()):
            decision: Literal["CANDIDATE_ELIGIBLE", "NO_GAIN", "REJECTED"] = "REJECTED"
        else:
            winner = max(METHODS, key=improvements.__getitem__)
            if improvements[winner] >= plan.minimum_relative_improvement:
                decision = "CANDIDATE_ELIGIBLE"
                selected = winner
            else:
                decision = "NO_GAIN"
        report = SandboxReport(
            marker=MARKER,
            decision=decision,
            selected_method=selected,
            formal_release_created=False,
            comparison_group=plan.comparison_group,
            dataset_digest=dataset.dataset_digest,
            base_config_digest=stage.base.config_digest,
            base_weights_digest=stage.base.weights_digest,
            mlflow_run_ids=runs,
            metrics=metrics,
            relative_improvements=improvements,
            gates=gates,
        )
        for method, run_id in runs.items():
            method_metrics = metrics[method]
            tracking_metrics = {
                "gold_loss": method_metrics.gold_loss,
                "gold_token_accuracy": method_metrics.gold_token_accuracy,
            }
            if method in stage.experiments:
                training = stage.experiments[method]
                tracking_metrics.update(
                    {
                        "train_loss": training.train_loss,
                        "validation_loss": training.validation_loss,
                        "optimizer_steps": float(training.optimizer_steps),
                    }
                )
            tracker.complete_run(run_id, metrics=tracking_metrics)
        _write_json(output_directory / "sandbox-report.json", _report_payload(report, plan, stage))
        _write_json(output_directory / "promotion-receipt.json", _receipt_payload(report, plan, stage))
        return report
    except Exception as exc:
        reason = type(exc).__name__
        for run_id in runs.values():
            try:
                tracker.fail_run(run_id, reason_code=reason)
            except Exception:
                pass
        raise


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def sha256_json(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _parse_cases(content: bytes, expected_split: str) -> tuple[DatasetCase, ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SandboxDataError("dataset_file_encoding_invalid") from exc
    cases: list[DatasetCase] = []
    for line in text.splitlines():
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SandboxDataError("dataset_case_json_invalid") from exc
        _require_keys(raw, CASE_FIELDS, "dataset_case_schema_invalid")
        if (
            raw["schema_version"] != CASE_SCHEMA
            or raw["split"] != expected_split
            or not _valid_text(raw["case_id"], 80)
            or not _valid_text(raw["prompt"], 300)
            or not _valid_text(raw["completion"], 300)
            or raw["risk_level"] not in RISK_LEVELS
        ):
            raise SandboxDataError("dataset_case_contract_invalid")
        cases.append(DatasetCase(**raw))
    if not cases:
        raise SandboxDataError("dataset_split_empty")
    return tuple(cases)


def _validate_training_evidence(
    evidence: RawTrainingEvidence, method: str, dataset: FrozenDataset
) -> None:
    if evidence.method != method:
        raise SandboxTrainingError("training_method_identity_mismatch")
    if evidence.optimizer_steps <= 0:
        raise SandboxTrainingError("training_optimizer_steps_invalid")
    if not math.isfinite(evidence.train_loss) or not math.isfinite(evidence.validation_loss):
        raise SandboxTrainingError("training_loss_non_finite")
    if (
        evidence.trainable_parameters <= 0
        or evidence.total_parameters < evidence.trainable_parameters
    ):
        raise SandboxTrainingError("training_parameter_counts_invalid")
    if not evidence.peft_only:
        raise SandboxTrainingError("training_updated_non_peft_parameters")
    if method == "LORA" and evidence.quantized_modules != 0:
        raise SandboxTrainingError("lora_must_not_quantize_base")
    if method == "QLORA" and evidence.quantized_modules <= 0:
        raise SandboxTrainingError("qlora_nf4_modules_missing")
    visited = set(evidence.visited_case_ids)
    train_ids = {case.case_id for case in dataset.train}
    gold_ids = {case.case_id for case in dataset.gold}
    if visited != train_ids or visited & gold_ids:
        raise SandboxTrainingError("training_case_access_invalid")
    if not evidence.runtime_versions or any(
        not _valid_text(key, 80) or not _valid_text(value, 120)
        for key, value in evidence.runtime_versions.items()
    ):
        raise SandboxTrainingError("training_runtime_identity_invalid")


def _evaluate_reloaded(
    dataset: FrozenDataset,
    plan: SandboxPlan,
    stage: TrainingStage,
    backend: SandboxBackend,
) -> tuple[dict[str, EvaluationMetrics], bool]:
    metrics: dict[str, EvaluationMetrics] = {}
    replay_gate = True
    for method in ("BASELINE", *METHODS):
        adapter = None
        if method in stage.experiments:
            manifest = stage.experiments[method].adapter_manifest
            verify_adapter_manifest(manifest)
            adapter = manifest.directory
        first = backend.evaluate(
            method=method,
            plan=plan,
            gold_cases=dataset.gold,
            base=stage.base,
            adapter_directory=adapter,
        )
        second = backend.evaluate(
            method=method,
            plan=plan,
            gold_cases=dataset.gold,
            base=stage.base,
            adapter_directory=adapter,
        )
        _validate_evaluation(first, len(dataset.gold))
        _validate_evaluation(second, len(dataset.gold))
        if (
            abs(first.gold_loss - second.gold_loss) > plan.replay_tolerance
            or abs(first.gold_token_accuracy - second.gold_token_accuracy)
            > plan.replay_tolerance
        ):
            replay_gate = False
        metrics[method] = first
    if metrics["BASELINE"].gold_loss <= 0:
        raise SandboxEvaluationError("baseline_gold_loss_invalid")
    return metrics, replay_gate


def _validate_evaluation(metrics: EvaluationMetrics, expected_cases: int) -> None:
    if (
        not math.isfinite(metrics.gold_loss)
        or metrics.gold_loss <= 0
        or not math.isfinite(metrics.gold_token_accuracy)
        or not 0 <= metrics.gold_token_accuracy <= 1
        or metrics.evaluated_cases != expected_cases
        or metrics.evaluated_completion_tokens <= 0
    ):
        raise SandboxEvaluationError("gold_evaluation_metrics_invalid")


def _artifact_files(directory: Path) -> tuple[ArtifactFile, ...]:
    if not directory.is_dir():
        raise SandboxArtifactError("adapter_artifact_directory_missing")
    files: list[ArtifactFile] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise SandboxArtifactError("adapter_artifact_symlink_forbidden")
        if not path.is_file() or path.name == "adapter-manifest.json":
            continue
        relative = path.relative_to(directory).as_posix()
        files.append(
            ArtifactFile(path=relative, size_bytes=path.stat().st_size, sha256=sha256_file(path))
        )
    return tuple(files)


def _tracking_params(
    method: str,
    dataset: FrozenDataset,
    plan: SandboxPlan,
    runtime: RuntimeIdentity,
) -> dict[str, str]:
    return {
        "ioap.method": method,
        "ioap.dataset_version": dataset.version,
        "ioap.dataset_digest": dataset.dataset_digest,
        "ioap.gold_digest": dataset.file_digests["gold"],
        "ioap.plan_digest": plan.digest,
        "ioap.base_config_digest": plan.base_config_digest,
        "ioap.seed": str(plan.seed),
        "ioap.image_digest": runtime.image_digest,
        "ioap.marker": MARKER,
    }


def _report_payload(
    report: SandboxReport, plan: SandboxPlan, stage: TrainingStage
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA,
        **_receipt_common(report, plan, stage),
        "training": {
            method: {
                "optimizer_steps": item.optimizer_steps,
                "train_loss": item.train_loss,
                "validation_loss": item.validation_loss,
                "trainable_parameters": item.trainable_parameters,
                "total_parameters": item.total_parameters,
                "quantized_modules": item.quantized_modules,
                "peft_only": item.peft_only,
                "visited_case_ids_digest": item.visited_case_ids_digest,
                "adapter_manifest_digest": item.adapter_manifest.digest,
                "runtime_versions": dict(item.runtime_versions),
            }
            for method, item in stage.experiments.items()
        },
    }


def _receipt_payload(
    report: SandboxReport, plan: SandboxPlan, stage: TrainingStage
) -> dict[str, Any]:
    return {"schema_version": PROMOTION_SCHEMA, **_receipt_common(report, plan, stage)}


def _receipt_common(
    report: SandboxReport, plan: SandboxPlan, stage: TrainingStage
) -> dict[str, Any]:
    selected_manifest = (
        stage.experiments[report.selected_method].adapter_manifest.digest
        if report.selected_method is not None
        else None
    )
    return {
        "marker": report.marker,
        "decision": report.decision,
        "selected_method": report.selected_method,
        "formal_release_created": report.formal_release_created,
        "comparison_group": report.comparison_group,
        "dataset_digest": report.dataset_digest,
        "base_config_digest": report.base_config_digest,
        "base_weights_digest": report.base_weights_digest,
        "selected_adapter_manifest_digest": selected_manifest,
        "mlflow_run_ids": dict(report.mlflow_run_ids),
        "metrics": {method: asdict(value) for method, value in report.metrics.items()},
        "relative_improvements": dict(report.relative_improvements),
        "minimum_relative_improvement": plan.minimum_relative_improvement,
        "gates": dict(report.gates),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_canonical_json(payload) + b"\n")
    temporary.replace(path)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_keys(value: object, keys: set[str] | frozenset[str], error: str) -> None:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise SandboxDataError(error)


def _valid_text(value: object, maximum: int) -> bool:
    return (
        isinstance(value, str)
        and value == value.strip()
        and 0 < len(value) <= maximum
        and "\x00" not in value
        and "\n" not in value
        and "\r" not in value
    )


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _token_id(token: str) -> int:
    return 4 + int.from_bytes(sha256(token.encode("utf-8")).digest()[:4], "big") % 124
