"""Strict distributed-training profiles and launcher contracts.

The experiment stores intent as JSON, but workers never pass that JSON directly to
DeepSpeed, FSDP, or ``torchrun``.  This module compiles the small reviewed surface
into deterministic runtime arguments.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

DistributedStrategy = Literal["single_gpu", "deepspeed", "fsdp"]


class DistributedProfileError(ValueError):
    """The frozen distributed profile is unsupported or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class CompiledDistributedProfile:
    strategy: DistributedStrategy
    world_size: int
    node_count: Literal[1]
    zero_stage: Literal[2, 3] | None = None
    offload_optimizer_device: Literal["none", "cpu"] = "none"
    offload_param_device: Literal["none", "cpu"] = "none"
    fsdp_sharding_strategy: Literal["FULL_SHARD", "SHARD_GRAD_OP"] | None = None
    transformer_layer_cls_to_wrap: tuple[str, ...] = ()
    activation_checkpointing: bool = False

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_distributed_profile(
    distributed: dict[str, Any],
    hardware: dict[str, Any],
    *,
    method: str,
    quantized: bool,
) -> CompiledDistributedProfile:
    """Compile one single-node profile; reject raw or ambiguous launcher options."""

    strategy = distributed.get("strategy")
    if strategy not in {"single_gpu", "deepspeed", "fsdp"}:
        raise DistributedProfileError("distributed_strategy_is_not_supported")
    world_size = _integer(distributed, "world_size", default=1)
    node_count = _integer(distributed, "node_count", default=1)
    if node_count != 1:
        raise DistributedProfileError("only_single_node_distributed_training_is_supported")
    if hardware.get("accelerator") != "NVIDIA_GPU":
        raise DistributedProfileError("nvidia_gpu_hardware_is_required")
    gpu_count = _integer(hardware, "count", default=1)
    if gpu_count != world_size:
        raise DistributedProfileError("hardware_gpu_count_must_equal_world_size")

    if strategy == "single_gpu":
        if world_size != 1:
            raise DistributedProfileError("single_gpu_profile_requires_world_size_one")
        return CompiledDistributedProfile("single_gpu", 1, 1)

    if method not in {"LORA", "QLORA"}:
        raise DistributedProfileError("distributed_profile_currently_requires_lora_or_qlora")
    if world_size < 2 or world_size > 8:
        raise DistributedProfileError("distributed_world_size_must_be_between_two_and_eight")

    if strategy == "deepspeed":
        zero_stage = _integer(distributed, "zero_stage", default=2)
        if zero_stage not in {2, 3}:
            raise DistributedProfileError("deepspeed_zero_stage_must_be_two_or_three")
        if quantized and zero_stage == 3:
            raise DistributedProfileError("qlora_requires_deepspeed_zero_stage_two")
        optimizer_offload = _choice(
            distributed, "offload_optimizer_device", {"none", "cpu"}, default="none"
        )
        parameter_offload = _choice(
            distributed, "offload_param_device", {"none", "cpu"}, default="none"
        )
        if zero_stage != 3 and parameter_offload != "none":
            raise DistributedProfileError("parameter_offload_requires_deepspeed_zero_stage_three")
        return CompiledDistributedProfile(
            "deepspeed",
            world_size,
            1,
            zero_stage=cast(Literal[2, 3], zero_stage),
            offload_optimizer_device=optimizer_offload,
            offload_param_device=parameter_offload,
        )

    if quantized:
        raise DistributedProfileError("qlora_fsdp_profile_is_not_supported")
    sharding = _choice(
        distributed,
        "sharding_strategy",
        {"FULL_SHARD", "SHARD_GRAD_OP"},
        default="FULL_SHARD",
    )
    layer_classes = distributed.get("transformer_layer_cls_to_wrap")
    if (
        not isinstance(layer_classes, list)
        or not layer_classes
        or not all(
            isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", value)
            for value in layer_classes
        )
    ):
        raise DistributedProfileError("fsdp_transformer_layer_classes_are_required")
    return CompiledDistributedProfile(
        "fsdp",
        world_size,
        1,
        fsdp_sharding_strategy=sharding,
        transformer_layer_cls_to_wrap=tuple(dict.fromkeys(layer_classes)),
        activation_checkpointing=_boolean(
            distributed, "activation_checkpointing", default=True
        ),
    )


def trainer_argument_overrides(
    profile: CompiledDistributedProfile,
    *,
    precision: Literal["bfloat16", "float16"],
) -> dict[str, Any]:
    """Return only reviewed Hugging Face Trainer distributed arguments."""

    if profile.strategy == "single_gpu":
        return {}
    if profile.strategy == "deepspeed":
        assert profile.zero_stage is not None
        zero: dict[str, Any] = {
            "stage": profile.zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
        }
        if profile.offload_optimizer_device == "cpu":
            zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        if profile.offload_param_device == "cpu":
            zero["offload_param"] = {"device": "cpu", "pin_memory": True}
        return {
            "deepspeed": {
                "bf16": {"enabled": precision == "bfloat16"},
                "fp16": {"enabled": precision == "float16"},
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": "auto",
                "gradient_clipping": "auto",
                "zero_optimization": zero,
                "steps_per_print": 100,
                "wall_clock_breakdown": False,
            }
        }
    assert profile.fsdp_sharding_strategy is not None
    return {
        "fsdp": "full_shard auto_wrap"
        if profile.fsdp_sharding_strategy == "FULL_SHARD"
        else "shard_grad_op auto_wrap",
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": list(profile.transformer_layer_cls_to_wrap),
            "backward_prefetch": "backward_pre",
            "forward_prefetch": False,
            "use_orig_params": True,
            "state_dict_type": "FULL_STATE_DICT",
            "cpu_ram_efficient_loading": True,
            "sync_module_states": True,
            "activation_checkpointing": profile.activation_checkpointing,
        },
    }


def build_torchrun_command(
    profile: CompiledDistributedProfile,
    *,
    module: str,
    module_arguments: list[str],
) -> list[str]:
    if not profile.distributed:
        raise DistributedProfileError("torchrun_requires_a_distributed_profile")
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={profile.world_size}",
        "--module",
        module,
        *module_arguments,
    ]


def process_rank() -> int:
    return _environment_integer("RANK", default=0)


def local_rank() -> int:
    return _environment_integer("LOCAL_RANK", default=0)


def launched_world_size() -> int:
    return _environment_integer("WORLD_SIZE", default=1)


def is_distributed_child() -> bool:
    return os.getenv("IOAP_DISTRIBUTED_TRAINING_CHILD") == "1"


def _integer(values: dict[str, Any], key: str, *, default: int) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DistributedProfileError(f"{key}_must_be_an_integer")
    return int(value)


def _environment_integer(key: str, *, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise DistributedProfileError(f"{key.lower()}_must_be_an_integer") from exc


def _choice(
    values: dict[str, Any], key: str, choices: set[str], *, default: str
) -> Any:
    value = values.get(key, default)
    if value not in choices:
        raise DistributedProfileError(f"{key}_is_not_supported")
    return value


def _boolean(values: dict[str, Any], key: str, *, default: bool) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise DistributedProfileError(f"{key}_must_be_boolean")
    return value
