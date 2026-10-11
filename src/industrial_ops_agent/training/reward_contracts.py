"""Versioned deterministic reward contracts for governed GRPO experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Literal, TypeAlias, cast

LEGACY_REWARD_CONTRACT_VERSION = "industrial-json-exact-v1"
CURRENT_REWARD_CONTRACT_VERSION = "industrial-json-structural-v2"

RewardContractVersion: TypeAlias = Literal[
    "industrial-json-exact-v1",
    "industrial-json-structural-v2",
]


class RewardContractError(ValueError):
    """A reward contract, target or migration is not registered."""


@dataclass(frozen=True, slots=True)
class RewardContractDescriptor:
    version: RewardContractVersion
    lifecycle: Literal["LEGACY_REPLAY_ONLY", "CURRENT"]
    scorer: Literal["json_structural_exact", "json_leaf_f1"]
    target_media_type: Literal["application/json"]
    minimum_score: float
    maximum_score: float
    invalid_json_score: float
    accepts_migrations_from: tuple[RewardContractVersion, ...]

    @property
    def digest(self) -> str:
        payload = json.dumps(
            asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return f"sha256:{sha256(payload).hexdigest()}"


@dataclass(frozen=True, slots=True)
class RewardTargetMigration:
    source_version: RewardContractVersion
    source_digest: str
    target_version: RewardContractVersion
    target_digest: str
    canonical_target: str
    canonical_target_hash: str
    migrated: bool

    def receipt(self) -> dict[str, Any]:
        return asdict(self)


_CONTRACTS: dict[RewardContractVersion, RewardContractDescriptor] = {
    LEGACY_REWARD_CONTRACT_VERSION: RewardContractDescriptor(
        version=LEGACY_REWARD_CONTRACT_VERSION,
        lifecycle="LEGACY_REPLAY_ONLY",
        scorer="json_structural_exact",
        target_media_type="application/json",
        minimum_score=0.0,
        maximum_score=1.0,
        invalid_json_score=0.0,
        accepts_migrations_from=(LEGACY_REWARD_CONTRACT_VERSION,),
    ),
    CURRENT_REWARD_CONTRACT_VERSION: RewardContractDescriptor(
        version=CURRENT_REWARD_CONTRACT_VERSION,
        lifecycle="CURRENT",
        scorer="json_leaf_f1",
        target_media_type="application/json",
        minimum_score=0.0,
        maximum_score=1.0,
        invalid_json_score=0.0,
        accepts_migrations_from=(
            LEGACY_REWARD_CONTRACT_VERSION,
            CURRENT_REWARD_CONTRACT_VERSION,
        ),
    ),
}

CURRENT_REWARD_CONTRACT_DIGEST = _CONTRACTS[CURRENT_REWARD_CONTRACT_VERSION].digest


def reward_contract(version: str) -> RewardContractDescriptor:
    try:
        return _CONTRACTS[cast(RewardContractVersion, version)]
    except KeyError as exc:
        raise RewardContractError("reward_contract_is_not_registered") from exc


def validate_reward_contract_digest(version: str, digest: str) -> RewardContractDescriptor:
    descriptor = reward_contract(version)
    if digest != descriptor.digest:
        raise RewardContractError("reward_contract_digest_mismatch")
    return descriptor


def require_current_reward_contract(version: Any, digest: Any) -> RewardContractDescriptor:
    if version != CURRENT_REWARD_CONTRACT_VERSION:
        raise RewardContractError("new_grpo_experiment_requires_current_reward_contract")
    if not isinstance(digest, str):
        raise RewardContractError("reward_contract_digest_is_required")
    return validate_reward_contract_digest(version, digest)


def migrate_reward_target(
    target: str,
    *,
    source_version: str,
    target_version: str,
) -> RewardTargetMigration:
    source = reward_contract(source_version)
    destination = reward_contract(target_version)
    if source.version not in destination.accepts_migrations_from:
        raise RewardContractError("reward_contract_migration_is_not_registered")
    canonical = _canonical_json(target)
    return RewardTargetMigration(
        source_version=source.version,
        source_digest=source.digest,
        target_version=destination.version,
        target_digest=destination.digest,
        canonical_target=canonical,
        canonical_target_hash=f"sha256:{sha256(canonical.encode()).hexdigest()}",
        migrated=source.version != destination.version,
    )


def score_reward(version: str, completion: Any, target: str) -> float:
    descriptor = reward_contract(version)
    try:
        candidate_value = json.loads(str(_completion_content(completion)))
        target_value = json.loads(target)
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return descriptor.invalid_json_score
    if descriptor.scorer == "json_structural_exact":
        return 1.0 if candidate_value == target_value else 0.0
    candidate_leaves = _flatten_json(candidate_value)
    target_leaves = _flatten_json(target_value)
    matched = sum(
        1
        for path, value in candidate_leaves.items()
        if path in target_leaves and target_leaves[path] == value
    )
    denominator = len(candidate_leaves) + len(target_leaves)
    return 1.0 if denominator == 0 else (2.0 * matched) / denominator


def summarize_reward_migrations(
    migrations: list[RewardTargetMigration],
) -> dict[str, Any]:
    receipts = sorted(
        (migration.receipt() for migration in migrations),
        key=lambda item: (
            item["source_version"],
            item["target_version"],
            item["canonical_target_hash"],
        ),
    )
    encoded = json.dumps(
        receipts, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "target_version": migrations[0].target_version if migrations else None,
        "target_digest": migrations[0].target_digest if migrations else None,
        "source_versions": sorted({item.source_version for item in migrations}),
        "source_digests": sorted({item.source_digest for item in migrations}),
        "sample_count": len(migrations),
        "migrated_sample_count": sum(item.migrated for item in migrations),
        "migration_receipt_digest": f"sha256:{sha256(encoded).hexdigest()}",
    }


def _canonical_json(value: str) -> str:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RewardContractError("reward_target_is_not_valid_json") from exc
    if not isinstance(parsed, (dict, list)):
        raise RewardContractError("reward_target_must_be_json_object_or_array")
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _completion_content(completion: Any) -> Any:
    if isinstance(completion, list) and completion:
        last_message = completion[-1]
        if isinstance(last_message, dict):
            return last_message.get("content")
    return completion


def _flatten_json(value: Any, path: str = "$") -> dict[str, str]:
    if isinstance(value, dict):
        if not value:
            return {path: "object:{}"}
        flattened: dict[str, str] = {}
        for key in sorted(value):
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            flattened.update(_flatten_json(value[key], f"{path}/{escaped}"))
        return flattened
    if isinstance(value, list):
        if not value:
            return {path: "array:[]"}
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(_flatten_json(item, f"{path}/{index}"))
        return flattened
    return {
        path: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    }
