"""Immutable deterministic reward profile for industrial Agent GRPO experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Literal, TypeAlias

from industrial_ops_agent.training.reward_contracts import (
    CURRENT_REWARD_CONTRACT_DIGEST,
    CURRENT_REWARD_CONTRACT_VERSION,
    score_reward,
)

LEGACY_PROFILE_VERSION: Literal["industrial-agent-json-grpo-v1"] = (
    "industrial-agent-json-grpo-v1"
)
PROFILE_VERSION: Literal["industrial-agent-json-grpo-v2"] = "industrial-agent-json-grpo-v2"
GrpoRewardProfileVersion: TypeAlias = Literal[
    "industrial-agent-json-grpo-v1",
    "industrial-agent-json-grpo-v2",
]


class GrpoRewardProfileError(ValueError):
    """The registered reward profile or one of its targets is invalid."""


@dataclass(frozen=True, slots=True)
class GrpoRewardProfileDescriptor:
    version: GrpoRewardProfileVersion
    lifecycle: Literal["LEGACY_REPLAY_ONLY", "CURRENT"]
    structural_contract_version: str
    structural_contract_digest: str
    structural_weight: float
    bounded_format_weight: float
    required_keys: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    allowed_priorities: tuple[str, ...]
    forbidden_fragments: tuple[str, ...]
    minimum_score: float
    maximum_score: float

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            asdict(self),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return f"sha256:{sha256(encoded).hexdigest()}"


LEGACY_GRPO_REWARD_PROFILE = GrpoRewardProfileDescriptor(
    version=LEGACY_PROFILE_VERSION,
    lifecycle="LEGACY_REPLAY_ONLY",
    structural_contract_version=CURRENT_REWARD_CONTRACT_VERSION,
    structural_contract_digest=CURRENT_REWARD_CONTRACT_DIGEST,
    structural_weight=0.8,
    bounded_format_weight=0.2,
    required_keys=("action", "approval_required", "equipment_id", "priority"),
    allowed_actions=("INSPECT", "HOLD_AND_ESCALATE"),
    allowed_priorities=("P1", "P2"),
    forbidden_fragments=(
        "bypass",
        "disable safeguard",
        "restart immediately",
        'approval_required":false',
    ),
    minimum_score=0.0,
    maximum_score=1.0,
)
LEGACY_GRPO_REWARD_PROFILE_DIGEST = LEGACY_GRPO_REWARD_PROFILE.digest

CURRENT_GRPO_REWARD_PROFILE = GrpoRewardProfileDescriptor(
    version=PROFILE_VERSION,
    lifecycle="CURRENT",
    structural_contract_version=CURRENT_REWARD_CONTRACT_VERSION,
    structural_contract_digest=CURRENT_REWARD_CONTRACT_DIGEST,
    structural_weight=0.8,
    bounded_format_weight=0.2,
    required_keys=("action", "approval_required", "equipment_id", "priority"),
    allowed_actions=("INSPECT", "HOLD_AND_ESCALATE"),
    allowed_priorities=("P1", "P2"),
    forbidden_fragments=(
        "bypass",
        "disable safeguard",
        "restart immediately",
        'approval_required":false',
    ),
    minimum_score=0.0,
    maximum_score=1.0,
)
CURRENT_GRPO_REWARD_PROFILE_DIGEST = CURRENT_GRPO_REWARD_PROFILE.digest


def require_current_grpo_reward_profile(
    version: str,
    digest: str,
) -> GrpoRewardProfileDescriptor:
    """Fail closed unless the caller binds the current immutable profile."""

    if version != PROFILE_VERSION:
        raise GrpoRewardProfileError("grpo_reward_profile_is_not_current")
    if digest != CURRENT_GRPO_REWARD_PROFILE_DIGEST:
        raise GrpoRewardProfileError("grpo_reward_profile_digest_mismatch")
    return CURRENT_GRPO_REWARD_PROFILE


def score_registered_grpo_reward(completion: Any, target: str) -> float:
    """Score one completion with structural truth plus bounded format shaping.

    The registered structural JSON contract owns 80% of the score. The remaining
    20% is deterministic shaping that gives sparse GRPO rollouts a bounded signal
    before they become valid JSON. Invalid or unsafe text can therefore never
    outrank a structurally correct completion.
    """

    return score_grpo_reward(PROFILE_VERSION, completion, target)


def score_grpo_reward(
    version: str,
    completion: Any,
    target: str,
) -> float:
    if version == LEGACY_PROFILE_VERSION:
        descriptor = LEGACY_GRPO_REWARD_PROFILE
        structural_input = completion
    elif version == PROFILE_VERSION:
        descriptor = CURRENT_GRPO_REWARD_PROFILE
        normalized = normalize_grpo_completion(completion)
        structural_input = completion if normalized is None else normalized
    else:
        raise GrpoRewardProfileError("grpo_reward_profile_is_not_registered")
    structural = score_reward(
        descriptor.structural_contract_version,
        structural_input,
        target,
    )
    shaped = _bounded_format_score(completion, target, descriptor)
    value = descriptor.structural_weight * structural + descriptor.bounded_format_weight * shaped
    return min(descriptor.maximum_score, max(descriptor.minimum_score, value))


def normalize_grpo_completion(completion: Any) -> str | None:
    """Extract and canonicalize the first balanced JSON object deterministically."""

    text = str(_completion_content(completion)).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
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
                except (TypeError, ValueError, json.JSONDecodeError):
                    return None
                if not isinstance(value, dict):
                    return None
                return json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            if depth < 0:
                return None
    return None


def _bounded_format_score(
    completion: Any,
    target: str,
    descriptor: GrpoRewardProfileDescriptor,
) -> float:
    text = str(_completion_content(completion)).strip()
    try:
        target_value = json.loads(target)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GrpoRewardProfileError("grpo_reward_target_is_not_valid_json") from exc
    if not isinstance(target_value, dict):
        raise GrpoRewardProfileError("grpo_reward_target_must_be_json_object")
    if set(target_value) != set(descriptor.required_keys):
        raise GrpoRewardProfileError("grpo_reward_target_schema_changed")
    if target_value.get("action") not in descriptor.allowed_actions:
        raise GrpoRewardProfileError("grpo_reward_target_action_is_not_allowed")
    if target_value.get("priority") not in descriptor.allowed_priorities:
        raise GrpoRewardProfileError("grpo_reward_target_priority_is_not_allowed")
    if target_value.get("approval_required") is not True:
        raise GrpoRewardProfileError("grpo_reward_target_requires_human_approval")

    score = 0.0
    if text.startswith("{") and text.endswith("}"):
        score += 0.1
    key_weight = 0.3 / len(descriptor.required_keys)
    score += sum(key_weight for key in descriptor.required_keys if f'"{key}"' in text)
    value_weight = 0.2 / len(descriptor.required_keys)
    score += sum(
        value_weight for value in target_value.values() if _canonical_scalar(value) in text
    )
    lowered = text.lower()
    if not any(fragment in lowered for fragment in descriptor.forbidden_fragments):
        score += 0.1
    try:
        candidate = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        candidate = None
    if isinstance(candidate, dict):
        score += 0.3
    return min(1.0, max(0.0, score))


def _completion_content(completion: Any) -> Any:
    if isinstance(completion, list) and completion:
        last_message = completion[-1]
        if isinstance(last_message, dict):
            return last_message.get("content")
    return completion


def _canonical_scalar(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{sha256(encoded).hexdigest()}"
