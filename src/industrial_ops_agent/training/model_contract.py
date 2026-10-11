"""Content-addressed model/tokenizer contract shared by inspection and training."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any


class ModelContractError(ValueError):
    pass


def expected_model_digest(revision: str) -> str:
    return f"hf-revision:{revision}"


def tokenizer_contract_digests(tokenizer: Any) -> tuple[str, str]:
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ModelContractError("base_tokenizer_has_no_chat_template")
    return tokenizer_content_digest(tokenizer), "sha256:" + sha256(
        chat_template.encode()
    ).hexdigest()


def tokenizer_content_digest(tokenizer: Any) -> str:
    """Hash tokenizer semantics without requiring a generative chat template."""

    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, dict) or not vocab:
        raise ModelContractError("base_tokenizer_has_no_vocabulary")
    added_tokens = {
        str(token_id): {
            "content": str(getattr(token, "content", token)),
            "special": bool(getattr(token, "special", False)),
        }
        for token_id, token in sorted(
            getattr(tokenizer, "added_tokens_decoder", {}).items(),
            key=lambda item: int(item[0]),
        )
    }
    tokenizer_payload = {
        "tokenizer_class": type(tokenizer).__name__,
        "vocab": sorted((str(token), int(token_id)) for token, token_id in vocab.items()),
        "special_tokens_map": _json_safe(getattr(tokenizer, "special_tokens_map", {})),
        "added_tokens": added_tokens,
    }
    return "sha256:" + _digest_json(tokenizer_payload)


def verify_model_contract(
    *,
    tokenizer: Any,
    revision: str,
    base_model_digest: str,
    tokenizer_digest: str,
    chat_template_digest: str,
) -> None:
    if base_model_digest != expected_model_digest(revision):
        raise ModelContractError("base_model_digest_does_not_bind_registered_revision")
    actual_tokenizer, actual_template = tokenizer_contract_digests(tokenizer)
    if tokenizer_digest != actual_tokenizer:
        raise ModelContractError("tokenizer_digest_mismatch")
    if chat_template_digest != actual_template:
        raise ModelContractError("chat_template_digest_mismatch")


def verify_encoder_model_contract(
    *,
    tokenizer: Any,
    revision: str,
    base_model_digest: str,
    tokenizer_digest: str,
    chat_template_digest: str,
) -> None:
    if base_model_digest != expected_model_digest(revision):
        raise ModelContractError("base_model_digest_does_not_bind_registered_revision")
    if tokenizer_digest != tokenizer_content_digest(tokenizer):
        raise ModelContractError("tokenizer_digest_mismatch")
    if chat_template_digest != "not-applicable":
        raise ModelContractError("encoder_chat_template_must_be_not_applicable")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _digest_json(value: Any) -> str:
    content = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(content).hexdigest()
