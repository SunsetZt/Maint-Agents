"""Resolve a fixed Hugging Face revision and print registration-ready digests."""

from __future__ import annotations

import argparse
import json

from industrial_ops_agent.training.model_contract import (
    expected_model_digest,
    tokenizer_content_digest,
    tokenizer_contract_digests,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-model-inspect")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--contract-profile",
        choices=("generative", "encoder"),
        default="generative",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.revision in {"main", "master", "latest"}:
        raise SystemExit("revision must be an immutable Hugging Face commit SHA")
    try:
        from huggingface_hub import model_info
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("install the training optional dependencies first") from exc
    info = model_info(args.model_id, revision=args.revision)
    resolved_revision = str(info.sha)
    if resolved_revision != args.revision:
        raise SystemExit("revision did not resolve to the exact requested commit")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=resolved_revision,
        trust_remote_code=False,
    )
    if args.contract_profile == "encoder":
        tokenizer_digest = tokenizer_content_digest(tokenizer)
        chat_template_digest = "not-applicable"
    else:
        tokenizer_digest, chat_template_digest = tokenizer_contract_digests(tokenizer)
    print(
        json.dumps(
            {
                "base_model_id": args.model_id,
                "base_model_revision": resolved_revision,
                "base_model_digest": expected_model_digest(resolved_revision),
                "tokenizer_digest": tokenizer_digest,
                "chat_template_digest": chat_template_digest,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
