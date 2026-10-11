"""CLI for Diffusers generation, expert review and bundle verification."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from industrial_ops_agent.media.content_credentials import (
    C2paContentCredentialInspector,
    C2paSdkContentCredentialSigner,
    C2paSigningMaterial,
)
from industrial_ops_agent.training.synthetic_media import (
    DiffusersInpaintingBackend,
    SyntheticGenerationRequest,
    SyntheticMediaError,
    credential_synthetic_media_bundle,
    generate_synthetic_media_bundle,
    review_synthetic_media_bundle,
    verify_synthetic_media_bundle,
)


def run() -> None:
    parser = argparse.ArgumentParser(
        prog="industrial-ops-synthetic-media",
        description="Generate and govern training-only synthetic industrial defect images.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    _add_bundle_inputs(generate, include_manifest=False)
    generate.add_argument("--output-image", type=Path, required=True)
    generate.add_argument("--output-manifest", type=Path, required=True)
    generate.add_argument("--asset-model", required=True)
    generate.add_argument("--defect-label", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--negative-prompt", default="")
    generate.add_argument("--model-id", required=True)
    generate.add_argument("--model-revision", required=True)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--steps", type=int, default=30)
    generate.add_argument("--guidance-scale", type=float, default=7.5)
    generate.add_argument("--generated-by", required=True)
    generate.add_argument("--device", choices=("cuda", "cpu"), default="cuda")

    review = commands.add_parser("review")
    _add_bundle_inputs(review, include_manifest=True)
    review.add_argument("--output-manifest", type=Path, required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--decision", choices=("APPROVED", "REJECTED"), required=True)
    review.add_argument("--notes", required=True)

    verify = commands.add_parser("verify")
    _add_bundle_inputs(verify, include_manifest=True)

    credential = commands.add_parser("credential")
    _add_bundle_inputs(credential, include_manifest=True)
    credential.add_argument("--output-image", type=Path, required=True)
    credential.add_argument("--output-manifest", type=Path, required=True)
    credential.add_argument("--certificate", type=Path, required=True)
    credential.add_argument("--private-key", type=Path, required=True)
    credential.add_argument("--algorithm", choices=("ES256", "PS256"), default="ES256")
    credential.add_argument("--timestamp-authority-url")
    credential.add_argument("--trust-anchors", type=Path)
    credential.add_argument(
        "--signing-environment",
        choices=("development", "test"),
        default="development",
        help="Local PEM signing is prohibited for production; use a KMS/HSM callback there.",
    )

    args = parser.parse_args()
    try:
        if args.command == "generate":
            _generate(args)
        elif args.command == "review":
            _review(args)
        elif args.command == "credential":
            _credential(args)
        else:
            _verify(args)
    except (OSError, SyntheticMediaError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


def _add_bundle_inputs(parser: argparse.ArgumentParser, *, include_manifest: bool) -> None:
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--mask-image", type=Path, required=True)
    if include_manifest:
        parser.add_argument("--image", type=Path, required=True)
        parser.add_argument("--manifest", type=Path, required=True)


def _generate(args: argparse.Namespace) -> None:
    _require_new_paths(args.output_image, args.output_manifest)
    request = SyntheticGenerationRequest(
        source_image=args.source_image.read_bytes(),
        mask_image=args.mask_image.read_bytes(),
        asset_model=args.asset_model,
        defect_label=args.defect_label,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generated_by_subject_id=args.generated_by,
        occurred_at=datetime.now(UTC),
    )
    bundle = generate_synthetic_media_bundle(
        request,
        DiffusersInpaintingBackend(
            model_id=args.model_id,
            model_revision=args.model_revision,
            device=args.device,
        ),
    )
    _write_new(args.output_image, bundle.image_png)
    _write_json_new(args.output_manifest, bundle.manifest)
    _print_result(bundle.manifest, image_path=args.output_image, manifest_path=args.output_manifest)


def _review(args: argparse.Namespace) -> None:
    if args.output_manifest.resolve() == args.manifest.resolve():
        raise SyntheticMediaError("review_manifest_must_preserve_the_generation_manifest")
    _require_new_paths(args.output_manifest)
    manifest = _read_manifest(args.manifest)
    reviewed = review_synthetic_media_bundle(
        manifest,
        image_png=args.image.read_bytes(),
        source_image=args.source_image.read_bytes(),
        mask_image=args.mask_image.read_bytes(),
        reviewer_subject_id=args.reviewer,
        decision=args.decision,
        notes=args.notes,
        occurred_at=datetime.now(UTC),
    )
    _write_json_new(args.output_manifest, reviewed)
    _print_result(reviewed, image_path=args.image, manifest_path=args.output_manifest)


def _verify(args: argparse.Namespace) -> None:
    manifest = _read_manifest(args.manifest)
    verify_synthetic_media_bundle(
        manifest,
        image_png=args.image.read_bytes(),
        source_image=args.source_image.read_bytes(),
        mask_image=args.mask_image.read_bytes(),
    )
    _print_result(manifest, image_path=args.image, manifest_path=args.manifest)


def _credential(args: argparse.Namespace) -> None:
    _require_new_paths(args.output_image, args.output_manifest)
    private_key = args.private_key.read_bytes()
    certificate = args.certificate.read_text(encoding="utf-8")
    trust_anchors = (
        args.trust_anchors.read_text(encoding="utf-8")
        if args.trust_anchors is not None
        else None
    )
    signer = C2paSdkContentCredentialSigner(
        material=C2paSigningMaterial(
            algorithm=args.algorithm,
            certificate_pem=certificate,
            sign_callback=_local_pem_sign_callback(private_key, args.algorithm),
            timestamp_authority_url=args.timestamp_authority_url,
        ),
        inspector=C2paContentCredentialInspector(trust_anchors_pem=trust_anchors),
    )
    credentialed = credential_synthetic_media_bundle(
        _read_manifest(args.manifest),
        image_png=args.image.read_bytes(),
        source_image=args.source_image.read_bytes(),
        mask_image=args.mask_image.read_bytes(),
        signer=signer,
    )
    _write_new(args.output_image, credentialed.image_png)
    _write_json_new(args.output_manifest, credentialed.manifest)
    _print_result(
        credentialed.manifest,
        image_path=args.output_image,
        manifest_path=args.output_manifest,
    )


def _local_pem_sign_callback(
    private_key: bytes,
    algorithm: str,
) -> Callable[[bytes], bytes]:
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa

        key = serialization.load_pem_private_key(private_key, password=None)
    except (ImportError, TypeError, ValueError) as exc:
        raise SyntheticMediaError("c2pa_local_private_key_invalid") from exc

    def sign(payload: bytes) -> bytes:
        if algorithm == "ES256" and isinstance(key, ec.EllipticCurvePrivateKey):
            return key.sign(payload, ec.ECDSA(hashes.SHA256()))
        if algorithm == "PS256" and isinstance(key, rsa.RSAPrivateKey):
            return key.sign(
                payload,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=hashes.SHA256().digest_size,
                ),
                hashes.SHA256(),
            )
        raise SyntheticMediaError("c2pa_private_key_algorithm_mismatch")

    return sign


def _read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SyntheticMediaError("synthetic_media_manifest_must_be_an_object")
    return value


def _require_new_paths(*paths: Path) -> None:
    if any(path.exists() for path in paths):
        raise SyntheticMediaError("synthetic_media_output_already_exists")


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(content)


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode()
    _write_new(path, encoded + b"\n")


def _print_result(manifest: dict[str, Any], *, image_path: Path, manifest_path: Path) -> None:
    print(
        json.dumps(
            {
                "status": "ok",
                "review_status": manifest["review"]["status"],
                "training_candidate_eligible": manifest["training_candidate_eligible"],
                "evidence_eligible": manifest["evidence_eligible"],
                "golden_dataset_eligible": manifest["golden_dataset_eligible"],
                "image": str(image_path),
                "manifest": str(manifest_path),
                "manifest_sha256": manifest["manifest_sha256"],
                "content_credential_status": (
                    manifest.get("content_credentials", {})
                    .get("report", {})
                    .get("status")
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
