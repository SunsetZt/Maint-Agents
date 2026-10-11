"""Trusted one-shot verifier for signed Staging dynamic acceptance manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.config import get_settings
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.secrets import SecretName, build_secret_provider
from industrial_ops_agent.security_audit import build_persistent_security_auditor
from industrial_ops_agent.staging_acceptance.attestation import (
    StagingAcceptanceAttestationService,
)
from industrial_ops_agent.staging_acceptance.cli import MAX_EVIDENCE_BYTES, _load_model
from industrial_ops_agent.staging_acceptance.dynamic_contracts import (
    StagingDynamicAcceptanceManifest,
)
from industrial_ops_agent.staging_acceptance.dynamic_manifest import (
    validate_dynamic_acceptance_manifest_identity,
)
from industrial_ops_agent.supply_chain.verifier import (
    CosignBlobVerifier,
    SupplyChainVerificationFailed,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-staging-attestation-verify")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--signature-bundle", required=True, type=Path)
    parser.add_argument(
        "--certificate-identity",
        default=os.getenv("IOAP_STAGING_ATTESTATION_CERTIFICATE_IDENTITY", ""),
    )
    parser.add_argument(
        "--certificate-oidc-issuer",
        default=os.getenv("IOAP_STAGING_ATTESTATION_CERTIFICATE_OIDC_ISSUER", ""),
    )
    parser.add_argument("--cosign-binary", default=os.getenv("IOAP_COSIGN_BINARY", "cosign"))
    parser.add_argument("--tenant-id", default=os.getenv("IOAP_STAGING_ATTESTATION_TENANT_ID", ""))
    parser.add_argument(
        "--subject-id",
        default=os.getenv(
            "IOAP_STAGING_ATTESTATION_SUBJECT_ID",
            "staging-attestation-verifier",
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest, _ = _load_model(
            args.manifest,
            StagingDynamicAcceptanceManifest,
            max_bytes=MAX_EVIDENCE_BYTES,
        )
        validate_dynamic_acceptance_manifest_identity(manifest)
        proof = CosignBlobVerifier(
            certificate_identity=args.certificate_identity,
            certificate_oidc_issuer=args.certificate_oidc_issuer,
            cosign_binary=args.cosign_binary,
        ).verify(args.manifest, args.signature_bundle)
    except (
        OSError,
        UnicodeError,
        ValueError,
        ValidationError,
        json.JSONDecodeError,
        SupplyChainVerificationFailed,
    ):
        _rejected("staging_attestation_verification_failed")
        return 2

    settings = get_settings()
    secrets = build_secret_provider(settings)
    database = Database(secrets.get(SecretName.DATABASE_URL).reveal())
    tenant_id = args.tenant_id or settings.deployment_controller_tenant_id
    try:
        record = StagingAcceptanceAttestationService(
            database,
            Authorizer(
                build_persistent_security_auditor(
                    database,
                    hash_key=secrets.get(SecretName.OIDC_CLIENT_SECRET).reveal().encode(),
                )
            ),
        ).record_verified(
            _identity(tenant_id, args.subject_id),
            manifest,
            proof,
            request_id=f"staging-attestation-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        )
    except (OSError, ValueError):
        _rejected("staging_attestation_record_failed")
        return 2
    finally:
        database.dispose()
    print(
        json.dumps(
            {
                "evidence_id": record.evidence_id,
                "manifest_id": record.manifest_id,
                "status": record.verification_status,
                "verification_hash": record.verification_hash,
            },
            sort_keys=True,
        )
    )
    return 0


def _identity(tenant_id: str, subject_id: str) -> IdentityContext:
    now = datetime.now(UTC)
    return IdentityContext(
        subject_id=subject_id,
        oidc_subject=f"workload:{subject_id}",
        tenant_id=tenant_id,
        roles=frozenset({Role.MODEL_ENGINEER}),
        asset_ids=frozenset(),
        site_ids=frozenset(),
        issued_at=now,
        expires_at=now + timedelta(minutes=10),
    )


def _rejected(reason: str) -> None:
    print(json.dumps({"status": "REJECTED", "reason": reason}, sort_keys=True), file=sys.stderr)


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()

