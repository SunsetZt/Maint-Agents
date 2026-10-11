"""Trusted one-shot verifier that records evidence only after Cosign succeeds."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.config import get_settings
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.secrets import SecretName, build_secret_provider
from industrial_ops_agent.security_audit import build_persistent_security_auditor
from industrial_ops_agent.supply_chain.service import SupplyChainEvidenceService
from industrial_ops_agent.supply_chain.verifier import (
    CosignEvidenceVerifier,
    SupplyChainVerificationFailed,
    VerificationFiles,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-supply-chain-verify")
    parser.add_argument("--statement", type=Path, required=True)
    parser.add_argument("--signature-bundle", type=Path, required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument("--vulnerability-report", type=Path, required=True)
    parser.add_argument("--license-report", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument(
        "--certificate-identity-regexp",
        default=os.getenv("IOAP_SUPPLY_CHAIN_CERTIFICATE_IDENTITY_REGEXP", ""),
    )
    parser.add_argument(
        "--certificate-oidc-issuer",
        default=os.getenv(
            "IOAP_SUPPLY_CHAIN_CERTIFICATE_OIDC_ISSUER",
            "https://token.actions.githubusercontent.com",
        ),
    )
    parser.add_argument("--cosign-binary", default=os.getenv("IOAP_COSIGN_BINARY", "cosign"))
    parser.add_argument("--tenant-id", default=os.getenv("IOAP_SUPPLY_CHAIN_TENANT_ID", ""))
    parser.add_argument(
        "--subject-id",
        default=os.getenv("IOAP_SUPPLY_CHAIN_SUBJECT_ID", "supply-chain-verifier"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.certificate_identity_regexp:
        raise SystemExit("certificate-identity-regexp is required")
    settings = get_settings()
    secrets = build_secret_provider(settings)
    database = Database(secrets.get(SecretName.DATABASE_URL).reveal())
    audit_key = secrets.get(SecretName.OIDC_CLIENT_SECRET).reveal().encode()
    tenant_id = args.tenant_id or settings.deployment_controller_tenant_id
    verifier = CosignEvidenceVerifier(
        certificate_identity_regexp=args.certificate_identity_regexp,
        certificate_oidc_issuer=args.certificate_oidc_issuer,
        cosign_binary=args.cosign_binary,
    )
    try:
        statement, proof = verifier.verify(
            VerificationFiles(
                statement=args.statement,
                signature_bundle=args.signature_bundle,
                sbom=args.sbom,
                vulnerability_report=args.vulnerability_report,
                license_report=args.license_report,
                provenance=args.provenance,
            )
        )
        record = SupplyChainEvidenceService(
            database,
            Authorizer(build_persistent_security_auditor(database, hash_key=audit_key)),
        ).record_verified(
            _identity(tenant_id, args.subject_id),
            statement,
            proof,
            request_id=f"supply-chain-verify-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        )
    except SupplyChainVerificationFailed as exc:
        print(json.dumps({"status": "REJECTED", "reason": str(exc)}, sort_keys=True))
        return 2
    finally:
        database.dispose()
    print(
        json.dumps(
            {
                "evidence_id": record.evidence_id,
                "image": f"{record.image_repository}@{record.image_digest}",
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


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
