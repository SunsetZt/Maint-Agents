"""Cosign-backed verification of a signed supply-chain statement and OCI image."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from industrial_ops_agent.supply_chain.service import (
    SupplyChainStatement,
    VerifiedSupplyChainProof,
    statement_from_document,
)


class SupplyChainVerificationFailed(Exception):
    pass


@dataclass(frozen=True, slots=True)
class VerifiedBlobProof:
    signed_blob_digest: str
    signature_bundle_digest: str
    certificate_identity: str
    certificate_oidc_issuer: str
    verifier_version: str


@dataclass(frozen=True, slots=True)
class VerificationFiles:
    statement: Path
    signature_bundle: Path
    sbom: Path
    vulnerability_report: Path
    license_report: Path
    provenance: Path


@dataclass(frozen=True, slots=True)
class CommandResult:
    stdout: bytes
    stderr: bytes


class CommandRunner(Protocol):
    def run(self, argv: list[str], *, timeout_seconds: int) -> CommandResult: ...


class SubprocessCommandRunner:
    def run(self, argv: list[str], *, timeout_seconds: int) -> CommandResult:
        try:
            completed = subprocess.run(  # noqa: S603 - argv is fixed and never uses a shell
                argv,
                check=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise SupplyChainVerificationFailed("cosign_verification_failed") from exc
        return CommandResult(stdout=completed.stdout, stderr=completed.stderr)


class CosignBlobVerifier:
    """Verify one signed blob against an exact workload certificate identity."""

    def __init__(
        self,
        *,
        certificate_identity: str,
        certificate_oidc_issuer: str,
        cosign_binary: str = "cosign",
        timeout_seconds: int = 120,
        runner: CommandRunner | None = None,
    ) -> None:
        if (
            not certificate_identity
            or certificate_identity.strip() != certificate_identity
            or len(certificate_identity) > 1024
            or any(character in certificate_identity for character in "\x00\r\n")
        ):
            raise ValueError("an exact certificate identity is required")
        if (
            not certificate_oidc_issuer.startswith("https://")
            or certificate_oidc_issuer.strip() != certificate_oidc_issuer
            or len(certificate_oidc_issuer) > 1024
            or any(character in certificate_oidc_issuer for character in "\x00\r\n")
        ):
            raise ValueError("an HTTPS certificate OIDC issuer is required")
        if not cosign_binary or any(character in cosign_binary for character in "\x00\r\n"):
            raise ValueError("cosign binary is invalid")
        if timeout_seconds <= 0 or timeout_seconds > 120:
            raise ValueError("Cosign timeout must be between 1 and 120 seconds")
        self._certificate_identity = certificate_identity
        self._certificate_oidc_issuer = certificate_oidc_issuer
        self._cosign_binary = cosign_binary
        self._timeout_seconds = timeout_seconds
        self._runner = runner or SubprocessCommandRunner()

    def verify(self, blob: Path, signature_bundle: Path) -> VerifiedBlobProof:
        signed_blob_digest = _file_digest(blob)
        signature_bundle_digest = _file_digest(signature_bundle)
        verification = self._runner.run(
            [
                self._cosign_binary,
                "verify-blob",
                "--certificate-identity",
                self._certificate_identity,
                "--certificate-oidc-issuer",
                self._certificate_oidc_issuer,
                "--bundle",
                str(signature_bundle),
                str(blob),
            ],
            timeout_seconds=self._timeout_seconds,
        )
        # Cosign v3 logs successful verify-blob to stderr via ui.Infof.
        # The runner still requires exit 0; warning text alone is not success.
        if not verification.stdout.strip() and b"Verified OK" not in (
            line.strip() for line in verification.stderr.splitlines()
        ):
            raise SupplyChainVerificationFailed("cosign_verification_output_missing")
        version = self._runner.run(
            [self._cosign_binary, "version", "--json"],
            timeout_seconds=min(self._timeout_seconds, 30),
        )
        return VerifiedBlobProof(
            signed_blob_digest=signed_blob_digest,
            signature_bundle_digest=signature_bundle_digest,
            certificate_identity=self._certificate_identity,
            certificate_oidc_issuer=self._certificate_oidc_issuer,
            verifier_version=_cosign_version(version.stdout),
        )


class CosignEvidenceVerifier:
    def __init__(
        self,
        *,
        certificate_identity_regexp: str,
        certificate_oidc_issuer: str,
        cosign_binary: str = "cosign",
        timeout_seconds: int = 120,
        runner: CommandRunner | None = None,
    ) -> None:
        if not certificate_identity_regexp or not certificate_oidc_issuer.startswith("https://"):
            raise ValueError("a certificate identity and HTTPS OIDC issuer are required")
        self._certificate_identity_regexp = certificate_identity_regexp
        self._certificate_oidc_issuer = certificate_oidc_issuer
        self._cosign_binary = cosign_binary
        self._timeout_seconds = timeout_seconds
        self._runner = runner or SubprocessCommandRunner()

    def verify(
        self, files: VerificationFiles
    ) -> tuple[SupplyChainStatement, VerifiedSupplyChainProof]:
        document = _json_object(files.statement)
        statement = statement_from_document(document)
        _require_digest(files.sbom, statement.sbom_digest, "sbom_digest_mismatch")
        _require_digest(
            files.vulnerability_report,
            statement.vulnerability_report_digest,
            "vulnerability_report_digest_mismatch",
        )
        _require_digest(
            files.license_report,
            statement.license_report_digest,
            "license_report_digest_mismatch",
        )
        _require_digest(
            files.provenance,
            statement.provenance_digest,
            "provenance_digest_mismatch",
        )
        _validate_sbom(files.sbom)
        _validate_vulnerability_report(files.vulnerability_report, statement)
        _validate_license_report(files.license_report, statement)

        common = [
            "--certificate-identity-regexp",
            self._certificate_identity_regexp,
            "--certificate-oidc-issuer",
            self._certificate_oidc_issuer,
        ]
        blob_result = self._runner.run(
            [
                self._cosign_binary,
                "verify-blob",
                *common,
                "--bundle",
                str(files.signature_bundle),
                str(files.statement),
            ],
            timeout_seconds=self._timeout_seconds,
        )
        image_result = self._runner.run(
            [
                self._cosign_binary,
                "verify",
                *common,
                "--output",
                "json",
                f"{statement.image_repository}@{statement.image_digest}",
            ],
            timeout_seconds=self._timeout_seconds,
        )
        version_result = self._runner.run(
            [self._cosign_binary, "version", "--json"],
            timeout_seconds=min(self._timeout_seconds, 30),
        )
        if not blob_result.stdout.strip() or not image_result.stdout.strip():
            raise SupplyChainVerificationFailed("cosign_verification_output_missing")
        verifier_version = _cosign_version(version_result.stdout)
        certificate_identity, certificate_issuer = _image_signer(image_result.stdout)
        return statement, VerifiedSupplyChainProof(
            image_signature_digest=_bytes_digest(image_result.stdout),
            model_signature_digest=_file_digest(files.signature_bundle),
            certificate_identity=certificate_identity,
            certificate_oidc_issuer=certificate_issuer,
            verifier_version=verifier_version,
        )


def _validate_sbom(path: Path) -> None:
    document = _json_object(path)
    if not str(document.get("spdxVersion", "")).startswith("SPDX-2."):
        raise SupplyChainVerificationFailed("sbom_is_not_spdx_json")


def _validate_vulnerability_report(
    path: Path, statement: SupplyChainStatement
) -> None:
    report = _json_object(path)
    matches = report.get("matches")
    if not isinstance(matches, list):
        raise SupplyChainVerificationFailed("vulnerability_report_is_not_grype_json")
    severity_rank = {
        "UNKNOWN": 0,
        "NEGLIGIBLE": 1,
        "LOW": 2,
        "MEDIUM": 3,
        "HIGH": 4,
        "CRITICAL": 5,
    }
    maximum = "UNKNOWN"
    for match in matches:
        if not isinstance(match, dict) or not isinstance(match.get("vulnerability"), dict):
            raise SupplyChainVerificationFailed("vulnerability_report_is_not_grype_json")
        severity = str(match["vulnerability"].get("severity", "UNKNOWN")).upper()
        if severity not in severity_rank:
            severity = "UNKNOWN"
        if severity_rank[severity] > severity_rank[maximum]:
            maximum = severity
    expected_status = "PASSED" if severity_rank[maximum] <= severity_rank["MEDIUM"] else "FAILED"
    if (
        maximum != statement.maximum_vulnerability_severity
        or expected_status != statement.vulnerability_scan_status
    ):
        raise SupplyChainVerificationFailed("vulnerability_conclusion_mismatch")


def _validate_license_report(path: Path, statement: SupplyChainStatement) -> None:
    report = _json_object(path)
    denied = report.get("denied_licenses")
    if not isinstance(denied, list) or not all(isinstance(item, str) for item in denied):
        raise SupplyChainVerificationFailed("license_report_is_invalid")
    expected_status = "APPROVED" if not denied else "REJECTED"
    if report.get("status") != expected_status or expected_status != statement.license_status:
        raise SupplyChainVerificationFailed("license_conclusion_mismatch")


def _json_object(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SupplyChainVerificationFailed("verification_file_is_invalid_json") from exc
    if not isinstance(document, dict):
        raise SupplyChainVerificationFailed("verification_file_must_be_a_json_object")
    return document


def _require_digest(path: Path, expected: str, reason: str) -> None:
    if _file_digest(path) != expected:
        raise SupplyChainVerificationFailed(reason)


def _file_digest(path: Path) -> str:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise SupplyChainVerificationFailed("verification_file_unreadable") from exc
    return _bytes_digest(content)


def _bytes_digest(content: bytes) -> str:
    return "sha256:" + sha256(content).hexdigest()


def _cosign_version(content: bytes) -> str:
    try:
        document = json.loads(content)
        version = document.get("gitVersion") or document.get("git_version")
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SupplyChainVerificationFailed("cosign_version_unreadable") from exc
    if not isinstance(version, str):
        raise SupplyChainVerificationFailed("cosign_version_unreadable")
    return version


def _image_signer(content: bytes) -> tuple[str, str]:
    try:
        document = json.loads(content)
        signature = document[0]
        optional = signature["optional"]
        subject = optional["Subject"]
        issuer = optional["Issuer"]
    except (UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise SupplyChainVerificationFailed("cosign_signer_identity_unreadable") from exc
    if not isinstance(subject, str) or not subject or not isinstance(issuer, str):
        raise SupplyChainVerificationFailed("cosign_signer_identity_unreadable")
    if not issuer.startswith("https://"):
        raise SupplyChainVerificationFailed("cosign_signer_identity_unreadable")
    return subject, issuer
