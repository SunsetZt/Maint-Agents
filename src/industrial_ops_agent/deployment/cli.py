"""In-cluster deployment reconciler and observation controller entrypoint."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.opa import OpaPolicyClient
from industrial_ops_agent.auth.policy import Authorizer
from industrial_ops_agent.config import Settings, get_settings
from industrial_ops_agent.deployment.controller import ControllerReport, DeploymentController
from industrial_ops_agent.deployment.kserve import KServeGatewayProvider, KubernetesHttpApi
from industrial_ops_agent.deployment.leader_election import (
    KubernetesLeaseElector,
    LeaseElectionConfig,
)
from industrial_ops_agent.deployment.prometheus import PrometheusObservationCollector
from industrial_ops_agent.deployment.service import ModelDeploymentService
from industrial_ops_agent.enterprise_assets import EnterpriseAutoShadowDispatcher
from industrial_ops_agent.persistence.database import Database
from industrial_ops_agent.secrets import SecretName, build_secret_provider, secret_from_file
from industrial_ops_agent.security_audit import build_persistent_security_auditor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ops-deployment-controller")
    parser.add_argument(
        "--once",
        action="store_true",
        help="run one reconciliation cycle and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    secrets = build_secret_provider(settings)
    database = Database(secrets.get(SecretName.DATABASE_URL).reveal())
    audit_key = secrets.get(SecretName.OIDC_CLIENT_SECRET).reveal().encode()
    authorizer = Authorizer(
        build_persistent_security_auditor(database, hash_key=audit_key),
        OpaPolicyClient(
            settings.opa_decision_url,
            timeout_seconds=settings.opa_timeout_seconds,
        ),
    )
    kubernetes = KubernetesHttpApi(
        settings.kubernetes_api_url,
        secret_from_file(settings.kubernetes_token_file).reveal(),
        ca_file=str(settings.kubernetes_ca_file),
        timeout_seconds=settings.kubernetes_timeout_seconds,
    )
    prometheus_token = (
        secret_from_file(settings.prometheus_token_file).reveal()
        if settings.prometheus_token_file is not None
        else None
    )
    controller = DeploymentController(
        service=ModelDeploymentService(database, authorizer),
        provider=KServeGatewayProvider(kubernetes),
        collector=PrometheusObservationCollector(
            settings.prometheus_url,
            bearer_token=prometheus_token,
            verify=(
                str(settings.prometheus_ca_file)
                if settings.prometheus_ca_file is not None
                else True
            ),
            timeout_seconds=settings.prometheus_timeout_seconds,
        ),
        minimum_collection_interval_seconds=settings.deployment_observation_interval_seconds,
    )
    auto_shadow_dispatcher = EnterpriseAutoShadowDispatcher(database, authorizer)
    elector = _build_leader_elector(settings, kubernetes)
    try:
        if elector is not None:
            elector.start()
        while True:
            identity = _identity(settings)
            auto_shadow_report = (
                auto_shadow_dispatcher.run_once(identity)
                if elector is None or elector.is_leader
                else None
            )
            report = _run_cycle_if_leader(controller, identity, elector)
            if report is not None:
                payload = asdict(report)
                if auto_shadow_report is not None:
                    payload["auto_shadow"] = asdict(auto_shadow_report)
                print(json.dumps(payload, sort_keys=True), flush=True)
            if args.once:
                if report is None:
                    return 3
                return 0 if report.dependency_failures == 0 else 2
            time.sleep(settings.deployment_controller_interval_seconds)
    except KeyboardInterrupt:
        return 0
    finally:
        if elector is not None:
            elector.stop()
        kubernetes.close()
        database.dispose()


def _build_leader_elector(
    settings: Settings,
    kubernetes: KubernetesHttpApi,
) -> KubernetesLeaseElector | None:
    if not settings.deployment_controller_leader_election_enabled:
        return None
    pod_uid = settings.deployment_controller_pod_uid
    if pod_uid is None:
        raise ValueError("Deployment controller Pod UID is required for leader election")
    return KubernetesLeaseElector(
        kubernetes,
        LeaseElectionConfig(
            namespace=settings.deployment_controller_lease_namespace,
            lease_name=settings.deployment_controller_lease_name,
            holder_identity=pod_uid,
            lease_duration_seconds=settings.deployment_controller_lease_duration_seconds,
            renew_deadline_seconds=settings.deployment_controller_renew_deadline_seconds,
            retry_period_seconds=settings.deployment_controller_retry_period_seconds,
        ),
    )


def _run_cycle_if_leader(
    controller: DeploymentController,
    identity: IdentityContext,
    elector: KubernetesLeaseElector | None,
) -> ControllerReport | None:
    if elector is not None and not elector.is_leader:
        return None
    return controller.run_once(identity)


def _identity(settings: Settings) -> IdentityContext:
    tenant_id = settings.deployment_controller_tenant_id
    subject_id = settings.deployment_controller_subject_id
    now = datetime.now(UTC)
    return IdentityContext(
        subject_id=subject_id,
        oidc_subject=f"workload:{subject_id}",
        tenant_id=tenant_id,
        roles=frozenset({Role.MODEL_DEPLOYMENT_CONTROLLER}),
        asset_ids=frozenset(),
        site_ids=frozenset(),
        issued_at=now,
        expires_at=now + timedelta(minutes=10),
    )


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
