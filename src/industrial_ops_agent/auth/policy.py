"""Single server-side policy boundary for M1 RBAC and device scope."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from industrial_ops_agent.auth.emergency import current_emergency_grant_id
from industrial_ops_agent.auth.errors import AuthorizationDenied
from industrial_ops_agent.auth.identity import IdentityContext, Role
from industrial_ops_agent.auth.opa import (
    OpaDecisionInput,
    OpaUnavailable,
    PolicyDecisionClient,
)
from industrial_ops_agent.security_audit import SecurityAuditor


class Action(StrEnum):
    VIEW_WORKSPACE = "workspace.view"
    READ_ASSET = "asset.read"
    CREATE_INCIDENT_DRAFT = "incident_draft.create"
    UPDATE_INCIDENT_DRAFT = "incident_draft.update"
    UPLOAD_MEDIA = "media.upload"
    RUN_RECOGNITION = "recognition.run"
    CONFIRM_RECOGNITION = "recognition.confirm"
    SUBMIT_INCIDENT = "incident.submit"
    TRIAGE_INCIDENT = "incident.triage"
    READ_INCIDENT_QUEUE = "incident_queue.read"
    CONTROL_INCIDENT = "incident.control"
    READ_CUSTOMER_CASE = "customer_case.read"
    UPDATE_CUSTOMER_CASE = "customer_case.update"
    CONFIRM_SERVICE_RESULT = "service_result.confirm"
    READ_CUSTOMER_UPDATE = "customer_update.read"
    START_DIAGNOSIS = "diagnosis.start"
    READ_DIAGNOSIS = "diagnosis.read"
    CREATE_DIAGNOSIS_FEEDBACK = "diagnosis_feedback.create"
    CONTROL_AGENT = "agent.control"
    TAKEOVER_DIAGNOSIS = "diagnosis.takeover"
    REVISE_DIAGNOSIS = "diagnosis.revise"
    SYNTHESIZE_DIAGNOSIS_SPEECH = "diagnosis_speech.synthesize"
    START_REALTIME_MEDIA = "realtime_media.start"
    CONFIRM_REALTIME_TRANSCRIPT = "realtime_transcript.confirm"
    READ_EXPERT_COLLABORATION = "expert_collaboration.read"
    CREATE_EXPERT_COLLABORATION = "expert_collaboration.create"
    ACCEPT_EXPERT_COLLABORATION = "expert_collaboration.accept"
    END_EXPERT_COLLABORATION = "expert_collaboration.end"
    SUBMIT_EXPERT_RECOMMENDATION = "expert_recommendation.submit"
    REVIEW_EXPERT_RECOMMENDATION = "expert_recommendation.review"
    READ_CITATION = "citation.read"
    EVALUATE_KNOWLEDGE_INDEX = "knowledge_index.evaluate"
    PUBLISH_KNOWLEDGE = "knowledge.publish"
    READ_KNOWLEDGE_GRAPH = "knowledge_graph.read"
    MANAGE_KNOWLEDGE_GRAPH = "knowledge_graph.manage"
    EVALUATE_KNOWLEDGE_GRAPH = "knowledge_graph.evaluate"
    ACTIVATE_KNOWLEDGE_GRAPH = "knowledge_graph.activate"
    READ_KNOWLEDGE_SEARCH = "knowledge_search.read"
    MANAGE_KNOWLEDGE_SEARCH = "knowledge_search.manage"
    EVALUATE_KNOWLEDGE_SEARCH = "knowledge_search.evaluate"
    ACTIVATE_KNOWLEDGE_SEARCH = "knowledge_search.activate"
    READ_EXTERNAL_REFERENCE = "external_reference.read"
    RUN_EXTERNAL_SEARCH = "external_search.run"
    MANAGE_EXTERNAL_SEARCH_POLICY = "external_search_policy.manage"
    READ_KNOWLEDGE_DELETION = "knowledge_deletion.read"
    REQUEST_KNOWLEDGE_DELETION = "knowledge_deletion.request"
    PROPOSE_ACTION = "action.propose"
    READ_APPROVAL = "approval.read"
    APPROVE_ACTION = "approval.decide"
    EXECUTE_ACTION = "action.execute"
    PROPOSE_REFUND_REQUEST = "refund_request.propose"
    READ_REFUND_REQUEST = "refund_request.read"
    READ_EXECUTION_RECONCILIATION = "execution_reconciliation.read"
    RECONCILE_EXECUTION = "execution_reconciliation.reconcile"
    READ_EQUIPMENT_CONTROL_HANDOFF = "equipment_control_handoff.read"
    READ_PURCHASE_REQUEST = "purchase_request.read"
    PROPOSE_SERVICE_QUOTATION = "service_quotation.propose"
    READ_SERVICE_QUOTATION = "service_quotation.read"
    READ_REPAIR_WORK_ORDER_AUTHORIZATION = "repair_work_order_authorization.read"
    PROPOSE_REPAIR_WORK_ORDER = "repair_work_order.propose"
    READ_CUSTOMER_SERVICE_QUOTATION = "service_quotation_customer.read"
    DECIDE_CUSTOMER_SERVICE_QUOTATION = "service_quotation_customer.decide"
    READ_WORK_ORDER = "work_order.read"
    ASSIGN_WORK_ORDER = "work_order.assign"
    EXECUTE_WORK_ORDER = "work_order.execute"
    CONTROL_WORK_ORDER = "work_order.control"
    ESCALATE_WORK_ORDER = "work_order.escalate"
    VERIFY_WORK_ORDER = "work_order.verify"
    CLOSE_WORK_ORDER = "work_order.close"
    READ_SECURITY_AUDIT = "security_audit.read"
    READ_EMERGENCY_ACCESS = "emergency_access.read"
    REQUEST_EMERGENCY_ACCESS = "emergency_access.request"
    DECIDE_EMERGENCY_ACCESS = "emergency_access.decide"
    REVOKE_EMERGENCY_ACCESS = "emergency_access.revoke"
    READ_TENANT_ADMINISTRATION = "tenant_administration.read"
    MANAGE_TENANT_ADMINISTRATION = "tenant_administration.manage"
    READ_DATA_FEEDBACK = "data_feedback.read"
    DECIDE_DATA_ELIGIBILITY = "data_eligibility.decide"
    RUN_DATA_DLP = "data_dlp.run"
    CREATE_ANNOTATION_TASK = "annotation_task.create"
    SYNC_ANNOTATION_TASK = "annotation_task.sync"
    START_CURATION_RUN = "curation_run.start"
    READ_CURATION_RUN = "curation_run.read"
    READ_DATASET_SNAPSHOT = "dataset_snapshot.read"
    DOWNLOAD_DATASET_MANIFEST = "dataset_manifest.download"
    READ_DATA_LINEAGE = "data_lineage.read"
    CREATE_TRAINING_EXPERIMENT = "training_experiment.create"
    READ_TRAINING_EXPERIMENT = "training_experiment.read"
    START_TRAINING_EXPERIMENT = "training_experiment.start"
    COMPLETE_TRAINING_EXPERIMENT = "training_experiment.complete"
    READ_EVALUATION_SUITE = "evaluation_suite.read"
    MANAGE_EVALUATION_SUITE = "evaluation_suite.manage"
    READ_EVALUATION_POLICY = "evaluation_policy.read"
    MANAGE_EVALUATION_POLICY = "evaluation_policy.manage"
    RUN_MODEL_EVALUATION = "model_evaluation.run"
    READ_MODEL_EVALUATION = "model_evaluation.read"
    VERIFY_SUPPLY_CHAIN_EVIDENCE = "supply_chain_evidence.verify"
    READ_SUPPLY_CHAIN_EVIDENCE = "supply_chain_evidence.read"
    VERIFY_STAGING_ACCEPTANCE_ATTESTATION = "staging_acceptance_attestation.verify"
    CREATE_MODEL_RELEASE = "model_release.create"
    READ_MODEL_RELEASE = "model_release.read"
    VALIDATE_MODEL_RELEASE = "model_release.validate"
    SUBMIT_MODEL_RELEASE_APPROVAL = "model_release.submit_approval"
    DECIDE_MODEL_RELEASE_APPROVAL = "model_release.decide_approval"
    READ_MODEL_DEPLOYMENT = "model_deployment.read"
    REQUEST_MODEL_DEPLOYMENT = "model_deployment.request"
    PROMOTE_MODEL_RELEASE = "model_release.promote"
    ROLLBACK_MODEL_RELEASE = "model_release.rollback"
    RECONCILE_MODEL_DEPLOYMENT = "model_deployment.reconcile"
    RECORD_RELEASE_OBSERVATION = "model_release.record_observation"
    READ_MODEL_GATEWAY = "model_gateway.read"
    READ_ENTERPRISE_PROJECT_ADOPTION = "enterprise_project_adoption.read"
    IMPORT_ENTERPRISE_MODEL_ASSET = "enterprise_model_asset.import"
    MANAGE_MODEL_GATEWAY_QUOTA = "model_gateway_quota.manage"
    READ_PROMPT_BUNDLE = "prompt_bundle.read"
    MANAGE_PROMPT_BUNDLE = "prompt_bundle.manage"
    REVIEW_PROMPT_BUNDLE = "prompt_bundle.review"
    READ_MEMORY = "memory.read"
    WRITE_MEMORY = "memory.write"
    REVOKE_MEMORY = "memory.revoke"
    READ_AGENT_COLLABORATION = "agent_collaboration.read"
    CREATE_AGENT_COLLABORATION = "agent_collaboration.create"
    DISPATCH_AGENT_COLLABORATION = "agent_collaboration.dispatch"
    REVIEW_AGENT_COLLABORATION = "agent_collaboration.review"
    READ_OPERATIONS = "operations.read"
    MANAGE_SERVICE_PERFORMANCE_BASELINE = "service_performance_baseline.manage"
    READ_TOOL_GOVERNANCE = "tool_governance.read"
    MANAGE_COST_POLICY = "cost_policy.manage"
    READ_RECOVERY = "recovery.read"
    MANAGE_RECOVERY = "recovery.manage"
    READ_ASSURANCE = "assurance.read"
    MANAGE_SECURITY_EXERCISE = "security_exercise.manage"
    MANAGE_PRODUCTION_ACCEPTANCE = "production_acceptance.manage"
    SIGN_PRODUCTION_ACCEPTANCE = "production_acceptance.sign"
    READ_PREDICTIVE_MAINTENANCE = "predictive_maintenance.read"
    INGEST_TELEMETRY = "telemetry.ingest"
    BUILD_TELEMETRY_WINDOW = "telemetry_window.build"
    BUILD_TELEMETRY_DATASET = "telemetry_dataset.build"
    BUILD_RUL_DATASET = "rul_dataset.build"
    RUN_ANOMALY_DETECTION = "anomaly_detection.run"
    DECIDE_ALERT_CANDIDATE = "alert_candidate.decide"
    REFER_ALERT_CANDIDATE = "alert_candidate.refer"
    RECORD_PREDICTIVE_OUTCOME = "predictive_outcome.record"
    GENERATE_RUL_FORECAST = "rul_forecast.generate"
    REVIEW_RUL_FORECAST = "rul_forecast.review"
    READ_MAINTENANCE_COUNCIL = "maintenance_council.read"
    REGISTER_MAINTENANCE_REVIEW_ROLLOUT = "maintenance_review_rollout.register"
    ACTIVATE_MAINTENANCE_REVIEW_POLICY = "maintenance_review_policy.activate"
    REQUEST_MAINTENANCE_COUNCIL = "maintenance_council.request"
    REVIEW_MAINTENANCE_COUNCIL = "maintenance_council.review"
    READ_DEVICE_FAMILY = "device_family.read"
    PROPOSE_DEVICE_FAMILY = "device_family.propose"
    REVIEW_DEVICE_FAMILY = "device_family.review"


ROLE_ACTIONS: dict[Role, frozenset[Action]] = {
    Role.CUSTOMER_CONTACT: frozenset(
        {
            Action.VIEW_WORKSPACE,
            Action.READ_ASSET,
            Action.CREATE_INCIDENT_DRAFT,
            Action.UPLOAD_MEDIA,
            Action.RUN_RECOGNITION,
            Action.CONFIRM_RECOGNITION,
            Action.SUBMIT_INCIDENT,
            Action.READ_CUSTOMER_CASE,
            Action.UPDATE_CUSTOMER_CASE,
            Action.CONFIRM_SERVICE_RESULT,
            Action.READ_DIAGNOSIS,
            Action.SYNTHESIZE_DIAGNOSIS_SPEECH,
            Action.START_REALTIME_MEDIA,
            Action.CONFIRM_REALTIME_TRANSCRIPT,
            Action.READ_CITATION,
            Action.READ_MEMORY,
            Action.WRITE_MEMORY,
            Action.REVOKE_MEMORY,
            Action.READ_APPROVAL,
            Action.READ_WORK_ORDER,
            Action.READ_CUSTOMER_SERVICE_QUOTATION,
            Action.DECIDE_CUSTOMER_SERVICE_QUOTATION,
        }
    ),
    Role.FIELD_ENGINEER: frozenset(
        {
            Action.VIEW_WORKSPACE,
            Action.READ_ASSET,
            Action.CREATE_INCIDENT_DRAFT,
            Action.UPDATE_INCIDENT_DRAFT,
            Action.UPLOAD_MEDIA,
            Action.RUN_RECOGNITION,
            Action.CONFIRM_RECOGNITION,
            Action.SUBMIT_INCIDENT,
            Action.START_DIAGNOSIS,
            Action.READ_DIAGNOSIS,
            Action.CREATE_DIAGNOSIS_FEEDBACK,
            Action.CONTROL_AGENT,
            Action.SYNTHESIZE_DIAGNOSIS_SPEECH,
            Action.START_REALTIME_MEDIA,
            Action.CONFIRM_REALTIME_TRANSCRIPT,
            Action.READ_EXPERT_COLLABORATION,
            Action.CREATE_EXPERT_COLLABORATION,
            Action.END_EXPERT_COLLABORATION,
            Action.REVIEW_EXPERT_RECOMMENDATION,
            Action.READ_CITATION,
            Action.READ_KNOWLEDGE_SEARCH,
            Action.READ_EXTERNAL_REFERENCE,
            Action.RUN_EXTERNAL_SEARCH,
            Action.READ_MEMORY,
            Action.WRITE_MEMORY,
            Action.REVOKE_MEMORY,
            Action.PROPOSE_ACTION,
            Action.READ_APPROVAL,
            Action.READ_EQUIPMENT_CONTROL_HANDOFF,
            Action.READ_WORK_ORDER,
            Action.READ_PURCHASE_REQUEST,
            Action.EXECUTE_WORK_ORDER,
            Action.CONTROL_WORK_ORDER,
            Action.READ_CUSTOMER_UPDATE,
            Action.READ_PREDICTIVE_MAINTENANCE,
            Action.READ_MAINTENANCE_COUNCIL,
            Action.READ_DEVICE_FAMILY,
        }
    ),
    Role.AFTER_SALES_ENGINEER: frozenset(
        {
            Action.VIEW_WORKSPACE,
            Action.READ_ASSET,
            Action.CREATE_INCIDENT_DRAFT,
            Action.UPDATE_INCIDENT_DRAFT,
            Action.UPLOAD_MEDIA,
            Action.RUN_RECOGNITION,
            Action.CONFIRM_RECOGNITION,
            Action.SUBMIT_INCIDENT,
            Action.TRIAGE_INCIDENT,
            Action.READ_INCIDENT_QUEUE,
            Action.CONTROL_INCIDENT,
            Action.START_DIAGNOSIS,
            Action.READ_DIAGNOSIS,
            Action.CREATE_DIAGNOSIS_FEEDBACK,
            Action.CONTROL_AGENT,
            Action.SYNTHESIZE_DIAGNOSIS_SPEECH,
            Action.START_REALTIME_MEDIA,
            Action.CONFIRM_REALTIME_TRANSCRIPT,
            Action.READ_CITATION,
            Action.READ_KNOWLEDGE_GRAPH,
            Action.READ_KNOWLEDGE_SEARCH,
            Action.READ_EXTERNAL_REFERENCE,
            Action.RUN_EXTERNAL_SEARCH,
            Action.READ_MEMORY,
            Action.WRITE_MEMORY,
            Action.REVOKE_MEMORY,
            Action.PROPOSE_ACTION,
            Action.PROPOSE_REFUND_REQUEST,
            Action.READ_APPROVAL,
            Action.READ_EQUIPMENT_CONTROL_HANDOFF,
            Action.READ_REFUND_REQUEST,
            Action.READ_EXECUTION_RECONCILIATION,
            Action.RECONCILE_EXECUTION,
            Action.APPROVE_ACTION,
            Action.EXECUTE_ACTION,
            Action.READ_WORK_ORDER,
            Action.READ_PURCHASE_REQUEST,
            Action.PROPOSE_SERVICE_QUOTATION,
            Action.READ_SERVICE_QUOTATION,
            Action.READ_REPAIR_WORK_ORDER_AUTHORIZATION,
            Action.PROPOSE_REPAIR_WORK_ORDER,
            Action.ASSIGN_WORK_ORDER,
            Action.READ_CUSTOMER_UPDATE,
            Action.CONTROL_WORK_ORDER,
            Action.ESCALATE_WORK_ORDER,
            Action.CLOSE_WORK_ORDER,
            Action.READ_PREDICTIVE_MAINTENANCE,
            Action.DECIDE_ALERT_CANDIDATE,
            Action.REFER_ALERT_CANDIDATE,
            Action.RECORD_PREDICTIVE_OUTCOME,
            Action.GENERATE_RUL_FORECAST,
            Action.READ_AGENT_COLLABORATION,
            Action.CREATE_AGENT_COLLABORATION,
            Action.DISPATCH_AGENT_COLLABORATION,
            Action.READ_MAINTENANCE_COUNCIL,
            Action.REQUEST_MAINTENANCE_COUNCIL,
            Action.READ_DEVICE_FAMILY,
        }
    ),
    Role.DOMAIN_EXPERT: frozenset(
        {
            Action.VIEW_WORKSPACE,
            Action.READ_ASSET,
            Action.CREATE_INCIDENT_DRAFT,
            Action.UPDATE_INCIDENT_DRAFT,
            Action.CONFIRM_RECOGNITION,
            Action.TRIAGE_INCIDENT,
            Action.READ_INCIDENT_QUEUE,
            Action.CONTROL_INCIDENT,
            Action.START_DIAGNOSIS,
            Action.READ_DIAGNOSIS,
            Action.CREATE_DIAGNOSIS_FEEDBACK,
            Action.CONTROL_AGENT,
            Action.TAKEOVER_DIAGNOSIS,
            Action.REVISE_DIAGNOSIS,
            Action.SYNTHESIZE_DIAGNOSIS_SPEECH,
            Action.START_REALTIME_MEDIA,
            Action.CONFIRM_REALTIME_TRANSCRIPT,
            Action.READ_EXPERT_COLLABORATION,
            Action.ACCEPT_EXPERT_COLLABORATION,
            Action.END_EXPERT_COLLABORATION,
            Action.SUBMIT_EXPERT_RECOMMENDATION,
            Action.READ_CITATION,
            Action.READ_MEMORY,
            Action.WRITE_MEMORY,
            Action.REVOKE_MEMORY,
            Action.EVALUATE_KNOWLEDGE_INDEX,
            Action.PUBLISH_KNOWLEDGE,
            Action.READ_KNOWLEDGE_GRAPH,
            Action.MANAGE_KNOWLEDGE_GRAPH,
            Action.READ_KNOWLEDGE_SEARCH,
            Action.MANAGE_KNOWLEDGE_SEARCH,
            Action.READ_EXTERNAL_REFERENCE,
            Action.RUN_EXTERNAL_SEARCH,
            Action.READ_KNOWLEDGE_DELETION,
            Action.REQUEST_KNOWLEDGE_DELETION,
            Action.PROPOSE_ACTION,
            Action.PROPOSE_REFUND_REQUEST,
            Action.READ_APPROVAL,
            Action.READ_EQUIPMENT_CONTROL_HANDOFF,
            Action.READ_REFUND_REQUEST,
            Action.READ_EXECUTION_RECONCILIATION,
            Action.RECONCILE_EXECUTION,
            Action.APPROVE_ACTION,
            Action.EXECUTE_ACTION,
            Action.READ_WORK_ORDER,
            Action.READ_PURCHASE_REQUEST,
            Action.PROPOSE_SERVICE_QUOTATION,
            Action.READ_SERVICE_QUOTATION,
            Action.READ_REPAIR_WORK_ORDER_AUTHORIZATION,
            Action.CONTROL_WORK_ORDER,
            Action.READ_CUSTOMER_UPDATE,
            Action.ESCALATE_WORK_ORDER,
            Action.VERIFY_WORK_ORDER,
            Action.CLOSE_WORK_ORDER,
            Action.READ_PREDICTIVE_MAINTENANCE,
            Action.DECIDE_ALERT_CANDIDATE,
            Action.REFER_ALERT_CANDIDATE,
            Action.RECORD_PREDICTIVE_OUTCOME,
            Action.GENERATE_RUL_FORECAST,
            Action.REVIEW_RUL_FORECAST,
            Action.READ_AGENT_COLLABORATION,
            Action.CREATE_AGENT_COLLABORATION,
            Action.DISPATCH_AGENT_COLLABORATION,
            Action.REVIEW_AGENT_COLLABORATION,
            Action.READ_MAINTENANCE_COUNCIL,
            Action.REQUEST_MAINTENANCE_COUNCIL,
            Action.REVIEW_MAINTENANCE_COUNCIL,
            Action.READ_DEVICE_FAMILY,
            Action.PROPOSE_DEVICE_FAMILY,
        }
    ),
    Role.TENANT_ADMIN: frozenset(
        {
            Action.VIEW_WORKSPACE,
            Action.READ_TENANT_ADMINISTRATION,
            Action.MANAGE_TENANT_ADMINISTRATION,
            Action.REGISTER_MAINTENANCE_REVIEW_ROLLOUT,
            Action.ACTIVATE_MAINTENANCE_REVIEW_POLICY,
            Action.READ_ASSET,
            Action.READ_INCIDENT_QUEUE,
            Action.CONTROL_INCIDENT,
            Action.READ_DIAGNOSIS,
            Action.SYNTHESIZE_DIAGNOSIS_SPEECH,
            Action.START_REALTIME_MEDIA,
            Action.CONFIRM_REALTIME_TRANSCRIPT,
            Action.READ_CITATION,
            Action.READ_MEMORY,
            Action.WRITE_MEMORY,
            Action.REVOKE_MEMORY,
            Action.EVALUATE_KNOWLEDGE_INDEX,
            Action.PUBLISH_KNOWLEDGE,
            Action.READ_KNOWLEDGE_GRAPH,
            Action.ACTIVATE_KNOWLEDGE_GRAPH,
            Action.READ_KNOWLEDGE_SEARCH,
            Action.ACTIVATE_KNOWLEDGE_SEARCH,
            Action.READ_EXTERNAL_REFERENCE,
            Action.MANAGE_EXTERNAL_SEARCH_POLICY,
            Action.READ_KNOWLEDGE_DELETION,
            Action.REQUEST_KNOWLEDGE_DELETION,
            Action.READ_APPROVAL,
            Action.READ_REFUND_REQUEST,
            Action.READ_EXECUTION_RECONCILIATION,
            Action.APPROVE_ACTION,
            Action.READ_WORK_ORDER,
            Action.READ_PURCHASE_REQUEST,
            Action.ASSIGN_WORK_ORDER,
            Action.READ_CUSTOMER_UPDATE,
            Action.CONTROL_WORK_ORDER,
            Action.ESCALATE_WORK_ORDER,
            Action.CLOSE_WORK_ORDER,
            Action.READ_ASSURANCE,
            Action.READ_ENTERPRISE_PROJECT_ADOPTION,
            Action.SIGN_PRODUCTION_ACCEPTANCE,
            Action.READ_PREDICTIVE_MAINTENANCE,
            Action.READ_AGENT_COLLABORATION,
            Action.READ_OPERATIONS,
            Action.MANAGE_SERVICE_PERFORMANCE_BASELINE,
            Action.READ_MAINTENANCE_COUNCIL,
            Action.READ_DEVICE_FAMILY,
            Action.REVIEW_DEVICE_FAMILY,
        }
    ),
    Role.SECURITY_AUDITOR: frozenset(
        {
            Action.VIEW_WORKSPACE,
            Action.READ_ASSET,
            Action.READ_SECURITY_AUDIT,
            Action.READ_EMERGENCY_ACCESS,
            Action.DECIDE_EMERGENCY_ACCESS,
            Action.REVOKE_EMERGENCY_ACCESS,
            Action.READ_KNOWLEDGE_DELETION,
            Action.READ_KNOWLEDGE_GRAPH,
            Action.READ_KNOWLEDGE_SEARCH,
            Action.READ_EXTERNAL_REFERENCE,
            Action.READ_TRAINING_EXPERIMENT,
            Action.READ_MODEL_EVALUATION,
            Action.READ_EVALUATION_SUITE,
            Action.READ_EVALUATION_POLICY,
            Action.READ_SUPPLY_CHAIN_EVIDENCE,
            Action.READ_MODEL_RELEASE,
            Action.READ_MODEL_DEPLOYMENT,
            Action.READ_MODEL_GATEWAY,
            Action.READ_ENTERPRISE_PROJECT_ADOPTION,
            Action.READ_PROMPT_BUNDLE,
            Action.READ_TOOL_GOVERNANCE,
            Action.READ_RECOVERY,
            Action.READ_ASSURANCE,
            Action.MANAGE_SECURITY_EXERCISE,
            Action.SIGN_PRODUCTION_ACCEPTANCE,
        }
    ),
    Role.DATA_STEWARD: frozenset(
        {
            Action.VIEW_WORKSPACE,
            Action.READ_DATA_FEEDBACK,
            Action.READ_KNOWLEDGE_DELETION,
            Action.REQUEST_KNOWLEDGE_DELETION,
            Action.DECIDE_DATA_ELIGIBILITY,
            Action.RUN_DATA_DLP,
            Action.CREATE_ANNOTATION_TASK,
            Action.START_CURATION_RUN,
            Action.READ_CURATION_RUN,
            Action.READ_DATASET_SNAPSHOT,
            Action.DOWNLOAD_DATASET_MANIFEST,
            Action.READ_DATA_LINEAGE,
            Action.READ_ENTERPRISE_PROJECT_ADOPTION,
            Action.MANAGE_EVALUATION_SUITE,
            Action.READ_EVALUATION_SUITE,
            Action.READ_PREDICTIVE_MAINTENANCE,
            Action.BUILD_TELEMETRY_DATASET,
            Action.BUILD_RUL_DATASET,
        }
    ),
    Role.ANNOTATION_ADMIN: frozenset(
        {
            Action.VIEW_WORKSPACE,
            Action.READ_DATA_FEEDBACK,
            Action.CREATE_ANNOTATION_TASK,
            Action.SYNC_ANNOTATION_TASK,
        }
    ),
    Role.MODEL_ENGINEER: frozenset(
        {
            Action.VIEW_WORKSPACE,
            Action.READ_CURATION_RUN,
            Action.READ_DATASET_SNAPSHOT,
            Action.DOWNLOAD_DATASET_MANIFEST,
            Action.READ_DATA_LINEAGE,
            Action.CREATE_TRAINING_EXPERIMENT,
            Action.READ_TRAINING_EXPERIMENT,
            Action.START_TRAINING_EXPERIMENT,
            Action.COMPLETE_TRAINING_EXPERIMENT,
            Action.READ_MODEL_EVALUATION,
            Action.READ_EVALUATION_SUITE,
            Action.READ_EVALUATION_POLICY,
            Action.VERIFY_SUPPLY_CHAIN_EVIDENCE,
            Action.READ_SUPPLY_CHAIN_EVIDENCE,
            Action.VERIFY_STAGING_ACCEPTANCE_ATTESTATION,
            Action.IMPORT_ENTERPRISE_MODEL_ASSET,
            Action.CREATE_MODEL_RELEASE,
            Action.READ_MODEL_RELEASE,
            Action.READ_MODEL_DEPLOYMENT,
            Action.READ_MODEL_GATEWAY,
            Action.READ_ENTERPRISE_PROJECT_ADOPTION,
            Action.READ_PROMPT_BUNDLE,
            Action.MANAGE_PROMPT_BUNDLE,
            Action.VALIDATE_MODEL_RELEASE,
            Action.SUBMIT_MODEL_RELEASE_APPROVAL,
            Action.READ_PREDICTIVE_MAINTENANCE,
            Action.BUILD_TELEMETRY_WINDOW,
            Action.RUN_ANOMALY_DETECTION,
            Action.BUILD_TELEMETRY_DATASET,
            Action.BUILD_RUL_DATASET,
        }
    ),
    Role.MODEL_EVALUATOR: frozenset(
        {
            Action.VIEW_WORKSPACE,
            Action.READ_DATASET_SNAPSHOT,
            Action.READ_DATA_LINEAGE,
            Action.READ_TRAINING_EXPERIMENT,
            Action.MANAGE_EVALUATION_SUITE,
            Action.READ_EVALUATION_SUITE,
            Action.MANAGE_EVALUATION_POLICY,
            Action.READ_EVALUATION_POLICY,
            Action.RUN_MODEL_EVALUATION,
            Action.READ_MODEL_EVALUATION,
            Action.READ_KNOWLEDGE_GRAPH,
            Action.EVALUATE_KNOWLEDGE_GRAPH,
            Action.READ_KNOWLEDGE_SEARCH,
            Action.EVALUATE_KNOWLEDGE_SEARCH,
            Action.READ_SUPPLY_CHAIN_EVIDENCE,
            Action.READ_MODEL_RELEASE,
            Action.READ_ENTERPRISE_PROJECT_ADOPTION,
            Action.READ_PROMPT_BUNDLE,
            Action.REVIEW_PROMPT_BUNDLE,
        }
    ),
    Role.MODEL_RELEASE_APPROVER: frozenset(
        {
            Action.VIEW_WORKSPACE,
            Action.READ_TRAINING_EXPERIMENT,
            Action.READ_MODEL_EVALUATION,
            Action.READ_EVALUATION_SUITE,
            Action.READ_EVALUATION_POLICY,
            Action.READ_SUPPLY_CHAIN_EVIDENCE,
            Action.READ_MODEL_RELEASE,
            Action.READ_ENTERPRISE_PROJECT_ADOPTION,
            Action.DECIDE_MODEL_RELEASE_APPROVAL,
            Action.READ_MODEL_DEPLOYMENT,
            Action.READ_MODEL_GATEWAY,
            Action.READ_PROMPT_BUNDLE,
            Action.REVIEW_PROMPT_BUNDLE,
        }
    ),
    Role.MODEL_RELEASE_OPERATOR: frozenset(
        {
            Action.VIEW_WORKSPACE,
            Action.READ_TRAINING_EXPERIMENT,
            Action.READ_MODEL_EVALUATION,
            Action.READ_EVALUATION_SUITE,
            Action.READ_EVALUATION_POLICY,
            Action.READ_SUPPLY_CHAIN_EVIDENCE,
            Action.READ_MODEL_RELEASE,
            Action.READ_ENTERPRISE_PROJECT_ADOPTION,
            Action.READ_MODEL_DEPLOYMENT,
            Action.READ_MODEL_GATEWAY,
            Action.READ_PROMPT_BUNDLE,
            Action.REQUEST_MODEL_DEPLOYMENT,
            Action.PROMOTE_MODEL_RELEASE,
            Action.ROLLBACK_MODEL_RELEASE,
            Action.MANAGE_MODEL_GATEWAY_QUOTA,
            Action.READ_ASSURANCE,
        }
    ),
    Role.MODEL_DEPLOYMENT_CONTROLLER: frozenset(
        {
            Action.READ_MODEL_RELEASE,
            Action.READ_MODEL_DEPLOYMENT,
            Action.REQUEST_MODEL_DEPLOYMENT,
            Action.RECONCILE_MODEL_DEPLOYMENT,
            Action.RECORD_RELEASE_OBSERVATION,
        }
    ),
    Role.PLATFORM_OPERATOR: frozenset(
        {
            Action.VIEW_WORKSPACE,
            Action.READ_EMERGENCY_ACCESS,
            Action.REQUEST_EMERGENCY_ACCESS,
            Action.READ_OPERATIONS,
            Action.READ_ENTERPRISE_PROJECT_ADOPTION,
            Action.MANAGE_SERVICE_PERFORMANCE_BASELINE,
            Action.READ_TOOL_GOVERNANCE,
            Action.READ_EXECUTION_RECONCILIATION,
            Action.READ_PROMPT_BUNDLE,
            Action.MANAGE_COST_POLICY,
            Action.READ_KNOWLEDGE_DELETION,
            Action.REQUEST_KNOWLEDGE_DELETION,
            Action.READ_RECOVERY,
            Action.MANAGE_RECOVERY,
            Action.READ_ASSURANCE,
            Action.MANAGE_PRODUCTION_ACCEPTANCE,
            Action.SIGN_PRODUCTION_ACCEPTANCE,
            Action.VERIFY_STAGING_ACCEPTANCE_ATTESTATION,
            Action.READ_PREDICTIVE_MAINTENANCE,
            Action.INGEST_TELEMETRY,
            Action.BUILD_TELEMETRY_WINDOW,
            Action.RUN_ANOMALY_DETECTION,
        }
    ),
}


@dataclass(frozen=True, slots=True)
class ResourceContext:
    tenant_id: str
    resource_id: str | None = None
    asset_id: str | None = None
    site_id: str | None = None


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason_code: str
    obligations: tuple[str, ...] = ()
    policy_version: str = "local-rbac-v1"
    emergency_grant_id: str | None = None


@dataclass(frozen=True, slots=True)
class EmergencyGrantResolution:
    allowed: bool
    reason_code: str


class EmergencyGrantPolicy(Protocol):
    def resolve(
        self,
        grant_id: str,
        identity: IdentityContext,
        action: Action,
        resource: ResourceContext,
    ) -> EmergencyGrantResolution: ...

    def record_usage(
        self,
        grant_id: str,
        identity: IdentityContext,
        action: Action,
        resource: ResourceContext,
        *,
        request_id: str,
    ) -> None: ...


OPA_ENFORCED_ACTIONS = frozenset(
    {
        Action.CONTROL_AGENT,
        Action.PUBLISH_KNOWLEDGE,
        Action.MANAGE_KNOWLEDGE_GRAPH,
        Action.ACTIVATE_KNOWLEDGE_GRAPH,
        Action.MANAGE_KNOWLEDGE_SEARCH,
        Action.MANAGE_EXTERNAL_SEARCH_POLICY,
        Action.ACTIVATE_KNOWLEDGE_SEARCH,
        Action.REQUEST_KNOWLEDGE_DELETION,
        Action.PROPOSE_ACTION,
        Action.PROPOSE_REFUND_REQUEST,
        Action.RECONCILE_EXECUTION,
        Action.APPROVE_ACTION,
        Action.EXECUTE_ACTION,
        Action.ASSIGN_WORK_ORDER,
        Action.EXECUTE_WORK_ORDER,
        Action.CONFIRM_SERVICE_RESULT,
        Action.CONTROL_INCIDENT,
        Action.CONTROL_WORK_ORDER,
        Action.ESCALATE_WORK_ORDER,
        Action.VERIFY_WORK_ORDER,
        Action.CLOSE_WORK_ORDER,
        Action.MANAGE_TENANT_ADMINISTRATION,
        Action.DECIDE_MODEL_RELEASE_APPROVAL,
        Action.REQUEST_MODEL_DEPLOYMENT,
        Action.PROMOTE_MODEL_RELEASE,
        Action.ROLLBACK_MODEL_RELEASE,
        Action.RECONCILE_MODEL_DEPLOYMENT,
        Action.RECORD_RELEASE_OBSERVATION,
        Action.MANAGE_MODEL_GATEWAY_QUOTA,
        Action.IMPORT_ENTERPRISE_MODEL_ASSET,
        Action.WRITE_MEMORY,
        Action.REVOKE_MEMORY,
        Action.MANAGE_RECOVERY,
        Action.MANAGE_COST_POLICY,
        Action.MANAGE_SECURITY_EXERCISE,
        Action.MANAGE_PRODUCTION_ACCEPTANCE,
        Action.SIGN_PRODUCTION_ACCEPTANCE,
        Action.REQUEST_EMERGENCY_ACCESS,
        Action.DECIDE_EMERGENCY_ACCESS,
        Action.REVOKE_EMERGENCY_ACCESS,
        Action.DECIDE_ALERT_CANDIDATE,
        Action.REFER_ALERT_CANDIDATE,
        Action.RECORD_PREDICTIVE_OUTCOME,
        Action.GENERATE_RUL_FORECAST,
        Action.REVIEW_RUL_FORECAST,
        Action.CREATE_AGENT_COLLABORATION,
        Action.DISPATCH_AGENT_COLLABORATION,
        Action.REVIEW_AGENT_COLLABORATION,
        Action.REQUEST_MAINTENANCE_COUNCIL,
        Action.REVIEW_MAINTENANCE_COUNCIL,
        Action.PROPOSE_DEVICE_FAMILY,
        Action.REVIEW_DEVICE_FAMILY,
    }
)


class Authorizer:
    def __init__(
        self,
        auditor: SecurityAuditor,
        policy_client: PolicyDecisionClient | None = None,
        emergency_grants: EmergencyGrantPolicy | None = None,
    ) -> None:
        self._auditor = auditor
        self._policy_client = policy_client
        self._emergency_grants = emergency_grants

    def decide(
        self,
        identity: IdentityContext,
        action: Action,
        resource: ResourceContext,
    ) -> PolicyDecision:
        if resource.tenant_id != identity.tenant_id:
            return PolicyDecision(False, "tenant_mismatch")
        emergency_grant_id = current_emergency_grant_id()
        if emergency_grant_id is not None:
            if self._emergency_grants is None:
                return PolicyDecision(False, "emergency_grant_resolver_unavailable")
            resolution = self._emergency_grants.resolve(
                emergency_grant_id,
                identity,
                action,
                resource,
            )
            return PolicyDecision(
                resolution.allowed,
                resolution.reason_code,
                (
                    "enforce_tenant_rls",
                    "record_emergency_access_usage",
                ),
                "emergency-access-v1",
                emergency_grant_id if resolution.allowed else None,
            )
        allowed_actions = frozenset().union(
            *(ROLE_ACTIONS.get(role, frozenset()) for role in identity.roles)
        )
        if action not in allowed_actions:
            return PolicyDecision(False, "role_denied")
        if resource.asset_id is not None and resource.asset_id not in identity.asset_ids:
            return PolicyDecision(False, "device_scope_denied")
        if resource.site_id is not None and resource.site_id not in identity.site_ids:
            return PolicyDecision(False, "device_scope_denied")
        if self._policy_client is None or action not in OPA_ENFORCED_ACTIONS:
            return PolicyDecision(True, "role_and_scope_allowed")
        try:
            external = self._policy_client.decide(
                OpaDecisionInput(
                    subject_id=identity.subject_id,
                    tenant_id=identity.tenant_id,
                    roles=tuple(sorted(role.value for role in identity.roles)),
                    asset_ids=tuple(sorted(identity.asset_ids)),
                    site_ids=tuple(sorted(identity.site_ids)),
                    action=action.value,
                    resource_tenant_id=resource.tenant_id,
                    resource_id=resource.resource_id,
                    resource_asset_id=resource.asset_id,
                    resource_site_id=resource.site_id,
                )
            )
        except OpaUnavailable:
            return PolicyDecision(False, "opa_unavailable_fail_closed")
        return PolicyDecision(
            external.allowed,
            external.reason_code,
            external.obligations,
            external.policy_version,
        )

    def require(
        self,
        identity: IdentityContext,
        action: Action,
        resource: ResourceContext,
        *,
        request_id: str,
    ) -> PolicyDecision:
        decision = self.decide(identity, action, resource)
        if decision.allowed and decision.emergency_grant_id is not None:
            assert self._emergency_grants is not None
            try:
                self._emergency_grants.record_usage(
                    decision.emergency_grant_id,
                    identity,
                    action,
                    resource,
                    request_id=request_id,
                )
            except Exception as exc:
                decision = PolicyDecision(
                    False,
                    "emergency_grant_usage_audit_failed",
                    policy_version="emergency-access-v1",
                )
                self._auditor.record(
                    identity=identity,
                    action=action.value,
                    decision="deny",
                    reason_code=decision.reason_code,
                    request_id=request_id,
                    resource_id=resource.resource_id or resource.asset_id,
                )
                raise AuthorizationDenied(decision.reason_code) from exc
        self._auditor.record(
            identity=identity,
            action=action.value,
            decision="allow" if decision.allowed else "deny",
            reason_code=decision.reason_code,
            request_id=request_id,
            resource_id=resource.resource_id or resource.asset_id,
        )
        if not decision.allowed:
            raise AuthorizationDenied(decision.reason_code)
        return decision

    def record_guard_decision(
        self,
        identity: IdentityContext,
        *,
        action: str,
        decision: str,
        reason_code: str,
        request_id: str,
        resource_id: str,
    ) -> None:
        """Record a post-authorization safety guard without business content."""

        self._auditor.record(
            identity=identity,
            action=action,
            decision=decision,
            reason_code=reason_code,
            request_id=request_id,
            resource_id=resource_id,
        )
