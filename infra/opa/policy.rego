package industrial_ops.authz

import rego.v1

role_actions := {
  "customer_contact": {"service_result.confirm", "memory.write", "memory.revoke"},
  "field_engineer": {
    "agent.control",
    "action.propose",
    "work_order.execute",
    "work_order.control",
    "memory.write",
    "memory.revoke",
  },
  "after_sales_engineer": {
    "agent.control",
    "incident.control",
    "action.propose",
    "refund_request.propose",
    "execution_reconciliation.reconcile",
    "approval.decide",
    "action.execute",
    "work_order.assign",
    "work_order.control",
    "work_order.escalate",
    "work_order.close",
    "alert_candidate.decide",
    "alert_candidate.refer",
    "predictive_outcome.record",
    "memory.write",
    "memory.revoke",
    "agent_collaboration.create",
    "agent_collaboration.dispatch",
    "agent_collaboration.review",
    "maintenance_council.request",
  },
  "domain_expert": {
    "agent.control",
    "incident.control",
    "knowledge.publish",
    "knowledge_graph.manage",
    "knowledge_search.manage",
    "knowledge_deletion.request",
    "action.propose",
    "refund_request.propose",
    "execution_reconciliation.reconcile",
    "approval.decide",
    "action.execute",
    "work_order.verify",
    "work_order.control",
    "work_order.escalate",
    "work_order.close",
    "alert_candidate.decide",
    "alert_candidate.refer",
    "predictive_outcome.record",
    "memory.write",
    "memory.revoke",
    "agent_collaboration.create",
    "agent_collaboration.dispatch",
    "agent_collaboration.review",
    "maintenance_council.request",
    "maintenance_council.review",
    "device_family.propose",
  },
  "tenant_admin": {
    "incident.control",
    "tenant_administration.manage",
    "knowledge.publish",
    "knowledge_graph.activate",
    "knowledge_search.activate",
    "external_search_policy.manage",
    "knowledge_deletion.request",
    "approval.decide",
    "work_order.assign",
    "work_order.control",
    "work_order.escalate",
    "work_order.close",
    "production_acceptance.sign",
    "device_family.review",
    "memory.write",
    "memory.revoke",
  },
  "security_auditor": {
    "security_exercise.manage",
    "production_acceptance.sign",
    "emergency_access.decide",
    "emergency_access.revoke",
  },
  "model_engineer": {"enterprise_model_asset.import"},
  "model_release_approver": {"model_release.decide_approval"},
  "model_release_operator": {
    "model_deployment.request",
    "model_release.promote",
    "model_release.rollback",
    "model_gateway_quota.manage",
  },
  "model_deployment_controller": {
    "model_deployment.request",
    "model_deployment.reconcile",
    "model_release.record_observation",
  },
  "platform_operator": {
    "cost_policy.manage",
    "knowledge_deletion.request",
    "recovery.manage",
    "production_acceptance.manage",
    "production_acceptance.sign",
    "emergency_access.request",
  },
}

tenant_allowed if input.tenant_id == input.resource_tenant_id

role_allowed if {
  some role in input.roles
  input.action in role_actions[role]
}

asset_allowed if input.resource_asset_id == null
asset_allowed if input.resource_asset_id in input.asset_ids

site_allowed if input.resource_site_id == null
site_allowed if input.resource_site_id in input.site_ids

allow if {
  tenant_allowed
  role_allowed
  asset_allowed
  site_allowed
}

result := {
  "allow": true,
  "reason": "opa_role_scope_allowed",
  "obligations": ["enforce_tenant_rls", "record_security_audit"],
  "policy_version": "m2-high-risk-v1",
} if allow

result := {
  "allow": false,
  "reason": "opa_role_or_scope_denied",
  "obligations": ["record_security_audit"],
  "policy_version": "m2-high-risk-v1",
} if not allow
