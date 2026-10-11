"""Closed versioned registry for model-visible tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    tool_id: str
    version: str
    risk_tier: str
    input_schema: dict[str, Any]
    required_action: str
    approval_required: bool
    idempotent: bool
    timeout_seconds: int
    max_attempts: int
    rate_limit_per_minute: int
    compensation: str

    def validate(self, parameters: dict[str, Any]) -> None:
        Draft202012Validator(self.input_schema).validate(parameters)


class ToolRegistry:
    def __init__(self, definitions: list[ToolDefinition]) -> None:
        self._definitions = {(item.tool_id, item.version): item for item in definitions}

    def get(self, tool_id: str, version: str) -> ToolDefinition:
        try:
            return self._definitions[(tool_id, version)]
        except KeyError as exc:
            if tool_id.startswith("equipment.control."):
                wildcard = self._definitions.get(("equipment.control.*", version))
                if wildcard is not None:
                    return wildcard
            raise ValueError("tool_not_registered") from exc

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return the immutable registry catalog in a stable presentation order."""

        return tuple(
            sorted(
                self._definitions.values(),
                key=lambda item: (item.risk_tier, item.tool_id, item.version),
            )
        )


def default_tool_registry() -> ToolRegistry:
    readonly = {
        "type": "object",
        "properties": {"asset_id": {"type": "string", "minLength": 1}},
        "required": ["asset_id"],
        "additionalProperties": False,
    }
    reserve = {
        "type": "object",
        "properties": {
            "incident_id": {"type": "string", "minLength": 1},
            "asset_id": {"type": "string", "minLength": 1},
            "part_number": {"type": "string", "pattern": "^[A-Z0-9-]{3,64}$"},
            "quantity": {"type": "integer", "minimum": 1, "maximum": 20},
            "inventory_snapshot": {
                "type": "object",
                "properties": {
                    "available_quantity": {"type": "integer", "minimum": 0},
                    "source": {"type": "string", "minLength": 1},
                    "source_record_id": {"type": "string", "minLength": 1},
                    "as_of": {"type": "string", "minLength": 1},
                },
                "required": ["available_quantity", "source", "source_record_id", "as_of"],
                "additionalProperties": False,
            },
            "compatibility_evidence": {"const": "INSTALLED_COMPONENT_MATCH"},
        },
        "required": ["incident_id", "asset_id", "part_number", "quantity"],
        "additionalProperties": False,
    }
    part_issue = {
        "type": "object",
        "properties": {
            "incident_id": {"type": "string", "minLength": 1},
            "asset_id": {"type": "string", "minLength": 1},
            "work_order_id": {"type": "string", "minLength": 1},
            "reservation_id": {"type": "string", "minLength": 1},
            "part_number": {"type": "string", "pattern": "^[A-Z0-9-]{3,64}$"},
            "quantity": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": [
            "incident_id",
            "asset_id",
            "work_order_id",
            "reservation_id",
            "part_number",
            "quantity",
        ],
        "additionalProperties": False,
    }
    part_movement_properties = {
        "incident_id": {"type": "string", "minLength": 1},
        "asset_id": {"type": "string", "minLength": 1},
        "work_order_id": {"type": "string", "minLength": 1},
        "reservation_id": {"type": "string", "minLength": 1},
        "part_issue_id": {"type": "string", "minLength": 1},
        "part_number": {"type": "string", "pattern": "^[A-Z0-9-]{3,64}$"},
        "quantity": {"type": "integer", "minimum": 1, "maximum": 20},
    }
    part_consumption = {
        "type": "object",
        "properties": {
            **part_movement_properties,
            "field_entry_id": {"type": "string", "minLength": 1},
            "movement_kind": {"const": "CONSUME"},
        },
        "required": [
            *part_movement_properties,
            "field_entry_id",
            "movement_kind",
        ],
        "additionalProperties": False,
    }
    part_return = {
        "type": "object",
        "properties": {
            **part_movement_properties,
            "movement_kind": {"const": "RETURN"},
        },
        "required": [*part_movement_properties, "movement_kind"],
        "additionalProperties": False,
    }
    customer_notification = {
        "type": "object",
        "properties": {
            "incident_id": {"type": "string", "minLength": 1},
            "asset_id": {"type": "string", "minLength": 1},
            "recipient_scope": {"const": "INCIDENT_REPORTER"},
            "recipient_subject_id": {"type": "string", "minLength": 1},
            "channel": {"const": "PORTAL"},
            "locale": {"const": "zh-CN"},
            "template_id": {
                "enum": [
                    "DIAGNOSIS_SUMMARY",
                    "INFORMATION_REQUEST",
                    "SERVICE_PROGRESS",
                ]
            },
            "content": {
                "type": "object",
                "properties": {
                    "headline": {"type": "string", "minLength": 3, "maxLength": 120},
                    "detail": {"type": "string", "minLength": 3, "maxLength": 1000},
                    "next_step": {"type": "string", "minLength": 3, "maxLength": 500},
                },
                "required": ["headline", "detail", "next_step"],
                "additionalProperties": False,
            },
        },
        "required": [
            "incident_id",
            "asset_id",
            "recipient_scope",
            "recipient_subject_id",
            "channel",
            "locale",
            "template_id",
            "content",
        ],
        "additionalProperties": False,
    }
    external_customer_notification = {
        "type": "object",
        "properties": {
            "incident_id": {"type": "string", "minLength": 1},
            "asset_id": {"type": "string", "minLength": 1},
            "recipient_scope": {"const": "INCIDENT_REPORTER"},
            "recipient_subject_id": {"type": "string", "minLength": 1},
            "channel": {"enum": ["PORTAL_AND_EMAIL", "PORTAL_AND_SMS"]},
            "locale": {"const": "zh-CN"},
            "template_id": {
                "enum": [
                    "DIAGNOSIS_SUMMARY",
                    "INFORMATION_REQUEST",
                    "SERVICE_PROGRESS",
                ]
            },
            "content": customer_notification["properties"]["content"],
            "contact_binding": {
                "type": "object",
                "properties": {
                    "contact_point_id": {"type": "string", "minLength": 1},
                    "channel": {"enum": ["EMAIL", "SMS"]},
                    "source_system": {"type": "string", "minLength": 1},
                    "source_record_id": {"type": "string", "minLength": 1},
                    "source_version": {"type": "string", "minLength": 1},
                    "as_of": {"type": "string", "minLength": 1},
                    "masked_destination": {
                        "type": "string",
                        "minLength": 3,
                        "maxLength": 320,
                    },
                    "verification_status": {"const": "VERIFIED"},
                    "consent_status": {"const": "OPTED_IN"},
                    "consent_basis_digest": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                    },
                    "expires_at": {
                        "anyOf": [
                            {"type": "string", "minLength": 1},
                            {"type": "null"},
                        ]
                    },
                },
                "required": [
                    "contact_point_id",
                    "channel",
                    "source_system",
                    "source_record_id",
                    "source_version",
                    "as_of",
                    "masked_destination",
                    "verification_status",
                    "consent_status",
                    "consent_basis_digest",
                    "expires_at",
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            "incident_id",
            "asset_id",
            "recipient_scope",
            "recipient_subject_id",
            "channel",
            "locale",
            "template_id",
            "content",
            "contact_binding",
        ],
        "additionalProperties": False,
    }
    purchase_request = {
        "type": "object",
        "properties": {
            "incident_id": {"type": "string", "minLength": 1},
            "asset_id": {"type": "string", "minLength": 1},
            "part_number": {"type": "string", "pattern": "^[A-Z0-9-]{3,64}$"},
            "quantity": {"type": "integer", "minimum": 1, "maximum": 100},
            "estimated_unit_cost_minor": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000000,
            },
            "estimated_total_cost_minor": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100000000,
            },
            "currency": {"enum": ["CNY", "USD", "EUR"]},
            "cost_center": {"type": "string", "pattern": "^[A-Z0-9_-]{2,64}$"},
            "justification": {"type": "string", "minLength": 3, "maxLength": 1000},
            "inventory_snapshot": {
                "type": "object",
                "properties": {
                    "available_quantity": {"type": "integer", "minimum": 0},
                    "source": {"type": "string", "minLength": 1},
                    "source_record_id": {"type": "string", "minLength": 1},
                    "as_of": {"type": "string", "minLength": 1},
                },
                "required": ["available_quantity", "source", "source_record_id", "as_of"],
                "additionalProperties": False,
            },
        },
        "required": [
            "incident_id",
            "asset_id",
            "part_number",
            "quantity",
            "estimated_unit_cost_minor",
            "estimated_total_cost_minor",
            "currency",
            "cost_center",
            "justification",
            "inventory_snapshot",
        ],
        "additionalProperties": False,
    }
    enterprise_purchase_request = {
        "type": "object",
        "properties": {
            **purchase_request["properties"],
            "delivery_target": {"const": "ENTERPRISE_ERP"},
            "erp_binding": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
                    },
                    "profile_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "profile_version": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                    "display_name": {"type": "string", "minLength": 1, "maxLength": 128},
                    "as_of": {"type": "string", "minLength": 1},
                    "expires_at": {"type": "string", "minLength": 1},
                },
                "required": [
                    "provider",
                    "profile_id",
                    "profile_version",
                    "display_name",
                    "as_of",
                    "expires_at",
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            *purchase_request["required"],
            "delivery_target",
            "erp_binding",
        ],
        "additionalProperties": False,
    }
    refund_request = {
        "type": "object",
        "properties": {
            "incident_id": {"type": "string", "minLength": 1},
            "asset_id": {"type": "string", "minLength": 1},
            "work_order_id": {"type": "string", "minLength": 1},
            "customer_subject_id": {"type": "string", "minLength": 1},
            "customer_confirmation_update_id": {"type": "string", "minLength": 1},
            "customer_result_accepted": {"type": "boolean"},
            "satisfaction_rating": {"type": "integer", "minimum": 1, "maximum": 5},
            "amount_minor": {"type": "integer", "minimum": 1, "maximum": 1000000},
            "currency": {"enum": ["CNY", "USD", "EUR"]},
            "cost_center": {"type": "string", "pattern": "^[A-Z0-9_-]{2,64}$"},
            "reason": {"type": "string", "minLength": 10, "maxLength": 2000},
            "compensation_policy_id": {"const": "customer-service-compensation-v1"},
        },
        "required": [
            "incident_id",
            "asset_id",
            "work_order_id",
            "customer_subject_id",
            "customer_confirmation_update_id",
            "customer_result_accepted",
            "satisfaction_rating",
            "amount_minor",
            "currency",
            "cost_center",
            "reason",
            "compensation_policy_id",
        ],
        "additionalProperties": False,
    }
    enterprise_refund_request = {
        "type": "object",
        "properties": {
            **refund_request["properties"],
            "delivery_target": {"const": "ENTERPRISE_FINANCE"},
            "reason_code": {
                "enum": [
                    "SERVICE_NOT_ACCEPTED",
                    "LOW_SATISFACTION",
                    "REWORK_COMPENSATION",
                    "OTHER_APPROVED_COMPENSATION",
                ]
            },
            "finance_binding": {
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "minLength": 1},
                    "profile_id": {"type": "string", "minLength": 1},
                    "profile_version": {"type": "string", "minLength": 1},
                    "display_name": {"type": "string", "minLength": 1},
                    "as_of": {"type": "string", "minLength": 1},
                    "expires_at": {"type": "string", "minLength": 1},
                },
                "required": [
                    "provider",
                    "profile_id",
                    "profile_version",
                    "display_name",
                    "as_of",
                    "expires_at",
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            *refund_request["required"],
            "delivery_target",
            "reason_code",
            "finance_binding",
        ],
        "additionalProperties": False,
    }
    service_quotation_publish = {
        "type": "object",
        "properties": {
            "incident_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "asset_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "incident_version": {"type": "integer", "minimum": 1},
            "diagnosis": {
                "type": "object",
                "properties": {
                    "diagnosis_run_id": {"type": "string", "minLength": 1},
                    "version": {"type": "integer", "minimum": 1},
                },
                "required": ["diagnosis_run_id", "version"],
                "additionalProperties": False,
            },
            "candidate": {
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "minLength": 1, "maxLength": 128},
                    "candidate_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "source_version": {"type": "string", "minLength": 1, "maxLength": 128},
                    "display_name": {"type": "string", "minLength": 1, "maxLength": 128},
                    "incident_ref": {"type": "string", "minLength": 1, "maxLength": 128},
                    "asset_ref": {"type": "string", "minLength": 1, "maxLength": 128},
                    "diagnosis_ref": {"type": "string", "minLength": 1, "maxLength": 128},
                    "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
                    "line_items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 100,
                        "items": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string", "minLength": 1, "maxLength": 64},
                                "description": {"type": "string", "minLength": 1, "maxLength": 255},
                                "quantity": {"type": "integer", "minimum": 1, "maximum": 10000},
                                "unit_price": {"type": "string", "pattern": "^(?:0|[1-9][0-9]{0,11})\\.[0-9]{2}$"},
                                "line_total": {"type": "string", "pattern": "^(?:0|[1-9][0-9]{0,11})\\.[0-9]{2}$"},
                            },
                            "required": ["code", "description", "quantity", "unit_price", "line_total"],
                            "additionalProperties": False,
                        },
                    },
                    "subtotal": {"type": "string", "pattern": "^(?:0|[1-9][0-9]{0,11})\\.[0-9]{2}$"},
                    "discount": {"type": "string", "pattern": "^(?:0|[1-9][0-9]{0,11})\\.[0-9]{2}$"},
                    "tax": {"type": "string", "pattern": "^(?:0|[1-9][0-9]{0,11})\\.[0-9]{2}$"},
                    "total": {"type": "string", "pattern": "^(?:0|[1-9][0-9]{0,11})\\.[0-9]{2}$"},
                    "as_of": {"type": "string", "minLength": 1},
                    "expires_at": {"type": "string", "minLength": 1},
                },
                "required": [
                    "provider", "candidate_id", "source_version", "display_name",
                    "incident_ref", "asset_ref", "diagnosis_ref", "currency",
                    "line_items", "subtotal", "discount", "tax", "total",
                    "as_of", "expires_at"
                ],
                "additionalProperties": False,
            },
            "entitlement": {
                "type": "object",
                "properties": {
                    "decision": {"enum": ["COVERED", "NOT_COVERED"]},
                    "reason_codes": {"type": "array", "items": {"type": "string"}},
                    "master_warranty": {"type": ["object", "null"]},
                    "live_entitlement": {"type": ["object", "null"]},
                },
                "required": ["decision", "reason_codes", "master_warranty", "live_entitlement"],
                "additionalProperties": False,
            },
        },
        "required": [
            "incident_id", "asset_id", "incident_version", "diagnosis",
            "candidate", "entitlement"
        ],
        "additionalProperties": False,
    }
    repair_work_order_create = {
        "type": "object",
        "properties": {
            "incident_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "asset_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "incident_version": {"type": "integer", "minimum": 1},
            "operation_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "authorization_type": {
                "enum": ["COVERED_SERVICE", "ACCEPTED_QUOTATION"]
            },
            "fulfillment_mode": {"const": "NO_INITIAL_PARTS"},
            "diagnosis": {
                "type": "object",
                "properties": {
                    "diagnosis_run_id": {"type": "string", "minLength": 1},
                    "version": {"type": "integer", "minimum": 1},
                },
                "required": ["diagnosis_run_id", "version"],
                "additionalProperties": False,
            },
            "entitlement": {
                "type": "object",
                "properties": {
                    "decision": {"enum": ["COVERED", "NOT_COVERED"]},
                    "reason_codes": {"type": "array", "items": {"type": "string"}},
                    "master_warranty": {"type": ["object", "null"]},
                    "live_entitlement": {"type": ["object", "null"]},
                },
                "required": ["decision", "reason_codes", "master_warranty", "live_entitlement"],
                "additionalProperties": False,
            },
            "quotation": {
                "oneOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {
                            "quotation_id": {"type": "string", "minLength": 1},
                            "quotation_version": {"type": "integer", "minimum": 1},
                            "state_version": {"type": "integer", "minimum": 1},
                            "status": {"const": "ACCEPTED"},
                            "diagnosis_run_id": {"type": "string", "minLength": 1},
                            "diagnosis_version": {"type": "integer", "minimum": 1},
                            "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
                            "total": {"type": "string", "pattern": "^(?:0|[1-9][0-9]{0,11})\\.[0-9]{2}$"},
                            "publication_proposal_id": {"type": "string", "minLength": 1},
                            "publication_approval_id": {"type": "string", "minLength": 1},
                            "customer_decision": {
                                "type": "object",
                                "properties": {
                                    "decision_id": {"type": "string", "minLength": 1},
                                    "decision": {"const": "ACCEPTED"},
                                    "reporter_subject_id": {"type": "string", "minLength": 1},
                                    "quotation_version": {"type": "integer", "minimum": 1},
                                    "decided_at": {"type": "string", "format": "date-time"},
                                },
                                "required": [
                                    "decision_id", "decision", "reporter_subject_id",
                                    "quotation_version", "decided_at"
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "required": [
                            "quotation_id", "quotation_version", "state_version", "status",
                            "diagnosis_run_id", "diagnosis_version", "currency", "total",
                            "publication_proposal_id", "publication_approval_id",
                            "customer_decision"
                        ],
                        "additionalProperties": False,
                    },
                ]
            },
        },
        "required": [
            "incident_id", "asset_id", "incident_version", "operation_id",
            "authorization_type", "fulfillment_mode", "diagnosis", "entitlement",
            "quotation"
        ],
        "additionalProperties": False,
    }
    work_order_assignment = {
        "type": "object",
        "properties": {
            "incident_id": {"type": "string", "minLength": 1},
            "asset_id": {"type": "string", "minLength": 1},
            "work_order_id": {"type": "string", "minLength": 1},
            "assignee_subject_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "reason": {"type": "string", "minLength": 3, "maxLength": 1000},
        },
        "required": [
            "incident_id",
            "asset_id",
            "work_order_id",
            "assignee_subject_id",
            "reason",
        ],
        "additionalProperties": False,
    }
    enterprise_work_order_assignment = {
        "type": "object",
        "properties": {
            **work_order_assignment["properties"],
            "site_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "assignment_target": {"const": "ENTERPRISE_FSM"},
            "fsm_binding": {
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "minLength": 1, "maxLength": 128},
                    "candidate_profile_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                    "profile_version": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                    "assignee_subject_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                    "display_name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                    "matched_skill_codes": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 64},
                        "minItems": 1,
                        "maxItems": 32,
                        "uniqueItems": True,
                    },
                    "service_window_start": {
                        "type": "string",
                        "format": "date-time",
                    },
                    "service_window_end": {
                        "type": "string",
                        "format": "date-time",
                    },
                    "travel_minutes": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10080,
                    },
                    "remaining_work_minutes": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10080,
                    },
                    "eligibility_code": {"const": "ELIGIBLE"},
                    "source_record_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 255,
                    },
                    "as_of": {"type": "string", "format": "date-time"},
                    "expires_at": {"type": "string", "format": "date-time"},
                },
                "required": [
                    "provider",
                    "candidate_profile_id",
                    "profile_version",
                    "assignee_subject_id",
                    "display_name",
                    "matched_skill_codes",
                    "service_window_start",
                    "service_window_end",
                    "travel_minutes",
                    "remaining_work_minutes",
                    "eligibility_code",
                    "source_record_id",
                    "as_of",
                    "expires_at",
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            *work_order_assignment["required"],
            "site_id",
            "assignment_target",
            "fsm_binding",
        ],
        "additionalProperties": False,
    }
    work_order_closure = {
        "type": "object",
        "properties": {
            "incident_id": {"type": "string", "minLength": 1},
            "asset_id": {"type": "string", "minLength": 1},
            "work_order_id": {"type": "string", "minLength": 1},
            "reason": {"type": "string", "minLength": 3, "maxLength": 1000},
        },
        "required": ["incident_id", "asset_id", "work_order_id", "reason"],
        "additionalProperties": False,
    }
    equipment_control_handoff = {
        "type": "object",
        "properties": {
            "incident_id": {"type": "string", "minLength": 1},
            "asset_id": {"type": "string", "minLength": 1},
            "requested_operation": {
                "enum": [
                    "STOP",
                    "REMOTE_START",
                    "PLC_PARAMETER_CHANGE",
                    "INTERLOCK_OVERRIDE",
                ]
            },
            "reason": {"type": "string", "minLength": 10, "maxLength": 2000},
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": 20,
                "uniqueItems": True,
            },
        },
        "required": [
            "incident_id",
            "asset_id",
            "requested_operation",
            "reason",
            "evidence_ids",
        ],
        "additionalProperties": False,
    }
    definitions = [
        ToolDefinition(tool, "1.0.0", "T0", readonly, "asset.read", False, True, 5, 2, 60, "none")
        for tool in (
            "asset.get",
            "warranty.get",
            "parts.availability",
            "schedule.availability",
            "work_orders.history",
        )
    ]
    definitions.extend(
        [
            ToolDefinition(
                "work_order.draft",
                "1.0.0",
                "T1",
                readonly,
                "work_order.read",
                False,
                True,
                5,
                2,
                30,
                "discard_draft",
            ),
            ToolDefinition(
                "parts.reserve",
                "1.0.0",
                "T2",
                reserve,
                "action.execute",
                True,
                True,
                15,
                1,
                10,
                "release_reservation",
            ),
            ToolDefinition(
                "parts.issue",
                "1.0.0",
                "T2",
                part_issue,
                "action.execute",
                True,
                True,
                15,
                1,
                10,
                "manual_return",
            ),
            ToolDefinition(
                "parts.consume",
                "1.0.0",
                "T2",
                part_consumption,
                "action.execute",
                True,
                True,
                15,
                1,
                10,
                "manual_reversal",
            ),
            ToolDefinition(
                "parts.return",
                "1.0.0",
                "T2",
                part_return,
                "action.execute",
                True,
                True,
                15,
                1,
                10,
                "manual_reissue",
            ),
            ToolDefinition(
                "customer.notify",
                "1.0.0",
                "T2",
                customer_notification,
                "action.execute",
                True,
                True,
                10,
                1,
                20,
                "publish_correction",
            ),
            ToolDefinition(
                "customer.notify",
                "1.1.0",
                "T2",
                external_customer_notification,
                "action.execute",
                True,
                True,
                10,
                1,
                20,
                "publish_correction",
            ),
            ToolDefinition(
                "purchase.request",
                "1.0.0",
                "T2",
                purchase_request,
                "action.execute",
                True,
                True,
                15,
                1,
                10,
                "cancel_purchase_request",
            ),
            ToolDefinition(
                "purchase.request",
                "1.1.0",
                "T2",
                enterprise_purchase_request,
                "action.execute",
                True,
                True,
                15,
                1,
                10,
                "cancel_purchase_request",
            ),
            ToolDefinition(
                "refund.request",
                "1.0.0",
                "T2",
                refund_request,
                "action.execute",
                True,
                True,
                10,
                1,
                10,
                "cancel_refund_request",
            ),
            ToolDefinition(
                "service.quote.publish",
                "1.0.0",
                "T2",
                service_quotation_publish,
                "action.execute",
                True,
                True,
                10,
                1,
                10,
                "publish_new_quotation_version",
            ),
            ToolDefinition(
                "work_order.create",
                "1.0.0",
                "T2",
                repair_work_order_create,
                "action.execute",
                True,
                True,
                10,
                1,
                10,
                "cancel_work_order_before_dispatch",
            ),
            ToolDefinition(
                "refund.request",
                "1.1.0",
                "T2",
                enterprise_refund_request,
                "action.execute",
                True,
                True,
                10,
                1,
                10,
                "cancel_refund_request",
            ),
            ToolDefinition(
                "work_order.assign",
                "1.0.0",
                "T2",
                work_order_assignment,
                "action.execute",
                True,
                True,
                10,
                1,
                20,
                "reassign_work_order",
            ),
            ToolDefinition(
                "work_order.assign",
                "1.1.0",
                "T2",
                enterprise_work_order_assignment,
                "action.execute",
                True,
                True,
                10,
                1,
                20,
                "reassign_work_order",
            ),
            ToolDefinition(
                "work_order.close",
                "1.0.0",
                "T2",
                work_order_closure,
                "action.execute",
                True,
                True,
                10,
                1,
                20,
                "manual_reopen_review",
            ),
            ToolDefinition(
                "equipment.control.*",
                "1.1.0",
                "T3",
                equipment_control_handoff,
                "external_safety_handoff",
                True,
                False,
                0,
                0,
                0,
                "external_handoff_only",
            ),
        ]
    )
    return ToolRegistry(definitions)
