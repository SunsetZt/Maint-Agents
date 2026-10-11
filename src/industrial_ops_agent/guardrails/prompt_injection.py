"""Prompt-injection gate that records fingerprints instead of sensitive content."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

PROMPT_INJECTION_POLICY_VERSION = "m6-prompt-injection-v1"
MAX_GUARDRAIL_TEXT_CHARACTERS = 64_000

_CONTROL_CHARACTERS = frozenset(
    {
        "\u061c",
        "\u200b",
        "\u200c",
        "\u200d",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2060",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
        "\ufeff",
    }
)


@dataclass(frozen=True, slots=True)
class _Rule:
    pattern_id: str
    category: str
    expression: re.Pattern[str]


@dataclass(frozen=True, slots=True)
class GuardrailFinding:
    phase: Literal["INPUT", "OUTPUT"]
    source_role: str
    pattern_id: str
    category: str
    severity: Literal["HIGH"]
    content_hash: str

    def audit_dict(self) -> dict[str, str]:
        return {
            "phase": self.phase,
            "source_role": self.source_role,
            "pattern_id": self.pattern_id,
            "category": self.category,
            "severity": self.severity,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    decision: Literal["ALLOWED", "BLOCKED"]
    policy_version: str
    findings: tuple[GuardrailFinding, ...]


class PromptInjectionGuard:
    """Block explicit instruction override, exfiltration and authority spoofing."""

    policy_version = PROMPT_INJECTION_POLICY_VERSION

    def inspect_messages(self, messages: tuple[dict[str, Any], ...]) -> GuardrailDecision:
        findings: list[GuardrailFinding] = []
        for message in messages:
            role = str(message.get("role", "unknown"))
            if role == "system":
                continue
            findings.extend(
                self._inspect_texts(
                    _message_texts(message.get("content")),
                    phase="INPUT",
                    source_role=role,
                )
            )
        return _decision(findings, self.policy_version)

    def inspect_text(self, text: str, *, source_role: str) -> GuardrailDecision:
        """Inspect one untrusted extracted text with the production policy."""

        return _decision(
            self._inspect_texts((text,), phase="INPUT", source_role=source_role),
            self.policy_version,
        )

    def inspect_output(self, content: dict[str, Any]) -> GuardrailDecision:
        encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return _decision(
            self._inspect_texts((encoded,), phase="OUTPUT", source_role="model"),
            self.policy_version,
        )

    def _inspect_texts(
        self,
        texts: tuple[str, ...],
        *,
        phase: Literal["INPUT", "OUTPUT"],
        source_role: str,
    ) -> list[GuardrailFinding]:
        findings: list[GuardrailFinding] = []
        for text in texts:
            if len(text) > MAX_GUARDRAIL_TEXT_CHARACTERS:
                findings.append(
                    _finding(
                        phase,
                        source_role,
                        "guardrail_text_limit",
                        "resource_exhaustion",
                        text,
                    )
                )
                continue
            normalized, contained_controls = _normalize(text)
            if contained_controls:
                findings.append(
                    _finding(
                        phase,
                        source_role,
                        "unicode_control_obfuscation",
                        "obfuscation",
                        normalized,
                    )
                )
            for rule in _RULES:
                if rule.expression.search(normalized):
                    findings.append(
                        _finding(
                            phase,
                            source_role,
                            rule.pattern_id,
                            rule.category,
                            normalized,
                        )
                    )
        unique = {
            (item.phase, item.source_role, item.pattern_id, item.content_hash): item
            for item in findings
        }
        return sorted(
            unique.values(),
            key=lambda item: (item.phase, item.source_role, item.pattern_id, item.content_hash),
        )


def _message_texts(content: object) -> tuple[str, ...]:
    if isinstance(content, str):
        return (content,)
    if not isinstance(content, list):
        return ()
    return tuple(
        str(item["text"])
        for item in content
        if (
            isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        )
    )


def _normalize(text: str) -> tuple[str, bool]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    contained_controls = any(character in _CONTROL_CHARACTERS for character in normalized)
    normalized = "".join(
        " " if character in _CONTROL_CHARACTERS else character for character in normalized
    )
    return re.sub(r"\s+", " ", normalized), contained_controls


def _finding(
    phase: Literal["INPUT", "OUTPUT"],
    source_role: str,
    pattern_id: str,
    category: str,
    normalized_text: str,
) -> GuardrailFinding:
    return GuardrailFinding(
        phase=phase,
        source_role=source_role[:32],
        pattern_id=pattern_id,
        category=category,
        severity="HIGH",
        content_hash=sha256(normalized_text.encode()).hexdigest(),
    )


def _decision(
    findings: list[GuardrailFinding], policy_version: str
) -> GuardrailDecision:
    return GuardrailDecision(
        decision="BLOCKED" if findings else "ALLOWED",
        policy_version=policy_version,
        findings=tuple(findings),
    )


_RULES = (
    _Rule(
        "instruction_override_en",
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,80}"
            r"\b(?:previous|prior|system|developer|safety|instructions?|rules?|prompt)\b",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "instruction_override_zh",
        "instruction_override",
        re.compile(r"(?:忽略|无视|绕过|覆盖|忘掉|废除).{0,40}(?:系统|开发者|上文|之前|安全|规则|指令|提示词)"),
    ),
    _Rule(
        "sensitive_exfiltration_en",
        "sensitive_exfiltration",
        re.compile(
            r"\b(?:reveal|show|print|return|expose|leak|read)\b.{0,60}"
            r"\b(?:system prompt|developer message|secret|api[ -]?key|access token|password)\b",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "sensitive_exfiltration_zh",
        "sensitive_exfiltration",
        re.compile(r"(?:显示|输出|泄露|打印|返回|读取).{0,40}(?:系统提示词|开发者指令|密钥|令牌|密码|secret)"),
    ),
    _Rule(
        "authority_escalation_en",
        "authority_escalation",
        re.compile(
            r"\b(?:you are now|act as|switch to)\b.{0,50}"
            r"\b(?:admin|administrator|root|developer|system)\b",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "authority_escalation_zh",
        "authority_escalation",
        re.compile(r"(?:你现在是|切换为|假装是|扮演).{0,30}(?:管理员|root|开发者|系统)"),
    ),
    _Rule(
        "tool_result_spoofing",
        "tool_result_spoofing",
        re.compile(
            r"(?:<tool_call>|<tool_result>|begin tool result|function_call|tool_result)",
            re.IGNORECASE,
        ),
    ),
)
