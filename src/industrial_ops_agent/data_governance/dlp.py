"""Presidio-based Chinese and industrial-sensitive-field redaction."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import spacy
from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
)
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

POLICY_VERSION = "m3-presidio-zh-industrial-v1"
ENTITY_TYPES = (
    "PERSON",
    "PHONE_NUMBER",
    "ADDRESS",
    "CUSTOMER_CODE",
    "DEVICE_SERIAL",
    "INTERNAL_ID",
)


@dataclass(frozen=True, slots=True)
class DlpOutput:
    status: str
    redacted_content: dict[str, str]
    findings: tuple[dict[str, Any], ...]
    residual_entity_types: tuple[str, ...]


class _RuleOnlyNlpEngine(NlpEngine):
    """Tokenization for Presidio pattern recognizers without a remote/model download."""

    def __init__(self) -> None:
        self._nlp = {"zh": spacy.blank("zh")}

    def load(self) -> None:
        pass

    def is_loaded(self) -> bool:
        return True

    def process_text(self, text: str, language: str) -> NlpArtifacts:
        doc = self._nlp[language](text)
        return NlpArtifacts(
            entities=[],
            tokens=doc,
            tokens_indices=[token.idx for token in doc],
            lemmas=[token.text for token in doc],
            nlp_engine=self,
            language=language,
        )

    def process_batch(
        self,
        texts: Iterable[str],
        language: str,
        batch_size: int = 1,
        n_process: int = 1,
        **kwargs: Any,
    ) -> Iterator[tuple[str, NlpArtifacts]]:
        del batch_size, n_process, kwargs
        for text in texts:
            yield text, self.process_text(text, language)

    def is_stopword(self, word: str, language: str) -> bool:
        del word, language
        return False

    def is_punct(self, word: str, language: str) -> bool:
        del language
        return all(not character.isalnum() for character in word)

    def get_supported_entities(self) -> list[str]:
        return []

    def get_supported_languages(self) -> list[str]:
        return ["zh"]


class PresidioDlpProcessor:
    """Detect and replace PII using Presidio plus enterprise-specific recognizers."""

    def __init__(self) -> None:
        registry = RecognizerRegistry(supported_languages=["zh"])
        for recognizer in _recognizers():
            registry.add_recognizer(recognizer)
        self._analyzer = AnalyzerEngine(
            registry=registry,
            nlp_engine=_RuleOnlyNlpEngine(),
            supported_languages=["zh"],
        )
        self._anonymizer = AnonymizerEngine()

    def process(self, content: Mapping[str, str]) -> DlpOutput:
        redacted: dict[str, str] = {}
        findings: list[dict[str, Any]] = []
        for field_name, text in sorted(content.items()):
            results = self._analyze(text)
            for result in results:
                findings.append(
                    {
                        "field": field_name,
                        "entity_type": result.entity_type,
                        "start": result.start,
                        "end": result.end,
                        "score": round(float(result.score), 4),
                        "replacement": f"<{result.entity_type}>",
                    }
                )
            redacted[field_name] = self._anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators={"DEFAULT": OperatorConfig("replace")},
            ).text

        residual = sorted(
            {
                result.entity_type
                for text in redacted.values()
                for result in self._analyze(text)
            }
        )
        return DlpOutput(
            status="PASSED" if not residual else "FAILED",
            redacted_content=redacted,
            findings=tuple(findings),
            residual_entity_types=tuple(residual),
        )

    def _analyze(self, text: str):
        return self._analyzer.analyze(
            text=text,
            language="zh",
            entities=list(ENTITY_TYPES),
        )


def _recognizers() -> tuple[PatternRecognizer, ...]:
    definitions = {
        "PHONE_NUMBER": [
            ("cn-mobile", r"(?<!\d)1[3-9]\d{9}(?!\d)", 0.95),
        ],
        "PERSON": [
            ("contact-name", r"(?<=联系人)[\u4e00-\u9fff]{2,4}", 0.9),
            (
                "confirmation-name",
                r"(?:赵|钱|孙|李|周|吴|郑|王|冯|陈|蒋|沈|韩|杨|朱|秦|许|何|吕|张|孔|曹|魏|姜)[\u4e00-\u9fff]{1,3}(?=通过)",
                0.85,
            ),
            ("action-name", r"(?<=为)[\u4e00-\u9fff]{2,4}(?=更换)", 0.85),
        ],
        "ADDRESS": [
            ("cn-address", r"(?<=地址：)(?!<)[^，。；;,\n]{4,80}", 0.9),
            ("cn-address-ascii-colon", r"(?<=地址:)(?!<)[^，。；;,\n]{4,80}", 0.9),
        ],
        "CUSTOMER_CODE": [
            ("customer-code", r"\bCUST-[A-Z0-9-]{2,64}\b", 0.95),
        ],
        "DEVICE_SERIAL": [
            ("device-serial", r"\bSN-[A-Z0-9-]{2,64}\b", 0.95),
        ],
        "INTERNAL_ID": [
            ("internal-id", r"\bINT-[A-Z0-9-]{2,64}\b", 0.95),
        ],
    }
    return tuple(
        PatternRecognizer(
            supported_entity=entity_type,
            supported_language="zh",
            patterns=[Pattern(name, expression, score) for name, expression, score in patterns],
            name=f"M3{entity_type.title().replace('_', '')}Recognizer",
            version="1.0.0",
        )
        for entity_type, patterns in definitions.items()
    )
