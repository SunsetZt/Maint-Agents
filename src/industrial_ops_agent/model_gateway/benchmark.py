"""Fair, content-bound benchmarks for OpenAI-compatible inference runtimes."""

from __future__ import annotations

import asyncio
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from time import monotonic
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

import httpx
from jsonschema import SchemaError, ValidationError, validate
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BENCHMARK_PLAN_SCHEMA = "industrial-inference-benchmark-plan-v1"
BENCHMARK_REPORT_SCHEMA = "industrial-inference-benchmark-report-v1"
BENCHMARK_COMPARISON_SCHEMA = "industrial-inference-benchmark-comparison-v1"
MAX_MEASURED_REQUESTS = 100_000

Sha256Digest = str
RuntimeEngine = Literal["vllm", "sglang", "tensorrt-llm"]
RuntimeProfile = Literal["PROD_GENERAL", "PROD_PREFIX_HEAVY", "PROD_NVIDIA_OPT"]

_PROFILE_ENGINES: dict[str, str] = {
    "PROD_GENERAL": "vllm",
    "PROD_PREFIX_HEAVY": "sglang",
    "PROD_NVIDIA_OPT": "tensorrt-llm",
}


class InferenceBenchmarkError(RuntimeError):
    """The benchmark plan, execution, or report cannot be trusted."""


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, max_length=128)
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=128)
    response_schema_name: str = Field(min_length=1, max_length=128)
    response_schema: dict[str, Any]
    max_output_tokens: int = Field(ge=1, le=32_768)

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for message in messages:
            if set(message) != {"role", "content"}:
                raise ValueError("benchmark messages only accept role and content")
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("benchmark message role is invalid")
            if not isinstance(message.get("content"), str) or not message["content"]:
                raise ValueError("benchmark message content must be non-empty text")
        return messages

    @field_validator("response_schema")
    @classmethod
    def validate_response_schema(cls, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            validate(instance={}, schema=schema)
        except ValidationError:
            # The empty object does not need to satisfy the schema. This only forces
            # jsonschema to validate the schema document itself.
            pass
        except SchemaError as exc:
            raise ValueError("benchmark response schema is invalid") from exc
        return schema


class BenchmarkRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: RuntimeProfile
    engine: RuntimeEngine
    endpoint_url: str = Field(min_length=8, max_length=2048)
    image_digest: str

    @field_validator("image_digest")
    @classmethod
    def validate_image_digest(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("runtime image digest must be sha256:<64 lowercase hex>")
        return value

    @model_validator(mode="after")
    def validate_profile_engine(self) -> BenchmarkRuntime:
        if _PROFILE_ENGINES[self.profile] != self.engine:
            raise ValueError("runtime profile and engine do not match")
        return self


class BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    served_model_id: str = Field(min_length=1, max_length=512)
    model_digest: str
    adapter_digest: str
    tokenizer_digest: str

    @field_validator("model_digest", "adapter_digest", "tokenizer_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("model artifact digest must be sha256:<64 lowercase hex>")
        return value


class BenchmarkHardware(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gpu_model: str = Field(min_length=1, max_length=255)
    gpu_count: int = Field(ge=1, le=128)
    memory_per_gpu_bytes: int = Field(ge=1)
    driver_version: str = Field(min_length=1, max_length=64)
    cuda_version: str = Field(min_length=1, max_length=32)


class BenchmarkExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    concurrency: int = Field(ge=1, le=4096)
    repeats: int = Field(ge=1, le=10_000)
    warmup_requests: int = Field(default=1, ge=0, le=10_000)
    timeout_seconds: float = Field(gt=0, le=3600)
    temperature: float = Field(default=0.0, ge=0, le=2)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    normalized_service_parameters: dict[str, Any] = Field(min_length=1)

    @field_validator("normalized_service_parameters")
    @classmethod
    def validate_service_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            _canonical_json(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("normalized service parameters must be JSON serializable") from exc
        return value


class BenchmarkThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_request_success_rate: float = Field(default=0.99, ge=0, le=1)
    minimum_structured_success_rate: float = Field(default=0.99, ge=0, le=1)
    maximum_error_rate: float = Field(default=0.01, ge=0, le=1)
    maximum_p95_latency_ms: float = Field(default=5000, gt=0)


class InferenceBenchmarkPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["industrial-inference-benchmark-plan-v1"] = (
        "industrial-inference-benchmark-plan-v1"
    )
    benchmark_id: str = Field(min_length=1, max_length=128)
    release_id: str = Field(min_length=1, max_length=128)
    model: BenchmarkModel
    runtime: BenchmarkRuntime
    hardware: BenchmarkHardware
    dataset_snapshot_id: str = Field(min_length=1, max_length=128)
    workload_digest: str = ""
    cases: list[BenchmarkCase] = Field(min_length=1, max_length=100_000)
    execution: BenchmarkExecution
    thresholds: BenchmarkThresholds = Field(default_factory=BenchmarkThresholds)

    @field_validator("workload_digest")
    @classmethod
    def validate_optional_workload_digest(cls, value: str) -> str:
        if value and not _is_sha256(value):
            raise ValueError("workload digest must be sha256:<64 lowercase hex>")
        return value

    @model_validator(mode="after")
    def bind_workload(self) -> InferenceBenchmarkPlan:
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("benchmark case ids must be unique")
        calculated = content_digest([case.model_dump(mode="json") for case in self.cases])
        if self.workload_digest and self.workload_digest != calculated:
            raise ValueError("workload digest does not match benchmark cases")
        if len(self.cases) * self.execution.repeats > MAX_MEASURED_REQUESTS:
            raise ValueError("benchmark measured request count exceeds safety limit")
        object.__setattr__(self, "workload_digest", calculated)
        return self

    def plan_digest(self) -> Sha256Digest:
        return content_digest(self.model_dump(mode="json"))

    def comparison_contract_digest(self) -> Sha256Digest:
        return content_digest(
            {
                "schema_version": self.schema_version,
                "benchmark_id": self.benchmark_id,
                "release_id": self.release_id,
                "model": self.model.model_dump(mode="json"),
                "hardware": self.hardware.model_dump(mode="json"),
                "dataset_snapshot_id": self.dataset_snapshot_id,
                "workload_digest": self.workload_digest,
                "execution": self.execution.model_dump(mode="json"),
                "thresholds": self.thresholds.model_dump(mode="json"),
            }
        )


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    case_id: str
    success: bool
    structured_output_valid: bool
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float | None
    tpot_ms: float | None
    e2e_latency_ms: float
    finish_reason: str | None
    response_hash: str | None
    failure_reason: str | None


class BenchmarkTransport(Protocol):
    async def complete(
        self, plan: InferenceBenchmarkPlan, case: BenchmarkCase
    ) -> BenchmarkSample: ...


class OpenAiStreamingBenchmarkTransport:
    """Streaming adapter shared by vLLM, SGLang, and TensorRT-LLM profiles."""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        *,
        timeout_seconds: float,
        allow_plain_http: bool = False,
    ) -> None:
        endpoint = _completion_endpoint(endpoint_url, allow_plain_http)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._endpoint = endpoint
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
        )

    async def __aenter__(self) -> OpenAiStreamingBenchmarkTransport:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    async def complete(
        self, plan: InferenceBenchmarkPlan, case: BenchmarkCase
    ) -> BenchmarkSample:
        payload = {
            "model": plan.model.served_model_id,
            "messages": case.messages,
            "max_tokens": case.max_output_tokens,
            "temperature": plan.execution.temperature,
            "seed": plan.execution.seed,
            "stream": True,
            "stream_options": {"include_usage": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": case.response_schema_name,
                    "schema": case.response_schema,
                    "strict": True,
                },
            },
        }
        started = monotonic()
        first_token_at: float | None = None
        finish_reason: str | None = None
        content_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        try:
            async with self._client.stream("POST", self._endpoint, json=payload) as response:
                if response.status_code != 200:
                    reason = (
                        "benchmark_endpoint_rejected"
                        if response.status_code < 500
                        else "benchmark_endpoint_failed"
                    )
                    raise InferenceBenchmarkError(reason)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise InferenceBenchmarkError("benchmark_stream_event_invalid") from exc
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        prompt_tokens = _non_negative_int(usage.get("prompt_tokens"))
                        completion_tokens = _non_negative_int(usage.get("completion_tokens"))
                    choices = event.get("choices", [])
                    if not isinstance(choices, list):
                        raise InferenceBenchmarkError("benchmark_stream_event_invalid")
                    for choice in choices:
                        if not isinstance(choice, dict):
                            raise InferenceBenchmarkError("benchmark_stream_event_invalid")
                        reason_value = choice.get("finish_reason")
                        if isinstance(reason_value, str):
                            finish_reason = reason_value
                        delta = choice.get("delta", {})
                        if not isinstance(delta, dict):
                            continue
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            if first_token_at is None:
                                first_token_at = monotonic()
                            content_parts.append(content)
        except httpx.HTTPError as exc:
            raise InferenceBenchmarkError("benchmark_endpoint_unavailable") from exc

        finished = monotonic()
        if first_token_at is None or completion_tokens <= 0 or not content_parts:
            raise InferenceBenchmarkError("benchmark_stream_result_incomplete")
        raw_content = "".join(content_parts)
        structured_valid = _structured_output_valid(raw_content, case.response_schema)
        tpot_ms = (
            max(0.0, (finished - first_token_at) * 1000 / (completion_tokens - 1))
            if completion_tokens > 1
            else 0.0
        )
        return BenchmarkSample(
            case_id=case.case_id,
            success=True,
            structured_output_valid=structured_valid,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=(first_token_at - started) * 1000,
            tpot_ms=tpot_ms,
            e2e_latency_ms=(finished - started) * 1000,
            finish_reason=finish_reason,
            response_hash=content_digest(raw_content),
            failure_reason=None,
        )


class InferenceBenchmarkRunner:
    async def run(
        self,
        plan: InferenceBenchmarkPlan,
        transport: BenchmarkTransport,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        warmup_failures: list[str] = []
        for index in range(plan.execution.warmup_requests):
            sample = await _capture_sample(transport, plan, plan.cases[index % len(plan.cases)])
            if not sample.success:
                warmup_failures.append(sample.failure_reason or "benchmark_warmup_failed")

        measured_cases = [case for _ in range(plan.execution.repeats) for case in plan.cases]
        random.Random(plan.execution.seed).shuffle(measured_cases)
        pending: asyncio.Queue[tuple[int, BenchmarkCase]] = asyncio.Queue()
        for index, case in enumerate(measured_cases):
            pending.put_nowait((index, case))
        samples_by_index: list[BenchmarkSample | None] = [None] * len(measured_cases)

        async def worker() -> None:
            while True:
                try:
                    index, case = pending.get_nowait()
                except asyncio.QueueEmpty:
                    return
                samples_by_index[index] = await _capture_sample(transport, plan, case)
                pending.task_done()

        started = monotonic()
        worker_count = min(plan.execution.concurrency, len(measured_cases))
        await asyncio.gather(*(worker() for _ in range(worker_count)))
        measured_wall_seconds = max(monotonic() - started, 0.000001)
        if any(sample is None for sample in samples_by_index):
            raise InferenceBenchmarkError("benchmark_worker_did_not_complete")
        samples = [sample for sample in samples_by_index if sample is not None]
        metrics = _metrics(samples, measured_wall_seconds)
        failed_gates = _failed_gates(plan, metrics, warmup_failures)
        report: dict[str, Any] = {
            "schema_version": BENCHMARK_REPORT_SCHEMA,
            "status": "PASSED" if not failed_gates else "FAILED",
            "generated_at": (now or datetime.now(UTC)).isoformat(),
            "benchmark_id": plan.benchmark_id,
            "release_id": plan.release_id,
            "plan_digest": plan.plan_digest(),
            "comparison_contract_digest": plan.comparison_contract_digest(),
            "runtime": {
                "profile": plan.runtime.profile,
                "engine": plan.runtime.engine,
                "image_digest": plan.runtime.image_digest,
                "endpoint_digest": content_digest(plan.runtime.endpoint_url.rstrip("/")),
            },
            "model": plan.model.model_dump(mode="json"),
            "hardware": plan.hardware.model_dump(mode="json"),
            "dataset_snapshot_id": plan.dataset_snapshot_id,
            "workload_digest": plan.workload_digest,
            "execution": plan.execution.model_dump(mode="json"),
            "thresholds": plan.thresholds.model_dump(mode="json"),
            "warmup": {
                "request_count": plan.execution.warmup_requests,
                "failure_count": len(warmup_failures),
                "failure_reasons": dict(sorted(Counter(warmup_failures).items())),
            },
            "metrics": metrics,
            "failed_gates": failed_gates,
            "samples": [asdict(sample) for sample in samples],
        }
        report["report_digest"] = benchmark_report_digest(report)
        return report


def compare_runtime_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    verified = [verify_benchmark_report(report) for report in reports]
    if len(verified) < 2:
        raise InferenceBenchmarkError("benchmark_comparison_requires_two_reports")
    contracts = {report["comparison_contract_digest"] for report in verified}
    if len(contracts) != 1:
        raise InferenceBenchmarkError("benchmark_comparison_contract_mismatch")
    engine_names = {report["runtime"]["engine"] for report in verified}
    if len(engine_names) != len(verified):
        raise InferenceBenchmarkError("benchmark_comparison_engine_is_duplicated")

    eligible = [report for report in verified if report["status"] == "PASSED"]
    ranked = sorted(
        eligible,
        key=lambda report: (
            _required_metric(report, "e2e_latency_ms", "p95"),
            _required_metric(report, "ttft_ms", "p95"),
            -_required_number(report["metrics"], "completion_tokens_per_second"),
            report["runtime"]["engine"],
        ),
    )
    conclusive = len(ranked) >= 2
    recommendation = (
        {
            "engine": ranked[0]["runtime"]["engine"],
            "profile": ranked[0]["runtime"]["profile"],
            "image_digest": ranked[0]["runtime"]["image_digest"],
            "report_digest": ranked[0]["report_digest"],
            "decision_scope": "benchmark_recommendation_only",
        }
        if conclusive
        else None
    )
    comparison: dict[str, Any] = {
        "schema_version": BENCHMARK_COMPARISON_SCHEMA,
        "status": "PASSED" if conclusive else "INCONCLUSIVE",
        "generated_at": datetime.now(UTC).isoformat(),
        "comparison_contract_digest": next(iter(contracts)),
        "report_digests": [report["report_digest"] for report in verified],
        "eligible_report_count": len(ranked),
        "ranking": [
            {
                "rank": index,
                "engine": report["runtime"]["engine"],
                "profile": report["runtime"]["profile"],
                "report_digest": report["report_digest"],
                "p95_e2e_latency_ms": _required_metric(report, "e2e_latency_ms", "p95"),
                "p95_ttft_ms": _required_metric(report, "ttft_ms", "p95"),
                "completion_tokens_per_second": _required_number(
                    report["metrics"], "completion_tokens_per_second"
                ),
            }
            for index, report in enumerate(ranked, start=1)
        ],
        "ineligible": [
            {
                "engine": report["runtime"]["engine"],
                "report_digest": report["report_digest"],
                "failed_gates": report["failed_gates"],
            }
            for report in verified
            if report["status"] != "PASSED"
        ],
        "recommendation": recommendation,
    }
    comparison["comparison_digest"] = benchmark_comparison_digest(comparison)
    return comparison


def benchmark_report_digest(report: Mapping[str, Any]) -> Sha256Digest:
    return content_digest({key: value for key, value in report.items() if key != "report_digest"})


def benchmark_comparison_digest(comparison: Mapping[str, Any]) -> Sha256Digest:
    return content_digest(
        {key: value for key, value in comparison.items() if key != "comparison_digest"}
    )


def verify_benchmark_report(report: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(report)
    if value.get("schema_version") != BENCHMARK_REPORT_SCHEMA:
        raise InferenceBenchmarkError("benchmark_report_schema_is_not_supported")
    if value.get("report_digest") != benchmark_report_digest(value):
        raise InferenceBenchmarkError("benchmark_report_digest_mismatch")
    if value.get("status") not in {"PASSED", "FAILED"}:
        raise InferenceBenchmarkError("benchmark_report_status_is_invalid")
    if not _is_sha256(value.get("comparison_contract_digest")):
        raise InferenceBenchmarkError("benchmark_comparison_contract_digest_is_invalid")
    runtime = value.get("runtime")
    metrics = value.get("metrics")
    if not isinstance(runtime, dict) or not isinstance(metrics, dict):
        raise InferenceBenchmarkError("benchmark_report_is_incomplete")
    if runtime.get("engine") not in _PROFILE_ENGINES.values():
        raise InferenceBenchmarkError("benchmark_report_engine_is_invalid")
    _required_number(metrics, "completion_tokens_per_second")
    if value["status"] == "PASSED":
        _required_metric(value, "e2e_latency_ms", "p95")
        _required_metric(value, "ttft_ms", "p95")
    return value


def content_digest(value: Any) -> Sha256Digest:
    return f"sha256:{sha256(_canonical_json(value)).hexdigest()}"


def write_benchmark_document(path: str, document: Mapping[str, Any]) -> None:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(target)


async def _capture_sample(
    transport: BenchmarkTransport,
    plan: InferenceBenchmarkPlan,
    case: BenchmarkCase,
) -> BenchmarkSample:
    started = monotonic()
    try:
        sample = await transport.complete(plan, case)
        if sample.case_id != case.case_id:
            raise InferenceBenchmarkError("benchmark_transport_case_mismatch")
        _validate_sample(sample)
        return sample
    except InferenceBenchmarkError as exc:
        reason = str(exc)[:255] or "benchmark_request_failed"
    except Exception:
        reason = "benchmark_transport_unexpected_failure"
    return BenchmarkSample(
        case_id=case.case_id,
        success=False,
        structured_output_valid=False,
        prompt_tokens=0,
        completion_tokens=0,
        ttft_ms=None,
        tpot_ms=None,
        e2e_latency_ms=(monotonic() - started) * 1000,
        finish_reason=None,
        response_hash=None,
        failure_reason=reason,
    )


def _metrics(samples: Sequence[BenchmarkSample], wall_seconds: float) -> dict[str, Any]:
    successes = [sample for sample in samples if sample.success]
    sample_count = len(samples)
    success_count = len(successes)
    structured_success_count = sum(sample.structured_output_valid for sample in successes)
    prompt_tokens = sum(sample.prompt_tokens for sample in successes)
    completion_tokens = sum(sample.completion_tokens for sample in successes)
    return {
        "sample_count": sample_count,
        "success_count": success_count,
        "failure_count": sample_count - success_count,
        "request_success_rate": success_count / sample_count,
        "structured_success_rate": structured_success_count / sample_count,
        "error_rate": (sample_count - success_count) / sample_count,
        "failure_reasons": dict(
            sorted(
                Counter(
                    sample.failure_reason or "benchmark_request_failed"
                    for sample in samples
                    if not sample.success
                ).items()
            )
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "requests_per_second": success_count / wall_seconds,
        "completion_tokens_per_second": completion_tokens / wall_seconds,
        "measured_wall_seconds": wall_seconds,
        "ttft_ms": _percentiles(
            [sample.ttft_ms for sample in successes if sample.ttft_ms is not None]
        ),
        "tpot_ms": _percentiles(
            [sample.tpot_ms for sample in successes if sample.tpot_ms is not None]
        ),
        "e2e_latency_ms": _percentiles([sample.e2e_latency_ms for sample in successes]),
    }


def _validate_sample(sample: BenchmarkSample) -> None:
    numbers = (sample.e2e_latency_ms,)
    optional_numbers = (sample.ttft_ms, sample.tpot_ms)
    if any(not math.isfinite(value) or value < 0 for value in numbers):
        raise InferenceBenchmarkError("benchmark_transport_sample_is_invalid")
    if any(
        value is not None and (not math.isfinite(value) or value < 0)
        for value in optional_numbers
    ):
        raise InferenceBenchmarkError("benchmark_transport_sample_is_invalid")
    if sample.prompt_tokens < 0 or sample.completion_tokens < 0:
        raise InferenceBenchmarkError("benchmark_transport_sample_is_invalid")
    if sample.success and (
        sample.ttft_ms is None
        or sample.tpot_ms is None
        or sample.completion_tokens <= 0
        or not _is_sha256(sample.response_hash)
        or sample.failure_reason is not None
    ):
        raise InferenceBenchmarkError("benchmark_transport_sample_is_invalid")
    if not sample.success and sample.failure_reason is None:
        raise InferenceBenchmarkError("benchmark_transport_sample_is_invalid")


def _failed_gates(
    plan: InferenceBenchmarkPlan,
    metrics: Mapping[str, Any],
    warmup_failures: Sequence[str],
) -> list[str]:
    failed: list[str] = []
    thresholds = plan.thresholds
    if warmup_failures:
        failed.append("warmup_requests_succeeded")
    if metrics["request_success_rate"] < thresholds.minimum_request_success_rate:
        failed.append("request_success_rate")
    if metrics["structured_success_rate"] < thresholds.minimum_structured_success_rate:
        failed.append("structured_success_rate")
    if metrics["error_rate"] > thresholds.maximum_error_rate:
        failed.append("error_rate")
    p95_latency = metrics["e2e_latency_ms"]["p95"]
    if p95_latency is None or p95_latency > thresholds.maximum_p95_latency_ms:
        failed.append("p95_latency")
    return failed


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    ordered = sorted(values)
    return {
        name: ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]
        for name, quantile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
    }


def _structured_output_valid(content: str, schema: Mapping[str, Any]) -> bool:
    try:
        parsed = json.loads(content)
        validate(instance=parsed, schema=dict(schema))
    except (json.JSONDecodeError, SchemaError, ValidationError):
        return False
    return isinstance(parsed, dict)


def _completion_endpoint(base_url: str, allow_plain_http: bool) -> str:
    parsed = urlparse(base_url)
    allowed = {"https"} | ({"http"} if allow_plain_http else set())
    if (
        parsed.scheme not in allowed
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise InferenceBenchmarkError("benchmark_endpoint_invalid")
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def _required_metric(report: Mapping[str, Any], metric: str, percentile: str) -> float:
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise InferenceBenchmarkError("benchmark_report_metrics_are_invalid")
    values = metrics.get(metric)
    if not isinstance(values, Mapping):
        raise InferenceBenchmarkError("benchmark_report_metrics_are_invalid")
    return _required_number(values, percentile)


def _required_number(values: Mapping[str, Any], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise InferenceBenchmarkError("benchmark_report_metrics_are_invalid")
    return float(value)


def _non_negative_int(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InferenceBenchmarkError("benchmark_usage_is_invalid")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    return all(character in "0123456789abcdef" for character in value[7:])
