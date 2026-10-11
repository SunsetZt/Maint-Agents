"""SLO evaluation, error budgets, release safety posture, and runbook catalog."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from industrial_ops_agent.operations.prometheus import (
    PrometheusAlert,
    PrometheusReader,
    PrometheusUnavailable,
)


class DataSourceStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


class SloStatus(StrEnum):
    HEALTHY = "HEALTHY"
    BREACHED = "BREACHED"
    NO_DATA = "NO_DATA"


@dataclass(frozen=True, slots=True)
class SloDefinition:
    slo_id: str
    name: str
    owner: str
    objective: float
    comparison: str
    unit: str
    window: str
    query: str
    runbook_id: str
    error_budget: bool = False


@dataclass(frozen=True, slots=True)
class SloObservation:
    slo_id: str
    name: str
    owner: str
    objective: float
    comparison: str
    unit: str
    window: str
    status: SloStatus
    current_value: float | None
    error_budget_remaining_fraction: float | None
    runbook_id: str


@dataclass(frozen=True, slots=True)
class RunbookDefinition:
    runbook_id: str
    title: str
    owner: str
    user_impact: str
    confirmation_queries: tuple[str, ...]
    safety_containment: tuple[str, ...]
    recovery: tuple[str, ...]
    rollback: tuple[str, ...]
    consistency_checks: tuple[str, ...]
    escalation: str
    postmortem: str


@dataclass(frozen=True, slots=True)
class ActiveAlert:
    name: str
    severity: str
    state: str
    active_at: datetime
    summary: str
    runbook_id: str
    owner: str
    security_hard_gate: bool


class RecoveryReleaseGate(Protocol):
    def release_gate_reasons(
        self,
        tenant_id: str,
        *,
        now: datetime | None = None,
    ) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class OperationsSnapshot:
    policy_version: str
    generated_at: datetime
    data_source_status: DataSourceStatus
    high_risk_release_allowed: bool
    release_gate_reasons: tuple[str, ...]
    slos: tuple[SloObservation, ...]
    alerts: tuple[ActiveAlert, ...]
    runbooks: tuple[RunbookDefinition, ...]


SLO_DEFINITIONS = (
    SloDefinition(
        slo_id="business-api-availability-30d",
        name="业务入口可用性",
        owner="platform-sre",
        objective=0.999,
        comparison=">=",
        unit="ratio",
        window="30d",
        query=(
            '1 - (sum(increase(ioap_http_requests_total{status=~"5.."}[30d])) '
            "/ sum(increase(ioap_http_requests_total[30d])))"
        ),
        runbook_id="api-slo-burn",
        error_budget=True,
    ),
    SloDefinition(
        slo_id="ordinary-api-latency-p95-5m",
        name="普通 API P95 延迟",
        owner="platform-sre",
        objective=0.3,
        comparison="<=",
        unit="seconds",
        window="5m",
        query=(
            "histogram_quantile(0.95, sum by (le) "
            "(rate(ioap_http_request_duration_seconds_bucket{path!~"
            '"/api/v1/(diagnoses|speech|realtime).*"}[5m])))'
        ),
        runbook_id="api-slo-burn",
    ),
    SloDefinition(
        slo_id="inference-availability-30d",
        name="模型推理可用性",
        owner="ml-platform",
        objective=0.999,
        comparison=">=",
        unit="ratio",
        window="30d",
        query=(
            'sum(increase(ioap_inference_requests_total{status="success"}[30d])) '
            "/ sum(increase(ioap_inference_requests_total[30d]))"
        ),
        runbook_id="inference-degradation",
        error_budget=True,
    ),
    SloDefinition(
        slo_id="cdc-freshness-p95-5m",
        name="CDC 事件新鲜度 P95",
        owner="data-platform",
        objective=10.0,
        comparison="<=",
        unit="seconds",
        window="5m",
        query=(
            "histogram_quantile(0.95, sum by (le) "
            "(rate(ioap_cdc_event_freshness_seconds_bucket[5m])))"
        ),
        runbook_id="event-pipeline-lag",
    ),
    SloDefinition(
        slo_id="dataset-dag-on-time-30d",
        name="数据集 DAG 准时完成率",
        owner="data-platform",
        objective=0.99,
        comparison=">=",
        unit="ratio",
        window="30d",
        query=(
            'sum(increase(ioap_dataset_dag_runs_total{status="on_time"}[30d])) '
            "/ sum(increase(ioap_dataset_dag_runs_total[30d]))"
        ),
        runbook_id="event-pipeline-lag",
        error_budget=True,
    ),
)


RUNBOOKS = (
    RunbookDefinition(
        "api-slo-burn",
        "API 错误预算燃烧",
        "platform-sre",
        "登录、工单、Agent 或治理入口可能超时或返回错误。",
        ("确认 5xx 比例、P95/P99 与受影响路由", "按版本、可用区和依赖拆分异常"),
        ("冻结高风险发布", "保留读路径，限制非必要批任务与重试风暴"),
        ("恢复异常实例或依赖", "容量不足时按已验证上限扩容"),
        ("回滚最近一次应用或配置变更",),
        ("核对幂等键、事务状态和消息积压", "确认错误预算停止继续燃烧"),
        "15 分钟未恢复升级至平台负责人，30 分钟通知业务负责人。",
        "24 小时内完成时间线、根因、用户影响和防复发项。",
    ),
    RunbookDefinition(
        "inference-degradation",
        "模型推理降级",
        "ml-platform",
        "诊断、识别、语音或 Agent 回答变慢、失败或转人工。",
        ("按模型、版本、后端检查成功率与首事件延迟", "检查 Guardrail 阻断与配额拒绝"),
        ("暂停模型晋级", "切至已验证基线模型并保持人工升级通道"),
        ("恢复健康副本、路由和模型缓存",),
        ("回滚模型版本、推理引擎或路由权重",),
        ("比对推理记录、引用证据与工单状态",),
        "关键诊断中断立即升级模型平台与售后值班负责人。",
        "复盘模型、引擎、数据和容量因素，补充回归评测。",
    ),
    RunbookDefinition(
        "event-pipeline-lag",
        "事件与数据管道积压",
        "data-platform",
        "反馈候选、标注、快照或训练数据出现延迟。",
        ("检查 Kafka consumer lag、Inbox 状态和 CDC 新鲜度", "检查 DAG 最近失败和血缘发送状态"),
        ("停止产生不可控下游副作用", "保留 Inbox 并禁止跳过治理门禁"),
        ("修复消费者或 DAG 后从安全位点重放",),
        ("回滚连接器、路由或任务代码",),
        ("核对重复/乱序事件、候选数量、Parquet 清单与血缘",),
        "积压超过 30 分钟升级数据平台；影响训练发布时同步模型负责人。",
        "记录积压范围、重放位点、重复处理证明和数据一致性结论。",
    ),
    RunbookDefinition(
        "temporal-workflow-stuck",
        "Temporal 工作流卡住",
        "platform-sre",
        "识别、诊断、工单或长流程无法推进。",
        ("检查 task queue backlog、workflow age 与失败 activity",),
        ("禁止直接改业务数据库推进状态", "隔离有副作用的重试"),
        ("恢复 worker 后通过 Temporal 重试或补偿",),
        ("回滚 worker 版本并保持 workflow 兼容性",),
        ("核对工作流历史、数据库状态与外部系统副作用",),
        "高风险动作卡住立即升级平台与业务所有者。",
        "保留 workflow history 并补充确定性/重放测试。",
    ),
    RunbookDefinition(
        "gpu-health",
        "GPU 推理节点异常",
        "ml-platform",
        "模型吞吐下降、推理失败或排队时间增加。",
        ("检查 DCGM XID、温度、显存、功耗和节点事件",),
        ("摘除异常节点并停止新调度", "避免自动重启掩盖硬件故障"),
        ("迁移负载并按厂商流程恢复节点",),
        ("回滚引擎参数或模型并发配置",),
        ("确认推理结果、路由权重和 GPU 健康恢复",),
        "XID 或持续过温立即升级基础设施与硬件支持。",
        "关联驱动、CUDA、模型、节点和硬件批次完成复盘。",
    ),
    RunbookDefinition(
        "security-hard-gate",
        "安全硬门禁触发",
        "security",
        "可能存在越权动作、敏感数据泄露或供应链完整性风险。",
        ("确认告警指纹、审计事件和受影响版本，不复制敏感正文",),
        ("立即冻结发布和高风险执行", "隔离凭据、版本或租户边界"),
        ("完成凭据轮换、策略修复和证据复核后人工解除",),
        ("回滚受影响模型、镜像、策略或配置",),
        ("核对审计链、数据边界、外部副作用和签名证据",),
        "立即升级安全负责人和事件指挥官。",
        "按安全事件流程保全证据、评估泄露范围并跟踪整改。",
    ),
    RunbookDefinition(
        "cost-anomaly",
        "模型与基础设施成本异常",
        "finops",
        "服务仍可能可用，但预算、配额或后续容量受到影响。",
        ("按租户、模型、令牌、GPU 和时间窗口确认成本偏差",),
        ("限制非必要批任务与异常调用方，不影响紧急售后",),
        ("修复重试、路由、配额或闲置资源",),
        ("回滚导致单位成本上升的模型或配置",),
        ("核对账单、用量记录、配额拒绝与业务请求量",),
        "超过日预算 20% 升级 FinOps、平台与业务所有者。",
        "复盘单位任务成本、异常来源和预算告警阈值。",
    ),
    RunbookDefinition(
        "operations-generic",
        "未分类生产告警",
        "platform-sre",
        "影响范围尚待确认。",
        ("使用告警名称和时间窗口定位指标，不依赖告警文本执行命令",),
        ("冻结相关高风险变更并保护数据一致性",),
        ("由对应服务所有者执行已审查恢复步骤",),
        ("回滚最近相关变更",),
        ("验证用户路径、状态机和审计证据",),
        "无法在 15 分钟内分类时升级事件指挥官。",
        "补齐专用告警标签与运行手册。",
    ),
)


class OperationsService:
    POLICY_VERSION = "operations-slo/v1"

    def __init__(
        self,
        prometheus: PrometheusReader,
        *,
        recovery_gate: RecoveryReleaseGate | None = None,
        recovery_tenant_id: str | None = None,
    ) -> None:
        self._prometheus = prometheus
        self._recovery_gate = recovery_gate
        self._recovery_tenant_id = recovery_tenant_id

    def snapshot(self, *, now: datetime | None = None) -> OperationsSnapshot:
        generated_at = (now or datetime.now(UTC)).astimezone(UTC)
        observations: list[SloObservation] = []
        source_status = DataSourceStatus.AVAILABLE
        source_failed = False
        for definition in SLO_DEFINITIONS:
            value: float | None = None
            if not source_failed:
                try:
                    value = self._prometheus.query_scalar(definition.query)
                except PrometheusUnavailable:
                    source_status = DataSourceStatus.UNAVAILABLE
                    source_failed = True
            observations.append(_observe(definition, value))

        raw_alerts: tuple[PrometheusAlert, ...] = ()
        if not source_failed:
            try:
                raw_alerts = self._prometheus.active_alerts()
            except PrometheusUnavailable:
                source_status = DataSourceStatus.DEGRADED
        alerts = tuple(ActiveAlert(**vars_from_alert(alert)) for alert in raw_alerts)

        reasons: list[str] = []
        if source_status is not DataSourceStatus.AVAILABLE:
            reasons.append("operations_evidence_unavailable")
        if any(item.status is SloStatus.NO_DATA for item in observations):
            reasons.append("required_slo_has_no_data")
        if any(
            item.status is SloStatus.BREACHED and item.error_budget_remaining_fraction == 0.0
            for item in observations
        ):
            reasons.append("error_budget_exhausted")
        if any(alert.security_hard_gate for alert in alerts if alert.state == "firing"):
            reasons.append("security_hard_gate_firing")
        if any(alert.severity == "critical" and alert.state == "firing" for alert in alerts):
            reasons.append("critical_alert_firing")
        if self._recovery_gate is not None and self._recovery_tenant_id is not None:
            try:
                reasons.extend(
                    self._recovery_gate.release_gate_reasons(
                        self._recovery_tenant_id,
                        now=generated_at,
                    )
                )
            except Exception:
                reasons.append("recovery_evidence_unavailable")
        return OperationsSnapshot(
            policy_version=self.POLICY_VERSION,
            generated_at=generated_at,
            data_source_status=source_status,
            high_risk_release_allowed=not reasons,
            release_gate_reasons=tuple(dict.fromkeys(reasons)),
            slos=tuple(observations),
            alerts=alerts,
            runbooks=RUNBOOKS,
        )


def _observe(definition: SloDefinition, value: float | None) -> SloObservation:
    status = SloStatus.NO_DATA
    budget: float | None = None
    if value is not None:
        meets = (
            value >= definition.objective
            if definition.comparison == ">="
            else value <= definition.objective
        )
        status = SloStatus.HEALTHY if meets else SloStatus.BREACHED
        if definition.error_budget:
            allowed_bad = 1.0 - definition.objective
            observed_bad = max(0.0, 1.0 - min(value, 1.0))
            budget = max(0.0, min(1.0, (allowed_bad - observed_bad) / allowed_bad))
    return SloObservation(
        slo_id=definition.slo_id,
        name=definition.name,
        owner=definition.owner,
        objective=definition.objective,
        comparison=definition.comparison,
        unit=definition.unit,
        window=definition.window,
        status=status,
        current_value=value,
        error_budget_remaining_fraction=budget,
        runbook_id=definition.runbook_id,
    )


def vars_from_alert(alert: PrometheusAlert) -> dict[str, object]:
    return {field: getattr(alert, field) for field in ActiveAlert.__dataclass_fields__}
