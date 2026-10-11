"use client";

import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Drawer,
  Form,
  Input,
  Modal,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
  message,
} from "antd";
import Link from "next/link";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  type PredictiveAlertCandidate,
  type PredictiveMaintenanceOverview,
  type PredictiveOutcomeInput,
  type RulForecast,
  type TelemetryWindow,
  buildRulDatasetSnapshot,
  buildTelemetryWindow,
  buildTelemetryDatasetSnapshot,
  createIncidentDraft,
  decidePredictiveAlert,
  detectTelemetryWindow,
  getPredictiveMaintenanceOverview,
  generateRulForecast,
  recordPredictiveOutcome,
  referPredictiveAlert,
  reviewRulForecast,
} from "@/lib/api/client";

const statusColors: Record<string, string> = {
  PENDING_CONFIRMATION: "orange",
  CONFIRMED: "blue",
  DISMISSED: "default",
  REFERRED: "green",
  READY: "green",
  DEGRADED: "orange",
  INSUFFICIENT: "red",
  ACCEPTED: "green",
  OUT_OF_ORDER_ACCEPTED: "gold",
  LATE_ACCEPTED: "gold",
  TOO_LATE: "red",
};

const statusLabels: Record<string, string> = {
  PENDING_CONFIRMATION: "待人工确认",
  CONFIRMED: "已确认",
  DISMISSED: "已排除",
  REFERRED: "已转入维护",
  READY: "可检测",
  DEGRADED: "缺失信号降级",
  INSUFFICIENT: "样本不足",
  ACCEPTED: "正常接收",
  OUT_OF_ORDER_ACCEPTED: "乱序已接收",
  LATE_ACCEPTED: "迟到已接收",
  TOO_LATE: "超水位线，仅留存",
};

type DecisionDialog = {
  candidate: PredictiveAlertCandidate;
  decision: "CONFIRMED" | "DISMISSED";
};

type OutcomeDialog = {
  candidate?: PredictiveAlertCandidate;
};

type RulReviewDialog = {
  forecast: RulForecast;
  decision: "ACCEPTED" | "REJECTED";
};

export default function PredictiveMaintenancePage() {
  const [overview, setOverview] = useState<PredictiveMaintenanceOverview>();
  const [legalActions, setLegalActions] = useState<string[]>([]);
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [busy, setBusy] = useState<string>();
  const [selected, setSelected] = useState<PredictiveAlertCandidate>();
  const [decisionDialog, setDecisionDialog] = useState<DecisionDialog>();
  const [decisionReason, setDecisionReason] = useState("");
  const [decisionNotes, setDecisionNotes] = useState("");
  const [outcomeDialog, setOutcomeDialog] = useState<OutcomeDialog>();
  const [rulReviewDialog, setRulReviewDialog] = useState<RulReviewDialog>();
  const [rulReviewReason, setRulReviewReason] = useState("");
  const [outcomeType, setOutcomeType] = useState<
    "TRUE_POSITIVE" | "FALSE_POSITIVE" | "MISSED_FAILURE"
  >("TRUE_POSITIVE");
  const [outcomeAssetId, setOutcomeAssetId] = useState("");
  const [failureAt, setFailureAt] = useState("");
  const [evidenceRef, setEvidenceRef] = useState("");
  const [evidenceDigest, setEvidenceDigest] = useState("");
  const [avoidedMinutes, setAvoidedMinutes] = useState("0");
  const [windowAssetId, setWindowAssetId] = useState("");
  const [windowStart, setWindowStart] = useState("");
  const [windowEnd, setWindowEnd] = useState("");
  const [includeTooLate, setIncludeTooLate] = useState(false);
  const [datasetStart, setDatasetStart] = useState("");
  const [datasetEnd, setDatasetEnd] = useState("");
  const [lastOperation, setLastOperation] = useState<string>();

  const allowed = useMemo(() => new Set(legalActions), [legalActions]);

  const load = useCallback(async () => {
    try {
      const result = await getPredictiveMaintenanceOverview();
      setOverview(result.overview);
      setLegalActions(result.legalActions);
      setRequestId(result.requestId);
      setError(undefined);
      setSelected((current) =>
        current
          ? result.overview.alert_candidates.find(
              (item) => item.alert_candidate_id === current.alert_candidate_id,
            )
          : undefined,
      );
    } catch (cause) {
      setError(cause);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function submitDecision() {
    if (!decisionDialog || !decisionReason.trim()) return;
    const key = `decision:${decisionDialog.candidate.alert_candidate_id}`;
    setBusy(key);
    setError(undefined);
    try {
      await decidePredictiveAlert(
        decisionDialog.candidate.alert_candidate_id,
        decisionDialog.candidate.version,
        {
          decision: decisionDialog.decision,
          reason_code: decisionReason.trim(),
          notes: decisionNotes.trim() || null,
        },
      );
      setDecisionDialog(undefined);
      setDecisionReason("");
      setDecisionNotes("");
      setLastOperation(
        decisionDialog.decision === "CONFIRMED"
          ? "候选告警已由人工确认；尚未自动创建工单。"
          : "候选告警已排除，原因已进入可追溯反馈。",
      );
      await load();
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function createMaintenanceReferral(candidate: PredictiveAlertCandidate) {
    const key = `referral:${candidate.alert_candidate_id}`;
    setBusy(key);
    setError(undefined);
    try {
      const idempotencyKey =
        globalThis.crypto?.randomUUID?.() ?? `predictive-draft-${Date.now()}`;
      const draftResult = await createIncidentDraft(
        {
          asset_id: candidate.asset_id,
          description: [
            "预测性维护候选已由领域人员确认。",
            `告警候选：${candidate.alert_candidate_id}`,
            `分析窗口：${candidate.window_start} - ${candidate.window_end}`,
            `异常分数：${candidate.anomaly_score.toFixed(3)}`,
            `证据：${candidate.supporting_signal_refs.join("、")}`,
            "请补充现场现象和媒体证据后，再按既有流程提交 Incident。",
          ].join("\n"),
        },
        idempotencyKey,
      );
      await referPredictiveAlert(
        candidate.alert_candidate_id,
        candidate.version,
        draftResult.draft.draft_id,
      );
      setLastOperation(`已创建并关联 Incident 草稿 ${draftResult.draft.draft_id}。`);
      message.success("已转入受控 Incident 草稿流程");
      await load();
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function runDetection(window: TelemetryWindow) {
    setBusy(`detect:${window.window_id}`);
    setError(undefined);
    try {
      const result = await detectTelemetryWindow(window.window_id);
      const detector = result.result.detector_kind === "TIMESERIES_TRANSFORMER"
        ? `生产时序模型 ${result.result.model_release_id}`
        : result.result.fallback_reason
          ? `规则降级（${result.result.fallback_reason}）`
          : "规则基线";
      setLastOperation(
        result.result.candidate
          ? `${detector} 检测产生候选 ${result.result.candidate.alert_candidate_id}，等待人工确认。`
          : `${detector} 异常分数 ${result.result.anomaly_score.toFixed(3)} 未达到候选阈值。`,
      );
      await load();
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function generateForecast(candidate: PredictiveAlertCandidate) {
    const key = `rul:${candidate.alert_candidate_id}`;
    setBusy(key);
    setError(undefined);
    try {
      const result = await generateRulForecast(
        candidate.alert_candidate_id,
        candidate.version,
        globalThis.crypto?.randomUUID?.() ?? `rul-forecast-${Date.now()}`,
      );
      setLastOperation(
        `RUL 候选 ${result.forecast.rul_forecast_id} 已按 ${result.forecast.historical_failure_count} 条真实故障提前量生成，等待不同领域专家复核。`,
      );
      await load();
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function submitRulReview() {
    if (!rulReviewDialog || !rulReviewReason.trim()) return;
    setBusy(`rul-review:${rulReviewDialog.forecast.rul_forecast_id}`);
    setError(undefined);
    try {
      await reviewRulForecast(
        rulReviewDialog.forecast.rul_forecast_id,
        rulReviewDialog.forecast.version,
        {
          decision: rulReviewDialog.decision,
          reason: rulReviewReason.trim(),
        },
      );
      setLastOperation(
        rulReviewDialog.decision === "ACCEPTED"
          ? "RUL 候选已由独立领域专家接受为维护计划参考；未自动创建工单或执行设备动作。"
          : "RUL 候选已由独立领域专家拒绝，原始候选和复核原因均保留。",
      );
      setRulReviewDialog(undefined);
      setRulReviewReason("");
      await load();
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function submitWindow(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!windowAssetId || !windowStart || !windowEnd) return;
    setBusy("build-window");
    setError(undefined);
    try {
      const result = await buildTelemetryWindow({
        asset_id: windowAssetId.trim(),
        schema_version: "rotating-equipment.telemetry.v1",
        window_start: new Date(windowStart).toISOString(),
        window_end: new Date(windowEnd).toISOString(),
        include_too_late: includeTooLate,
      });
      setLastOperation(
        `特征快照 ${result.window.feature_snapshot_id} 已生成，质量状态 ${result.window.quality_status}。`,
      );
      await load();
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function submitDataset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!datasetStart || !datasetEnd) return;
    setBusy("build-telemetry-dataset");
    setError(undefined);
    try {
      const result = await buildTelemetryDatasetSnapshot({
        window_start: new Date(datasetStart).toISOString(),
        window_end: new Date(datasetEnd).toISOString(),
        engine: "local",
      });
      setLastOperation(
        result.snapshot.training_eligible
          ? `时序训练快照 ${result.snapshot.snapshot_id} 已发布，共 ${result.snapshot.row_count} 个窗口，可在实验中心登记 TIMESERIES_TRANSFORMER。`
          : `时序快照 ${result.snapshot.snapshot_id} 已发布但未达到训练门槛：${result.snapshot.blocker_codes.join("、")}。`,
      );
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function submitRulDataset() {
    if (!datasetStart || !datasetEnd) return;
    setBusy("build-rul-dataset");
    setError(undefined);
    try {
      const result = await buildRulDatasetSnapshot({
        window_start: new Date(datasetStart).toISOString(),
        window_end: new Date(datasetEnd).toISOString(),
        engine: "local",
      });
      setLastOperation(
        result.snapshot.training_eligible
          ? `RUL 监督训练快照 ${result.snapshot.snapshot_id} 已发布，共 ${result.snapshot.row_count} 条真实故障提前量，可在实验中心登记 RUL_TRANSFORMER。`
          : `RUL 快照 ${result.snapshot.snapshot_id} 已冻结但未达到训练门槛：${result.snapshot.blocker_codes.join("、")}。`,
      );
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  function openOutcome(candidate?: PredictiveAlertCandidate) {
    setOutcomeDialog({ candidate });
    setOutcomeType(candidate ? "TRUE_POSITIVE" : "MISSED_FAILURE");
    setOutcomeAssetId(candidate?.asset_id ?? "");
    setFailureAt("");
    setEvidenceRef("");
    setEvidenceDigest("");
    setAvoidedMinutes("0");
  }

  async function submitOutcome() {
    if (!outcomeDialog || !outcomeAssetId.trim() || !evidenceRef.trim()) return;
    if (!/^sha256:[0-9a-f]{64}$/.test(evidenceDigest)) return;
    const requiresFailure = outcomeType !== "FALSE_POSITIVE";
    if (requiresFailure && !failureAt) return;
    setBusy("record-outcome");
    setError(undefined);
    try {
      const now = new Date().toISOString();
      const input: PredictiveOutcomeInput = {
        asset_id: outcomeAssetId.trim(),
        alert_candidate_id: outcomeDialog.candidate?.alert_candidate_id ?? null,
        outcome_type: outcomeType,
        observed_at: now,
        failure_observed_at: requiresFailure ? new Date(failureAt).toISOString() : null,
        work_order_id: null,
        avoided_downtime_minutes: Number.parseInt(avoidedMinutes || "0", 10),
        evidence_ref: evidenceRef.trim(),
        evidence_digest: evidenceDigest,
      };
      await recordPredictiveOutcome(
        input,
        globalThis.crypto?.randomUUID?.() ?? `predictive-outcome-${Date.now()}`,
      );
      setOutcomeDialog(undefined);
      setLastOperation("真实维护结果已记录，误报、漏报和提前量指标已重新计算。");
      await load();
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  const metrics = overview?.metrics;
  const forecastedAlertIds = useMemo(
    () => new Set(overview?.rul_forecasts.map((item) => item.alert_candidate_id) ?? []),
    [overview?.rul_forecasts],
  );
  const retrainingRecommendations = useMemo(
    () => overview?.rul_release_calibrations.filter((item) => item.retraining_recommended) ?? [],
    [overview?.rul_release_calibrations],
  );

  return (
    <AppShell>
      <div className="page-stack">
        <div>
          <Typography.Title level={2}>设备遥测与预测性维护</Typography.Title>
          <Typography.Paragraph type="secondary">
            Kafka 遥测经过 Schema 校验、去重、水位线和事件时间窗处理后，规则或模型只能产生
            AlertCandidate。高分不会直接停机或创建工单，必须由领域人员确认后进入既有 Incident、诊断、审批与工单流程。
          </Typography.Paragraph>
        </div>

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!overview && !error ? <LoadingState label="正在加载预测维护证据" /> : null}
        {lastOperation ? <Alert showIcon type="success" message={lastOperation} closable /> : null}

        {overview ? (
          <>
            <Alert
              showIcon
              type={metrics?.rul_claim_allowed ? "success" : "warning"}
              message={
                metrics?.rul_claim_allowed
                  ? "故障标签达到当前 RUL 研究门槛"
                  : "当前仅提供规则、趋势和异常辅助证据"
              }
              description={
                metrics?.rul_claim_allowed
                  ? `仍需独立离线评测和生产门禁；当前漂移状态：${metrics.drift_status}。`
                  : `故障结果与已标注告警不足，不宣称能够准确预测剩余寿命；当前漂移状态：${metrics?.drift_status ?? "NO_BASELINE"}。`
              }
            />

            <Row gutter={[16, 16]}>
              <Col xs={24} sm={12} xl={4}>
                <Card><Statistic title="候选告警" value={metrics?.alert_candidate_count ?? 0} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={4}>
                <Card><Statistic title="待人工确认" value={metrics?.pending_confirmation_count ?? 0} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={4}>
                <Card><Statistic title="工单转化率" value={percent(metrics?.work_order_conversion_rate)} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={4}>
                <Card><Statistic title="误报率" value={percent(metrics?.false_positive_rate)} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={4}>
                <Card><Statistic title="漏报率" value={percent(metrics?.miss_rate)} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={4}>
                <Card><Statistic title="中位提前量" value={minutes(metrics?.median_lead_time_minutes)} /></Card>
              </Col>
            </Row>

            <Card
              title="AlertCandidate 人工确认队列"
              extra={
                allowed.has("predictive_outcome.record") ? (
                  <Button onClick={() => openOutcome()}>记录漏报故障</Button>
                ) : null
              }
            >
              {overview.alert_candidates.length === 0 ? (
                <EmptyState description="当前没有达到阈值的候选告警" />
              ) : (
                <Table<PredictiveAlertCandidate>
                  rowKey="alert_candidate_id"
                  dataSource={overview.alert_candidates}
                  pagination={{ pageSize: 10 }}
                  scroll={{ x: 1200 }}
                  columns={[
                    { title: "设备", dataIndex: "asset_id", width: 160 },
                    {
                      title: "异常分数",
                      width: 110,
                      render: (_, item) => item.anomaly_score.toFixed(3),
                    },
                    {
                      title: "窗口",
                      width: 210,
                      render: (_, item) => `${formatTime(item.window_start)} 至 ${formatTime(item.window_end)}`,
                    },
                    {
                      title: "状态",
                      dataIndex: "status",
                      width: 130,
                      render: (value: string) => <Tag color={statusColors[value]}>{statusLabels[value] ?? value}</Tag>,
                    },
                    {
                      title: "触发依据",
                      width: 230,
                      render: (_, item) => item.explanation_codes.join("、"),
                    },
                    {
                      title: "操作",
                      width: 330,
                      fixed: "right",
                      render: (_, item) => (
                        <Space wrap>
                          <Button size="small" onClick={() => setSelected(item)}>详情</Button>
                          {item.status === "PENDING_CONFIRMATION" && allowed.has("alert_candidate.decide") ? (
                            <>
                              <Button
                                size="small"
                                type="primary"
                                onClick={() => setDecisionDialog({ candidate: item, decision: "CONFIRMED" })}
                              >
                                人工确认
                              </Button>
                              <Button
                                size="small"
                                onClick={() => setDecisionDialog({ candidate: item, decision: "DISMISSED" })}
                              >
                                排除
                              </Button>
                            </>
                          ) : null}
                          {item.status === "CONFIRMED" && allowed.has("alert_candidate.refer") ? (
                            <Button
                              size="small"
                              loading={busy === `referral:${item.alert_candidate_id}`}
                              onClick={() => void createMaintenanceReferral(item)}
                            >
                              创建维护草稿
                            </Button>
                          ) : null}
                          {item.status !== "PENDING_CONFIRMATION"
                            && item.status !== "DISMISSED"
                            && metrics?.rul_claim_allowed
                            && allowed.has("rul_forecast.generate")
                            && !forecastedAlertIds.has(item.alert_candidate_id) ? (
                              <Button
                                size="small"
                                loading={busy === `rul:${item.alert_candidate_id}`}
                                onClick={() => void generateForecast(item)}
                              >
                                生成 RUL 候选
                              </Button>
                            ) : null}
                          {item.status !== "PENDING_CONFIRMATION" && allowed.has("predictive_outcome.record") ? (
                            <Button size="small" onClick={() => openOutcome(item)}>记录结果</Button>
                          ) : null}
                        </Space>
                      ),
                    },
                  ]}
                />
              )}
            </Card>

            <Card
              title="剩余寿命（RUL）候选与独立复核"
              extra={<Tag color="purple">经验提前量基线 v1</Tag>}
            >
              <Alert
                showIcon
                type="info"
                style={{ marginBottom: 16 }}
                message="RUL 是维护计划参考，不是自动维修指令"
                description="区间来自同一遥测 Schema 下已人工标注的真实故障提前量 P10/P50/P90。只有通过研究门槛才生成，且必须由不同领域专家复核；接受后仍不自动停机、创建 Incident、工单或备件动作。"
              />
              {overview.rul_forecasts.length === 0 ? (
                <EmptyState description="尚无达到研究门槛并提交复核的 RUL 候选" />
              ) : (
                <Table<RulForecast>
                  rowKey="rul_forecast_id"
                  dataSource={overview.rul_forecasts}
                  pagination={{ pageSize: 10 }}
                  scroll={{ x: 1250 }}
                  columns={[
                    { title: "设备", dataIndex: "asset_id", width: 150 },
                    {
                      title: "预计剩余时间",
                      width: 170,
                      render: (_, item) => minutes(item.estimate_minutes),
                    },
                    {
                      title: "P10–P90 区间",
                      width: 220,
                      render: (_, item) => `${minutes(item.lower_bound_minutes)} – ${minutes(item.upper_bound_minutes)}`,
                    },
                    {
                      title: "建议复检时间",
                      dataIndex: "recommended_inspection_at",
                      width: 180,
                      render: formatTime,
                    },
                    {
                      title: "历史故障样本",
                      dataIndex: "historical_failure_count",
                      width: 120,
                    },
                    {
                      title: "状态",
                      dataIndex: "status",
                      width: 140,
                      render: (value: string) => (
                        <Tag color={rulStatusColor(value)}>{rulStatusLabel(value)}</Tag>
                      ),
                    },
                    {
                      title: "证据摘要",
                      dataIndex: "cohort_digest",
                      width: 220,
                      ellipsis: true,
                    },
                    {
                      title: "操作",
                      fixed: "right",
                      width: 170,
                      render: (_, item) => item.status === "PENDING_REVIEW"
                        && allowed.has("rul_forecast.review") ? (
                          <Space>
                            <Button
                              size="small"
                              type="primary"
                              onClick={() => setRulReviewDialog({ forecast: item, decision: "ACCEPTED" })}
                            >接受</Button>
                            <Button
                              size="small"
                              onClick={() => setRulReviewDialog({ forecast: item, decision: "REJECTED" })}
                            >拒绝</Button>
                          </Space>
                        ) : "—",
                    },
                  ]}
                />
              )}
            </Card>

            <Card
              title="RUL 在线结果校准与再训练门禁"
              extra={retrainingRecommendations.length > 0
                ? <Tag color="red">{retrainingRecommendations.length} 个 Release 建议再训练</Tag>
                : <Tag color="blue">仅由真实故障结果驱动</Tag>}
            >
              <Alert
                showIcon
                type={retrainingRecommendations.length > 0 ? "warning" : "info"}
                style={{ marginBottom: 16 }}
                message={retrainingRecommendations.length > 0
                  ? "近期 RUL 准确率或区间覆盖已越过治理门槛"
                  : "真实故障会自动形成不可变校准证据"}
                description={retrainingRecommendations.length > 0
                  ? "建议在下方发布新的 RUL 监督快照并进入实验中心重新训练、独立评测和晋级；系统不会自动训练、切换模型或创建维修动作。"
                  : "每条校准绑定 Forecast、Outcome、Release、模型版本和证据摘要。每个版本至少积累 30 条结果后，才比较历史参考窗口与最近 10 条结果。"}
              />
              {overview.rul_release_calibrations.length === 0 ? (
                <EmptyState description="尚无同时具备 RUL 预测和真实故障结果的校准证据" />
              ) : (
                <Table
                  rowKey={(item) => `${item.source_model_release_id}:${item.forecast_model_version}`}
                  dataSource={overview.rul_release_calibrations}
                  pagination={false}
                  scroll={{ x: 1450 }}
                  columns={[
                    { title: "Release", dataIndex: "source_model_release_id", width: 190, ellipsis: true },
                    { title: "模型版本", dataIndex: "forecast_model_version", width: 210, ellipsis: true },
                    { title: "已标注结果", dataIndex: "labeled_outcome_count", width: 105 },
                    {
                      title: "总体中位 MAE",
                      width: 145,
                      render: (_, item) => minutes(item.median_absolute_error_minutes),
                    },
                    {
                      title: "近期中位 MAE",
                      width: 145,
                      render: (_, item) => minutes(item.recent_median_absolute_error_minutes),
                    },
                    {
                      title: "近期区间覆盖",
                      width: 140,
                      render: (_, item) => percent(item.recent_interval_coverage),
                    },
                    {
                      title: "门禁状态",
                      dataIndex: "status",
                      width: 160,
                      render: (value: string) => (
                        <Tag color={rulCalibrationStatusColor(value)}>
                          {rulCalibrationStatusLabel(value)}
                        </Tag>
                      ),
                    },
                    {
                      title: "判定依据",
                      width: 260,
                      render: (_, item) => item.reason_codes.map(rulCalibrationReasonLabel).join("、"),
                    },
                    { title: "聚合证据摘要", dataIndex: "evidence_digest", ellipsis: true, width: 230 },
                  ]}
                />
              )}
              {overview.rul_calibrations.length > 0 ? (
                <Table
                  style={{ marginTop: 20 }}
                  rowKey="calibration_id"
                  dataSource={overview.rul_calibrations}
                  pagination={{ pageSize: 8 }}
                  scroll={{ x: 1200 }}
                  columns={[
                    { title: "设备", dataIndex: "asset_id", width: 150 },
                    { title: "模型版本", dataIndex: "forecast_model_version", width: 210, ellipsis: true },
                    {
                      title: "真实剩余时间",
                      width: 150,
                      render: (_, item) => minutes(item.actual_minutes),
                    },
                    {
                      title: "绝对误差",
                      width: 130,
                      render: (_, item) => minutes(item.absolute_error_minutes),
                    },
                    {
                      title: "区间命中",
                      dataIndex: "interval_covered",
                      width: 105,
                      render: (value: boolean) => <Tag color={value ? "green" : "red"}>{value ? "是" : "否"}</Tag>,
                    },
                    { title: "故障时间", dataIndex: "evaluated_at", width: 180, render: formatTime },
                    { title: "证据摘要", dataIndex: "evidence_digest", ellipsis: true },
                  ]}
                />
              ) : null}
            </Card>

            {allowed.has("telemetry_window.build") ? (
              <Card title="构建设备事件时间窗">
                <form className="form-grid" onSubmit={submitWindow}>
                  <label htmlFor="window-asset">
                    设备 ID
                    <input id="window-asset" value={windowAssetId} onChange={(event) => setWindowAssetId(event.target.value)} />
                  </label>
                  <label htmlFor="window-start">
                    窗口开始
                    <input id="window-start" type="datetime-local" value={windowStart} onChange={(event) => setWindowStart(event.target.value)} />
                  </label>
                  <label htmlFor="window-end">
                    窗口结束
                    <input id="window-end" type="datetime-local" value={windowEnd} onChange={(event) => setWindowEnd(event.target.value)} />
                  </label>
                  <label htmlFor="include-too-late">
                    <input id="include-too-late" type="checkbox" checked={includeTooLate} onChange={(event) => setIncludeTooLate(event.target.checked)} />
                    受控历史回放：纳入超水位线事件
                  </label>
                  <Button type="primary" htmlType="submit" loading={busy === "build-window"}>
                    生成不可变特征快照
                  </Button>
                </form>
              </Card>
            ) : null}

            {allowed.has("telemetry_dataset.build") || allowed.has("rul_dataset.build") ? (
              <Card
                title="发布受治理时序训练快照"
                extra={<Link href="/ai/experiments">进入训练实验中心</Link>}
              >
                <Alert
                  showIcon
                  type="info"
                  message="异常检测与 RUL 使用独立数据契约"
                  description="两类快照都按设备分组并冻结 Parquet、质量报告和 OpenLineage。RUL 快照只纳入证据摘要合法、时间顺序正确的真实故障提前量，不会把未标注窗口伪造成监督标签。"
                  style={{ marginBottom: 16 }}
                />
                <form className="form-grid" onSubmit={submitDataset}>
                  <label htmlFor="dataset-start">
                    数据窗口开始
                    <input id="dataset-start" type="datetime-local" value={datasetStart} onChange={(event) => setDatasetStart(event.target.value)} />
                  </label>
                  <label htmlFor="dataset-end">
                    数据窗口结束
                    <input id="dataset-end" type="datetime-local" value={datasetEnd} onChange={(event) => setDatasetEnd(event.target.value)} />
                  </label>
                  <Space wrap>
                    {allowed.has("telemetry_dataset.build") ? (
                      <Button type="primary" htmlType="submit" loading={busy === "build-telemetry-dataset"}>
                        发布异常检测快照
                      </Button>
                    ) : null}
                    {allowed.has("rul_dataset.build") ? (
                      <Button
                        htmlType="button"
                        loading={busy === "build-rul-dataset"}
                        onClick={() => void submitRulDataset()}
                      >
                        发布 RUL 监督快照
                      </Button>
                    ) : null}
                  </Space>
                </form>
              </Card>
            ) : null}

            <Card title="最近设备时间窗">
              <Table<TelemetryWindow>
                rowKey="window_id"
                dataSource={overview.windows}
                pagination={{ pageSize: 8 }}
                scroll={{ x: 1050 }}
                columns={[
                  { title: "设备", dataIndex: "asset_id", width: 160 },
                  { title: "样本", dataIndex: "sample_count", width: 80 },
                  {
                    title: "质量",
                    dataIndex: "quality_status",
                    width: 140,
                    render: (value: string) => <Tag color={statusColors[value]}>{statusLabels[value] ?? value}</Tag>,
                  },
                  {
                    title: "缺失信号",
                    width: 220,
                    render: (_, item) => item.missing_signals.join("、") || "无",
                  },
                  { title: "特征快照", dataIndex: "feature_snapshot_id", ellipsis: true },
                  {
                    title: "操作",
                    width: 120,
                    render: (_, item) => allowed.has("anomaly_detection.run") ? (
                      <Button
                        size="small"
                        disabled={item.quality_status === "INSUFFICIENT"}
                        loading={busy === `detect:${item.window_id}`}
                        onClick={() => void runDetection(item)}
                      >
                        运行检测
                      </Button>
                    ) : null,
                  },
                ]}
              />
            </Card>

            <Card title="最近遥测接收状态">
              <Table
                rowKey="event_id"
                dataSource={overview.events}
                pagination={{ pageSize: 8 }}
                scroll={{ x: 900 }}
                columns={[
                  { title: "设备", dataIndex: "asset_id", width: 160 },
                  { title: "序列", dataIndex: "sequence", width: 90 },
                  { title: "事件时间", dataIndex: "event_time", width: 200, render: (value: string) => formatTime(value) },
                  { title: "来源", dataIndex: "source_system", width: 150 },
                  {
                    title: "接收判定",
                    dataIndex: "ingestion_status",
                    width: 180,
                    render: (value: string) => <Tag color={statusColors[value]}>{statusLabels[value] ?? value}</Tag>,
                  },
                ]}
              />
            </Card>

            <Typography.Text type="secondary" className="request-id">
              请求标识：{requestId} · 遥测契约：{overview.telemetry_contract_version} · 阈值策略：{overview.threshold_policy_id}
            </Typography.Text>
          </>
        ) : null}
      </div>

      <Drawer
        width={640}
        title="候选告警证据"
        open={Boolean(selected)}
        onClose={() => setSelected(undefined)}
      >
        {selected ? (
          <div className="page-stack">
            <Descriptions bordered column={1} size="small">
              <Descriptions.Item label="候选标识">{selected.alert_candidate_id}</Descriptions.Item>
              <Descriptions.Item label="设备">{selected.asset_id}</Descriptions.Item>
              <Descriptions.Item label="状态"><Tag color={statusColors[selected.status]}>{statusLabels[selected.status] ?? selected.status}</Tag></Descriptions.Item>
              <Descriptions.Item label="检测器">{selected.detector_kind} · {selected.model_release_id}</Descriptions.Item>
              <Descriptions.Item label="分数 / 阈值">{selected.anomaly_score.toFixed(3)} / {selected.threshold.toFixed(3)}</Descriptions.Item>
              <Descriptions.Item label="特征快照">{selected.feature_snapshot_id}</Descriptions.Item>
              <Descriptions.Item label="人工原因">{selected.decision_reason_code ?? "尚未确认"}</Descriptions.Item>
              <Descriptions.Item label="说明">{selected.decision_notes ?? "—"}</Descriptions.Item>
            </Descriptions>
            <Card size="small" title="传感器证据引用">
              {selected.supporting_signal_refs.map((item) => <div key={item}>{item}</div>)}
            </Card>
            {selected.referral ? (
              <Alert
                showIcon
                type="success"
                message="已进入现有维护流程"
                description={<Link href={`/incidents/drafts/${selected.referral.incident_draft_id}`}>打开 Incident 草稿 {selected.referral.incident_draft_id}</Link>}
              />
            ) : null}
          </div>
        ) : null}
      </Drawer>

      <Modal
        title={decisionDialog?.decision === "CONFIRMED" ? "确认候选告警" : "排除候选告警"}
        open={Boolean(decisionDialog)}
        onCancel={() => setDecisionDialog(undefined)}
        onOk={() => void submitDecision()}
        okButtonProps={{ disabled: !decisionReason.trim(), loading: busy?.startsWith("decision:") }}
      >
        <Form layout="vertical">
          <Form.Item label="原因代码" required>
            <Input value={decisionReason} onChange={(event) => setDecisionReason(event.target.value)} placeholder="例如 vibration_confirmed 或 sensor_fault" />
          </Form.Item>
          <Form.Item label="现场判断说明">
            <Input.TextArea rows={4} value={decisionNotes} onChange={(event) => setDecisionNotes(event.target.value)} />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title={outcomeDialog?.candidate ? "记录候选告警真实结果" : "记录漏报故障"}
        open={Boolean(outcomeDialog)}
        onCancel={() => setOutcomeDialog(undefined)}
        onOk={() => void submitOutcome()}
        okButtonProps={{ loading: busy === "record-outcome" }}
      >
        <Form layout="vertical">
          <Form.Item label="结果类型" required>
            <Select
              value={outcomeType}
              disabled={!outcomeDialog?.candidate}
              onChange={setOutcomeType}
              options={outcomeDialog?.candidate ? [
                { value: "TRUE_POSITIVE", label: "真实故障 / 正确告警" },
                { value: "FALSE_POSITIVE", label: "误报" },
              ] : [{ value: "MISSED_FAILURE", label: "漏报故障" }]}
            />
          </Form.Item>
          <Form.Item label="设备 ID" required>
            <Input value={outcomeAssetId} disabled={Boolean(outcomeDialog?.candidate)} onChange={(event) => setOutcomeAssetId(event.target.value)} />
          </Form.Item>
          {outcomeType !== "FALSE_POSITIVE" ? (
            <Form.Item label="故障发生时间" required>
              <Input type="datetime-local" value={failureAt} onChange={(event) => setFailureAt(event.target.value)} />
            </Form.Item>
          ) : null}
          <Form.Item label="避免停机时长（分钟）">
            <Input type="number" min="0" value={avoidedMinutes} onChange={(event) => setAvoidedMinutes(event.target.value)} />
          </Form.Item>
          <Form.Item label="结果证据引用" required>
            <Input value={evidenceRef} onChange={(event) => setEvidenceRef(event.target.value)} placeholder="workorder://... 或 historian://..." />
          </Form.Item>
          <Form.Item label="证据 SHA-256" required>
            <Input value={evidenceDigest} onChange={(event) => setEvidenceDigest(event.target.value)} placeholder="sha256:64位十六进制" />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title={rulReviewDialog?.decision === "ACCEPTED" ? "接受 RUL 候选" : "拒绝 RUL 候选"}
        open={Boolean(rulReviewDialog)}
        okText="提交复核"
        cancelText="取消"
        onCancel={() => setRulReviewDialog(undefined)}
        onOk={() => void submitRulReview()}
        okButtonProps={{
          disabled: !rulReviewReason.trim(),
          loading: busy?.startsWith("rul-review:"),
        }}
      >
        <Alert
          showIcon
          type="warning"
          style={{ marginBottom: 16 }}
          message="复核结论不会触发设备或工单副作用"
        />
        <Form layout="vertical">
          <Form.Item label="独立复核依据" htmlFor="rul-review-reason" required>
            <Input.TextArea
              id="rul-review-reason"
              rows={4}
              value={rulReviewReason}
              onChange={(event) => setRulReviewReason(event.target.value)}
              placeholder="说明样本代表性、现场工况、区间可用性及维护计划判断"
            />
          </Form.Item>
        </Form>
      </Modal>
    </AppShell>
  );
}

function formatTime(value: string): string {
  return new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "short",
    timeStyle: "short",
    hour12: false,
  }).format(new Date(value));
}

function percent(value: number | null | undefined): string {
  return value == null ? "无证据" : `${(value * 100).toFixed(1)}%`;
}

function minutes(value: number | null | undefined): string {
  return value == null ? "无证据" : `${value.toFixed(1)} 分钟`;
}

function rulStatusColor(value: string): string {
  if (value === "ACCEPTED") return "green";
  if (value === "REJECTED") return "red";
  return "orange";
}

function rulStatusLabel(value: string): string {
  if (value === "ACCEPTED") return "已独立接受";
  if (value === "REJECTED") return "已拒绝";
  return "等待独立复核";
}

function rulCalibrationStatusColor(value: string): string {
  if (value === "STABLE") return "green";
  if (value === "RETRAINING_RECOMMENDED") return "red";
  return "gold";
}

function rulCalibrationStatusLabel(value: string): string {
  if (value === "STABLE") return "校准稳定";
  if (value === "RETRAINING_RECOMMENDED") return "建议再训练";
  return "证据不足";
}

function rulCalibrationReasonLabel(value: string): string {
  const labels: Record<string, string> = {
    MINIMUM_30_LABELED_OUTCOMES_NOT_MET: "不足 30 条真实结果",
    ONLINE_CALIBRATION_WITHIN_GATES: "近期误差与覆盖在门禁内",
    RECENT_MEDIAN_MAE_REGRESSION: "近期中位 MAE 回退",
    RECENT_INTERVAL_COVERAGE_BELOW_FLOOR: "近期 P10–P90 覆盖低于 70%",
  };
  return labels[value] ?? value;
}
