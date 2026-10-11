"use client";

import { Alert, Button, Card, Col, Input, Row, Space, Statistic, Table, Tag, Typography } from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  activateServicePerformanceBaseline,
  createServicePerformanceBaseline,
  getServicePerformanceV3,
  listServicePerformanceBaselines,
  type ServicePerformanceBaseline,
  type ServicePerformanceV3,
} from "@/lib/api/client";

type PerformanceSlice = ServicePerformanceV3["slices"][number];
type TargetComparison = ServicePerformanceV3["target_comparisons"][number];

export default function ServicePerformancePage() {
  const defaults = useMemo(() => defaultWindow(), []);
  const [windowStart, setWindowStart] = useState(defaults.start);
  const [windowEnd, setWindowEnd] = useState(defaults.end);
  const [performance, setPerformance] = useState<ServicePerformanceV3>();
  const [baselines, setBaselines] = useState<ServicePerformanceBaseline[]>([]);
  const [baselineLegalActions, setBaselineLegalActions] = useState<string[]>([]);
  const [baselineName, setBaselineName] = useState("");
  const [activationReasons, setActivationReasons] = useState<Record<string, string>>({});
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setBusy(true);
    setError(undefined);
    try {
      const [result, baselineResult] = await Promise.all([
        getServicePerformanceV3(atLocalMidnight(windowStart), atLocalMidnight(windowEnd)),
        listServicePerformanceBaselines(),
      ]);
      setPerformance(result.performance);
      setBaselines(baselineResult.baselines);
      setBaselineLegalActions(baselineResult.legalActions);
      setRequestId(result.requestId);
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }, [windowEnd, windowStart]);

  useEffect(() => { void load(); }, [load]);

  const createBaseline = async () => {
    if (!baselineName.trim()) return;
    setError(undefined);
    try {
      await createServicePerformanceBaseline({
        name: baselineName.trim(),
        reference_window_start: atLocalMidnight(windowStart),
        reference_window_end: atLocalMidnight(windowEnd),
        minimum_work_order_samples: 1,
        minimum_resolution_samples: 1,
        targets: {
          mttr_reduction_rate: 0.2,
          first_time_fix_lift: 0.1,
          remote_resolution_lift: 0.15,
        },
      });
      setBaselineName("");
      await load();
    } catch (caught) { setError(caught); }
  };

  const activateBaseline = async (baseline: ServicePerformanceBaseline) => {
    const reason = activationReasons[baseline.baseline_id]?.trim() ?? "";
    if (reason.length < 5) return;
    setError(undefined);
    try {
      await activateServicePerformanceBaseline(baseline.baseline_id, baseline.version, reason);
      setActivationReasons((current) => ({ ...current, [baseline.baseline_id]: "" }));
      await load();
    } catch (caught) { setError(caught); }
  };

  const siteSlices = performance?.slices.filter((item) => item.dimension === "SITE") ?? [];
  const categorySlices = performance?.slices.filter(
    (item) => item.dimension === "INCIDENT_CATEGORY",
  ) ?? [];

  return (
    <AppShell>
      <div className="page-stack">
        <div>
          <Typography.Title level={2}>售后服务绩效与返工分析</Typography.Title>
          <Typography.Paragraph type="secondary">
            指标直接读取已关闭工单、IncidentResolution、维修轮次、独立验收和 Incident 关闭控制事实。工单指标按关闭时间、解决指标按解决时间筛选，结束日期均为不包含边界。
          </Typography.Paragraph>
        </div>

        <Card title="统计窗口">
          <Space wrap>
            <label>开始日期 <Input type="date" value={windowStart} onChange={(event) => setWindowStart(event.target.value)} /></label>
            <label>结束日期（不含） <Input type="date" value={windowEnd} onChange={(event) => setWindowEnd(event.target.value)} /></label>
            <Button type="primary" loading={busy} disabled={!windowStart || !windowEnd || windowStart >= windowEnd} onClick={() => void load()}>重新计算</Button>
          </Space>
        </Card>

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!performance && !error ? <LoadingState label="正在计算服务业务指标" /> : null}

        {performance ? (
          <>
            {performance.data_quality.status !== "COMPLETE" ? (
              <Alert
                type="warning"
                showIcon
                message={`数据质量：${qualityLabel(performance.data_quality.status)}`}
                description={`缺少完工 ${performance.data_quality.missing_completion_count}；缺少通过验收 ${performance.data_quality.missing_passed_verification_count}；工单时间线异常 ${performance.data_quality.invalid_timeline_count}；已关闭但缺关闭控制 ${performance.data_quality.missing_close_control_count}；关闭早于解决 ${performance.data_quality.invalid_close_timeline_count}。不完整记录不会进入对应指标分母。`}
              />
            ) : (
              <Alert type="success" showIcon message="当前窗口业务事实完整" description="纳入统计的关闭工单具备完工与独立验收，解决样本的确认时间线也完整。" />
            )}
            <BaselineGovernance
              performance={performance}
              baselines={baselines}
              legalActions={baselineLegalActions}
              baselineName={baselineName}
              activationReasons={activationReasons}
              busy={busy}
              onNameChange={setBaselineName}
              onReasonChange={(baselineId, value) => setActivationReasons(
                (current) => ({ ...current, [baselineId]: value }),
              )}
              onCreate={createBaseline}
              onActivate={activateBaseline}
            />

            <Row gutter={[16, 16]}>
              <Metric title="已关闭工单" value={performance.summary.closed_work_order_count} />
              <Metric title="一次修复率" value={percentage(performance.summary.first_time_fix_rate)} suffix="%" />
              <Metric title="返工率" value={percentage(performance.summary.rework_rate)} suffix="%" />
              <Metric title="平均维修轮次" value={decimal(performance.summary.average_repair_rounds)} />
              <Metric title="平均 MTTR" value={decimal(performance.summary.mean_mttr_minutes)} suffix="分钟" />
              <Metric title="P90 MTTR" value={decimal(performance.summary.p90_mttr_minutes)} suffix="分钟" />
              <Metric title="远程解决率" value={percentage(performance.summary.remote_resolution_rate)} suffix="%" />
              <Metric title="待确认解决" value={performance.summary.pending_confirmation_count} />
              <Metric title="平均确认耗时" value={decimal(performance.summary.mean_confirmation_minutes)} suffix="分钟" />
              <Metric title="P90 确认耗时" value={decimal(performance.summary.p90_confirmation_minutes)} suffix="分钟" />
            </Row>

            <Card title="指标口径与样本">
              <Space wrap>
                <Tag color="blue">一次修复样本 {performance.summary.first_time_fix_eligible_count}</Tag>
                <Tag color="green">一次修复 {performance.summary.first_time_fix_count}</Tag>
                <Tag color="orange">发生返工 {performance.summary.reworked_work_order_count}</Tag>
                <Tag color="purple">MTTR 样本 {performance.summary.mttr_sample_count}</Tag>
                <Tag>P50 {formatMinutes(performance.summary.p50_mttr_minutes)}</Tag>
              </Space>
              <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
                MTTR：Incident 创建到首次通过独立验收；一次修复：关闭工单没有进入任何返工轮次。指标不会通过删除失败记录得到改善。
              </Typography.Paragraph>
            </Card>

            <Card title="远程解决与确认口径">
              <Space wrap>
                <Tag color="blue">解决样本 {performance.summary.resolution_sample_count}</Tag>
                <Tag color="cyan">远程解决 {performance.summary.remote_resolution_count}</Tag>
                <Tag color="geekblue">工单解决 {performance.summary.work_order_resolution_count}</Tag>
                <Tag color="green">已确认关闭 {performance.summary.closed_resolution_count}</Tag>
                <Tag color="gold">待确认 {performance.summary.pending_confirmation_count}</Tag>
                <Tag color="purple">确认耗时样本 {performance.summary.confirmation_sample_count}</Tag>
                <Tag>P50 {formatMinutes(performance.summary.p50_confirmation_minutes)}</Tag>
              </Space>
              <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
                远程解决率只按已登记的 REMOTE/WORK_ORDER 解决事实计算；确认耗时从解决时点到首条合法 CLOSE 控制。未解决 Incident 不进入分母，缺失或倒置关闭事实不生成推测耗时。
              </Typography.Paragraph>
            </Card>

            <SliceTable title="按站点切片" rows={siteSlices} />
            <SliceTable title="按故障分类切片" rows={categorySlices} />

            <Typography.Text type="secondary">
              合同 {performance.metric_contract_version} · 生成时间 {new Date(performance.generated_at).toLocaleString("zh-CN", { hour12: false })} · 请求 ID {requestId}
            </Typography.Text>
          </>
        ) : null}
      </div>
    </AppShell>
  );
}

function BaselineGovernance({ performance, baselines, legalActions, baselineName, activationReasons,
  busy, onNameChange, onReasonChange, onCreate, onActivate }: {
  performance: ServicePerformanceV3;
  baselines: ServicePerformanceBaseline[];
  legalActions: string[];
  baselineName: string;
  activationReasons: Record<string, string>;
  busy: boolean;
  onNameChange: (value: string) => void;
  onReasonChange: (baselineId: string, value: string) => void;
  onCreate: () => Promise<void>;
  onActivate: (baseline: ServicePerformanceBaseline) => Promise<void>;
}) {
  const baseline = performance.baseline;
  const candidates = baselines.filter((item) => item.legal_actions?.includes("ACTIVATE"));
  const canCreate = legalActions.includes("CREATE");
  return (
    <>
      <Alert
        type={baseline ? "success" : "info"}
        showIcon
        message={baseline ? `当前活动基线：${baseline.name}` : baselineStatus(performance.baseline_status)}
        description={baseline
          ? `参考窗口 ${formatDate(baseline.reference_window_start)} 至 ${formatDate(baseline.reference_window_end)}；内容摘要 ${baseline.content_digest}`
          : "没有适用于当前窗口的已审批基线，平台不会自动使用上一周期或零值补齐。"}
      />
      {baseline ? (
        <Space wrap>
          <Tag color="blue">参考工单 {baseline.samples.closed_work_order_count}</Tag>
          <Tag color="purple">参考 MTTR 样本 {baseline.samples.mttr_sample_count}</Tag>
          <Tag color="green">参考一次修复样本 {baseline.samples.first_time_fix_sample_count}</Tag>
          <Tag color="cyan">参考解决样本 {baseline.samples.resolution_sample_count}</Tag>
        </Space>
      ) : null}
      <Row gutter={[16, 16]}>
        {performance.target_comparisons.map((item) => (
          <ComparisonCard key={item.metric} comparison={item} />
        ))}
      </Row>
      <Card title="企业基线管理">
        <Space direction="vertical" style={{ width: "100%" }}>
          <Typography.Text type="secondary">
            仅固化当前查询窗口的服务端权威指标；默认目标为 MTTR 降低 20%、一次修复率提升 10 个百分点、远程解决率提升 15 个百分点。
          </Typography.Text>
          {canCreate ? (
            <>
              <label htmlFor="service-performance-baseline-name">基线名称</label>
              <Input id="service-performance-baseline-name" value={baselineName}
                onChange={(event) => onNameChange(event.target.value)} placeholder="例如：2026 Q2 审批参考" />
              <Button type="primary" disabled={!baselineName.trim()} loading={busy}
                onClick={() => void onCreate()}>从当前窗口固化基线</Button>
            </>
          ) : null}
          {candidates.map((item) => (
            <Card size="small" key={item.baseline_id} title={`${item.name} · DRAFT v${item.version}`}>
              <Space direction="vertical" style={{ width: "100%" }}>
                <label htmlFor={`activation-reason-${item.baseline_id}`}>候选基线激活理由</label>
                <Input id={`activation-reason-${item.baseline_id}`}
                  value={activationReasons[item.baseline_id] ?? ""}
                  onChange={(event) => onReasonChange(item.baseline_id, event.target.value)} />
                <Button disabled={(activationReasons[item.baseline_id]?.trim().length ?? 0) < 5}
                  loading={busy} onClick={() => void onActivate(item)}>激活候选基线</Button>
              </Space>
            </Card>
          ))}
        </Space>
      </Card>
    </>
  );
}

function ComparisonCard({ comparison }: { comparison: TargetComparison }) {
  return (
    <Col xs={24} lg={8}>
      <Card title={comparisonLabel(comparison.metric)}
        extra={<Tag color={comparison.status === "MET" ? "green" : comparison.status === "NOT_MET" ? "red" : "default"}>
          {comparisonStatus(comparison.status)}
        </Tag>}>
        <Statistic title="改善值" value={formatImprovement(comparison.improvement)} />
        <Typography.Paragraph type="secondary">
          基线 {comparisonValue(comparison.metric, comparison.baseline_value)} · 当前 {comparisonValue(comparison.metric, comparison.current_value)} · 目标 {formatImprovement(comparison.target)}
        </Typography.Paragraph>
        {comparison.reason ? <Typography.Text type="secondary">{comparisonReason(comparison.reason)}</Typography.Text> : null}
      </Card>
    </Col>
  );
}

function Metric({ title, value, suffix }: { title: string; value: number | string; suffix?: string }) {
  return (
    <Col xs={24} sm={12} xl={8}>
      <Card><Statistic title={title} value={value} suffix={suffix} /></Card>
    </Col>
  );
}

function SliceTable({ title, rows }: { title: string; rows: PerformanceSlice[] }) {
  return (
    <Card title={title}>
      <Table<PerformanceSlice>
        rowKey={(item) => `${item.dimension}-${item.key}`}
        pagination={false}
        dataSource={rows}
        locale={{ emptyText: "当前窗口没有可统计的关闭工单或解决样本" }}
        scroll={{ x: 1420 }}
        columns={[
          { title: "范围", dataIndex: "label", width: 220 },
          { title: "关闭工单", dataIndex: "closed_work_order_count", width: 110 },
          { title: "有效样本", dataIndex: "first_time_fix_eligible_count", width: 110 },
          { title: "一次修复", dataIndex: "first_time_fix_count", width: 110 },
          { title: "一次修复率", width: 130, render: (_, item) => formatRate(item.first_time_fix_rate) },
          { title: "返工工单", dataIndex: "reworked_work_order_count", width: 110 },
          { title: "平均 MTTR", width: 150, render: (_, item) => formatMinutes(item.mean_mttr_minutes) },
          { title: "解决样本", dataIndex: "resolution_sample_count", width: 110 },
          { title: "远程解决", dataIndex: "remote_resolution_count", width: 110 },
          { title: "远程率", width: 130, render: (_, item) => formatRate(item.remote_resolution_rate) },
          { title: "待确认", dataIndex: "pending_confirmation_count", width: 100 },
        ]}
      />
    </Card>
  );
}

function defaultWindow(): { start: string; end: string } {
  const end = new Date();
  end.setDate(end.getDate() + 1);
  const start = new Date(end);
  start.setDate(start.getDate() - 30);
  return { start: localDate(start), end: localDate(end) };
}

function localDate(value: Date): string {
  const local = new Date(value.getTime() - value.getTimezoneOffset() * 60 * 1000);
  return local.toISOString().slice(0, 10);
}

function atLocalMidnight(value: string): string {
  return new Date(`${value}T00:00:00`).toISOString();
}

function percentage(value: number | null): number | string {
  return value === null ? "—" : Number((value * 100).toFixed(1));
}

function decimal(value: number | null): number | string {
  return value === null ? "—" : Number(value.toFixed(1));
}

function formatRate(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function formatMinutes(value: number | null): string {
  return value === null ? "—" : `${value.toFixed(1)} 分钟`;
}

function qualityLabel(status: string): string {
  return status === "NO_DATA" ? "无样本" : "部分可用";
}

function baselineStatus(status: ServicePerformanceV3["baseline_status"]): string {
  return status === "NOT_EFFECTIVE" ? "企业基线尚未对当前窗口生效" : "尚未配置企业基线";
}

function comparisonLabel(metric: TargetComparison["metric"]): string {
  if (metric === "MEAN_MTTR") return "MTTR 降幅";
  if (metric === "FIRST_TIME_FIX_RATE") return "一次修复率提升";
  return "远程解决率提升";
}

function comparisonStatus(status: TargetComparison["status"]): string {
  if (status === "MET") return "已达标";
  if (status === "NOT_MET") return "未达标";
  return "证据不足";
}

function comparisonReason(reason: string): string {
  const labels: Record<string, string> = {
    baseline_not_configured: "尚未配置已审批基线",
    baseline_not_effective: "已审批基线尚未对当前窗口生效",
    baseline_mttr_non_positive: "审批基线的 MTTR 必须大于零",
    current_samples_insufficient: "当前窗口样本不足或数据质量不完整",
  };
  return labels[reason] ?? reason;
}

function comparisonValue(metric: TargetComparison["metric"], value: number | null): string {
  return metric === "MEAN_MTTR" ? formatMinutes(value) : formatRate(value);
}

function formatImprovement(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function formatDate(value: string): string {
  return new Date(value).toLocaleDateString("zh-CN");
}
