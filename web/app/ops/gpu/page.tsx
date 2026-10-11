"use client";

import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Progress,
  Row,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from "antd";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  getGpuOperations,
  type GpuOperations,
  type RuntimeGpuPosture,
} from "@/lib/api/client";

const healthColors: Record<string, string> = {
  HEALTHY: "green",
  DEGRADED: "orange",
  CRITICAL: "red",
  NO_DATA: "default",
};

const healthLabels: Record<string, string> = {
  HEALTHY: "健康",
  DEGRADED: "降级",
  CRITICAL: "严重",
  NO_DATA: "无证据",
};

const advisoryColors: Record<string, string> = {
  HOLD: "green",
  SCALE_OUT_RECOMMENDED: "orange",
  SCALE_IN_CANDIDATE: "blue",
  INVESTIGATE: "red",
  NO_DATA: "default",
};

const advisoryLabels: Record<string, string> = {
  HOLD: "保持容量",
  SCALE_OUT_RECOMMENDED: "建议扩容",
  SCALE_IN_CANDIDATE: "缩容候选",
  INVESTIGATE: "优先排障",
  NO_DATA: "证据不足",
};

const reasonLabels: Record<string, string> = {
  gpu_metrics_unavailable: "GPU 指标不可用",
  gpu_xid_errors_detected: "检测到 GPU XID 错误",
  gpu_ecc_double_bit_errors_detected: "检测到 ECC 双比特错误",
  gpu_temperature_critical: "GPU 温度达到严重阈值",
  gpu_temperature_warning: "GPU 持续高温",
  gpu_memory_pressure: "GPU 显存压力较高",
  gpu_capacity_exhausted: "集群 GPU 可调度余量耗尽",
  dcgm_target_unavailable: "部分 DCGM Exporter 不可用",
  gpu_metrics_partial: "GPU 指标不完整",
  runtime_metrics_unavailable: "Runtime 实时指标不可用",
  runtime_metrics_partial: "Runtime 实时指标不完整",
  inference_error_rate_high: "推理错误率偏高",
  inference_queue_age_high: "推理队列等待时间偏高",
  gpu_utilization_high: "GPU 利用率偏高",
  gpu_utilization_low: "GPU 利用率持续偏低",
  inference_queue_empty: "推理队列基本为空",
  runtime_within_capacity_policy: "当前负载在容量策略范围内",
};

export default function GpuOperationsPage() {
  const [operations, setOperations] = useState<GpuOperations>();
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async (refresh = false) => {
    if (refresh) setRefreshing(true);
    try {
      const result = await getGpuOperations();
      setOperations(result.operations);
      setRequestId(result.requestId);
      setError(undefined);
    } catch (cause) {
      setError(cause);
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const advisoryCounts = useMemo(() => ({
    scaleOut: operations?.runtimes.filter(
      (runtime) => runtime.advisory === "SCALE_OUT_RECOMMENDED",
    ).length ?? 0,
    investigate: operations?.runtimes.filter(
      (runtime) => runtime.advisory === "INVESTIGATE",
    ).length ?? 0,
    noData: operations?.runtimes.filter(
      (runtime) => runtime.advisory === "NO_DATA",
    ).length ?? 0,
  }), [operations]);

  return (
    <AppShell>
      <div className="page-stack">
        <Space align="start" style={{ justifyContent: "space-between", width: "100%" }}>
          <div>
            <Typography.Title level={2}>GPU 推理与容量运维中心</Typography.Title>
            <Typography.Paragraph type="secondary">
              聚合 GPU Operator、DCGM、KServe、Prometheus 与模型发布记录，定位硬件异常、显存压力、推理排队和容量风险。所有查询由服务端固化并受租户 RBAC 保护。
            </Typography.Paragraph>
          </div>
          <Button loading={refreshing} onClick={() => void load(true)}>刷新证据</Button>
        </Space>

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!operations && !error ? <LoadingState label="正在聚合 GPU 与推理运行证据" /> : null}

        {operations ? (
          <>
            {operations.data_source_status !== "AVAILABLE" ? (
              <Alert
                showIcon
                type="warning"
                message="GPU 运维证据不完整"
                description={`监控源状态为 ${operations.data_source_status}。缺失指标显示为“—”，不会被折算为健康或用于自动缩容。`}
              />
            ) : null}

            <Alert
              showIcon
              type={operations.cluster.health === "HEALTHY" ? "success" : operations.cluster.health === "CRITICAL" ? "error" : "warning"}
              message={`集群 GPU 状态：${healthLabels[operations.cluster.health] ?? operations.cluster.health}`}
              description={operations.cluster.reasons.length
                ? operations.cluster.reasons.map(reasonText).join("；")
                : "DCGM 健康、硬件错误、温度、显存和集群容量均处于当前策略范围内。"}
              action={<Button size="small"><Link href="/ops">查看 GPU 运行手册</Link></Button>}
            />

            <Row gutter={[16, 16]}>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="DCGM 发现 GPU" value={metric(operations.cluster.discovered_gpu_count)} suffix="张" /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="可调度 GPU" value={metric(operations.cluster.allocatable_gpu_count)} suffix="张" /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="已申请 GPU" value={metric(operations.cluster.requested_gpu_count)} suffix="张" /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card>
                  <Statistic
                    title="容量余量"
                    value={metric(operations.cluster.capacity_headroom_gpu)}
                    suffix="张"
                    valueStyle={operations.cluster.capacity_headroom_gpu != null && operations.cluster.capacity_headroom_gpu <= 0 ? { color: "#cf1322" } : undefined}
                  />
                </Card>
              </Col>
            </Row>

            <Row gutter={[16, 16]}>
              <Col xs={24} lg={12}>
                <Card title="集群负载与显存">
                  <MetricProgress label="平均 GPU 利用率" value={operations.cluster.average_utilization} />
                  <MetricProgress label="集群显存利用率" value={operations.cluster.memory_utilization} />
                  <MetricProgress label="DCGM 采集目标可用率" value={operations.cluster.dcgm_targets_up_ratio} />
                </Card>
              </Col>
              <Col xs={24} lg={12}>
                <Card title="硬件健康信号">
                  <Descriptions column={2} bordered size="small">
                    <Descriptions.Item label="最高温度">{formatNumber(operations.cluster.max_temperature_celsius, "°C")}</Descriptions.Item>
                    <Descriptions.Item label="总功耗">{formatNumber(operations.cluster.power_usage_watts, " W")}</Descriptions.Item>
                    <Descriptions.Item label="15 分钟 XID">{formatNumber(operations.cluster.xid_errors_15m, " 次")}</Descriptions.Item>
                    <Descriptions.Item label="15 分钟 ECC DBE">{formatNumber(operations.cluster.ecc_dbe_errors_15m, " 次")}</Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>
            </Row>

            <Row gutter={[16, 16]}>
              <Col xs={24} sm={8}>
                <Card><Statistic title="建议扩容 Runtime" value={advisoryCounts.scaleOut} valueStyle={{ color: "#d46b08" }} /></Card>
              </Col>
              <Col xs={24} sm={8}>
                <Card><Statistic title="优先排障 Runtime" value={advisoryCounts.investigate} valueStyle={{ color: "#cf1322" }} /></Card>
              </Col>
              <Col xs={24} sm={8}>
                <Card><Statistic title="证据不足 Runtime" value={advisoryCounts.noData} valueStyle={{ color: "#8c8c8c" }} /></Card>
              </Col>
            </Row>

            <Card
              title={`KServe Runtime 容量建议（${operations.runtimes.length}）`}
              extra={<Typography.Text type="secondary">扩缩容须经模型发布控制器执行</Typography.Text>}
            >
              <Table<RuntimeGpuPosture>
                rowKey="deployment_id"
                dataSource={operations.runtimes}
                pagination={{ pageSize: 10, hideOnSinglePage: true }}
                locale={{ emptyText: "当前租户没有模型部署 Runtime" }}
                scroll={{ x: 1780 }}
                columns={[
                  {
                    title: "发布 / Runtime",
                    fixed: "left",
                    width: 245,
                    render: (_, runtime) => (
                      <Space direction="vertical" size={0}>
                        <Typography.Text strong>{runtime.release_id}</Typography.Text>
                        <Typography.Text type="secondary">{runtime.namespace}/{runtime.service_name}</Typography.Text>
                      </Space>
                    ),
                  },
                  {
                    title: "发布阶段",
                    width: 155,
                    render: (_, runtime) => (
                      <Space direction="vertical" size={0}>
                        <Tag color={runtime.deployment_status === "READY" ? "green" : "blue"}>{runtime.deployment_status}</Tag>
                        <Typography.Text type="secondary">{runtime.current_stage} · {runtime.observed_traffic_percent}%</Typography.Text>
                      </Space>
                    ),
                  },
                  {
                    title: "推理配置",
                    width: 185,
                    render: (_, runtime) => (
                      <Space direction="vertical" size={0}>
                        <Typography.Text>{runtime.runtime_engine ?? "未声明引擎"}</Typography.Text>
                        <Typography.Text type="secondary">{runtime.gpu_model ?? "GPU 型号未声明"} · {formatNumber(runtime.gpu_per_replica, " GPU/副本")}</Typography.Text>
                      </Space>
                    ),
                  },
                  { title: "运行副本", dataIndex: "running_replicas", width: 100, render: (value) => numberCell(value) },
                  { title: "请求/秒", dataIndex: "request_rate_per_second", width: 110, render: (value) => numberCell(value, 2) },
                  { title: "错误率", dataIndex: "error_rate", width: 105, render: ratioCell },
                  { title: "P95", dataIndex: "p95_latency_ms", width: 110, render: (value) => formatNumber(value, " ms") },
                  { title: "GPU 利用率", dataIndex: "gpu_utilization", width: 120, render: ratioCell },
                  { title: "显存利用率", dataIndex: "gpu_memory_utilization", width: 120, render: ratioCell },
                  { title: "队列最老等待", dataIndex: "queue_age_seconds", width: 125, render: (value) => formatNumber(value, " 秒") },
                  { title: "XID/15m", dataIndex: "xid_errors_15m", width: 95, render: (value) => numberCell(value) },
                  {
                    title: "最近发布观测",
                    width: 180,
                    render: (_, runtime) => runtime.latest_observation_at
                      ? `${runtime.latest_observation_decision ?? "—"} · ${new Date(runtime.latest_observation_at).toLocaleString()}`
                      : "暂无",
                  },
                  {
                    title: "容量决策建议",
                    fixed: "right",
                    width: 250,
                    render: (_, runtime) => (
                      <Space direction="vertical" size={2}>
                        <Tag color={advisoryColors[runtime.advisory]}>{advisoryLabels[runtime.advisory] ?? runtime.advisory}</Tag>
                        <Typography.Text type="secondary">
                          {runtime.advisory_reasons.map(reasonText).join("；")}
                        </Typography.Text>
                      </Space>
                    ),
                  },
                ]}
              />
            </Card>

            <Alert
              showIcon
              type="info"
              message="容量建议不是自动变更"
              description={(
                <span>
                  本页只形成可审计的诊断与建议。确认节点健康、配额、发布门禁和回滚路径后，请在 <Link href="/ai/releases">模型发布中心</Link> 通过既有 Deployment/KServe 控制器变更生产状态。
                </span>
              )}
            />
            <Typography.Text type="secondary">
              策略 {operations.policy_version} · 证据时间 {new Date(operations.generated_at).toLocaleString()} · 请求 ID {requestId}
            </Typography.Text>
          </>
        ) : null}
      </div>
    </AppShell>
  );
}

function MetricProgress({ label, value }: { label: string; value: number | null }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <Space style={{ justifyContent: "space-between", width: "100%" }}>
        <Typography.Text>{label}</Typography.Text>
        <Typography.Text>{value == null ? "—" : `${(value * 100).toFixed(1)}%`}</Typography.Text>
      </Space>
      <Progress
        percent={value == null ? 0 : Math.max(0, Math.min(100, value * 100))}
        status={value == null ? "normal" : value >= 0.9 ? "exception" : "active"}
        showInfo={false}
      />
    </div>
  );
}

function metric(value: number | null): number | string {
  return value == null ? "—" : Number(value.toFixed(2));
}

function numberCell(value: number | null, precision = 0): string {
  return value == null ? "—" : value.toFixed(precision);
}

function ratioCell(value: number | null): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value: number | null, suffix: string): string {
  return value == null ? "—" : `${Number(value.toFixed(2))}${suffix}`;
}

function reasonText(reason: string): string {
  return reasonLabels[reason] ?? reason;
}
