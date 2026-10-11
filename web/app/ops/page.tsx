"use client";

import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Drawer,
  List,
  Row,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import { RecoveryGovernancePanel } from "@/components/m6/RecoveryGovernancePanel";
import {
  getOperationsOverview,
  type OperationsAlert,
  type OperationsOverview,
  type OperationsRunbook,
  type SloObservation,
} from "@/lib/api/client";

const statusColors: Record<string, string> = {
  HEALTHY: "green",
  BREACHED: "red",
  NO_DATA: "default",
};

const statusLabels: Record<string, string> = {
  HEALTHY: "健康",
  BREACHED: "已违反",
  NO_DATA: "无证据",
};

const releaseReasons: Record<string, string> = {
  operations_evidence_unavailable: "监控证据不可用",
  required_slo_has_no_data: "必要 SLO 没有数据",
  error_budget_exhausted: "错误预算已经耗尽",
  security_hard_gate_firing: "安全硬门禁正在触发",
  critical_alert_firing: "存在关键生产告警",
  recovery_evidence_unavailable: "恢复治理证据服务不可用",
  recovery_evidence_noncompliant: "备份或恢复证据不合规",
  postgres_restore_drill_overdue: "PostgreSQL 月度恢复验证已到期",
  cross_component_recovery_drill_overdue: "跨组件季度恢复演练已到期",
};

export default function OperationsWorkspacePage() {
  const [overview, setOverview] = useState<OperationsOverview>();
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [selectedRunbook, setSelectedRunbook] = useState<OperationsRunbook>();

  const load = useCallback(async () => {
    try {
      const result = await getOperationsOverview();
      setOverview(result.overview);
      setRequestId(result.requestId);
      setError(undefined);
    } catch (cause) {
      setError(cause);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const runbooks = useMemo(
    () => new Map(overview?.runbooks.map((item) => [item.runbook_id, item])),
    [overview],
  );
  const counts = useMemo(() => ({
    healthy: overview?.slos.filter((item) => item.status === "HEALTHY").length ?? 0,
    breached: overview?.slos.filter((item) => item.status === "BREACHED").length ?? 0,
    noData: overview?.slos.filter((item) => item.status === "NO_DATA").length ?? 0,
    critical: overview?.alerts.filter(
      (item) => item.severity === "critical" && item.state === "firing",
    ).length ?? 0,
  }), [overview]);

  function openRunbook(runbookId: string) {
    setSelectedRunbook(runbooks.get(runbookId) ?? runbooks.get("operations-generic"));
  }

  return (
    <AppShell>
      <div className="page-stack">
        <div>
          <Typography.Title level={2}>生产运行、SLO 与故障处置</Typography.Title>
          <Typography.Paragraph type="secondary">
            浏览器只访问受 OIDC 和 RBAC 保护的运维聚合 API，不直接连接 Prometheus。这里展示可验证的服务目标、错误预算、活动告警和对应运行手册。
          </Typography.Paragraph>
        </div>

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!overview && !error ? <LoadingState label="正在聚合生产运维证据" /> : null}

        {overview ? (
          <>
            {overview.data_source_status !== "AVAILABLE" || counts.noData > 0 ? (
              <Alert
                showIcon
                type="warning"
                message="当前证据不完整，不能判定系统健康"
                description={`监控源状态：${overview.data_source_status}；无数据 SLO：${counts.noData}。NO_DATA 不会被折算成健康，且会暂停高风险发布。`}
              />
            ) : null}
            <Alert
              showIcon
              type={overview.high_risk_release_allowed ? "success" : "error"}
              message={overview.high_risk_release_allowed ? "高风险发布门禁允许" : "高风险发布门禁暂停"}
              description={overview.high_risk_release_allowed
                ? "监控证据、必要 SLO、错误预算和活动关键告警均满足当前策略。"
                : overview.release_gate_reasons.map((reason) => releaseReasons[reason] ?? reason).join("；")}
            />

            <Row gutter={[16, 16]}>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="健康 SLO" value={counts.healthy} valueStyle={{ color: "#389e0d" }} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="违反 SLO" value={counts.breached} valueStyle={{ color: "#cf1322" }} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="无证据 SLO" value={counts.noData} valueStyle={{ color: "#8c8c8c" }} /></Card>
              </Col>
              <Col xs={24} sm={12} xl={6}>
                <Card><Statistic title="活动关键告警" value={counts.critical} valueStyle={{ color: "#cf1322" }} /></Card>
              </Col>
            </Row>

            <RecoveryGovernancePanel />

            <Card title="服务等级目标与错误预算">
              <Table<SloObservation>
                rowKey="slo_id"
                pagination={false}
                dataSource={overview.slos}
                scroll={{ x: 1050 }}
                columns={[
                  { title: "服务目标", dataIndex: "name", width: 220 },
                  { title: "负责人", dataIndex: "owner", width: 140 },
                  { title: "窗口", dataIndex: "window", width: 90 },
                  {
                    title: "目标",
                    width: 145,
                    render: (_, item) => `${item.comparison} ${formatValue(item.objective, item.unit)}`,
                  },
                  {
                    title: "当前值",
                    width: 145,
                    render: (_, item) => item.current_value == null
                      ? "—"
                      : formatValue(item.current_value, item.unit),
                  },
                  {
                    title: "状态",
                    dataIndex: "status",
                    width: 110,
                    render: (value: string) => <Tag color={statusColors[value]}>{statusLabels[value] ?? value}</Tag>,
                  },
                  {
                    title: "剩余错误预算",
                    width: 150,
                    render: (_, item) => item.error_budget_remaining_fraction == null
                      ? "不适用"
                      : `${(item.error_budget_remaining_fraction * 100).toFixed(1)}%`,
                  },
                  {
                    title: "处置",
                    width: 110,
                    render: (_, item) => <Button size="small" onClick={() => openRunbook(item.runbook_id)}>运行手册</Button>,
                  },
                ]}
              />
            </Card>

            <Card title={`活动告警（${overview.alerts.length}）`}>
              <Table<OperationsAlert>
                rowKey={(item) => `${item.name}-${item.active_at}`}
                pagination={false}
                dataSource={overview.alerts}
                locale={{ emptyText: "当前没有活动告警" }}
                scroll={{ x: 980 }}
                columns={[
                  {
                    title: "级别",
                    dataIndex: "severity",
                    width: 100,
                    render: (value: string) => <Tag color={value === "critical" ? "red" : value === "warning" ? "orange" : "blue"}>{value.toUpperCase()}</Tag>,
                  },
                  { title: "告警", dataIndex: "name", width: 260 },
                  { title: "摘要", dataIndex: "summary" },
                  { title: "负责人", dataIndex: "owner", width: 140 },
                  {
                    title: "开始时间",
                    dataIndex: "active_at",
                    width: 190,
                    render: (value: string) => new Date(value).toLocaleString(),
                  },
                  {
                    title: "处置",
                    width: 110,
                    render: (_, item) => <Button size="small" onClick={() => openRunbook(item.runbook_id)}>运行手册</Button>,
                  },
                ]}
              />
            </Card>

            <Typography.Text type="secondary">
              策略 {overview.policy_version} · 证据时间 {new Date(overview.generated_at).toLocaleString()} · 请求 ID {requestId}
            </Typography.Text>
          </>
        ) : null}

        <Drawer
          width={680}
          title={selectedRunbook?.title ?? "运行手册"}
          open={Boolean(selectedRunbook)}
          onClose={() => setSelectedRunbook(undefined)}
        >
          {selectedRunbook ? <RunbookDetail runbook={selectedRunbook} /> : null}
        </Drawer>
      </div>
    </AppShell>
  );
}

function RunbookDetail({ runbook }: { runbook: OperationsRunbook }) {
  return (
    <Space direction="vertical" size="large" style={{ width: "100%" }}>
      <Descriptions column={1} bordered size="small">
        <Descriptions.Item label="运行手册 ID">{runbook.runbook_id}</Descriptions.Item>
        <Descriptions.Item label="负责人">{runbook.owner}</Descriptions.Item>
        <Descriptions.Item label="用户影响">{runbook.user_impact}</Descriptions.Item>
      </Descriptions>
      <RunbookList title="确认与诊断" items={runbook.confirmation_queries} />
      <RunbookList title="安全止损" items={runbook.safety_containment} />
      <RunbookList title="恢复步骤" items={runbook.recovery} />
      <RunbookList title="回滚步骤" items={runbook.rollback} />
      <RunbookList title="一致性核验" items={runbook.consistency_checks} />
      <Alert type="warning" showIcon message="升级机制" description={runbook.escalation} />
      <Alert type="info" showIcon message="事后复盘" description={runbook.postmortem} />
    </Space>
  );
}

function RunbookList({ title, items }: { title: string; items: string[] }) {
  return (
    <Card size="small" title={title}>
      <List size="small" dataSource={items} renderItem={(item) => <List.Item>{item}</List.Item>} />
    </Card>
  );
}

function formatValue(value: number, unit: string): string {
  if (unit === "ratio") return `${(value * 100).toFixed(3)}%`;
  if (unit === "seconds") return `${value.toFixed(3)} 秒`;
  return String(value);
}
