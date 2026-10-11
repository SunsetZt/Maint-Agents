"use client";

import {
  Alert,
  Button,
  Card,
  Col,
  Form,
  Input,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmergencyAccessPanel } from "@/components/m6/EmergencyAccessPanel";
import { ProductionAssurancePanel } from "@/components/m6/ProductionAssurancePanel";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  type SecurityAuditEvent,
  type SecurityAuditSummary,
  getSecurityAuditSummary,
  listSecurityAuditEvents,
} from "@/lib/api/client";

type AuditFilters = {
  decision?: "allow" | "deny";
  action?: string;
  windowHours: number;
};

const PAGE_SIZE = 30;

export default function SecurityWorkspacePage() {
  const [events, setEvents] = useState<SecurityAuditEvent[]>();
  const [summary, setSummary] = useState<SecurityAuditSummary>();
  const [nextCursor, setNextCursor] = useState<string>();
  const [cursorHistory, setCursorHistory] = useState<(string | undefined)[]>([
    undefined,
  ]);
  const [filters, setFilters] = useState<AuditFilters>({ windowHours: 24 });
  const [error, setError] = useState<unknown>();
  const [activeGrantId, setActiveGrantId] = useState<string>();
  const [form] = Form.useForm<AuditFilters>();
  const cursor = cursorHistory[cursorHistory.length - 1];

  const load = useCallback(async () => {
    try {
      const [eventResult, summaryResult] = await Promise.all([
        listSecurityAuditEvents({
          limit: PAGE_SIZE,
          cursor,
          decision: filters.decision,
          action: filters.action,
        }, activeGrantId),
        getSecurityAuditSummary(filters.windowHours, activeGrantId),
      ]);
      setEvents(eventResult.events);
      setNextCursor(eventResult.nextCursor);
      setSummary(summaryResult.summary);
      setError(undefined);
    } catch (cause) {
      setError(cause);
    }
  }, [activeGrantId, cursor, filters]);

  useEffect(() => {
    void load();
  }, [load]);

  const actionCounts = useMemo(
    () => Object.entries(summary?.action_counts ?? {}).sort((left, right) => right[1] - left[1]),
    [summary],
  );

  function applyFilters(values: AuditFilters) {
    setFilters({
      decision: values.decision,
      action: values.action?.trim() || undefined,
      windowHours: values.windowHours,
    });
    setCursorHistory([undefined]);
  }

  return (
    <AppShell>
      <div className="page-stack">
        <div>
          <Typography.Title level={2}>安全审计与 AI Guardrail</Typography.Title>
          <Typography.Paragraph type="secondary">
            查看当前租户的授权决策、Prompt Injection 拦截和模型调用安全结论。审计只保存主体、租户、资源和内容指纹，不展示 Prompt、Secret 或原始业务正文。
          </Typography.Paragraph>
        </div>

        <Alert
          type="info"
          showIcon
          message="纵深防御边界"
          description="Guardrail 负责阻断指令覆盖、敏感信息套取、权限冒充和工具结果伪造；OIDC、RLS、OPA、人工审批和 Tool Gateway 仍分别执行身份、数据与副作用控制。"
        />

        <ProductionAssurancePanel />

        <EmergencyAccessPanel
          activeGrantId={activeGrantId}
          onUseGrant={setActiveGrantId}
        />

        {summary ? (
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={12} xl={6}>
              <Card><Statistic title="观察窗口事件" value={summary.total_events} /></Card>
            </Col>
            <Col xs={24} sm={12} xl={6}>
              <Card><Statistic title="允许" value={summary.allowed_events} valueStyle={{ color: "#389e0d" }} /></Card>
            </Col>
            <Col xs={24} sm={12} xl={6}>
              <Card><Statistic title="拒绝" value={summary.denied_events} valueStyle={{ color: "#cf1322" }} /></Card>
            </Col>
            <Col xs={24} sm={12} xl={6}>
              <Card><Statistic title="Guardrail 拦截" value={summary.guardrail_blocks} valueStyle={{ color: "#d46b08" }} /></Card>
            </Col>
          </Row>
        ) : null}

        <Card title="审计筛选">
          <Form<AuditFilters>
            form={form}
            layout="inline"
            initialValues={{ windowHours: 24 }}
            onFinish={applyFilters}
          >
            <Form.Item name="decision" label="决策">
              <Select
                allowClear
                style={{ width: 130 }}
                options={[
                  { label: "允许", value: "allow" },
                  { label: "拒绝", value: "deny" },
                ]}
              />
            </Form.Item>
            <Form.Item name="action" label="动作">
              <Input placeholder="如 model_gateway.guardrail" style={{ width: 260 }} />
            </Form.Item>
            <Form.Item name="windowHours" label="汇总窗口">
              <Select
                style={{ width: 130 }}
                options={[
                  { label: "最近 24 小时", value: 24 },
                  { label: "最近 7 天", value: 168 },
                  { label: "最近 30 天", value: 720 },
                ]}
              />
            </Form.Item>
            <Form.Item><Button htmlType="submit" type="primary">查询</Button></Form.Item>
          </Form>
        </Card>

        {actionCounts.length ? (
          <Card title="高频安全动作">
            <Space wrap>
              {actionCounts.map(([action, count]) => (
                <Tag key={action}>{action} · {count}</Tag>
              ))}
            </Space>
          </Card>
        ) : null}

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!events && !error ? <LoadingState label="正在读取安全审计" /> : null}
        {events && !events.length && !error ? <EmptyState description="当前筛选条件下没有审计事件" /> : null}
        {events?.length ? (
          <Card title="不可变审计事件">
            <Table<SecurityAuditEvent>
              rowKey="event_id"
              pagination={false}
              dataSource={events}
              scroll={{ x: 1100 }}
              columns={[
                {
                  title: "发生时间",
                  dataIndex: "occurred_at",
                  width: 190,
                  render: (value: string) => new Date(value).toLocaleString(),
                },
                {
                  title: "决策",
                  dataIndex: "decision",
                  width: 90,
                  render: (value: string) => (
                    <Tag color={value === "allow" ? "green" : "red"}>{value.toUpperCase()}</Tag>
                  ),
                },
                { title: "动作", dataIndex: "action", width: 230 },
                { title: "原因码", dataIndex: "reason_code", width: 250 },
                {
                  title: "主体指纹",
                  dataIndex: "subject_hash",
                  width: 145,
                  render: (value: string) => <Typography.Text code>{value.slice(0, 12)}</Typography.Text>,
                },
                {
                  title: "资源指纹",
                  dataIndex: "resource_hash",
                  width: 145,
                  render: (value: string | null) => value
                    ? <Typography.Text code>{value.slice(0, 12)}</Typography.Text>
                    : "—",
                },
                { title: "请求 ID", dataIndex: "request_id", width: 220 },
              ]}
            />
            <Space style={{ marginTop: 16 }}>
              <Button
                disabled={cursorHistory.length === 1}
                onClick={() => setCursorHistory((current) => current.slice(0, -1))}
              >上一页</Button>
              <Button
                disabled={!nextCursor}
                onClick={() => nextCursor && setCursorHistory((current) => [...current, nextCursor])}
              >下一页</Button>
              <Typography.Text type="secondary">第 {cursorHistory.length} 页</Typography.Text>
            </Space>
          </Card>
        ) : null}
      </div>
    </AppShell>
  );
}
