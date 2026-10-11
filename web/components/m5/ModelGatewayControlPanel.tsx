"use client";

import {
  Alert,
  Button,
  Card,
  Col,
  Form,
  InputNumber,
  Modal,
  Row,
  Select,
  Space,
  Statistic,
  Switch,
  Table,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";

import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  type ModelInference,
  type ModelQuotaInput,
  type ModelRoute,
  listModelInferences,
  listModelRoutes,
  updateModelQuota,
} from "@/lib/api/client";

export function ModelGatewayControlPanel() {
  const [routes, setRoutes] = useState<ModelRoute[]>();
  const [inferences, setInferences] = useState<ModelInference[]>();
  const [editing, setEditing] = useState<ModelRoute>();
  const [error, setError] = useState<unknown>();
  const [busy, setBusy] = useState(false);
  const [form] = Form.useForm<ModelQuotaInput>();

  const load = useCallback(async () => {
    try {
      const [routeData, inferenceData] = await Promise.all([
        listModelRoutes(),
        listModelInferences(50),
      ]);
      setRoutes(routeData);
      setInferences(inferenceData);
      setError(undefined);
    } catch (cause) {
      setError(cause);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const totals = useMemo(() => ({
    succeeded: inferences?.filter((item) => item.status === "SUCCEEDED").length ?? 0,
    failed: inferences?.filter((item) => item.status === "FAILED").length ?? 0,
    tokens: inferences?.reduce(
      (sum, item) => sum + item.usage_prompt_tokens + item.usage_completion_tokens,
      0,
    ) ?? 0,
  }), [inferences]);

  function edit(route: ModelRoute) {
    if (!route.quota) return;
    setEditing(route);
    form.setFieldsValue({
      requests_per_minute: route.quota.requests_per_minute,
      tokens_per_day: route.quota.tokens_per_day,
      max_output_tokens: route.quota.max_output_tokens,
      allowed_request_classes: route.quota.allowed_request_classes,
      enabled: route.quota.enabled,
    });
  }

  async function save(values: ModelQuotaInput) {
    if (!editing?.quota) return;
    setBusy(true);
    try {
      await updateModelQuota(editing.alias, editing.quota.version, values);
      setEditing(undefined);
      await load();
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="生产模型路由与调用审计">
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <Alert
          type="info"
          showIcon
          message="诊断只调用当前生产别名绑定的已审批版本"
          description="发布或回滚确认后原子切换路由；请求受租户配额、固定版本、超时、Prompt Injection Guardrail、结构化输出和引用白名单约束。每次调用保存不含正文的 Context Manifest，固定 Prompt、Release、证据、工具结果、预算、裁剪事实及摘要。"
        />
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!routes || !inferences ? <LoadingState /> : (
          <>
            <Row gutter={[16, 16]}>
              <Col xs={12} lg={6}><Card size="small"><Statistic title="生产路由" value={routes.filter((item) => item.status === "ACTIVE").length} /></Card></Col>
              <Col xs={12} lg={6}><Card size="small"><Statistic title="成功调用" value={totals.succeeded} /></Card></Col>
              <Col xs={12} lg={6}><Card size="small"><Statistic title="失败调用" value={totals.failed} /></Card></Col>
              <Col xs={12} lg={6}><Card size="small"><Statistic title="近期 Token" value={totals.tokens} /></Card></Col>
            </Row>
            {routes.length === 0 ? <EmptyState description="尚无通过生产门禁的模型路由" /> : (
              <Table rowKey="alias" pagination={false} dataSource={routes} columns={[
                { title: "别名", dataIndex: "alias", render: (value: string) => <Typography.Text code>{value}</Typography.Text> },
                { title: "生产发布", dataIndex: "active_release_id", ellipsis: true },
                { title: "状态", dataIndex: "status", render: (value: string) => <Tag color={value === "ACTIVE" ? "green" : "red"}>{value}</Tag> },
                { title: "运行时", dataIndex: "runtime_profile" },
                { title: "RPM", render: (_, row) => row.quota?.requests_per_minute ?? "—" },
                { title: "日 Token", render: (_, row) => row.quota?.tokens_per_day ?? "—" },
                { title: "最大输出", render: (_, row) => row.quota?.max_output_tokens ?? "—" },
                { title: "配额", render: (_, row) => <Space><Tag color={row.quota?.enabled ? "green" : "red"}>{row.quota?.enabled ? "启用" : "停用"}</Tag>{row.quota ? <Button type="link" onClick={() => edit(row)}>调整</Button> : null}</Space> },
              ]} />
            )}
            <Table
              rowKey="inference_request_id"
              size="small"
              pagination={{ pageSize: 10 }}
              dataSource={inferences}
              scroll={{ x: 1350 }}
              columns={[
                { title: "状态", dataIndex: "status", render: (value: string) => <Tag color={value === "SUCCEEDED" ? "green" : value === "FAILED" ? "red" : "gold"}>{value}</Tag> },
                { title: "调用 ID", dataIndex: "inference_request_id", ellipsis: true },
                { title: "发布版本", dataIndex: "resolved_release_id", ellipsis: true },
                { title: "类别", dataIndex: "request_class" },
                { title: "安全结论", dataIndex: "safety_decision" },
                { title: "Guardrail 策略", dataIndex: "guardrail_policy_version", ellipsis: true },
                { title: "命中", render: (_, row) => row.guardrail_findings.length },
                { title: "Token", render: (_, row) => `${row.usage_prompt_tokens} + ${row.usage_completion_tokens}` },
                { title: "上下文", render: (_, row) => row.context_hash ? <Typography.Text code>{row.context_hash.slice(0, 18)}</Typography.Text> : <Tag color="orange">缺失</Tag> },
                { title: "上下文来源", render: (_, row) => row.context_manifest ? `${row.context_manifest.evidence.length} 证据 / ${row.context_manifest.tool_results.length} 工具 / ${row.context_manifest.memories.length} 记忆` : "—" },
                { title: "裁剪", render: (_, row) => row.context_manifest?.truncated_items.length ?? "—" },
                { title: "Prompt hash", dataIndex: "prompt_hash", render: (value: string) => <Typography.Text code>{value.slice(0, 16)}</Typography.Text> },
                { title: "失败原因", dataIndex: "failure_reason", ellipsis: true, render: (value?: string | null) => value ?? "—" },
                { title: "时间", dataIndex: "created_at", render: (value: string) => new Date(value).toLocaleString() },
              ]}
            />
          </>
        )}
      </Space>

      <Modal
        title={`调整模型配额${editing ? ` · ${editing.alias}` : ""}`}
        open={Boolean(editing)}
        onCancel={() => setEditing(undefined)}
        onOk={() => form.submit()}
        confirmLoading={busy}
      >
        <Form form={form} layout="vertical" onFinish={(values) => void save(values)}>
          <Form.Item name="requests_per_minute" label="每分钟请求数" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item>
          <Form.Item name="tokens_per_day" label="每日 Token" rules={[{ required: true }]}><InputNumber min={1} style={{ width: "100%" }} /></Form.Item>
          <Form.Item name="max_output_tokens" label="单次最大输出 Token" rules={[{ required: true }]}><InputNumber min={1} max={32768} style={{ width: "100%" }} /></Form.Item>
          <Form.Item name="allowed_request_classes" label="允许的调用类别" rules={[{ required: true }]}><Select mode="multiple" options={["DIAGNOSIS", "VLM", "ASR", "TTS"].map((value) => ({ value }))} /></Form.Item>
          <Form.Item name="enabled" label="启用配额与路由" valuePropName="checked"><Switch /></Form.Item>
        </Form>
      </Modal>
    </Card>
  );
}
