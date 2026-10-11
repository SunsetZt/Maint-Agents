"use client";

import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  InputNumber,
  Modal,
  Select,
  Space,
  Table,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useState } from "react";

import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  type EmergencyAccessGrant,
  type EmergencyAccessRequestInput,
  decideEmergencyAccessGrant,
  listEmergencyAccessGrants,
  requestEmergencyAccessGrant,
  revokeEmergencyAccessGrant,
} from "@/lib/api/client";

type DecisionValues = {
  decision: "APPROVE" | "REJECT";
  reason: string;
};

type RevokeValues = { reason: string };

const STATUS_COLORS: Record<string, string> = {
  PENDING: "gold",
  APPROVED: "green",
  REJECTED: "red",
  REVOKED: "default",
  EXPIRED: "default",
};

export function EmergencyAccessPanel({
  activeGrantId,
  onUseGrant,
}: {
  activeGrantId?: string;
  onUseGrant: (grantId?: string) => void;
}) {
  const [grants, setGrants] = useState<EmergencyAccessGrant[]>();
  const [error, setError] = useState<unknown>();
  const [receipt, setReceipt] = useState<string>();
  const [requestOpen, setRequestOpen] = useState(false);
  const [decisionTarget, setDecisionTarget] = useState<EmergencyAccessGrant>();
  const [revokeTarget, setRevokeTarget] = useState<EmergencyAccessGrant>();
  const [submitting, setSubmitting] = useState(false);
  const [requestForm] = Form.useForm<EmergencyAccessRequestInput>();
  const [decisionForm] = Form.useForm<DecisionValues>();
  const [revokeForm] = Form.useForm<RevokeValues>();

  const load = useCallback(async () => {
    try {
      const result = await listEmergencyAccessGrants();
      setGrants(result.grants);
      setError(undefined);
      if (
        activeGrantId
        && !result.grants.some(
          (item) => item.grant_id === activeGrantId && item.status === "APPROVED",
        )
      ) {
        onUseGrant(undefined);
      }
    } catch (cause) {
      setError(cause);
    }
  }, [activeGrantId, onUseGrant]);

  useEffect(() => {
    void load();
  }, [load]);

  async function submitRequest(values: EmergencyAccessRequestInput) {
    setSubmitting(true);
    try {
      const grant = await requestEmergencyAccessGrant(values);
      setReceipt(`紧急授权申请 ${grant.grant_id} 已提交，等待独立安全审计员处理。`);
      setRequestOpen(false);
      requestForm.resetFields();
      await load();
    } catch (cause) {
      setError(cause);
    } finally {
      setSubmitting(false);
    }
  }

  async function submitDecision(values: DecisionValues) {
    if (!decisionTarget) return;
    setSubmitting(true);
    try {
      const grant = await decideEmergencyAccessGrant(
        decisionTarget.grant_id,
        decisionTarget.version,
        values,
      );
      setReceipt(`申请 ${grant.grant_id} 已${values.decision === "APPROVE" ? "批准" : "拒绝"}。`);
      setDecisionTarget(undefined);
      decisionForm.resetFields();
      await load();
    } catch (cause) {
      setError(cause);
    } finally {
      setSubmitting(false);
    }
  }

  async function submitRevoke(values: RevokeValues) {
    if (!revokeTarget) return;
    setSubmitting(true);
    try {
      const grant = await revokeEmergencyAccessGrant(
        revokeTarget.grant_id,
        revokeTarget.version,
        values,
      );
      setReceipt(`授权 ${grant.grant_id} 已立即撤销。`);
      if (activeGrantId === grant.grant_id) onUseGrant(undefined);
      setRevokeTarget(undefined);
      revokeForm.resetFields();
      await load();
    } catch (cause) {
      setError(cause);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card
      title="IAM-004 紧急访问治理"
      extra={<Button type="primary" onClick={() => setRequestOpen(true)}>申请临时访问</Button>}
    >
      <Space direction="vertical" size="middle" style={{ width: "100%" }}>
        <Alert
          type="warning"
          showIcon
          message="仅限已登记安全事件的只读审计访问"
          description="申请绑定事件编号、security_audit.read、固定资源和最长 60 分钟有效期。请求人不能自批，授权不会跨租户，也不能执行写操作或设备控制。"
        />
        {activeGrantId ? (
          <Alert
            type="success"
            showIcon
            message={`当前审计查询使用授权 ${activeGrantId}`}
            action={<Button size="small" onClick={() => onUseGrant(undefined)}>停止使用</Button>}
          />
        ) : null}
        {receipt ? <Alert closable type="success" message={receipt} onClose={() => setReceipt(undefined)} /> : null}
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!grants && !error ? <LoadingState label="正在读取紧急授权" /> : null}
        {grants ? (
          <Table<EmergencyAccessGrant>
            rowKey="grant_id"
            pagination={false}
            dataSource={grants}
            scroll={{ x: 1200 }}
            columns={[
              {
                title: "状态",
                dataIndex: "status",
                width: 110,
                render: (value: string) => <Tag color={STATUS_COLORS[value]}>{value}</Tag>,
              },
              { title: "事件编号", dataIndex: "incident_number", width: 170 },
              { title: "申请人", dataIndex: "requester_subject_id", width: 150 },
              { title: "动作", dataIndex: "action", width: 170 },
              {
                title: "过期时间",
                dataIndex: "requested_expires_at",
                width: 190,
                render: (value: string) => new Date(value).toLocaleString(),
              },
              { title: "使用次数", dataIndex: "usage_count", width: 100 },
              {
                title: "版本",
                dataIndex: "version",
                width: 80,
                render: (value: number) => <Typography.Text code>{value}</Typography.Text>,
              },
              {
                title: "操作",
                key: "actions",
                fixed: "right",
                width: 250,
                render: (_, grant) => (
                  <Space wrap>
                    {grant.legal_actions.includes("DECIDE") ? (
                      <Button size="small" onClick={() => setDecisionTarget(grant)}>处理</Button>
                    ) : null}
                    {grant.legal_actions.includes("REVOKE") ? (
                      <Button danger size="small" onClick={() => setRevokeTarget(grant)}>撤销</Button>
                    ) : null}
                    {grant.legal_actions.includes("USE_SECURITY_AUDIT") ? (
                      <Button
                        size="small"
                        type={activeGrantId === grant.grant_id ? "primary" : "default"}
                        onClick={() => onUseGrant(grant.grant_id)}
                      >用于审计查询</Button>
                    ) : null}
                  </Space>
                ),
              },
            ]}
            expandable={{
              expandedRowRender: (grant) => (
                <Space direction="vertical">
                  <Typography.Text>申请理由：{grant.justification}</Typography.Text>
                  {grant.decisions.map((item) => (
                    <Typography.Text key={item.decision_id} type="secondary">
                      {new Date(item.occurred_at).toLocaleString()} · {item.decision} · {item.actor_subject_id} · {item.reason}
                    </Typography.Text>
                  ))}
                  {grant.usage_history.map((item) => (
                    <Typography.Text key={item.usage_id} type="secondary">
                      实际使用 · {new Date(item.occurred_at).toLocaleString()} · {item.subject_id} · {item.action} · {item.resource_id} · {item.request_id}
                    </Typography.Text>
                  ))}
                </Space>
              ),
            }}
          />
        ) : null}
      </Space>

      <Modal
        title="申请紧急只读访问"
        open={requestOpen}
        onCancel={() => setRequestOpen(false)}
        onOk={() => requestForm.submit()}
        confirmLoading={submitting}
      >
        <Form<EmergencyAccessRequestInput>
          form={requestForm}
          layout="vertical"
          initialValues={{
            action: "security_audit.read",
            resource_id: "security-audit",
            ttl_minutes: 30,
          }}
          onFinish={(values) => void submitRequest(values)}
        >
          <Form.Item name="incident_number" label="安全事件编号" rules={[{ required: true }]}>
            <Input placeholder="SEC-2026-0001" />
          </Form.Item>
          <Form.Item name="action" label="限定动作"><Input disabled /></Form.Item>
          <Form.Item name="resource_id" label="限定资源"><Input disabled /></Form.Item>
          <Form.Item name="ttl_minutes" label="有效期（分钟）" rules={[{ required: true }]}>
            <InputNumber min={5} max={60} style={{ width: "100%" }} />
          </Form.Item>
          <Form.Item name="justification" label="必要性说明" rules={[{ required: true, min: 10 }]}>
            <Input.TextArea rows={4} maxLength={1000} showCount />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title={`处理申请 ${decisionTarget?.grant_id ?? ""}`}
        open={Boolean(decisionTarget)}
        onCancel={() => setDecisionTarget(undefined)}
        onOk={() => decisionForm.submit()}
        confirmLoading={submitting}
      >
        <Form<DecisionValues>
          form={decisionForm}
          layout="vertical"
          initialValues={{ decision: "APPROVE" }}
          onFinish={(values) => void submitDecision(values)}
        >
          <Form.Item name="decision" label="决定" rules={[{ required: true }]}>
            <Select options={[{ label: "批准", value: "APPROVE" }, { label: "拒绝", value: "REJECT" }]} />
          </Form.Item>
          <Form.Item name="reason" label="决定依据" rules={[{ required: true, min: 3 }]}>
            <Input.TextArea rows={3} maxLength={500} showCount />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title={`撤销授权 ${revokeTarget?.grant_id ?? ""}`}
        open={Boolean(revokeTarget)}
        onCancel={() => setRevokeTarget(undefined)}
        onOk={() => revokeForm.submit()}
        confirmLoading={submitting}
      >
        <Form<RevokeValues>
          form={revokeForm}
          layout="vertical"
          onFinish={(values) => void submitRevoke(values)}
        >
          <Form.Item name="reason" label="撤销原因" rules={[{ required: true, min: 3 }]}>
            <Input.TextArea rows={3} maxLength={500} showCount />
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
}
