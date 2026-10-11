"use client";

import { Alert, Button, Card, Form, Input, List, Modal, Select, Space, Tag, Typography } from "antd";
import { useCallback, useEffect, useState } from "react";

import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  ApiClientError,
  type EquipmentControlHandoff,
  createEquipmentControlHandoff,
  listEquipmentControlHandoffs,
} from "@/lib/api/client";

type HandoffValues = {
  requested_operation: "STOP" | "REMOTE_START" | "PLC_PARAMETER_CHANGE" | "INTERLOCK_OVERRIDE";
  reason: string;
  evidence_ids_text: string;
};

const operations = [
  { value: "STOP", label: "请求外部系统评估停机" },
  { value: "REMOTE_START", label: "请求外部系统评估远程启动" },
  { value: "PLC_PARAMETER_CHANGE", label: "请求外部系统评估 PLC 参数变更" },
  { value: "INTERLOCK_OVERRIDE", label: "请求外部系统评估联锁处置" },
] as const;

export function EquipmentControlHandoffPanel({
  incidentId,
  incidentVersion,
  incidentStatus,
  evidenceBundleId,
}: {
  incidentId: string;
  incidentVersion: number;
  incidentStatus: string;
  evidenceBundleId: string;
}) {
  const [form] = Form.useForm<HandoffValues>();
  const [handoffs, setHandoffs] = useState<EquipmentControlHandoff[]>();
  const [available, setAvailable] = useState(true);
  const [error, setError] = useState<unknown>();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setError(undefined);
    try {
      setHandoffs((await listEquipmentControlHandoffs(incidentId)).handoffs);
      setAvailable(true);
    } catch (caught) {
      if (caught instanceof ApiClientError && caught.status === 403) {
        setAvailable(false);
        return;
      }
      setError(caught);
    }
  }, [incidentId]);

  useEffect(() => { void load(); }, [load]);

  async function submit(values: HandoffValues) {
    setBusy(true);
    setError(undefined);
    try {
      const evidenceIds = Array.from(new Set(
        values.evidence_ids_text.split(/[\n,]/).map((value) => value.trim()).filter(Boolean),
      ));
      await createEquipmentControlHandoff(
        incidentId,
        {
          incident_version: incidentVersion,
          requested_operation: values.requested_operation,
          reason: values.reason.trim(),
          evidence_ids: evidenceIds,
        },
        crypto.randomUUID(),
      );
      setOpen(false);
      form.resetFields();
      await load();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  if (!available) return null;
  const active = !["CLOSED", "CANCELLED"].includes(incidentStatus);

  return (
    <Card title="设备控制外部安全交接（T3）">
      <div className="page-stack">
        <Alert
          type="error"
          showIcon
          message="本平台不会执行停机、启动、PLC 参数修改或解除联锁"
          description="这里只生成带 Incident 版本和已确认证据的外部复核交接单。普通审批、Agent 输出和本页面操作都不会产生设备控制凭证。"
        />
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!handoffs && !error ? <LoadingState label="正在读取外部安全交接记录" /> : null}
        <Space wrap>
          <Button danger type="primary" disabled={!active} onClick={() => setOpen(true)}>
            创建外部安全交接单
          </Button>
          {!active ? <Typography.Text type="secondary">已关闭或取消的故障单不能创建新交接。</Typography.Text> : null}
        </Space>

        {handoffs?.length === 0 ? <EmptyState description="当前故障单暂无设备控制交接" /> : null}
        <List
          dataSource={handoffs}
          renderItem={(handoff) => (
            <List.Item>
              <List.Item.Meta
                title={(
                  <Space wrap>
                    <Typography.Text strong>{handoff.requested_operation}</Typography.Text>
                    <Tag color="red">{handoff.status}</Tag>
                  </Space>
                )}
                description={(
                  <Space direction="vertical" size={2}>
                    <span>{handoff.handoff_id} · {handoff.target_system}</span>
                    <span>ToolCall {handoff.tool_call_id} · Incident v{handoff.incident_version}</span>
                    <span>证据：{handoff.evidence_ids.join(", ")}</span>
                    <span>{handoff.reason}</span>
                  </Space>
                )}
              />
            </List.Item>
          )}
        />
      </div>

      <Modal
        open={open}
        title="创建 T3 外部安全交接单"
        okText="生成交接单"
        cancelText="取消"
        confirmLoading={busy}
        onCancel={() => setOpen(false)}
        onOk={() => form.submit()}
        destroyOnHidden
      >
        <Form<HandoffValues>
          form={form}
          layout="vertical"
          initialValues={{
            requested_operation: "STOP",
            evidence_ids_text: evidenceBundleId,
          }}
          onFinish={(values) => void submit(values)}
        >
          <Form.Item name="requested_operation" label="请求外部系统评估的动作" rules={[{ required: true }]}>
            <Select options={[...operations]} />
          </Form.Item>
          <Form.Item
            name="evidence_ids_text"
            label="已确认 Evidence ID"
            rules={[{ required: true, message: "至少填写一个已确认的证据 ID" }]}
          >
            <Input.TextArea rows={2} placeholder="多个 ID 使用换行或逗号分隔" />
          </Form.Item>
          <Form.Item
            name="reason"
            label="外部安全复核原因"
            rules={[{ required: true, min: 10, max: 2000 }]}
          >
            <Input.TextArea rows={5} showCount maxLength={2000} />
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
}
