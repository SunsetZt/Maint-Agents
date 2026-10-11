"use client";

import { Alert, Button, Card, Input, List, Rate, Space, Tag, Typography } from "antd";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import { FactGrid, StatusTag } from "@/components/m2/BusinessState";
import { ServiceQuotationPanel } from "@/components/m2/ServiceQuotationPanel";
import {
  type CustomerCase,
  type CustomerCaseUpdate,
  appendCustomerCaseUpdate,
  getCustomerCase,
  listCustomerCaseUpdates,
} from "@/lib/api/client";

export default function CustomerCasePage() {
  const { incidentId } = useParams<{ incidentId: string }>();
  const [customerCase, setCustomerCase] = useState<CustomerCase>();
  const [updates, setUpdates] = useState<CustomerCaseUpdate[]>();
  const [message, setMessage] = useState("");
  const [evidenceId, setEvidenceId] = useState("");
  const [rating, setRating] = useState(5);
  const [confirmationMessage, setConfirmationMessage] = useState("现场复测正常，确认本次服务结果");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>();

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const [caseResult, updateResult] = await Promise.all([
        getCustomerCase(incidentId),
        listCustomerCaseUpdates(incidentId),
      ]);
      setCustomerCase(caseResult.customerCase);
      setUpdates(updateResult.updates);
    } catch (caught) {
      setError(caught);
    }
  }, [incidentId]);

  useEffect(() => { void load(); }, [load]);

  async function appendInformation() {
    if (!message.trim()) return;
    await submit({
      client_operation_id: operationId(),
      update_type: "INFORMATION",
      message: message.trim(),
      evidence_ids: [],
    });
    setMessage("");
  }

  async function appendEvidence() {
    if (!evidenceId.trim()) return;
    await submit({
      client_operation_id: operationId(),
      update_type: "EVIDENCE",
      message: message.trim() || "客户补充现场证据",
      evidence_ids: [evidenceId.trim()],
    });
    setEvidenceId("");
    setMessage("");
  }

  async function confirmResult(workOrderId: string | undefined, accepted: boolean) {
    const input: Parameters<typeof appendCustomerCaseUpdate>[1] = {
      client_operation_id: operationId(),
      update_type: "RESULT_CONFIRMATION",
      message: confirmationMessage.trim(),
      evidence_ids: [],
      result_accepted: accepted,
      satisfaction_rating: rating,
    };
    if (workOrderId) input.work_order_id = workOrderId;
    await submit(input);
  }

  async function submit(input: Parameters<typeof appendCustomerCaseUpdate>[1]) {
    setBusy(true);
    setError(undefined);
    try {
      await appendCustomerCaseUpdate(incidentId, input);
      await load();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>服务案例进度</Typography.Title>
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!customerCase && !error ? <LoadingState label="正在读取服务进度" /> : null}
        {customerCase ? (
          <>
            <Card
              title={customerCase.asset_display_name ?? customerCase.asset_id}
              extra={<StatusTag status={customerCase.status} version={customerCase.version} />}
            >
              <Typography.Paragraph>{customerCase.description}</Typography.Paragraph>
              <FactGrid facts={[
                ["服务编号", customerCase.incident_id],
                ["设备", customerCase.asset_id],
                ["型号", customerCase.model_code],
                ["站点", customerCase.site_name ?? customerCase.site_id],
                ["发起时间", formatTimestamp(customerCase.created_at)],
                ["最后更新", formatTimestamp(customerCase.updated_at)],
              ]} />
              <Link href="/portal">返回客户服务门户</Link>
            </Card>

            <ServiceQuotationPanel incidentId={incidentId} mode="customer" />

            {customerCase.resolution?.mode === "REMOTE" ? (
              <Card title="远程解决结果">
                <div className="page-stack">
                  <Space wrap>
                    <Tag color="cyan">REMOTE</Tag>
                    {customerCase.resolution.result_accepted === true ? <Tag color="green">已接受</Tag> : null}
                    {customerCase.resolution.result_accepted === false ? <Tag color="red">仍需处理</Tag> : null}
                  </Space>
                  <Typography.Paragraph>{customerCase.resolution.summary}</Typography.Paragraph>
                  <FactGrid facts={[
                    ["解决时间", formatTimestamp(customerCase.resolution.resolved_at)],
                    ["解决证据", customerCase.resolution.evidence_ids.join("、") || "—"],
                    ["确认时间", formatTimestamp(customerCase.resolution.confirmation_occurred_at)],
                  ]} />
                  {customerCase.legal_actions.includes("CONFIRM_REMOTE_RESULT") ? (
                    <div className="form-grid">
                      <Input.TextArea
                        aria-label="远程结果确认说明"
                        rows={2}
                        value={confirmationMessage}
                        onChange={(event) => setConfirmationMessage(event.target.value)}
                        placeholder="说明现场复测结果"
                      />
                      <Space><span>满意度</span><Rate value={rating} onChange={setRating} /></Space>
                      <Space wrap>
                        <Button type="primary" loading={busy} disabled={!confirmationMessage.trim()} onClick={() => void confirmResult(undefined, true)}>接受远程解决结果</Button>
                        <Button danger loading={busy} disabled={!confirmationMessage.trim()} onClick={() => void confirmResult(undefined, false)}>仍有问题，继续处理</Button>
                      </Space>
                    </div>
                  ) : null}
                </div>
              </Card>
            ) : null}

            <Card title="关联工单与服务结果">
              {customerCase.work_orders.length === 0 ? <Alert type="info" showIcon message={customerCase.resolution?.mode === "REMOTE" ? "本次故障通过远程协作解决，无需现场工单" : "服务团队正在受理，尚未创建现场工单"} /> : null}
              <List
                dataSource={customerCase.work_orders}
                renderItem={(work) => (
                  <List.Item>
                    <div className="page-stack" style={{ width: "100%" }}>
                      <Space wrap>
                        <Typography.Text strong>{work.work_order_id}</Typography.Text>
                        <Tag color="blue">{work.status}</Tag>
                        <Tag>{work.priority}</Tag>
                        {work.result_accepted === true ? <Tag color="green">客户已接受</Tag> : null}
                        {work.result_accepted === false ? <Tag color="red">客户要求继续处理</Tag> : null}
                      </Space>
                      <FactGrid facts={[
                        ["SLA 截止", formatTimestamp(work.sla_due_at)],
                        ["最后更新", formatTimestamp(work.updated_at)],
                      ]} />
                      {work.legal_actions.includes("CONFIRM_RESULT") ? (
                        <div className="form-grid">
                          <Input.TextArea rows={2} value={confirmationMessage} onChange={(event) => setConfirmationMessage(event.target.value)} placeholder="结果确认说明" />
                          <Space><span>满意度</span><Rate value={rating} onChange={setRating} /></Space>
                          <Space wrap>
                            <Button type="primary" loading={busy} disabled={!confirmationMessage.trim()} onClick={() => void confirmResult(work.work_order_id, true)}>接受服务结果</Button>
                            <Button danger loading={busy} disabled={!confirmationMessage.trim()} onClick={() => void confirmResult(work.work_order_id, false)}>仍有问题，要求继续处理</Button>
                          </Space>
                        </div>
                      ) : null}
                    </div>
                  </List.Item>
                )}
              />
            </Card>

            {customerCase.legal_actions.includes("ADD_UPDATE") ? (
              <Card title="补充资料">
                <div className="form-grid">
                  <Input.TextArea rows={3} value={message} onChange={(event) => setMessage(event.target.value)} placeholder="补充故障现象、可联系时间或现场变化" />
                  <Button loading={busy} disabled={!message.trim()} onClick={() => void appendInformation()}>追加说明</Button>
                  <Space.Compact block>
                    <Input value={evidenceId} onChange={(event) => setEvidenceId(event.target.value)} placeholder="已上传的 Evidence ID" />
                    <Button loading={busy} disabled={!evidenceId.trim()} onClick={() => void appendEvidence()}>关联补充证据</Button>
                  </Space.Compact>
                  <Typography.Text type="secondary">照片、录音或视频需先通过受控上传与安全扫描，本页只关联 Evidence ID。</Typography.Text>
                </div>
              </Card>
            ) : null}

            <Card title="我提交的更新">
              {updates?.length === 0 ? <EmptyState description="尚未补充资料或确认结果" /> : null}
              <List
                dataSource={updates}
                renderItem={(update) => (
                  <List.Item>
                    <List.Item.Meta
                      title={<Space><Tag>{update.update_type}</Tag>{update.work_order_id ? <span>{update.work_order_id}</span> : null}</Space>}
                      description={`${formatTimestamp(update.occurred_at)} · ${update.message}${update.satisfaction_rating ? ` · 满意度 ${update.satisfaction_rating}/5` : ""}`}
                    />
                  </List.Item>
                )}
              />
            </Card>
          </>
        ) : null}
      </div>
    </AppShell>
  );
}

function operationId(): string {
  return `portal-${crypto.randomUUID()}`;
}

function formatTimestamp(value: string | null): string {
  return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "—";
}
