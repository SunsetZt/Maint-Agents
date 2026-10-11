"use client";

import { Alert, Button, Card, Form, Input, InputNumber, List, Modal, Select, Space, Tag, Typography } from "antd";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  ApiClientError,
  type CustomerCaseUpdate,
  type RefundRequest,
  type RefundRequestOption,
  type RefundRequestProposal,
  getWorkOrder,
  listRefundRequestOptions,
  listRefundRequests,
  proposeRefundRequest,
} from "@/lib/api/client";

type RefundValues = {
  work_order_id: string;
  delivery_target: "INTERNAL" | "ENTERPRISE_FINANCE";
  amount: number;
  currency: "CNY" | "USD" | "EUR";
  cost_center: string;
  reason: string;
};

export function RefundRequestPanel({
  incidentId,
  customerUpdates,
}: {
  incidentId: string;
  customerUpdates: CustomerCaseUpdate[];
}) {
  const [form] = Form.useForm<RefundValues>();
  const [requests, setRequests] = useState<RefundRequest[]>();
  const [options, setOptions] = useState<RefundRequestOption[]>();
  const [financeUnavailableReason, setFinanceUnavailableReason] = useState<string | null>();
  const [proposal, setProposal] = useState<RefundRequestProposal>();
  const [available, setAvailable] = useState(true);
  const [error, setError] = useState<unknown>();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  const eligibleConfirmations = useMemo(() => {
    const latestByWorkOrder = new Map<string, CustomerCaseUpdate>();
    for (const update of customerUpdates) {
      if (update.update_type === "RESULT_CONFIRMATION" && update.work_order_id) {
        latestByWorkOrder.set(update.work_order_id, update);
      }
    }
    return Array.from(latestByWorkOrder.values()).filter(
      (update) => update.result_accepted === false
        || (update.satisfaction_rating !== null && update.satisfaction_rating <= 2),
    );
  }, [customerUpdates]);

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const history = await listRefundRequests(incidentId);
      const workOrderId = eligibleConfirmations[0]?.work_order_id;
      const deliveryOptions = workOrderId
        ? await listRefundRequestOptions(workOrderId)
        : undefined;
      setRequests(history.refundRequests);
      setOptions(deliveryOptions?.options ?? []);
      setFinanceUnavailableReason(deliveryOptions?.financeUnavailableReason ?? null);
      setAvailable(true);
    } catch (caught) {
      if (caught instanceof ApiClientError && caught.status === 403) {
        setAvailable(false);
        return;
      }
      setError(caught);
    }
  }, [eligibleConfirmations, incidentId]);

  useEffect(() => { void load(); }, [load]);

  async function submit(values: RefundValues) {
    setBusy(true);
    setError(undefined);
    try {
      const work = (await getWorkOrder(values.work_order_id)).workOrder;
      const selected = options?.find(
        (option) => option.delivery_target === values.delivery_target,
      );
      if (!selected) throw new Error("退款交接目标已失效，请刷新后重试");
      const confirmation = eligibleConfirmations.find(
        (item) => item.work_order_id === values.work_order_id,
      );
      if (!confirmation) throw new Error("客户退款资格已失效，请刷新后重试");
      const result = await proposeRefundRequest(values.work_order_id, {
        work_order_version: work.version,
        amount_minor: Math.round(values.amount * 100),
        currency: values.currency,
        cost_center: values.cost_center.trim().toUpperCase(),
        reason: values.reason.trim(),
        delivery_target: selected.delivery_target,
        profile_id: selected.delivery_target === "ENTERPRISE_FINANCE"
          ? selected.profile_id
          : undefined,
        reason_code: selected.delivery_target === "ENTERPRISE_FINANCE"
          ? refundReasonCode(confirmation)
          : undefined,
      });
      setProposal(result.proposal);
      setOpen(false);
      form.resetFields();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  async function selectWorkOrder(workOrderId: string) {
    setError(undefined);
    try {
      const result = await listRefundRequestOptions(workOrderId);
      setOptions(result.options);
      setFinanceUnavailableReason(result.financeUnavailableReason);
      form.setFieldValue("delivery_target", "INTERNAL");
    } catch (caught) {
      setError(caught);
    }
  }

  if (!available) return null;

  return (
    <Card title="客户退款与服务补偿（T2）">
      <div className="page-stack">
        <Alert
          type="warning"
          showIcon
          message="只有客户拒绝服务结果或最新满意度为 1–2 星时才能提案"
          description="平台会绑定最新客户确认、工单版本、金额和服务端授权的财务 Profile。财务接收不等于已支付；平台不提供直接付款入口。"
        />
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!requests && !error ? <LoadingState label="正在读取退款申请" /> : null}
        {proposal ? (
          <Alert
            type="success"
            showIcon
            message={`退款提案 ${proposal.proposal_id} 已进入审批箱`}
            description={<Link href="/approvals">前往独立审批并提交内部退款申请</Link>}
          />
        ) : null}
        <Space wrap>
          <Button type="primary" disabled={eligibleConfirmations.length === 0} onClick={() => setOpen(true)}>
            创建退款申请提案
          </Button>
          {eligibleConfirmations.length === 0 ? (
            <Typography.Text type="secondary">当前没有符合条件的最新客户结果确认。</Typography.Text>
          ) : null}
        </Space>
        {options ? (
          <div>
            <Typography.Text strong>服务端授权的交接目标：</Typography.Text>
            <Space wrap style={{ marginInlineStart: 8 }}>
              {options.map((option) => (
                <Tag
                  key={`${option.delivery_target}:${option.profile_id ?? "internal"}`}
                  color={option.delivery_target === "ENTERPRISE_FINANCE" ? "blue" : "default"}
                >
                  {option.display_name}
                </Tag>
              ))}
            </Space>
          </div>
        ) : null}
        {financeUnavailableReason ? (
          <Alert
            type="info"
            showIcon
            message="企业财务交接当前不可用"
            description={`仍可创建平台内部申请；财务不可用原因：${financeUnavailableReason}`}
          />
        ) : null}
        {requests?.length === 0 ? <EmptyState description="当前故障单暂无退款申请" /> : null}
        <List
          dataSource={requests}
          renderItem={(request) => (
            <List.Item>
              <List.Item.Meta
                title={(
                  <Space wrap>
                    <Typography.Text strong>{formatMoney(request.amount_minor, request.currency)}</Typography.Text>
                    <Tag color={refundStatus(request).color}>{refundStatus(request).label}</Tag>
                    <span>{request.work_order_id}</span>
                  </Space>
                )}
                description={(
                  <Space direction="vertical" size={2}>
                    <span>{request.refund_request_id} · {request.cost_center} · {request.reason}</span>
                    {request.delivery_target === "ENTERPRISE_FINANCE" ? (
                      <span>
                        {request.profile_display_name ?? request.provider}
                        {request.external_request_id
                          ? ` · 外部申请 ${maskExternalIdentity(request.external_request_id)}`
                          : " · 尚无财务外部申请标识"}
                      </span>
                    ) : null}
                    {request.status === "RECONCILING" ? (
                      <span>
                        结果未知时只查询原操作号，
                        <Link href={request.reconciliation_id
                          ? `/reconciliations#${request.reconciliation_id}`
                          : "/reconciliations"}
                        >
                          前往对账中心
                        </Link>
                      </span>
                    ) : null}
                    {request.status === "ACCEPTED" && request.payment_status !== "PAID" ? (
                      <span>财务接收不等于已支付；只有受信回执显示 PAID 才会标记已支付。</span>
                    ) : null}
                    {request.status === "REJECTED"
                      || request.status === "CANCELLED"
                      || request.payment_status === "FAILED" ? (
                      <Space wrap>
                        <span>
                          原批准提交不会自动重提或发起付款；请人工核对后创建新的审批提案。
                          {request.provider_reason ? ` 原因：${request.provider_reason}` : ""}
                        </span>
                        <Button size="small" onClick={() => setOpen(true)}>创建新退款提案</Button>
                      </Space>
                    ) : null}
                  </Space>
                )}
              />
            </List.Item>
          )}
        />
      </div>
      <Modal
        open={open}
        title="创建退款申请提案"
        okText="提交审批"
        cancelText="取消"
        confirmLoading={busy}
        onCancel={() => setOpen(false)}
        onOk={() => form.submit()}
        destroyOnHidden
      >
        <Form<RefundValues>
          form={form}
          layout="vertical"
          initialValues={{
            work_order_id: eligibleConfirmations[0]?.work_order_id,
            delivery_target: "INTERNAL",
            currency: "CNY",
          }}
          onFinish={(values) => void submit(values)}
        >
          <Form.Item name="work_order_id" label="客户拒绝或低评分的工单" rules={[{ required: true }]}>
            <Select
              options={eligibleConfirmations.map((update) => ({
                value: update.work_order_id!,
                label: `${update.work_order_id} · ${update.result_accepted === false ? "拒绝结果" : `${update.satisfaction_rating} 星`}`,
              }))}
              onChange={(value: string) => void selectWorkOrder(value)}
            />
          </Form.Item>
          <Form.Item name="delivery_target" label="交接目标" rules={[{ required: true }]}>
            <Select options={options?.map((option) => ({
              value: option.delivery_target,
              label: option.display_name,
            }))} />
          </Form.Item>
          <Form.Item name="amount" label="申请金额" rules={[{ required: true, type: "number", min: 0.01, max: 10000 }]}>
            <InputNumber min={0.01} max={10000} precision={2} style={{ width: "100%" }} />
          </Form.Item>
          <Form.Item name="currency" label="币种" rules={[{ required: true }]}>
            <Select options={["CNY", "USD", "EUR"].map((value) => ({ value }))} />
          </Form.Item>
          <Form.Item
            name="cost_center"
            label="成本中心"
            rules={[
              { required: true, min: 2, max: 64 },
              { pattern: /^[A-Za-z0-9_-]+$/, message: "仅允许字母、数字、下划线和连字符" },
            ]}
          >
            <Input maxLength={64} placeholder="例如 SERVICE_EAST" />
          </Form.Item>
          <Form.Item name="reason" label="补偿理由" rules={[{ required: true, min: 10, max: 2000 }]}>
            <Input.TextArea rows={5} showCount maxLength={2000} />
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
}

function formatMoney(minor: number, currency: string): string {
  return new Intl.NumberFormat("zh-CN", { style: "currency", currency }).format(minor / 100);
}

function refundReasonCode(update: CustomerCaseUpdate) {
  return update.result_accepted === false
    ? "SERVICE_NOT_ACCEPTED" as const
    : "LOW_SATISFACTION" as const;
}

function maskExternalIdentity(value: string): string {
  return value.length <= 6 ? "******" : `…${value.slice(-6)}`;
}

function refundStatus(request: RefundRequest): { label: string; color: string } {
  if (request.delivery_target === "INTERNAL" && request.status === "SUBMITTED") {
    return { label: "平台内部申请已提交", color: "default" };
  }
  if (request.status === "REJECTED") return { label: "财务已拒绝", color: "error" };
  if (request.status === "CANCELLED") return { label: "财务已取消", color: "default" };
  if (request.status === "RECONCILING") return { label: "财务结果核对中", color: "warning" };
  if (request.status === "SUBMITTING") return { label: "正在提交财务", color: "processing" };
  if (request.status === "ACCEPTED") {
    return {
      NOT_STARTED: { label: "财务已接收，待处理", color: "processing" },
      PROCESSING: { label: "支付处理中", color: "processing" },
      PAID: { label: "已支付", color: "success" },
      FAILED: { label: "支付失败", color: "error" },
    }[request.payment_status] ?? { label: request.payment_status, color: "default" };
  }
  return { label: request.status, color: "default" };
}
