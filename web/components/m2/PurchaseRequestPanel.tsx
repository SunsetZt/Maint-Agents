"use client";

import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  InputNumber,
  List,
  Modal,
  Select,
  Space,
  Tag,
  Typography,
} from "antd";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import { FactGrid } from "@/components/m2/BusinessState";
import {
  type PurchaseFulfillment,
  type PurchaseRequest,
  type PurchaseRequestOption,
  type PurchaseRequestProposal,
  type PurchaseReservationReadiness,
  type SparePartsLookup,
  listPurchaseRequestOptions,
  listPurchaseRequests,
  lookupSpareParts,
  proposePurchaseRequest,
} from "@/lib/api/client";

type PurchaseValues = {
  delivery_target: "INTERNAL" | "ENTERPRISE_ERP";
  part_number: string;
  quantity: number;
  estimated_unit_cost: number;
  currency: "CNY" | "USD" | "EUR";
  cost_center: string;
  justification: string;
};

export function PurchaseRequestPanel({
  incidentId,
  incidentVersion,
  incidentStatus,
  assetId,
}: {
  incidentId: string;
  incidentVersion: number;
  incidentStatus: string;
  assetId: string;
}) {
  const [form] = Form.useForm<PurchaseValues>();
  const [lookup, setLookup] = useState<SparePartsLookup>();
  const [requests, setRequests] = useState<PurchaseRequest[]>();
  const [options, setOptions] = useState<PurchaseRequestOption[]>();
  const [erpUnavailableReason, setErpUnavailableReason] = useState<string | null>();
  const [proposal, setProposal] = useState<PurchaseRequestProposal>();
  const [error, setError] = useState<unknown>();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const [inventory, history, deliveryOptions] = await Promise.all([
        lookupSpareParts(assetId),
        listPurchaseRequests(incidentId),
        listPurchaseRequestOptions(incidentId),
      ]);
      setLookup(inventory.lookup);
      setRequests(history.purchaseRequests);
      setOptions(deliveryOptions.options);
      setErpUnavailableReason(deliveryOptions.erpUnavailableReason);
    } catch (caught) {
      setError(caught);
    }
  }, [assetId, incidentId]);

  useEffect(() => { void load(); }, [load]);

  async function submit(values: PurchaseValues) {
    setBusy(true);
    setError(undefined);
    try {
      const selected = options?.find(
        (option) => option.delivery_target === values.delivery_target,
      );
      if (!selected) throw new Error("采购交接目标已失效，请刷新后重试");
      const result = await proposePurchaseRequest(incidentId, {
        incident_version: incidentVersion,
        part_number: values.part_number,
        quantity: values.quantity,
        estimated_unit_cost_minor: Math.round(values.estimated_unit_cost * 100),
        currency: values.currency,
        cost_center: values.cost_center.trim().toUpperCase(),
        justification: values.justification,
        delivery_target: selected.delivery_target,
        profile_id: selected.profile_id,
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

  const currentAvailability = lookup?.availability;
  const canPropose = incidentStatus === "DIAGNOSED" && Boolean(currentAvailability);

  return (
    <Card title="库存缺口与采购申请（T2）">
      <div className="page-stack">
        <Alert
          type="warning"
          showIcon
          message="只有权威库存无法满足需求时才能创建采购提案"
          description="提交提案会绑定库存快照、数量、预计金额、币种、成本中心和理由；审批后执行时再次查询 WMS，库存已经满足则停止采购。"
        />
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {(!lookup || !options) && !error ? (
          <LoadingState label="正在读取备件、采购记录和交接配置" />
        ) : null}
        {currentAvailability ? (
          <FactGrid facts={[
            ["WMS 料号", currentAvailability.part_number],
            ["当前可用数量", currentAvailability.available_quantity],
            ["库存来源", currentAvailability.source],
            ["数据时间", currentAvailability.as_of],
          ]} />
        ) : null}
        {proposal ? (
          <Alert
            type="success"
            showIcon
            message={`采购提案 ${proposal.proposal_id} 已进入审批箱`}
            description={<Link href="/approvals">前往审批与提交采购申请</Link>}
          />
        ) : null}
        {options ? (
          <div>
            <Typography.Text strong>服务端授权的交接目标：</Typography.Text>
            <Space wrap style={{ marginInlineStart: 8 }}>
              {options.map((option) => (
                <Tag
                  key={`${option.delivery_target}:${option.profile_id ?? "internal"}`}
                  color={option.delivery_target === "ENTERPRISE_ERP" ? "blue" : "default"}
                >
                  {option.display_name}
                </Tag>
              ))}
            </Space>
          </div>
        ) : null}
        {erpUnavailableReason ? (
          <Alert
            type="info"
            showIcon
            message="企业 ERP 交接当前不可用"
            description={`仍可创建平台内部申请；ERP 不可用原因：${erpUnavailableReason}`}
          />
        ) : null}
        <Space wrap>
          <Button type="primary" disabled={!canPropose} onClick={() => setOpen(true)}>
            创建采购申请提案
          </Button>
          {incidentStatus !== "DIAGNOSED" ? (
            <Typography.Text type="secondary">仅已完成诊断的故障单可申请采购。</Typography.Text>
          ) : null}
        </Space>

        <Typography.Title level={5}>已提交采购申请</Typography.Title>
        {requests?.length === 0 ? <EmptyState description="当前故障单暂无采购申请" /> : null}
        <List
          dataSource={requests}
          renderItem={(request) => (
            <List.Item>
              <List.Item.Meta
                title={(
                  <Space wrap>
                    <Typography.Text strong>{request.part_number}</Typography.Text>
                    <Tag color={purchaseStatus(request).color}>
                      {purchaseStatus(request).label}
                    </Tag>
                    <span>{request.quantity} 件</span>
                  </Space>
                )}
                description={(
                  <Space direction="vertical" size={2}>
                    <span>{request.purchase_request_id} · 成本中心 {request.cost_center}</span>
                    <span>
                      预计总额 {formatMoney(request.estimated_total_cost_minor, request.currency)}
                      {` · 提交时库存 ${request.available_quantity_at_submission}`}
                    </span>
                    <span>{request.justification}</span>
                    {request.delivery_target === "ENTERPRISE_ERP" ? (
                      <span>
                        {request.profile_display_name ?? request.provider}
                        {request.external_request_id
                          ? ` · ERP 申请标识 ${request.external_request_id}`
                          : " · 尚无 ERP 外部申请标识"}
                      </span>
                    ) : null}
                    {request.status === "RECONCILING" ? (
                      <span>
                        结果未知只会使用原操作号查询，
                        <Link
                          href={request.reconciliation_id
                            ? `/reconciliations#${request.reconciliation_id}`
                            : "/reconciliations"}
                        >
                          前往对账中心
                        </Link>
                        {request.reconciliation_id
                          ? `（${request.reconciliation_id}）`
                          : null}
                      </span>
                    ) : null}
                    {request.status === "ACCEPTED" ? (
                      <span>ERP 已确认受理采购申请；这不代表已选供应商或已生成采购订单。</span>
                    ) : null}
                    <ProcurementFulfillmentState request={request} assetId={assetId}
                      incidentVersion={incidentVersion} />
                    {request.status === "REJECTED" || request.status === "CANCELLED" ? (
                      <Space wrap>
                        <span>
                          原批准提交不会自动重放；如需调整参数，请重新创建提案并完成新的审批。
                          {request.reason ? ` 原因：${request.reason}` : ""}
                        </span>
                        <Button
                          size="small"
                          disabled={!canPropose}
                          onClick={() => setOpen(true)}
                        >
                          创建新采购提案
                        </Button>
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
        title="创建采购申请提案"
        okText="提交审批"
        cancelText="取消"
        confirmLoading={busy}
        onCancel={() => setOpen(false)}
        onOk={() => form.submit()}
        destroyOnHidden
      >
        <Form<PurchaseValues>
          form={form}
          layout="vertical"
          validateTrigger="onBlur"
          initialValues={{
            delivery_target: "INTERNAL",
            part_number: currentAvailability?.part_number,
            quantity: (currentAvailability?.available_quantity ?? 0) + 1,
            currency: "CNY",
          }}
          onFinish={(values) => void submit(values)}
        >
          <Form.Item name="delivery_target" label="交接目标" rules={[{ required: true }]}>
            <Select
              options={options?.map((option) => ({
                value: option.delivery_target,
                label: option.display_name,
              }))}
            />
          </Form.Item>
          <Form.Item name="part_number" label="采购料号" rules={[{ required: true }]}>
            <Select
              showSearch
              optionFilterProp="label"
              options={lookup?.components.map((component) => ({
                value: component.part_number,
                label: `${component.part_number} · ${component.part_name}`,
              }))}
            />
          </Form.Item>
          <Form.Item
            name="quantity"
            label="需求数量"
            rules={[
              { required: true, type: "number", min: 1, max: 100 },
              {
                validator: async (_, value: number | undefined) => {
                  if (
                    value !== undefined
                    && currentAvailability
                    && value <= currentAvailability.available_quantity
                  ) {
                    throw new Error("需求数量必须大于当前可用库存");
                  }
                },
              },
            ]}
          >
            <InputNumber min={1} max={100} style={{ width: "100%" }} />
          </Form.Item>
          <Form.Item
            name="estimated_unit_cost"
            label="预计单价"
            rules={[{ required: true, type: "number", min: 0, max: 10000 }]}
          >
            <InputNumber min={0} max={10000} precision={2} style={{ width: "100%" }} />
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
            <Input placeholder="例如 SERVICE_EAST" maxLength={64} />
          </Form.Item>
          <Form.Item
            name="justification"
            label="采购理由"
            rules={[{ required: true, min: 3, max: 1000 }]}
          >
            <Input.TextArea rows={4} showCount maxLength={1000} />
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
}

function formatMoney(minor: number, currency: string): string {
  return new Intl.NumberFormat("zh-CN", {
    style: "currency",
    currency,
  }).format(minor / 100);
}

function purchaseStatus(request: PurchaseRequest): { label: string; color: string } {
  if (request.delivery_target === "INTERNAL" && request.status === "SUBMITTED") {
    return { label: "平台内部申请已提交", color: "default" };
  }
  return ({
    SUBMITTING: { label: "ERP 正在提交", color: "processing" },
    RECONCILING: { label: "ERP 结果核对中", color: "warning" },
    ACCEPTED: { label: "ERP 已受理", color: "success" },
    REJECTED: { label: "ERP 已拒绝", color: "error" },
    CANCELLED: { label: "ERP 已取消", color: "default" },
  }[request.status] ?? { label: request.status, color: "default" });
}

function ProcurementFulfillmentState({
  request,
  assetId,
  incidentVersion,
}: {
  request: PurchaseRequest;
  assetId: string;
  incidentVersion: number;
}) {
  const fulfillment = request.fulfillment;
  const readiness = request.reservation_readiness;
  if (!fulfillment && !readiness) return null;

  return (
    <Space orientation="vertical" size={6} style={{ width: "100%" }}>
      {fulfillment ? <FulfillmentFacts fulfillment={fulfillment} /> : null}
      {readiness ? (
        <ReservationReadiness
          readiness={readiness}
          assetId={assetId}
          incidentId={request.incident_id}
          incidentVersion={incidentVersion}
          partNumber={request.part_number}
          quantity={request.quantity}
        />
      ) : null}
    </Space>
  );
}

function FulfillmentFacts({ fulfillment }: { fulfillment: PurchaseFulfillment }) {
  return (
    <Space orientation="vertical" size={2}>
      <Typography.Text strong>
        {`ERP 履约：${fulfillmentStatusLabel(fulfillment.status)}`}
      </Typography.Text>
      <span>{`外部订单 ${fulfillment.external_order_id}`}</span>
      <span>
        {`累计收货 ${fulfillment.received_quantity} / ${fulfillment.requested_quantity} 件`}
      </span>
      <span>
        {fulfillment.expected_delivery_at
          ? `预计到货 ${fulfillment.expected_delivery_at}`
          : "ERP 未提供预计到货时间"}
      </span>
      <span>{`最后履约事件 ${fulfillment.last_event_occurred_at}`}</span>
      {fulfillment.reason ? <span>{`ERP 原因：${fulfillment.reason}`}</span> : null}
      <Typography.Text type="secondary">
        ERP 履约进度不等于 WMS 可用库存，平台不会用 ERP 数量覆盖库存事实。
      </Typography.Text>
    </Space>
  );
}

function ReservationReadiness({
  readiness,
  assetId,
  incidentId,
  incidentVersion,
  partNumber,
  quantity,
}: {
  readiness: PurchaseReservationReadiness;
  assetId: string;
  incidentId: string;
  incidentVersion: number;
  partNumber: string;
  quantity: number;
}) {
  const presentation = readinessPresentation(readiness.status);
  const hasWmsFact = readiness.source && readiness.source_record_id && readiness.as_of;
  return (
    <Alert
      type={presentation.type}
      showIcon
      message={presentation.title}
      description={(
        <Space orientation="vertical" size={2}>
          <span>{presentation.description}</span>
          {hasWmsFact ? (
            <span>
              {`WMS 来源 ${readiness.source} · 源记录 ${readiness.source_record_id}`}
              {readiness.available_quantity !== null
                ? ` · 当前可用 ${readiness.available_quantity} 件`
                : ""}
              {` · 数据时间 ${readiness.as_of}`}
            </span>
          ) : null}
          {readiness.status === "READY_FOR_RESERVATION" && incidentVersion ? (
            <Link href={partsReservationHref({
              assetId, incidentId, incidentVersion, partNumber, quantity,
            })}>
              前往备件查询并发起 T2 预留审批
            </Link>
          ) : null}
        </Space>
      )}
    />
  );
}

function partsReservationHref(context: {
  assetId: string;
  incidentId: string;
  incidentVersion: number;
  partNumber: string;
  quantity: number;
}): string {
  const query = new URLSearchParams({
    assetId: context.assetId,
    incidentId: context.incidentId,
    incidentVersion: String(context.incidentVersion),
    partNumber: context.partNumber,
    quantity: String(context.quantity),
  });
  return `/parts?${query.toString()}`;
}

function fulfillmentStatusLabel(status: PurchaseFulfillment["status"]): string {
  return ({
    APPROVED: "采购已批准",
    ORDERED: "已下单",
    IN_TRANSIT: "运输中",
    PARTIALLY_RECEIVED: "部分收货",
    RECEIVED: "全部收货",
    REJECTED: "履约已拒绝",
    CANCELLED: "履约已取消",
  } as const)[status];
}

function readinessPresentation(
  status: PurchaseReservationReadiness["status"],
): { title: string; description: string; type: "success" | "info" | "warning" | "error" } {
  return ({
    AWAITING_ERP_FULFILLMENT: {
      title: "等待 ERP 履约",
      description: "ERP 只确认受理了采购申请，平台正在等待可信订单与收货事件。",
      type: "info",
    },
    NOT_RECEIVED: {
      title: "ERP 尚未全部收货",
      description: "ERP 履约进度不等于 WMS 可用库存；全部收货前不会声明可发起预留。",
      type: "info",
    },
    AWAITING_WMS_SYNC: {
      title: "等待 WMS 库存同步",
      description: "ERP 已报告全部收货，但 ERP 收货不等于 WMS 可用库存；当前 WMS 时点或数量尚未满足申请。",
      type: "warning",
    },
    FACTS_UNAVAILABLE: {
      title: "WMS 库存事实不可用",
      description: "当前无法取得可信 WMS 事实，请稍后刷新；平台不会回退使用 ERP 收货数量。",
      type: "error",
    },
    REVIEW_REQUIRED: {
      title: "WMS 料号需要人工复核",
      description: "WMS 返回的料号与采购料号冲突，请由库存主数据责任人复核映射。",
      type: "warning",
    },
    READY_FOR_RESERVATION: {
      title: "WMS 已确认可发起备件预留",
      description: "服务端已核对收货时点、料号和可用数量；仍需沿用既有 T2 提案、审批和执行流程。",
      type: "success",
    },
  } as const)[status];
}
