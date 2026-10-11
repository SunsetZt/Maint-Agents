"use client";

import { Alert, Button, Card, List, Select, Space, Tag, Typography } from "antd";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import { FactGrid } from "@/components/m2/BusinessState";
import {
  type AssetSummary,
  type PartsReservationProposal,
  type SparePartsLookup,
  listAuthorizedAssets,
  lookupSpareParts,
  proposePartReservation,
} from "@/lib/api/client";

type ReservationContext = {
  assetId: string;
  incidentId: string;
  incidentVersion: number;
  partNumber: string;
  quantity: number;
};

export default function PartsPage() {
  const [assets, setAssets] = useState<AssetSummary[]>();
  const [assetId, setAssetId] = useState<string>();
  const [lookup, setLookup] = useState<SparePartsLookup>();
  const [requestId, setRequestId] = useState<string>();
  const [reservationContext, setReservationContext] = useState<ReservationContext>();
  const [reservationProposal, setReservationProposal] = useState<PartsReservationProposal>();
  const [proposalError, setProposalError] = useState<unknown>();
  const [proposalBusy, setProposalBusy] = useState(false);
  const [idempotencyKey, setIdempotencyKey] = useState(createIdempotencyKey);
  const [error, setError] = useState<unknown>();
  const [loading, setLoading] = useState(false);

  const loadAssets = useCallback(async () => {
    setError(undefined);
    try {
      const result = await listAuthorizedAssets();
      setAssets(result.assets);
      const search = new URLSearchParams(window.location.search);
      const requested = search.get("assetId");
      setReservationContext(parseReservationContext(search));
      const selected = result.assets.some((asset) => asset.asset_id === requested)
        ? requested ?? undefined
        : result.assets[0]?.asset_id;
      setAssetId((current) => current ?? selected);
    } catch (caught) {
      setError(caught);
    }
  }, []);

  const loadLookup = useCallback(async () => {
    if (!assetId) return;
    setLoading(true);
    setError(undefined);
    setLookup(undefined);
    try {
      const result = await lookupSpareParts(assetId);
      setLookup(result.lookup);
      setRequestId(result.requestId);
    } catch (caught) {
      setError(caught);
      setLookup(undefined);
    } finally {
      setLoading(false);
    }
  }, [assetId]);

  useEffect(() => { void loadAssets(); }, [loadAssets]);
  useEffect(() => { void loadLookup(); }, [loadLookup]);

  const canPropose = isReservationEligible(reservationContext, lookup);

  async function submitReservation() {
    if (!reservationContext || !canPropose) return;
    setProposalBusy(true);
    setProposalError(undefined);
    try {
      const result = await proposePartReservation(
        reservationContext.incidentId,
        {
          incident_version: reservationContext.incidentVersion,
          part_number: reservationContext.partNumber,
          quantity: reservationContext.quantity,
        },
        idempotencyKey,
      );
      setReservationProposal(result.proposal);
      setIdempotencyKey(createIdempotencyKey());
    } catch (caught) {
      setProposalError(caught);
    } finally {
      setProposalBusy(false);
    }
  }

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>备件查询与设备适配中心</Typography.Title>
        <Alert
          type="info"
          showIcon
          message="EAM 装机事实与 WMS 实时库存分开呈现"
          description="库存存在不代表备件一定适配。页面只把与装机部件料号一致的结果标记为已匹配，其他结果必须由工程师复核。"
        />

        <Card title="选择授权设备" extra={<Button loading={loading} onClick={() => void loadLookup()}>刷新实时库存</Button>}>
          <Select
            showSearch
            style={{ width: "100%", maxWidth: 560 }}
            placeholder="选择设备"
            value={assetId}
            optionFilterProp="label"
            options={assets?.map((asset) => ({
              value: asset.asset_id,
              label: `${asset.display_name ?? asset.asset_id} · ${asset.model_code ?? "未知型号"}`,
            }))}
            onChange={setAssetId}
          />
        </Card>

        {error ? <ErrorState error={error} onRetry={() => void loadLookup()} /> : null}
        {!assets && !error ? <LoadingState label="正在读取授权设备" /> : null}
        {assets?.length === 0 ? <EmptyState description="当前身份没有授权设备" /> : null}
        {loading && !lookup ? <LoadingState label="正在查询 EAM 与实时 WMS" /> : null}

        {lookup ? (
          <>
            <Card title={lookup.asset_display_name ?? lookup.asset_id}>
              <FactGrid facts={[
                ["设备 ID", lookup.asset_id],
                ["型号", lookup.model_code],
                ["序列号", lookup.serial_number],
                ["站点", lookup.site_name ?? lookup.site_id],
                ["请求标识", requestId],
              ]} />
            </Card>

            <Card title="EAM 装机部件/BOM">
              {lookup.components.length === 0 ? <EmptyState description="EAM 未返回可见装机部件" /> : null}
              <List
                dataSource={lookup.components}
                renderItem={(component) => (
                  <List.Item>
                    <List.Item.Meta
                      title={<Space><Typography.Text strong>{component.part_name}</Typography.Text><Tag>{component.part_number}</Tag></Space>}
                      description={(
                        <Space wrap>
                          <span>装机数量 {component.installed_quantity}</span>
                          <Tag>{component.component_status}</Tag>
                          <span>{component.source}</span>
                          <span>{component.source_record_id}</span>
                          <span>as_of {formatTimestamp(component.as_of)}</span>
                          <Tag color={component.freshness === "current" ? "green" : "orange"}>{component.freshness}</Tag>
                        </Space>
                      )}
                    />
                  </List.Item>
                )}
              />
            </Card>

            <Card title="WMS 实时库存">
              {lookup.availability ? (
                <div className="page-stack">
                  <Space wrap>
                    <Tag color={lookup.availability.stock_status === "IN_STOCK" ? "green" : "red"}>{lookup.availability.stock_status}</Tag>
                    <Tag color={lookup.availability.compatibility_evidence === "INSTALLED_COMPONENT_MATCH" ? "blue" : "orange"}>
                      {lookup.availability.compatibility_evidence === "INSTALLED_COMPONENT_MATCH" ? "装机料号已匹配" : "适配性需人工复核"}
                    </Tag>
                  </Space>
                  <FactGrid facts={[
                    ["料号", lookup.availability.part_number],
                    ["可用数量", lookup.availability.available_quantity],
                    ["事实来源", lookup.availability.source],
                    ["源记录", lookup.availability.source_record_id],
                    ["数据时间", formatTimestamp(lookup.availability.as_of)],
                    ["业务所有者", lookup.availability.authority_owner],
                    ["工具调用", lookup.availability.tool_call_id],
                  ]} />
                </div>
              ) : (
                <Alert type="warning" showIcon message="实时库存暂不可用" description={lookup.availability_failure} />
              )}
            </Card>

            {reservationContext ? (
              <Card title="备件预留提案（T2）">
                {canPropose ? (
                  <div className="page-stack">
                    <FactGrid facts={[
                      ["故障单", reservationContext.incidentId],
                      ["故障单版本", reservationContext.incidentVersion],
                      ["预留料号", reservationContext.partNumber],
                      ["预留数量", reservationContext.quantity],
                    ]} />
                    {proposalError ? <ErrorState error={proposalError} /> : null}
                    {reservationProposal ? (
                      <Alert
                        type="success"
                        showIcon
                        message={`${reservationProposal.proposal_id} 已进入审批箱`}
                        description={(
                          <Space orientation="vertical" size={2}>
                            <span>{`审批标识 ${reservationProposal.approval_id}`}</span>
                            <span>当前尚未预留库存，也未创建工单；审批通过后才会执行 WMS 预留。</span>
                            <Link href="/approvals">前往审批箱</Link>
                          </Space>
                        )}
                      />
                    ) : (
                      <Button type="primary" loading={proposalBusy}
                        onClick={() => void submitReservation()}>
                        提交 T2 预留审批
                      </Button>
                    )}
                  </div>
                ) : (
                  <Alert type="warning" showIcon message="当前上下文不满足预留提案门禁"
                    description="设备、装机料号、实时 WMS 料号、适配证据或可用数量不匹配，本页保持只读。" />
                )}
              </Card>
            ) : null}

            <Alert
              type="warning"
              showIcon
              message="领料/预留属于 T2 业务副作用"
              description="本页只查询，不直接占用库存。实际预留必须从诊断提案进入人工审批，审批通过后再由幂等执行器调用 WMS。"
              action={<Link href="/approvals"><Button>查看审批箱</Button></Link>}
            />
          </>
        ) : null}
      </div>
    </AppShell>
  );
}

function formatTimestamp(value: string | null): string {
  return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "—";
}

function parseReservationContext(search: URLSearchParams): ReservationContext | undefined {
  const assetId = search.get("assetId");
  const incidentId = search.get("incidentId");
  const incidentVersion = Number(search.get("incidentVersion"));
  const partNumber = search.get("partNumber");
  const quantity = Number(search.get("quantity"));
  if (!assetId || !incidentId || !partNumber || !Number.isInteger(incidentVersion)
    || incidentVersion < 1 || !Number.isInteger(quantity) || quantity < 1) return undefined;
  return { assetId, incidentId, incidentVersion, partNumber, quantity };
}

function isReservationEligible(
  context: ReservationContext | undefined,
  lookup: SparePartsLookup | undefined,
): boolean {
  const availability = lookup?.availability;
  return Boolean(context && lookup && availability
    && lookup.asset_id === context.assetId
    && availability.part_number === context.partNumber
    && availability.available_quantity >= context.quantity
    && availability.compatibility_evidence === "INSTALLED_COMPONENT_MATCH"
    && lookup.components.some((component) => component.part_number === context.partNumber
      && component.component_status === "INSTALLED"));
}

function createIdempotencyKey(): string {
  const token = globalThis.crypto?.randomUUID?.()
    ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  return `parts-reservation-${token}`;
}
