"use client";

import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Row,
  Select,
  Space,
  Tag,
  Typography,
} from "antd";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  type AssetSummary,
  type ServiceEntitlement,
  listAuthorizedAssets,
  lookupServiceEntitlement,
} from "@/lib/api/client";

export default function ServiceEntitlementsPage() {
  const [assets, setAssets] = useState<AssetSummary[]>();
  const [assetId, setAssetId] = useState<string>();
  const [entitlement, setEntitlement] = useState<ServiceEntitlement>();
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [loading, setLoading] = useState(false);

  const loadAssets = useCallback(async () => {
    setError(undefined);
    try {
      const result = await listAuthorizedAssets();
      setAssets(result.assets);
      const requested = new URLSearchParams(window.location.search).get("assetId");
      const selected = result.assets.some((asset) => asset.asset_id === requested)
        ? requested ?? undefined
        : result.assets[0]?.asset_id;
      setAssetId((current) => current ?? selected);
    } catch (cause) {
      setError(cause);
    }
  }, []);

  const lookup = useCallback(async () => {
    if (!assetId) return;
    setLoading(true);
    setError(undefined);
    setEntitlement(undefined);
    try {
      const result = await lookupServiceEntitlement(assetId);
      setEntitlement(result.entitlement);
      setRequestId(result.requestId);
    } catch (cause) {
      setError(cause);
    } finally {
      setLoading(false);
    }
  }, [assetId]);

  useEffect(() => { void loadAssets(); }, [loadAssets]);
  useEffect(() => { void lookup(); }, [lookup]);

  return (
    <AppShell>
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <Space align="start" style={{ justifyContent: "space-between", width: "100%" }}>
          <div>
            <Typography.Title level={2}>售后服务权益核验中心</Typography.Title>
            <Typography.Paragraph type="secondary">
              将 EAM 保修主数据与合同权益系统的实时事实交叉核验，供报修受理、派工和报价前人工确认。
            </Typography.Paragraph>
          </div>
          <Space>
            <Link href="/assets/select">查看设备档案</Link>
            <Link href="/incidents/new"><Button type="primary">创建故障草稿</Button></Link>
          </Space>
        </Space>

        <Alert
          type="warning"
          showIcon
          message="核验结论不自动拒绝服务或产生费用"
          description="实时事实缺失、过期或与 EAM 主数据冲突时必须转人工复核。本页不能直接修改合同、工单、报价或账单。"
        />

        <Card
          title="选择授权设备"
          extra={<Button loading={loading} onClick={() => void lookup()}>刷新实时权益</Button>}
        >
          <Select
            showSearch
            style={{ width: "100%", maxWidth: 620 }}
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

        {error ? <ErrorState error={error} onRetry={() => void lookup()} /> : null}
        {!assets && !error ? <LoadingState label="正在读取授权设备" /> : null}
        {assets?.length === 0 ? <EmptyState description="当前身份没有授权设备" /> : null}
        {loading && !entitlement ? <LoadingState label="正在核验 EAM 与合同权益系统" /> : null}

        {entitlement ? (
          <>
            <Card
              title={entitlement.asset_display_name ?? entitlement.asset_id}
              extra={<Tag color={decisionColor(entitlement.decision)}>{decisionLabel(entitlement.decision)}</Tag>}
            >
              <Row gutter={[16, 16]}>
                <Col xs={24} lg={12}>
                  <Descriptions bordered size="small" column={1} title="设备与客户">
                    <Descriptions.Item label="设备 ID">{entitlement.asset_id}</Descriptions.Item>
                    <Descriptions.Item label="型号 / 序列号">
                      {entitlement.model_code ?? "—"} / {entitlement.serial_number ?? "—"}
                    </Descriptions.Item>
                    <Descriptions.Item label="生命周期">{entitlement.lifecycle_status ?? "—"}</Descriptions.Item>
                    <Descriptions.Item label="客户">{entitlement.customer_name ?? "—"}</Descriptions.Item>
                    <Descriptions.Item label="站点">{entitlement.site_name ?? entitlement.site_id ?? "—"}</Descriptions.Item>
                  </Descriptions>
                </Col>
                <Col xs={24} lg={12}>
                  <Descriptions bordered size="small" column={1} title="核验结论">
                    <Descriptions.Item label="结论">
                      <Tag color={decisionColor(entitlement.decision)}>{decisionLabel(entitlement.decision)}</Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="依据">
                      <Space wrap>
                        {entitlement.reason_codes.map((code) => <Tag key={code}>{reasonLabel(code)}</Tag>)}
                      </Space>
                    </Descriptions.Item>
                    <Descriptions.Item label="下一步">{nextActionLabel(entitlement.next_action)}</Descriptions.Item>
                    <Descriptions.Item label="请求标识">{requestId ?? "—"}</Descriptions.Item>
                  </Descriptions>
                </Col>
              </Row>
            </Card>

            <Row gutter={[16, 16]}>
              <Col xs={24} xl={12}>
                <Card title="EAM / 合同主数据快照" style={{ height: "100%" }}>
                  {entitlement.master_warranty ? (
                    <Descriptions bordered size="small" column={1}>
                      <Descriptions.Item label="合同号">{entitlement.master_warranty.contract_number}</Descriptions.Item>
                      <Descriptions.Item label="状态"><Tag>{entitlement.master_warranty.status}</Tag></Descriptions.Item>
                      <Descriptions.Item label="覆盖期">
                        {formatDate(entitlement.master_warranty.coverage_start)} 至 {formatDate(entitlement.master_warranty.coverage_end)}
                      </Descriptions.Item>
                      <Descriptions.Item label="服务等级">{entitlement.master_warranty.service_level ?? "—"}</Descriptions.Item>
                      <Descriptions.Item label="来源">{entitlement.master_warranty.source}</Descriptions.Item>
                      <Descriptions.Item label="源记录">{entitlement.master_warranty.source_record_id}</Descriptions.Item>
                      <Descriptions.Item label="数据时间">{formatTime(entitlement.master_warranty.as_of)}</Descriptions.Item>
                      <Descriptions.Item label="新鲜度"><Tag>{entitlement.master_warranty.freshness}</Tag></Descriptions.Item>
                    </Descriptions>
                  ) : <EmptyState description="EAM 未提供保修主数据" />}
                </Card>
              </Col>
              <Col xs={24} xl={12}>
                <Card title="合同权益系统实时事实" style={{ height: "100%" }}>
                  {entitlement.live_entitlement ? (
                    <Descriptions bordered size="small" column={1}>
                      <Descriptions.Item label="实时有效">
                        <Tag color={entitlement.live_entitlement.valid ? "green" : "default"}>
                          {entitlement.live_entitlement.valid ? "是" : "否"}
                        </Tag>
                      </Descriptions.Item>
                      <Descriptions.Item label="覆盖范围">{entitlement.live_entitlement.coverage}</Descriptions.Item>
                      <Descriptions.Item label="业务所有者">{entitlement.live_entitlement.authority_owner}</Descriptions.Item>
                      <Descriptions.Item label="来源">{entitlement.live_entitlement.source}</Descriptions.Item>
                      <Descriptions.Item label="源记录">{entitlement.live_entitlement.source_record_id}</Descriptions.Item>
                      <Descriptions.Item label="数据时间">{formatTime(entitlement.live_entitlement.as_of)}</Descriptions.Item>
                      <Descriptions.Item label="工具调用">{entitlement.live_entitlement.tool_call_id}</Descriptions.Item>
                    </Descriptions>
                  ) : (
                    <Alert
                      type="error"
                      showIcon
                      message="实时合同权益事实不可用"
                      description={entitlement.live_fact_failure ?? "未返回可用事实"}
                    />
                  )}
                </Card>
              </Col>
            </Row>

            <Alert
              type="info"
              showIcon
              message="受控业务动作边界"
              description={`自动拒绝服务：否；自动计费：否；自动修改工单：否；必经流程：${entitlement.action_policy.required_flow}`}
            />
          </>
        ) : null}
      </Space>
    </AppShell>
  );
}

function decisionColor(value: string) {
  if (value === "COVERED") return "green";
  if (value === "NOT_COVERED") return "default";
  if (value === "REVIEW_REQUIRED") return "orange";
  return "red";
}

function decisionLabel(value: string) {
  const labels: Record<string, string> = {
    COVERED: "权益有效",
    NOT_COVERED: "未覆盖（待人工确认）",
    REVIEW_REQUIRED: "来源冲突，需复核",
    FACTS_UNAVAILABLE: "事实不可用",
  };
  return labels[value] ?? value;
}

function reasonLabel(value: string) {
  const labels: Record<string, string> = {
    AUTHORITATIVE_AND_MASTERED_FACTS_AGREE: "实时事实与主数据一致",
    LIVE_ENTITLEMENT_UNAVAILABLE: "实时权益不可用",
    LIVE_ENTITLEMENT_STALE: "实时权益已过期",
    LIVE_ENTITLEMENT_FUTURE_TIMESTAMP: "实时时间异常",
    MASTER_WARRANTY_MISSING: "缺少保修主数据",
    MASTER_WARRANTY_NOT_CURRENT: "保修主数据不新鲜",
    WARRANTY_SOURCES_CONFLICT: "两处保修事实冲突",
  };
  return labels[value] ?? value;
}

function nextActionLabel(value: string) {
  const labels: Record<string, string> = {
    RETRY_OR_CONTACT_CONTRACT_OWNER: "重试实时查询或联系合同权益负责人",
    CONTACT_CONTRACT_OWNER_AND_CONFIRM_SERVICE_SCOPE: "联系合同权益负责人并确认服务范围",
    HUMAN_CONFIRM_COVERAGE_BEFORE_DISPATCH: "派工前由售后人员确认覆盖范围",
    HUMAN_CONFIRM_BEFORE_QUOTE_OR_SERVICE_DENIAL: "报价或拒绝服务前必须人工确认",
  };
  return labels[value] ?? value;
}

function formatDate(value: string | null) {
  return value ? new Date(value).toLocaleDateString("zh-CN") : "长期/未提供";
}

function formatTime(value: string | null) {
  return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "未知";
}

