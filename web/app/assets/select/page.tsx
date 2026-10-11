"use client";

import {
  Alert,
  Button,
  Card,
  Descriptions,
  Drawer,
  List,
  Space,
  Table,
  Tag,
  Typography,
} from "antd";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  type AssetDetail,
  type AssetSummary,
  getAuthorizedAsset,
  listAuthorizedAssets,
} from "@/lib/api/client";

export default function AssetSelectPage() {
  const [assets, setAssets] = useState<AssetSummary[]>();
  const [selected, setSelected] = useState<AssetDetail>();
  const [error, setError] = useState<unknown>();
  const [loadingDetail, setLoadingDetail] = useState(false);

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const response = await listAuthorizedAssets();
      setAssets(response.assets);
    } catch (caught) {
      setError(caught);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function showDetail(assetId: string) {
    setSelected(undefined);
    setError(undefined);
    setLoadingDetail(true);
    try {
      const response = await getAuthorizedAsset(assetId);
      setSelected(response.asset);
    } catch (caught) {
      setError(caught);
    } finally {
      setLoadingDetail(false);
    }
  }

  return (
    <AppShell>
      <div className="page-stack">
        <Space align="center" style={{ justifyContent: "space-between", width: "100%" }}>
          <div>
            <Typography.Title level={2}>授权设备与售后权益</Typography.Title>
            <Typography.Paragraph type="secondary">
              查看设备型号、客户、安装站点、在保状态和装机部件；每项事实均保留权威来源与更新时间。
            </Typography.Paragraph>
          </div>
          <Space>
            <Button onClick={() => void load()}>刷新主数据</Button>
            <Link href="/incidents/new"><Button type="primary">新建故障草稿</Button></Link>
          </Space>
        </Space>

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!assets && !error ? <LoadingState label="正在读取授权设备主数据" /> : null}
        {assets?.length === 0 ? <EmptyState description="当前身份没有设备或站点授权" /> : null}
        {assets?.length ? (
          <Card>
            <Table
              rowKey="asset_id"
              dataSource={assets}
              pagination={{ pageSize: 20 }}
              columns={[
                {
                  title: "设备",
                  render: (_, item) => (
                    <Space direction="vertical" size={0}>
                      <Typography.Text strong>{item.display_name ?? item.asset_id}</Typography.Text>
                      <Typography.Text type="secondary" code>{item.asset_id}</Typography.Text>
                    </Space>
                  ),
                },
                { title: "型号", render: (_, item) => item.model_code ?? "—" },
                { title: "客户", render: (_, item) => item.customer_name ?? "—" },
                { title: "站点", render: (_, item) => item.site_name ?? "—" },
                {
                  title: "保修",
                  render: (_, item) => item.warranty_status
                    ? <Tag color={item.warranty_status === "ACTIVE" ? "green" : "default"}>{item.warranty_status}</Tag>
                    : "—",
                },
                {
                  title: "数据状态",
                  render: (_, item) => (
                    <Space wrap>
                      <Tag color={item.freshness === "current" ? "green" : "orange"}>{item.freshness}</Tag>
                      <Tag color={item.access_basis === "SITE_SCOPE" ? "blue" : "default"}>
                        {item.access_basis === "SITE_SCOPE" ? "站点授权" : "设备授权"}
                      </Tag>
                    </Space>
                  ),
                },
                {
                  title: "操作",
                  render: (_, item) => (
                    <Button onClick={() => void showDetail(item.asset_id)}>查看档案与来源</Button>
                  ),
                },
              ]}
            />
          </Card>
        ) : null}
      </div>

      <Drawer
        title="设备主数据与售后权益"
        width={760}
        open={loadingDetail || Boolean(selected)}
        onClose={() => setSelected(undefined)}
      >
        {loadingDetail ? <LoadingState label="正在重新校验设备或站点授权" /> : null}
        {selected ? <AssetDetailPanel asset={selected} /> : null}
      </Drawer>
    </AppShell>
  );
}

function AssetDetailPanel({ asset }: { asset: AssetDetail }) {
  const components = asset.components ?? [];
  const sourceTrace = asset.source_trace ?? [];
  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      <Alert
        type="info"
        showIcon
        message={asset.access_basis === "SITE_SCOPE" ? "通过站点范围授权" : "通过设备范围授权"}
        description="详情请求已由服务端重新执行租户、角色及设备/站点范围校验。"
      />
      <Descriptions bordered column={2} size="small" title="设备实例">
        <Descriptions.Item label="设备 ID">{asset.asset_id}</Descriptions.Item>
        <Descriptions.Item label="名称">{asset.display_name ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="型号">{asset.model_code ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="序列号">{asset.serial_number ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="生命周期">{asset.lifecycle_status ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="主数据版本">v{asset.version}</Descriptions.Item>
      </Descriptions>

      <Descriptions bordered column={1} size="small" title="客户与安装站点">
        <Descriptions.Item label="客户">
          {asset.customer ? `${asset.customer.customer_name}（${asset.customer.customer_id}）` : "—"}
        </Descriptions.Item>
        <Descriptions.Item label="站点">
          {asset.site ? `${asset.site.site_name}（${asset.site.site_id}）` : "—"}
        </Descriptions.Item>
        <Descriptions.Item label="安装地址">{asset.site?.address ?? "—"}</Descriptions.Item>
      </Descriptions>

      <Descriptions bordered column={1} size="small" title="保修与服务等级">
        <Descriptions.Item label="状态">
          {asset.warranty
            ? <Tag color={asset.warranty.status === "ACTIVE" ? "green" : "default"}>{asset.warranty.status}</Tag>
            : "—"}
        </Descriptions.Item>
        <Descriptions.Item label="合同号">{asset.warranty?.contract_number ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="覆盖期">
          {asset.warranty
            ? `${formatDate(asset.warranty.coverage_start)} 至 ${formatDate(asset.warranty.coverage_end)}`
            : "—"}
        </Descriptions.Item>
        <Descriptions.Item label="服务等级">{asset.warranty?.service_level ?? "—"}</Descriptions.Item>
      </Descriptions>

      <List
        bordered
        header={`装机部件（${components.length}）`}
        locale={{ emptyText: "暂无可见装机部件" }}
        dataSource={components}
        renderItem={(component) => (
          <List.Item>
            <List.Item.Meta
              title={`${component.part_name} · ${component.part_number}`}
              description={
                <Space wrap>
                  <span>序列号 {component.serial_number ?? "—"}</span>
                  <span>数量 {component.quantity}</span>
                  <Tag>{component.status}</Tag>
                  <span>安装 {formatDate(component.installed_at)}</span>
                </Space>
              }
            />
          </List.Item>
        )}
      />

      <List
        bordered
        header="权威来源追溯"
        dataSource={sourceTrace}
        renderItem={(source) => (
          <List.Item>
            <Space direction="vertical" size={0}>
              <Space wrap>
                <Tag color={source.source_kind === "enterprise" ? "blue" : "default"}>{source.scope}</Tag>
                <Typography.Text strong>{source.source_system}</Typography.Text>
                <Typography.Text code copyable>{source.source_record_id}</Typography.Text>
              </Space>
              <Typography.Text type="secondary">
                v{source.version} · as_of {formatDateTime(source.as_of)} · {source.freshness}
              </Typography.Text>
            </Space>
          </List.Item>
        )}
      />
    </Space>
  );
}

function formatDate(value: string | null | undefined) {
  return value ? new Date(value).toLocaleDateString("zh-CN") : "长期/未提供";
}

function formatDateTime(value: string | null | undefined) {
  return value ? new Date(value).toLocaleString("zh-CN") : "未知";
}
