"use client";

import { Alert, Button, Card, List, Pagination, Select, Space, Tag, Typography } from "antd";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import { FactGrid, StatusTag } from "@/components/m2/BusinessState";
import {
  type CustomerCase,
  type CustomerCaseStatus,
  listCustomerCases,
} from "@/lib/api/client";

const PAGE_SIZE = 20;
const STATUS_OPTIONS: Array<{ label: string; value: CustomerCaseStatus }> = [
  { label: "已提交", value: "SUBMITTED" },
  { label: "待补充", value: "NEEDS_INFORMATION" },
  { label: "已受理", value: "TRIAGED" },
  { label: "诊断中", value: "DIAGNOSING" },
  { label: "已诊断", value: "DIAGNOSED" },
  { label: "服务执行中", value: "WORK_ORDER_CREATED" },
  { label: "已解决", value: "RESOLVED" },
  { label: "已升级", value: "ESCALATED" },
  { label: "已关闭", value: "CLOSED" },
];

export default function CustomerPortalPage() {
  const [cases, setCases] = useState<CustomerCase[]>();
  const [status, setStatus] = useState<CustomerCaseStatus>();
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const result = await listCustomerCases({
        ...(status ? { status } : {}),
        limit: PAGE_SIZE,
        offset,
      });
      setCases(result.cases);
      setTotal(result.total);
      setRequestId(result.requestId);
    } catch (caught) {
      setError(caught);
    }
  }, [offset, status]);

  useEffect(() => { void load(); }, [load]);

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>客户服务门户</Typography.Title>
        <Alert
          type="info"
          showIcon
          message="只展示由当前账号报告且仍在授权设备范围内的服务案例"
          description="你可以查看服务进度、补充资料并确认维修结果；内部推理、其他客户数据和高风险工具不会在此门户暴露。"
          action={<Link href="/incidents/new"><Button type="primary">发起新报障</Button></Link>}
        />

        <Card title="我的服务案例" extra={<Button onClick={() => void load()}>刷新</Button>}>
          <Space wrap>
            <Select<CustomerCaseStatus>
              allowClear
              style={{ minWidth: 180 }}
              placeholder="全部服务进度"
              options={STATUS_OPTIONS}
              value={status}
              onChange={(value) => { setStatus(value); setOffset(0); }}
            />
            <Typography.Text type="secondary">共 {total} 条 · 请求 {requestId ?? "—"}</Typography.Text>
          </Space>
        </Card>

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!cases && !error ? <LoadingState label="正在读取客户服务案例" /> : null}
        {cases?.length === 0 ? <EmptyState description="当前没有服务案例" /> : null}
        <List
          dataSource={cases}
          renderItem={(customerCase) => (
            <List.Item>
              <Card
                style={{ width: "100%" }}
                title={customerCase.asset_display_name ?? customerCase.asset_id}
                extra={<StatusTag status={customerCase.status} version={customerCase.version} />}
              >
                <div className="page-stack">
                  <Typography.Text>{customerCase.description}</Typography.Text>
                  <FactGrid facts={[
                    ["服务编号", customerCase.incident_id],
                    ["设备型号", customerCase.model_code],
                    ["服务站点", customerCase.site_name ?? customerCase.site_id],
                    ["关联工单", customerCase.work_orders.length],
                    ["最后更新", formatTimestamp(customerCase.updated_at)],
                  ]} />
                  <Space wrap>
                    {customerCase.work_orders.map((work) => (
                      <Tag key={work.work_order_id} color={work.result_accepted === false ? "red" : "blue"}>
                        工单 {work.status}{work.result_accepted === true ? " · 已确认" : ""}
                      </Tag>
                    ))}
                  </Space>
                  <Link href={`/portal/cases/${customerCase.incident_id}`}>查看进度与补充资料</Link>
                </div>
              </Card>
            </List.Item>
          )}
        />
        {total > PAGE_SIZE ? (
          <Pagination
            current={Math.floor(offset / PAGE_SIZE) + 1}
            pageSize={PAGE_SIZE}
            total={total}
            showSizeChanger={false}
            onChange={(page) => setOffset((page - 1) * PAGE_SIZE)}
          />
        ) : null}
      </div>
    </AppShell>
  );
}

function formatTimestamp(value: string): string {
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}
