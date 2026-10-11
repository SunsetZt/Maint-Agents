"use client";

import { Alert, Button, Card, Descriptions, List, Space, Table, Tag, Typography } from "antd";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  getWorkOrderClosureReport,
  type WorkOrderClosureReport,
} from "@/lib/api/client";

type ClosurePart = WorkOrderClosureReport["parts"][number];

export default function WorkOrderClosureReportPage() {
  const { workOrderId } = useParams<{ workOrderId: string }>();
  const [report, setReport] = useState<WorkOrderClosureReport>();
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const result = await getWorkOrderClosureReport(workOrderId);
      setReport(result.report);
      setRequestId(result.requestId);
    } catch (caught) {
      setError(caught);
    }
  }, [workOrderId]);

  useEffect(() => { void load(); }, [load]);

  return (
    <AppShell>
      <div className="page-stack">
        <Space wrap style={{ justifyContent: "space-between", width: "100%" }}>
          <div>
            <Typography.Title level={2}>维修服务关闭报告</Typography.Title>
            <Typography.Paragraph type="secondary">
              本报告由已关闭工单的权威完工、验收、备件、客户确认和关闭事件实时投影。
            </Typography.Paragraph>
          </div>
          <Space className="report-actions">
            <Link href={`/work-orders/${workOrderId}`}>返回工单详情</Link>
            <Button type="primary" disabled={!report} onClick={() => window.print()}>打印报告</Button>
          </Space>
        </Space>

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!report && !error ? <LoadingState label="正在生成关闭报告" /> : null}

        {report ? (
          <>
            <Alert
              type="success"
              showIcon
              message="关闭事实完整且已通过独立验收"
              description={`合同 ${report.report_contract_version}；关闭事实摘要 ${report.closure_facts_digest}`}
            />

            <Card title="工单与设备">
              <Descriptions bordered column={{ xs: 1, md: 2 }} size="small">
                <Descriptions.Item label="工单">{report.work_order_id}</Descriptions.Item>
                <Descriptions.Item label="故障单">{report.incident_id}</Descriptions.Item>
                <Descriptions.Item label="设备">{report.asset_display_name ?? report.asset_id}</Descriptions.Item>
                <Descriptions.Item label="型号 / 序列号">{report.model_code ?? "—"} / {report.serial_number ?? "—"}</Descriptions.Item>
                <Descriptions.Item label="站点">{report.site_name ?? report.site_id ?? "—"}</Descriptions.Item>
                <Descriptions.Item label="关闭时间">{formatTimestamp(report.closed_at)}</Descriptions.Item>
                <Descriptions.Item label="关闭执行人">{report.closed_by_subject_id}</Descriptions.Item>
                <Descriptions.Item label="关闭原因">{report.close_reason ?? "人工直接关闭"}</Descriptions.Item>
              </Descriptions>
            </Card>

            <Card title="最终维修结果" extra={<Space><Tag color="blue">第 {report.completion_round_number} 轮</Tag><Tag color={report.rework_count ? "orange" : "green"}>返工 {report.rework_count} 次</Tag></Space>}>
              <Descriptions bordered column={1} size="small">
                <Descriptions.Item label="根因">{report.root_cause}</Descriptions.Item>
                <Descriptions.Item label="处理动作">
                  <List size="small" dataSource={report.actions} renderItem={(item) => <List.Item>{item}</List.Item>} />
                </Descriptions.Item>
                <Descriptions.Item label="现场证据">
                  <Space wrap>{report.evidence_ids.map((item) => <Tag key={item}>{item}</Tag>)}</Space>
                </Descriptions.Item>
                <Descriptions.Item label="费用">{report.cost_amount ?? "—"}</Descriptions.Item>
                <Descriptions.Item label="现场客户签字">{report.field_customer_confirmation ?? "—"}</Descriptions.Item>
                <Descriptions.Item label="完工人 / 时间">{report.completed_by_subject_id} · {formatTimestamp(report.completed_at)}</Descriptions.Item>
              </Descriptions>
            </Card>

            <Card title="实际使用备件">
              <Table<ClosurePart>
                rowKey="reservation_id"
                pagination={false}
                dataSource={report.parts}
                columns={[
                  { title: "预留记录", dataIndex: "reservation_id" },
                  { title: "料号", dataIndex: "part_number" },
                  { title: "数量", dataIndex: "quantity", width: 90 },
                  { title: "状态", dataIndex: "status", width: 120 },
                  { title: "来源", dataIndex: "source", width: 160 },
                  { title: "数据时点", dataIndex: "as_of", render: (value: string) => formatTimestamp(value) },
                ]}
              />
            </Card>

            <Card title="独立验收与客户确认">
              <Descriptions bordered column={{ xs: 1, md: 2 }} size="small">
                <Descriptions.Item label="验收结论"><Tag color="green">通过</Tag></Descriptions.Item>
                <Descriptions.Item label="验收人">{report.verifier_subject_id}</Descriptions.Item>
                <Descriptions.Item label="验收说明">{report.verification_reason}</Descriptions.Item>
                <Descriptions.Item label="验收时间">{formatTimestamp(report.verified_at)}</Descriptions.Item>
                <Descriptions.Item label="客户接受结果"><Tag color={report.customer_result_accepted ? "green" : "red"}>{report.customer_result_accepted ? "已接受" : "未接受"}</Tag></Descriptions.Item>
                <Descriptions.Item label="满意度">{report.customer_satisfaction_rating ?? "—"}</Descriptions.Item>
                <Descriptions.Item label="客户确认人">{report.customer_confirmed_by_subject_id ?? "现场签字确认"}</Descriptions.Item>
                <Descriptions.Item label="客户确认时间">{formatTimestamp(report.customer_confirmed_at)}</Descriptions.Item>
              </Descriptions>
            </Card>

            <Card title="关闭审计关联">
              <Descriptions bordered column={1} size="small">
                <Descriptions.Item label="Completion">{report.completion_id}</Descriptions.Item>
                <Descriptions.Item label="Verification">{report.verification_id}</Descriptions.Item>
                <Descriptions.Item label="关闭 Proposal">{report.close_proposal_id ?? "人工关闭，无提案"}</Descriptions.Item>
                <Descriptions.Item label="关闭 Operation">{report.close_operation_id ?? "人工关闭，无操作号"}</Descriptions.Item>
                <Descriptions.Item label="事实摘要">{report.closure_facts_digest}</Descriptions.Item>
                <Descriptions.Item label="请求 ID">{requestId}</Descriptions.Item>
              </Descriptions>
            </Card>
          </>
        ) : null}
      </div>
    </AppShell>
  );
}

function formatTimestamp(value: string | null): string {
  if (!value) return "—";
  const timestamp = new Date(value);
  return Number.isNaN(timestamp.getTime())
    ? value
    : timestamp.toLocaleString("zh-CN", { hour12: false });
}
