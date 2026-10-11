"use client";

import { Alert, Button, Card, Descriptions, List, Space, Tag, Typography } from "antd";

import type { CurationRun } from "@/lib/api/client";

type Props = {
  runs: CurationRun[];
  busyRunId?: string;
  onInspect: (runId: string) => void;
  onOpenSnapshot: (snapshotId: string) => void;
};

export function CurationRunCatalog({
  runs,
  busyRunId,
  onInspect,
  onOpenSnapshot,
}: Props) {
  return (
    <List
      grid={{ gutter: 16, xs: 1, xl: 2 }}
      dataSource={runs}
      renderItem={(run) => {
        const noData = isNoDataRun(run);
        return (
          <List.Item>
            <Card
              title={<Typography.Text code>{run.run_id}</Typography.Text>}
              extra={<Tag color={statusColor(run, noData)}>{statusLabel(run, noData)}</Tag>}
            >
              <Descriptions
                size="small"
                column={1}
                items={[
                  { key: "engine", label: "执行引擎", children: run.engine },
                  {
                    key: "window",
                    label: "固定窗口",
                    children: `${formatTime(run.window_start)} — ${formatTime(run.window_end)}`,
                  },
                  { key: "inputs", label: "输入样本", children: run.input_count },
                  {
                    key: "exclusions",
                    label: "排除记录",
                    children: run.exclusion_report.length,
                  },
                  { key: "contract", label: "数据契约", children: run.contract_version },
                ]}
              />
              {noData ? (
                <Alert
                  showIcon
                  type="info"
                  title="本次无可策展数据"
                  description={
                    run.exclusion_report.length > 0
                      ? `所选窗口中的 ${run.exclusion_report.length} 条候选均未通过治理门禁，因此未生成数据集快照。`
                      : "所选时间窗口内没有反馈候选，因此未启动策展处理，也未生成空数据集快照。"
                  }
                />
              ) : run.failure_reason ? (
                <Alert
                  showIcon
                  type={run.status === "LINEAGE_PENDING" ? "warning" : "error"}
                  title={run.status === "LINEAGE_PENDING" ? "等待血缘补发" : "策展运行失败"}
                  description={run.failure_reason}
                />
              ) : null}
              <Space wrap>
                <Button
                  loading={busyRunId === run.run_id}
                  onClick={() => onInspect(run.run_id)}
                >
                  刷新与审查
                </Button>
                {run.snapshot_id ? (
                  <Button
                    onClick={() => {
                      if (run.snapshot_id) onOpenSnapshot(run.snapshot_id);
                    }}
                  >
                    查看关联快照
                  </Button>
                ) : null}
              </Space>
            </Card>
          </List.Item>
        );
      }}
    />
  );
}

function formatTime(value: string): string {
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

function isNoDataRun(run: CurationRun): boolean {
  return run.status === "NO_DATA" || run.failure_reason === "no_eligible_inputs";
}

function statusLabel(run: CurationRun, noData: boolean): string {
  if (noData) return "无可用数据";
  if (run.status === "COMPLETED") return "已完成";
  if (run.status === "LINEAGE_PENDING") return "等待血缘补发";
  if (run.status === "FAILED") return "失败";
  return run.status;
}

function statusColor(run: CurationRun, noData: boolean): string {
  if (noData) return "blue";
  if (run.status === "COMPLETED") return "green";
  if (run.status === "LINEAGE_PENDING") return "orange";
  if (run.status === "FAILED") return "red";
  return "blue";
}
