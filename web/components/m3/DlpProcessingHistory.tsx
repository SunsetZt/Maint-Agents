"use client";

import { Alert, Card, Descriptions, List, Space, Tag, Typography } from "antd";

import type { DlpProcessingHistory as DlpProcessingHistoryItem } from "@/lib/api/client";

export function DlpProcessingHistory({
  runs,
}: {
  runs: DlpProcessingHistoryItem[];
}) {
  return (
    <Card title={`DLP 处理历史 (${runs.length})`} size="small">
      <List
        dataSource={runs}
        locale={{ emptyText: "尚未执行 DLP 脱敏" }}
        renderItem={(run, index) => (
          <List.Item>
            <div style={{ width: "100%" }}>
              <Space wrap style={{ marginBottom: 8 }}>
                <Typography.Text strong>第 {index + 1} 次处理</Typography.Text>
                <Tag color={run.status === "PASSED" ? "green" : "red"}>{run.status}</Tag>
                <Tag>候选 v{run.candidate_version}</Tag>
              </Space>
              <Descriptions
                size="small"
                column={1}
                items={[
                  { key: "policy", label: "DLP 策略", children: run.policy_version },
                  {
                    key: "source-hash",
                    label: "源内容哈希",
                    children: <Typography.Text code copyable>{run.source_content_hash}</Typography.Text>,
                  },
                  {
                    key: "output-hash",
                    label: "脱敏内容哈希",
                    children: <Typography.Text code copyable>{run.content_hash}</Typography.Text>,
                  },
                  {
                    key: "findings",
                    label: "识别实体",
                    children: Object.keys(run.finding_counts).length ? (
                      <Space wrap>
                        {Object.entries(run.finding_counts).map(([entityType, count]) => (
                          <Tag key={entityType}>{entityType} × {count}</Tag>
                        ))}
                      </Space>
                    ) : "未识别到敏感实体",
                  },
                  { key: "actor", label: "执行主体", children: run.processed_by_subject_id },
                  {
                    key: "time",
                    label: "执行时间",
                    children: new Date(run.created_at).toLocaleString("zh-CN"),
                  },
                  {
                    key: "id",
                    label: "处理标识",
                    children: <Typography.Text code copyable>{run.dlp_result_id}</Typography.Text>,
                  },
                ]}
              />
              {run.residual_entity_types.length ? (
                <Alert
                  showIcon
                  type="error"
                  title="脱敏后仍检测到禁止实体"
                  description={run.residual_entity_types.join("、")}
                  style={{ marginTop: 12 }}
                />
              ) : null}
            </div>
          </List.Item>
        )}
      />
    </Card>
  );
}
