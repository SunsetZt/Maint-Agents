"use client";

import { Card, Descriptions, List, Space, Tag, Typography } from "antd";

import type { EligibilityDecision } from "@/lib/api/client";

export function EligibilityDecisionHistory({
  decisions,
}: {
  decisions: EligibilityDecision[];
}) {
  return (
    <Card title={`用途与许可决策历史 (${decisions.length})`} size="small">
      <List
        dataSource={decisions}
        locale={{ emptyText: "尚未作出用途与许可决定" }}
        renderItem={(decision, index) => (
          <List.Item>
            <div style={{ width: "100%" }}>
              <Space wrap style={{ marginBottom: 8 }}>
                <Typography.Text strong>第 {index + 1} 次决策</Typography.Text>
                <Tag color={decision.allow_training ? "green" : "red"}>
                  {decision.allow_training ? "批准" : "拒绝或不满足门禁"}
                </Tag>
                <Tag>候选 v{decision.candidate_version}</Tag>
              </Space>
              <Descriptions
                size="small"
                column={1}
                items={[
                  { key: "purpose", label: "限定用途", children: decision.purpose },
                  {
                    key: "basis",
                    label: "决策依据",
                    children: decision.consent_basis ?? "未提供",
                  },
                  { key: "license", label: "许可证", children: decision.license_status },
                  {
                    key: "retention",
                    label: "保留截止",
                    children: decision.retention_until
                      ? new Date(decision.retention_until).toLocaleString("zh-CN")
                      : "不适用",
                  },
                  { key: "policy", label: "策略版本", children: decision.policy_version },
                  { key: "actor", label: "决策主体", children: decision.decided_by_subject_id },
                  {
                    key: "time",
                    label: "决策时间",
                    children: new Date(decision.created_at).toLocaleString("zh-CN"),
                  },
                  {
                    key: "id",
                    label: "决策标识",
                    children: <Typography.Text code copyable>{decision.decision_id}</Typography.Text>,
                  },
                ]}
              />
            </div>
          </List.Item>
        )}
      />
    </Card>
  );
}
