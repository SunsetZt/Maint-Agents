"use client";

import { Button, Card, List, Select, Space, Tag, Typography } from "antd";
import { useMemo, useState } from "react";

import { governanceReasonLabel } from "@/components/m3/GovernanceBlockers";
import type { CurationRun } from "@/lib/api/client";

type Exclusion = CurationRun["exclusion_report"][number];

export function CurationExclusionReport({
  exclusions,
  onOpenCandidate,
}: {
  exclusions: Exclusion[];
  onOpenCandidate: (candidateId: string) => void;
}) {
  const [reason, setReason] = useState<string>();
  const summary = useMemo(() => summarizeExclusions(exclusions), [exclusions]);
  const visible = reason
    ? exclusions.filter((exclusion) => exclusion.reason === reason)
    : exclusions;

  return (
    <Card title={`排除报告 (${exclusions.length})`} size="small">
      {exclusions.length ? (
        <div className="page-stack">
          <Space wrap>
            {summary.map((item) => (
              <Tag key={item.reason} color="orange">
                {governanceReasonLabel(item.reason)} × {item.count}
              </Tag>
            ))}
          </Space>
          <Select
            allowClear
            value={reason}
            placeholder="按排除原因筛选"
            style={{ width: "100%" }}
            options={summary.map((item) => ({
              value: item.reason,
              label: `${governanceReasonLabel(item.reason)} (${item.count})`,
            }))}
            onChange={setReason}
          />
          <List
            size="small"
            dataSource={visible}
            renderItem={(exclusion) => (
              <List.Item
                actions={[
                  <Button
                    key="inspect"
                    type="link"
                    onClick={() => onOpenCandidate(exclusion.candidate_id)}
                  >
                    打开候选审计
                  </Button>,
                ]}
              >
                <List.Item.Meta
                  title={<Typography.Text code>{exclusion.candidate_id}</Typography.Text>}
                  description={(
                    <Space wrap>
                      <Tag color="orange">{governanceReasonLabel(exclusion.reason)}</Tag>
                      <Typography.Text type="secondary" code>{exclusion.reason}</Typography.Text>
                    </Space>
                  )}
                />
              </List.Item>
            )}
          />
        </div>
      ) : (
        <Typography.Text type="secondary">本次运行没有排除记录。</Typography.Text>
      )}
    </Card>
  );
}

export function summarizeExclusions(exclusions: Exclusion[]) {
  const counts = new Map<string, number>();
  for (const exclusion of exclusions) {
    counts.set(exclusion.reason, (counts.get(exclusion.reason) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([reason, count]) => ({ reason, count }))
    .sort((left, right) => right.count - left.count || left.reason.localeCompare(right.reason));
}
