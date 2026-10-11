"use client";

import { Tag, Timeline, Typography } from "antd";

import type { LineageStage } from "@/lib/api/client";

const STAGE_LABELS: Record<string, string> = {
  WORK_ORDER: "工单闭环",
  OUTBOX_EVENT: "事务 Outbox",
  INBOX_EVENT: "事务 Inbox",
  FEEDBACK_CANDIDATE: "反馈候选",
  ELIGIBILITY: "用途授权",
  DLP: "DLP 脱敏",
  ANNOTATION: "标注复核",
  CURATION_RUN: "策展运行",
  DATASET_SNAPSHOT: "数据集快照",
  OPENLINEAGE: "OpenLineage 血缘",
};

type Stage = Pick<LineageStage, "stage" | "status" | "resource_id"> &
  Partial<Pick<LineageStage, "occurred_at" | "facts">>;

export function DataLineageTimeline({ stages }: { stages: Stage[] }) {
  return (
    <Timeline
      items={stages.map((stage) => ({
        color: stageColor(stage.status),
        children: (
          <div className="m3-lineage-stage">
            <div>
              <Typography.Text strong>
                {STAGE_LABELS[stage.stage] ?? stage.stage}
              </Typography.Text>{" "}
              <Tag>{stage.status}</Tag>
            </div>
            <Typography.Text code>{stage.resource_id}</Typography.Text>
            {stage.occurred_at ? (
              <Typography.Text type="secondary">
                {new Date(stage.occurred_at).toLocaleString("zh-CN")}
              </Typography.Text>
            ) : null}
            {stage.facts && Object.keys(stage.facts).length ? (
              <pre className="json-report">{JSON.stringify(stage.facts, null, 2)}</pre>
            ) : null}
          </div>
        ),
      }))}
    />
  );
}

function stageColor(status: string): "red" | "orange" | "green" {
  const normalized = status.toUpperCase();
  if (["FAILED", "DENIED", "CONFLICT", "INCOMPLETE"].some((value) => normalized.includes(value))) {
    return "red";
  }
  if (["PENDING", "BLOCKED", "NOT_RECORDED"].some((value) => normalized.includes(value))) {
    return "orange";
  }
  return "green";
}
