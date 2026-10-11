"use client";

import { Space, Tag, Typography } from "antd";

import type {
  DatasetSnapshot,
  DatasetTrainingBlockerCode,
} from "@/lib/api/client";

const TRAINING_BLOCKER_LABELS: Record<DatasetTrainingBlockerCode, string> = {
  snapshot_not_candidate: "快照尚未进入候选状态",
  lineage_not_confirmed: "OpenLineage 血缘尚未确认",
  manifest_not_published: "Manifest 尚未发布",
  quality_report_missing: "质量报告缺失",
  dataset_empty: "数据集没有可训练样本",
  training_eligibility_not_granted: "训练资格尚未授予",
};

export function trainingBlockerLabel(blocker: DatasetTrainingBlockerCode): string {
  return TRAINING_BLOCKER_LABELS[blocker];
}

const TRAINING_BLOCKER_CODES = [
  "snapshot_not_candidate",
  "lineage_not_confirmed",
  "manifest_not_published",
  "quality_report_missing",
  "dataset_empty",
  "training_eligibility_not_granted",
] as const satisfies readonly DatasetTrainingBlockerCode[];

export const DATASET_TRAINING_BLOCKER_OPTIONS = TRAINING_BLOCKER_CODES.map((value) => ({
  value,
  label: trainingBlockerLabel(value),
}));

export function DatasetTrainingBlockers({
  blockers,
}: {
  blockers: DatasetSnapshot["training_blockers"];
}) {
  if (!blockers.length) return <Tag color="green">全部训练门禁已通过</Tag>;
  return (
    <Space wrap size={[4, 4]}>
      <Typography.Text type="secondary">训练阻断：</Typography.Text>
      {blockers.map((blocker) => (
        <Tag key={blocker} color="orange" title={blocker}>
          {trainingBlockerLabel(blocker)}
        </Tag>
      ))}
    </Space>
  );
}
