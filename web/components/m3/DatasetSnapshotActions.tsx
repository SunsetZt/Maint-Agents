"use client";

import { Button, Space } from "antd";

import type { DatasetSnapshot } from "@/lib/api/client";

type SnapshotActionFacts = Pick<DatasetSnapshot, "legal_actions">;

export function DatasetSnapshotActions({
  snapshot,
  busy = false,
  onInspect,
  onOpenRun,
  onCreateExperiment,
  onDownload,
}: {
  snapshot: SnapshotActionFacts;
  busy?: boolean;
  onInspect?: () => void;
  onOpenRun?: () => void;
  onCreateExperiment?: () => void;
  onDownload: () => void;
}) {
  const canDownload = snapshot.legal_actions.includes("DOWNLOAD_MANIFEST");
  const canCreateExperiment = snapshot.legal_actions.includes("CREATE_EXPERIMENT");
  if (!onInspect && !onOpenRun && !canDownload && !canCreateExperiment) return null;
  return (
    <Space wrap aria-busy={busy}>
      {onInspect ? <Button disabled={busy} onClick={onInspect}>详情与血缘</Button> : null}
      {onOpenRun ? (
        <Button disabled={busy} onClick={onOpenRun}>查看来源策展运行</Button>
      ) : null}
      {canCreateExperiment && onCreateExperiment ? (
        <Button type="primary" disabled={busy} onClick={onCreateExperiment}>
          基于此快照登记训练实验
        </Button>
      ) : null}
      {canDownload ? (
        <Button disabled={busy} onClick={onDownload}>下载已校验清单</Button>
      ) : null}
    </Space>
  );
}
