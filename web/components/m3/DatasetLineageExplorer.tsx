"use client";

import { Alert, Card, Select, Space, Typography } from "antd";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { DataLineageTimeline } from "@/components/m3/DataLineageTimeline";
import type { DataLineage } from "@/lib/api/client";

export function DatasetLineageExplorer({ lineage }: { lineage: DataLineage }) {
  const [workOrderId, setWorkOrderId] = useState(
    lineage.source_chains[0]?.work_order_id,
  );

  useEffect(() => {
    setWorkOrderId(lineage.source_chains[0]?.work_order_id);
  }, [lineage.lineage_id, lineage.source_chains]);

  const selectedChain = useMemo(
    () => lineage.source_chains.find((chain) => chain.work_order_id === workOrderId),
    [lineage.source_chains, workOrderId],
  );
  const stages = [
    ...(selectedChain?.stages ?? []),
    ...lineage.pipeline_stages,
  ];

  return (
    <div className="page-stack">
      {lineage.failure_reason ? (
        <Alert
          showIcon
          type="warning"
          title="血缘发布尚未完成"
          description={lineage.failure_reason}
        />
      ) : null}
      <Card title={`源工单追溯 (${lineage.source_chains.length})`} size="small">
        {lineage.source_chains.length ? (
          <Space direction="vertical" size="middle" style={{ width: "100%" }}>
            <Select
              showSearch
              value={workOrderId}
              style={{ width: "100%" }}
              placeholder="选择一条源工单"
              optionFilterProp="label"
              options={lineage.source_chains.map((chain) => ({
                value: chain.work_order_id,
                label: chain.work_order_id,
              }))}
              onChange={setWorkOrderId}
            />
            {selectedChain ? (
              <Space wrap>
                <Link href={`/work-orders/${encodeURIComponent(selectedChain.work_order_id)}`}>
                  进入原始工单
                </Link>
                <Typography.Text type="secondary">
                  反馈候选：{selectedChain.candidate_id ?? "治理链缺失"}
                </Typography.Text>
              </Space>
            ) : null}
          </Space>
        ) : (
          <Typography.Text type="secondary">该血缘记录没有源工单。</Typography.Text>
        )}
      </Card>
      <DataLineageTimeline stages={stages} />
    </div>
  );
}
