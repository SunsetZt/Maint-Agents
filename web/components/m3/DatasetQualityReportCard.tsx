"use client";

import { Alert, Button, Card, Descriptions, Space, Statistic, Tag, Typography } from "antd";
import { useEffect, useState } from "react";

import { governanceReasonLabel } from "@/components/m3/GovernanceBlockers";
import { ErrorState } from "@/components/RequestState";
import {
  type DatasetQualityReport,
  getDatasetQualityReport,
} from "@/lib/api/client";

type QualityReportLoader = (
  snapshotId: string,
) => Promise<{ report: DatasetQualityReport; requestId: string }>;

export function DatasetQualityReportCard({
  snapshotId,
  available,
  loadReport = getDatasetQualityReport,
}: {
  snapshotId: string;
  available: boolean;
  loadReport?: QualityReportLoader;
}) {
  const [report, setReport] = useState<DatasetQualityReport>();
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setReport(undefined);
    setRequestId(undefined);
    setError(undefined);
  }, [snapshotId]);

  async function load() {
    setLoading(true);
    setError(undefined);
    try {
      const result = await loadReport(snapshotId);
      setReport(result.report);
      setRequestId(result.requestId);
    } catch (loadError) {
      setError(loadError);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card
      title="数据质量报告"
      size="small"
      extra={available ? (
        <Button loading={loading} onClick={() => void load()}>
          {report ? "重新核验" : "读取并核验"}
        </Button>
      ) : null}
    >
      {!available ? (
        <Alert
          showIcon
          type="warning"
          title="质量报告制品尚未登记"
          description="该快照不能作为训练输入。"
        />
      ) : error ? (
        <ErrorState error={error} onRetry={() => void load()} />
      ) : report ? (
        <div className="page-stack">
          <Alert
            showIcon
            type="success"
            title="Pandera 数据质量门禁已通过"
            description="报告字节、目录哈希、Schema、行数、切分及分组泄漏均已由服务端复验。"
          />
          <Space wrap size="large">
            <Statistic title="样本总数" value={report.row_count} />
            <Statistic title="分组泄漏" value={report.group_leakage_count} />
            <Statistic title="排除记录" value={report.exclusions.length} />
          </Space>
          <Descriptions
            size="small"
            column={1}
            items={[
              { key: "contract", label: "数据契约", children: report.contract_version },
              {
                key: "splits",
                label: "切分计数",
                children: (
                  <Space wrap>
                    {Object.entries(report.split_counts).map(([split, count]) => (
                      <Tag key={split}>{split}: {count}</Tag>
                    ))}
                  </Space>
                ),
              },
              {
                key: "hash",
                label: "报告内容哈希",
                children: <Typography.Text code copyable>{report.content_hash}</Typography.Text>,
              },
            ]}
          />
          {qualityExclusionSummary(report).length ? (
            <Space wrap>
              <Typography.Text type="secondary">排除原因：</Typography.Text>
              {qualityExclusionSummary(report).map(({ reason, count }) => (
                <Tag key={reason} color="orange">{governanceReasonLabel(reason)} × {count}</Tag>
              ))}
            </Space>
          ) : (
            <Typography.Text type="secondary">本报告没有排除记录。</Typography.Text>
          )}
          {requestId ? (
            <Typography.Text type="secondary" className="request-id">
              请求标识：{requestId}
            </Typography.Text>
          ) : null}
        </div>
      ) : (
        <Typography.Text type="secondary">
          按需读取对象存储中的报告，并由服务端完成完整性与语义复验。
        </Typography.Text>
      )}
    </Card>
  );
}

function qualityExclusionSummary(report: DatasetQualityReport) {
  const counts = new Map<string, number>();
  for (const exclusion of report.exclusions) {
    const reason = exclusion.reason ?? "unknown";
    counts.set(reason, (counts.get(reason) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([reason, count]) => ({ reason, count }))
    .sort((left, right) => right.count - left.count || left.reason.localeCompare(right.reason));
}
