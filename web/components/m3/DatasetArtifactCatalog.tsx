"use client";

import { Alert, Card, Descriptions, List, Select, Space, Statistic, Tag, Typography } from "antd";
import { useMemo, useState } from "react";

import type { DatasetSnapshot } from "@/lib/api/client";

type Artifact = NonNullable<DatasetSnapshot["artifacts"]>[number];

const KIND_LABELS: Record<string, string> = {
  parquet: "Parquet 数据分片",
  quality_report: "质量报告",
  media_image: "图像训练制品",
  media_audio: "音频训练制品",
};

export function DatasetArtifactCatalog({ artifacts = [] }: { artifacts?: Artifact[] }) {
  const [kind, setKind] = useState<string>();
  const kinds = useMemo(
    () => [...new Set(artifacts.map((artifact) => artifact.kind))].sort(),
    [artifacts],
  );
  const visible = kind
    ? artifacts.filter((artifact) => artifact.kind === kind)
    : artifacts;
  const totalBytes = artifacts.reduce((total, artifact) => total + artifact.size_bytes, 0);
  const invalidHashes = artifacts.filter(
    (artifact) => !isSha256(artifact.content_hash) || !isSha256(artifact.schema_hash),
  );

  return (
    <Card title={`不可变制品完整性 (${artifacts.length})`} size="small">
      {artifacts.length ? (
        <div className="page-stack">
          <Space wrap size="large">
            <Statistic title="制品数量" value={artifacts.length} />
            <Statistic title="制品总大小" value={formatBytes(totalBytes)} />
            <Statistic title="制品类型" value={kinds.length} />
          </Space>
          {invalidHashes.length ? (
            <Alert
              showIcon
              type="error"
              title="存在格式异常的完整性哈希"
              description={`${invalidHashes.length} 个制品需要重新核验，当前页面不会把它们视为可信制品。`}
            />
          ) : (
            <Alert
              showIcon
              type="success"
              title="内容哈希与 Schema 哈希格式完整"
              description="这里验证目录中的 SHA-256 证据格式；实际下载仍以服务端 Manifest 和对象校验为准。"
            />
          )}
          <Select
            allowClear
            value={kind}
            placeholder="按制品类型筛选"
            style={{ width: "100%" }}
            options={kinds.map((value) => ({
              value,
              label: KIND_LABELS[value] ?? value,
            }))}
            onChange={setKind}
          />
          <List
            dataSource={visible}
            renderItem={(artifact) => (
              <List.Item>
                <div style={{ width: "100%" }}>
                  <Space wrap style={{ marginBottom: 8 }}>
                    <Tag color="blue">{KIND_LABELS[artifact.kind] ?? artifact.kind}</Tag>
                    {artifact.split ? <Tag>{artifact.split}</Tag> : null}
                    <Typography.Text code>{artifact.artifact_id}</Typography.Text>
                  </Space>
                  <Descriptions
                    size="small"
                    column={1}
                    items={[
                      { key: "rows", label: "行数", children: artifact.row_count },
                      { key: "size", label: "大小", children: formatBytes(artifact.size_bytes) },
                      {
                        key: "content-hash",
                        label: "内容哈希",
                        children: <Typography.Text code copyable>{artifact.content_hash}</Typography.Text>,
                      },
                      {
                        key: "schema-hash",
                        label: "Schema 哈希",
                        children: <Typography.Text code copyable>{artifact.schema_hash}</Typography.Text>,
                      },
                    ]}
                  />
                </div>
              </List.Item>
            )}
          />
        </div>
      ) : (
        <Typography.Text type="secondary">该快照尚未登记可消费制品。</Typography.Text>
      )}
    </Card>
  );
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

function isSha256(value: string): boolean {
  return /^sha256:[a-f0-9]{64}$/i.test(value);
}
