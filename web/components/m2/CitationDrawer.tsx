"use client";

import {
  Alert,
  Button,
  Descriptions,
  Divider,
  Drawer,
  List,
  Space,
  Tag,
  Typography,
} from "antd";
import { useState } from "react";

import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  type Citation,
  downloadCitationSource,
  getCitation,
} from "@/lib/api/client";

export function CitationDrawer({ citationIds }: { citationIds: string[] }) {
  const [open, setOpen] = useState(false);
  const [citation, setCitation] = useState<Citation>();
  const [error, setError] = useState<unknown>();
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);

  async function show(citationId: string) {
    setOpen(true);
    setCitation(undefined);
    setError(undefined);
    setLoading(true);
    try {
      // Deliberately refetch every click: the server re-authorizes the exact
      // document version against the user's current tenant and asset scope.
      const response = await getCitation(citationId);
      setCitation(response.citation);
    } catch (caught) {
      setError(caught);
    } finally {
      setLoading(false);
    }
  }

  async function downloadOriginal() {
    if (!citation?.provenance.source.download_path) return;
    setError(undefined);
    setDownloading(true);
    try {
      const source = await downloadCitationSource(citation.citation_id);
      const url = URL.createObjectURL(source.blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = citation.provenance.source.source_filename ?? `${citation.citation_id}-source`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (caught) {
      setError(caught);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <>
      <List
        header="可追溯引用（点击后实时重新授权）"
        locale={{ emptyText: "本次诊断没有可见引用" }}
        dataSource={citationIds}
        renderItem={(id) => (
          <List.Item actions={[<Button key={id} onClick={() => void show(id)}>查看原文锚点</Button>]}>
            <Typography.Text code>{id}</Typography.Text>
          </List.Item>
        )}
      />
      <Drawer title="引用原文与全链路溯源" width={720} open={open} onClose={() => setOpen(false)}>
        {loading ? <LoadingState label="正在重新授权引用" /> : null}
        {error ? <ErrorState error={error} /> : null}
        {citation ? (
          <Space direction="vertical" size="middle" style={{ width: "100%" }}>
            <Alert
              type={citation.provenance.integrity.status === "VERIFIED" ? "success" : "error"}
              showIcon
              message={
                citation.provenance.integrity.status === "VERIFIED"
                  ? "处理链完整性校验通过"
                  : "处理链完整性校验异常"
              }
              description={
                <Typography.Text code copyable>
                  {citation.provenance.integrity.chain_digest}
                </Typography.Text>
              }
            />

            <Descriptions bordered column={1} size="small" title="授权原文锚点">
              <Descriptions.Item label="标题">{citation.title}</Descriptions.Item>
              <Descriptions.Item label="文档版本">v{citation.document_version}</Descriptions.Item>
              <Descriptions.Item label="锚点位置">
                {citation.provenance.anchor.anchor_kind} · 页码 {citation.page_number ?? "—"} ·
                字符 {citation.start_offset}–{citation.end_offset}
              </Descriptions.Item>
              <Descriptions.Item label="授权摘录">
                {citation.content_excerpt}
              </Descriptions.Item>
              <Descriptions.Item label="摘录校验和">
                <Typography.Text code copyable>{citation.provenance.anchor.excerpt_checksum}</Typography.Text>
              </Descriptions.Item>
            </Descriptions>

            <Descriptions bordered column={1} size="small" title="1. 原始来源与采集">
              <Descriptions.Item label="来源类型">
                <Tag color={citation.provenance.source.origin === "MANAGED_FILE" ? "blue" : "default"}>
                  {citation.provenance.source.origin === "MANAGED_FILE" ? "平台托管文件" : "直接录入"}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="原始位置">
                <Typography.Text copyable>{citation.provenance.source.source_uri}</Typography.Text>
              </Descriptions.Item>
              <Descriptions.Item label="原始文件">
                {citation.provenance.source.source_filename ?? "—"}
                {citation.provenance.source.size_bytes
                  ? ` · ${citation.provenance.source.size_bytes.toLocaleString()} 字节`
                  : ""}
              </Descriptions.Item>
              <Descriptions.Item label="文件类型">
                {citation.provenance.source.detected_mime
                  ?? citation.provenance.source.declared_mime
                  ?? "—"}
              </Descriptions.Item>
              <Descriptions.Item label="采集任务">
                {citation.provenance.source.ingestion_id ? (
                  <Space wrap>
                    <Typography.Text code copyable>
                      {citation.provenance.source.ingestion_id}
                    </Typography.Text>
                    <Tag>{citation.provenance.source.ingestion_status}</Tag>
                    <span>{citation.provenance.source.ingestion_attempt_count} 次尝试</span>
                  </Space>
                ) : "直接录入，无文件扫描任务"}
              </Descriptions.Item>
              <Descriptions.Item label="来源 SHA-256">
                <Typography.Text code copyable>{citation.source_checksum}</Typography.Text>
              </Descriptions.Item>
              {citation.provenance.source.download_path ? (
                <Descriptions.Item label="原文件操作">
                  <Button type="primary" loading={downloading} onClick={() => void downloadOriginal()}>
                    重新授权、校验并下载原文件
                  </Button>
                </Descriptions.Item>
              ) : null}
            </Descriptions>

            {citation.provenance.source.attempts.length ? (
              <List
                size="small"
                bordered
                header="扫描与解析尝试"
                dataSource={citation.provenance.source.attempts}
                renderItem={(attempt) => (
                  <List.Item>
                    <Space wrap>
                      <Tag color={attempt.status === "DRAFT_READY" ? "green" : "orange"}>
                        #{attempt.attempt_number} {attempt.status}
                      </Tag>
                      <span>解析器 {attempt.parser_version ?? "—"}</span>
                      <span>{formatDateTime(attempt.completed_at ?? attempt.started_at)}</span>
                      {attempt.failure_reason ? <Typography.Text type="danger">{attempt.failure_reason}</Typography.Text> : null}
                    </Space>
                  </List.Item>
                )}
              />
            ) : null}

            <Descriptions bordered column={1} size="small" title="2. 文档解析与切块">
              <Descriptions.Item label="文档版本 ID">
                <Typography.Text code copyable>{citation.document_version_id}</Typography.Text>
              </Descriptions.Item>
              <Descriptions.Item label="治理状态">
                <Space wrap>
                  <Tag color="green">{citation.provenance.document.version_status}</Tag>
                  <span>{citation.provenance.document.classification}</span>
                </Space>
              </Descriptions.Item>
              <Descriptions.Item label="解析器 / 切块器">
                {citation.provenance.document.parser_version} / {citation.provenance.document.chunker_version}
              </Descriptions.Item>
              <Descriptions.Item label="适用设备">
                {citation.provenance.document.device_models.join("、") || "全部设备"}
              </Descriptions.Item>
              <Descriptions.Item label="切块">
                第 {citation.provenance.chunk.ordinal} 块 · {citation.provenance.chunk.token_count} tokens ·
                {citation.provenance.chunk.embedding_dimensions} 维向量
              </Descriptions.Item>
              <Descriptions.Item label="切块 SHA-256">
                <Typography.Text code copyable>{citation.provenance.chunk.content_checksum}</Typography.Text>
              </Descriptions.Item>
            </Descriptions>

            <Descriptions bordered column={1} size="small" title="3. 索引评测与发布">
              <Descriptions.Item label="索引发布">
                {citation.provenance.index_release.name} v{citation.provenance.index_release.version}
              </Descriptions.Item>
              <Descriptions.Item label="发布状态">
                <Space wrap>
                  <Tag color="green">{citation.provenance.index_release.status}</Tag>
                  {citation.provenance.index_release.is_active ? <Tag color="blue">当前生效</Tag> : <Tag>历史版本</Tag>}
                </Space>
              </Descriptions.Item>
              <Descriptions.Item label="发布人 / 时间">
                {citation.provenance.index_release.published_by_subject_id ?? "—"} ·
                {formatDateTime(citation.provenance.index_release.published_at)}
              </Descriptions.Item>
              <Descriptions.Item label="索引内容校验和">
                <Typography.Text code copyable>{citation.provenance.index_release.content_checksum}</Typography.Text>
              </Descriptions.Item>
            </Descriptions>

            <Divider style={{ margin: 0 }} />
            <Space wrap>
              <IntegrityTag label="来源" valid={citation.provenance.integrity.source_checksum_consistent} />
              <IntegrityTag label="抽取" valid={citation.provenance.integrity.extraction_checksum_valid} />
              <IntegrityTag label="切块" valid={citation.provenance.integrity.chunk_checksum_valid} />
              <IntegrityTag label="锚点" valid={citation.provenance.integrity.citation_checksum_valid} />
              <IntegrityTag label="关联" valid={citation.provenance.integrity.document_links_valid} />
              <IntegrityTag label="发布" valid={citation.provenance.integrity.release_published} />
            </Space>
          </Space>
        ) : null}
      </Drawer>
    </>
  );
}

function IntegrityTag({ label, valid }: { label: string; valid: boolean }) {
  return <Tag color={valid ? "green" : "red"}>{label} {valid ? "通过" : "异常"}</Tag>;
}

function formatDateTime(value: string | null | undefined) {
  return value ? new Date(value).toLocaleString("zh-CN") : "—";
}
