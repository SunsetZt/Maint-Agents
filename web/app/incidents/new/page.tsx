"use client";

import { Alert, Button, Card, Typography } from "antd";
import Link from "next/link";
import { FormEvent, useCallback, useEffect, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  type AssetSummary,
  createIncidentDraft,
  listAuthorizedAssets,
} from "@/lib/api/client";

export default function NewIncidentPage() {
  const [assets, setAssets] = useState<AssetSummary[] | null>(null);
  const [assetId, setAssetId] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<unknown>();
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<{ draftId: string; requestId: string }>();

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const response = await listAuthorizedAssets();
      setAssets(response.assets);
      setAssetId((current) => current || response.assets[0]?.asset_id || "");
    } catch (caught) {
      setError(caught);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!assetId || !description.trim()) return;
    setSubmitting(true);
    setError(undefined);
    try {
      const key = globalThis.crypto?.randomUUID?.() ?? `draft-${Date.now()}`;
      const response = await createIncidentDraft(
        { asset_id: assetId, description: description.trim() },
        key,
      );
      setResult({ draftId: response.draft.draft_id, requestId: response.requestId });
    } catch (caught) {
      setError(caught);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>新建故障草稿</Typography.Title>
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {assets === null && !error ? <LoadingState label="正在读取授权设备" /> : null}
        {assets?.length === 0 ? (
          <EmptyState description="当前身份没有可报障的授权设备" />
        ) : null}
        {assets && assets.length > 0 ? (
          <Card>
            <form className="form-grid" onSubmit={submit}>
              <label htmlFor="asset-id">
                授权设备
                <select
                  id="asset-id"
                  value={assetId}
                  onChange={(event) => setAssetId(event.target.value)}
                >
                  {assets.map((asset) => (
                    <option key={asset.asset_id} value={asset.asset_id}>
                      {asset.display_name ?? asset.asset_id} · {asset.model_code ?? "型号未知"} ·
                      {asset.site_name ?? "站点未知"} · {asset.warranty_status ?? "保修未知"}
                    </option>
                  ))}
                </select>
              </label>
              <label htmlFor="description">
                故障描述
                <textarea
                  id="description"
                  rows={6}
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  placeholder="描述现象、工况和发生时间，不要粘贴密码或凭据"
                />
              </label>
              <Button
                type="primary"
                htmlType="submit"
                loading={submitting}
                disabled={!assetId || !description.trim()}
              >
                保存故障草稿
              </Button>
            </form>
          </Card>
        ) : null}
        {result ? (
          <Alert
            type="success"
            showIcon
            message="草稿已由服务端确认"
            description={
              <div className="page-stack">
                <span className="request-id">请求标识：{result.requestId}</span>
                <Link href={`/incidents/drafts/${result.draftId}`}>
                  继续上传媒体并查看状态
                </Link>
              </div>
            }
          />
        ) : null}
      </div>
    </AppShell>
  );
}
