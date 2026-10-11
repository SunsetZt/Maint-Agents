"use client";

import { Card, Empty, Timeline, Typography } from "antd";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import { type TimelineEvent, getDraftTimeline } from "@/lib/api/client";

export default function DraftTimelinePage() {
  const params = useParams<{ draftId: string }>();
  const [events, setEvents] = useState<TimelineEvent[]>();
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const response = await getDraftTimeline(params.draftId);
      setEvents([...response.events].sort((left, right) => left.sequence - right.sequence));
      setRequestId(response.requestId);
    } catch (caught) {
      setError(caught);
    }
  }, [params.draftId]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>服务端事实时间线</Typography.Title>
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!events && !error ? <LoadingState /> : null}
        {events?.length === 0 ? <Empty description="尚无时间线事件" /> : null}
        {events && events.length > 0 ? (
          <Card>
            <Timeline
              items={events.map((event) => ({
                children: (
                  <div>
                    <strong>#{event.sequence} · {event.event_type}</strong>
                    <div>{new Date(event.occurred_at).toLocaleString("zh-CN")}</div>
                  </div>
                ),
              }))}
            />
          </Card>
        ) : null}
        {requestId ? <span className="request-id">请求标识：{requestId}</span> : null}
      </div>
    </AppShell>
  );
}
