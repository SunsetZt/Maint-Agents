import { Alert, Button, Empty, Spin } from "antd";

import { ApiClientError } from "@/lib/api/client";

export function LoadingState({ label = "正在加载服务端事实" }: { label?: string }) {
  return <Spin tip={label} size="large" />;
}

export function EmptyState({ description }: { description: string }) {
  return <Empty description={description} />;
}

export function ErrorState({
  error,
  onRetry,
}: {
  error: unknown;
  onRetry?: () => void;
}) {
  const apiError = error instanceof ApiClientError ? error : undefined;
  const forbidden = apiError?.status === 401 || apiError?.status === 403;
  const title = forbidden ? "当前身份无权访问" : "服务请求未完成";
  const description = apiError?.message ?? "网络或依赖暂时不可用，请安全重试。";
  return (
    <Alert
      type={forbidden ? "warning" : "error"}
      showIcon
      message={title}
      description={
        <div className="page-stack">
          <span>{description}</span>
          {apiError?.requestId ? (
            <span className="request-id">请求标识：{apiError.requestId}</span>
          ) : null}
          {onRetry && (apiError?.retryable ?? true) ? (
            <Button onClick={onRetry}>重试</Button>
          ) : null}
        </div>
      }
    />
  );
}
