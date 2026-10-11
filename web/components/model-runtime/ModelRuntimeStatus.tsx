"use client";

import { Alert, Card, Descriptions, Space, Spin, Tag, Typography } from "antd";
import { useEffect, useMemo, useState } from "react";

import {
  type ModelExecution,
  type ModelRuntimeStatus as ModelRuntimeStatusData,
  getModelRuntimeStatus,
} from "@/lib/api/client";

type ActionState = "idle" | "loading" | "success" | "failure";
type ModelExecutionView = Omit<ModelExecution, "component" | "target_environment"> & {
  component: string;
  target_environment: string;
};

type ModelRuntimeStatusProps = {
  requiredComponents: string[];
  executions?: ModelExecutionView[];
  actionState?: ActionState;
  actionLabel?: string;
  actionError?: unknown;
};

export function ModelRuntimeStatus({
  requiredComponents,
  executions = [],
  actionState = "idle",
  actionLabel = "模型操作",
  actionError,
}: ModelRuntimeStatusProps) {
  const [runtime, setRuntime] = useState<ModelRuntimeStatusData>();
  const [runtimeError, setRuntimeError] = useState<unknown>();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setRuntimeError(undefined);
    void getModelRuntimeStatus()
      .then((value) => {
        if (active) setRuntime(value);
      })
      .catch((caught: unknown) => {
        if (active) setRuntimeError(caught);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const components = useMemo(
    () => requiredComponents.map((name) => [name, runtime?.components[name]] as const),
    [requiredComponents, runtime],
  );
  const ready = Boolean(
    runtime
      && runtime.status === "READY"
      && components.every(([, component]) => component?.binding_status === "READY"),
  );
  const currentExecutions = useMemo(
    () => executions.filter((execution) => {
      const component = runtime?.components[execution.component];
      return ready
        && requiredComponents.includes(execution.component)
        && execution.guardrail_decision === "ALLOWED"
        && execution.target_environment === runtime?.target_environment
        && execution.model_alias === component?.alias
        && execution.release_id === component?.release_id
        && execution.deployment_id === component?.deployment_id;
    }),
    [executions, ready, requiredComponents, runtime],
  );

  if (loading) {
    return (
      <Card size="small" title="受治理模型运行状态">
        <Space aria-live="polite"><Spin size="small" />正在核验真实模型运行状态</Space>
      </Card>
    );
  }

  const runtimeLabel = ready
    ? runtime?.target_environment === "PRODUCTION"
      ? "生产真实模型"
      : "项目暂存真实模型"
    : "模型不可用";
  const unavailableReasons = components
    .filter(([, component]) => component?.binding_status !== "READY")
    .map(([name, component]) => `${name}: ${component?.reason ?? "component_binding_missing"}`);

  return (
    <Card
      size="small"
      title="受治理模型运行状态"
      extra={ready ? <Tag color="green">{runtimeLabel}</Tag> : undefined}
    >
      <div className="page-stack" aria-live="polite">
        {!ready ? (
          <Alert
            type="error"
            showIcon
            title="模型不可用"
            description={[
              errorMessage(runtimeError),
              ...unavailableReasons,
              runtime?.request_id ? `request_id: ${runtime.request_id}` : undefined,
            ].filter(Boolean).join(" · ")}
          />
        ) : null}
        {actionState === "loading" ? (
          <Alert type="info" showIcon title={`${actionLabel}处理中`} />
        ) : null}
        {actionState === "success" ? (
          <Alert type="success" showIcon title={`${actionLabel}已完成`} />
        ) : null}
        {actionState === "failure" ? (
          <Alert
            type="error"
            showIcon
            title={`${actionLabel}失败`}
            description={errorMessage(actionError)}
          />
        ) : null}

        {ready ? (
          <div className="page-stack">
            {components.map(([name, component]) => {
              if (!component) return null;
              const hasExecution = currentExecutions.some((item) => item.component === name);
              const processor = component.processor_version
                ?? (!hasExecution ? component.model_version : null);
              return (
                <div key={name}>
                  <Descriptions size="small" column={4} title={name.toUpperCase()}>
                    <Descriptions.Item label="Alias">{component.alias ?? "—"}</Descriptions.Item>
                    <Descriptions.Item label="Release">{component.release_id ?? "—"}</Descriptions.Item>
                    <Descriptions.Item label="Deployment">{component.deployment_id ?? "—"}</Descriptions.Item>
                    <Descriptions.Item label="处理器">{processor ?? "—"}</Descriptions.Item>
                  </Descriptions>
                </div>
              );
            })}
          </div>
        ) : null}

        {currentExecutions.length > 0 ? (
          <>
            <Alert type="success" showIcon title="真实模型已参与" />
            <div className="page-stack">
              {currentExecutions.map((execution) => (
                <div key={execution.inference_request_id}>
                  <Descriptions size="small" column={4} title={execution.component.toUpperCase()}>
                    <Descriptions.Item label="Inference">
                      {execution.inference_request_id}
                    </Descriptions.Item>
                    <Descriptions.Item label="模型/处理器">
                      {execution.processor_version}
                    </Descriptions.Item>
                    <Descriptions.Item label="Release">{execution.release_id}</Descriptions.Item>
                    <Descriptions.Item label="Deployment">{execution.deployment_id}</Descriptions.Item>
                    <Descriptions.Item label="耗时">{`${execution.latency_ms} ms`}</Descriptions.Item>
                    <Descriptions.Item label="Token">
                      {`${execution.prompt_tokens} / ${execution.completion_tokens}`}
                    </Descriptions.Item>
                    <Descriptions.Item label="Guardrail">
                      {execution.guardrail_decision}
                    </Descriptions.Item>
                    <Descriptions.Item label="请求标识">{execution.request_id}</Descriptions.Item>
                  </Descriptions>
                </div>
              ))}
            </div>
          </>
        ) : (
          <Typography.Text type="secondary">尚无当前成功推理证据</Typography.Text>
        )}
      </div>
    </Card>
  );
}

function errorMessage(error: unknown): string | undefined {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  return error == null ? undefined : String(error);
}
