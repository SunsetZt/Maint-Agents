"use client";

import {
  Alert,
  Button,
  Card,
  Descriptions,
  Drawer,
  Dropdown,
  Form,
  Input,
  InputNumber,
  Modal,
  Progress,
  Space,
  Table,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";

import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  type ModelDeployment,
  type ModelDeploymentPlanInput,
  type ModelRelease,
  listModelDeployments,
  promoteModelRelease,
  requestModelDeployment,
  rollbackModelRelease,
} from "@/lib/api/client";

type DeploymentPlanValues = ModelDeploymentPlanInput;

export function DeploymentControlPanel({
  releases,
  onChanged,
}: {
  releases: ModelRelease[];
  onChanged: () => Promise<void>;
}) {
  const [deployments, setDeployments] = useState<ModelDeployment[]>();
  const [collectionActions, setCollectionActions] = useState<string[]>([]);
  const [error, setError] = useState<unknown>();
  const [commandError, setCommandError] = useState<unknown>();
  const [busy, setBusy] = useState<string>();
  const [requestTarget, setRequestTarget] = useState<ModelRelease>();
  const [rollbackTarget, setRollbackTarget] = useState<ModelDeployment>();
  const [rollbackReason, setRollbackReason] = useState("operator_requested_rollback");
  const [selected, setSelected] = useState<ModelDeployment>();
  const [form] = Form.useForm<DeploymentPlanValues>();
  const routingMode = Form.useWatch("routing_mode", form) ?? "STANDARD";
  const autoscalingMode = Form.useWatch("autoscaling_mode", form) ?? "FIXED";

  const load = useCallback(async () => {
    try {
      const result = await listModelDeployments();
      setDeployments(result.deployments);
      setCollectionActions(result.legalActions);
      setError(undefined);
    } catch (cause) {
      setError(cause);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const byRelease = useMemo(
    () => new Map((deployments ?? []).map((item) => [item.release_id, item])),
    [deployments],
  );
  const rows = useMemo(
    () => releases
      .filter((release) => release.approval?.status === "APPROVED" || byRelease.has(release.release_id))
      .map((release) => ({ release, deployment: byRelease.get(release.release_id) })),
    [byRelease, releases],
  );

  async function request(values: DeploymentPlanValues) {
    if (!requestTarget) return;
    setBusy(`request-${requestTarget.release_id}`);
    setCommandError(undefined);
    try {
      const {
        endpoint_picker_service_name: endpointPickerServiceName,
        endpoint_picker_service_port: endpointPickerServicePort,
        autoscaling_mode: requestedAutoscalingMode,
        autoscaling_min_replicas: autoscalingMinReplicas,
        autoscaling_max_replicas: autoscalingMaxReplicas,
        autoscaling_target_running_requests: autoscalingTargetRunningRequests,
        ...baseValues
      } = values;
      const routingInput: ModelDeploymentPlanInput = values.routing_mode === "INFERENCE_POOL"
        ? {
            ...baseValues,
            routing_mode: "INFERENCE_POOL",
            endpoint_picker_service_name: endpointPickerServiceName,
            endpoint_picker_service_port: endpointPickerServicePort,
            autoscaling_mode: requestedAutoscalingMode,
          }
        : {
            ...baseValues,
            routing_mode: "STANDARD",
            autoscaling_mode: requestedAutoscalingMode,
          };
      const input: ModelDeploymentPlanInput = requestedAutoscalingMode === "KEDA_VLLM"
        ? {
            ...routingInput,
            autoscaling_mode: "KEDA_VLLM",
            autoscaling_min_replicas: autoscalingMinReplicas,
            autoscaling_max_replicas: autoscalingMaxReplicas,
            autoscaling_target_running_requests: autoscalingTargetRunningRequests,
          }
        : { ...routingInput, autoscaling_mode: "FIXED" };
      await requestModelDeployment(requestTarget.release_id, requestTarget.version, input);
      setRequestTarget(undefined);
      form.resetFields();
      await Promise.all([load(), onChanged()]);
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function promote(deployment: ModelDeployment) {
    setBusy(`promote-${deployment.deployment_id}`);
    setCommandError(undefined);
    try {
      await promoteModelRelease(deployment.release_id, deployment.version);
      await Promise.all([load(), onChanged()]);
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function rollback() {
    if (!rollbackTarget) return;
    setBusy(`rollback-${rollbackTarget.deployment_id}`);
    setCommandError(undefined);
    try {
      await rollbackModelRelease(
        rollbackTarget.release_id,
        rollbackTarget.version,
        rollbackReason,
      );
      setRollbackTarget(undefined);
      await Promise.all([load(), onChanged()]);
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  return (
    <Card title="在线部署、灰度与回滚" className="deployment-control-card">
      <Space direction="vertical" size="middle" style={{ width: "100%" }}>
        <Alert
          type="info"
          showIcon
          message="期望状态与真实状态分离"
          description="操作员只提交 Shadow、晋级或回滚请求；机器控制器确认 KServe、HTTPRoute、Transformer 在线质量与 GPU 指标后，才更新实际阶段和流量。"
        />
        {commandError ? (
          <ErrorState error={commandError} onRetry={() => setCommandError(undefined)} />
        ) : null}
        {error ? (
          <ErrorState error={error} onRetry={() => void load()} />
        ) : deployments === undefined ? (
          <LoadingState />
        ) : rows.length === 0 ? (
          <EmptyState description="暂无审批通过、可部署的模型版本" />
        ) : (
          <Table
            rowKey={({ release }) => release.release_id}
            dataSource={rows}
            pagination={false}
            scroll={{ x: 1280 }}
            columns={[
              {
                title: "发布版本",
                render: (_, row) => <Typography.Text code>{row.release.release_id}</Typography.Text>,
              },
              {
                title: "控制器",
                render: (_, row) => row.deployment
                  ? <Tag color={controllerColor(row.deployment.status)}>{row.deployment.status}</Tag>
                  : <Tag>未申请</Tag>,
              },
              {
                title: "阶段",
                render: (_, row) => row.deployment
                  ? <Space><Tag>{row.deployment.current_stage}</Tag><span>→</span><Tag color="blue">{row.deployment.desired_stage}</Tag></Space>
                  : "—",
              },
              {
                title: "流量",
                width: 180,
                render: (_, row) => row.deployment ? (
                  <Progress
                    percent={row.deployment.observed_traffic_percent}
                    success={{ percent: row.deployment.observed_traffic_percent }}
                    format={() => `${row.deployment?.observed_traffic_percent}% / 期望 ${row.deployment?.desired_traffic_percent}%`}
                  />
                ) : "0%",
              },
              {
                title: "最新观察",
                render: (_, row) => {
                  const observation = row.deployment?.observations.at(-1);
                  return observation
                    ? <Tag color={decisionColor(observation.decision)}>{observation.stage} · {observation.decision}</Tag>
                    : "—";
                },
              },
              {
                title: "失败原因",
                render: (_, row) => row.deployment?.failure_reason ?? "—",
              },
              {
                title: "操作",
                fixed: "right",
                width: 320,
                render: (_, row) => {
                  const deployment = row.deployment;
                  return (
                    <Space wrap>
                      {!deployment
                        && collectionActions.includes("REQUEST_MODEL_DEPLOYMENT")
                        && row.release.approval?.status === "APPROVED" ? (
                          <Button type="primary" onClick={() => setRequestTarget(row.release)}>
                            发起 Shadow
                          </Button>
                        ) : null}
                      {deployment ? <Button type="link" onClick={() => setSelected(deployment)}>证据详情</Button> : null}
                      {deployment?.legal_actions.includes("PROMOTE_MODEL_RELEASE") ? (
                        <Button
                          type="primary"
                          loading={busy === `promote-${deployment.deployment_id}`}
                          onClick={() => void promote(deployment)}
                        >
                          晋级下一阶段
                        </Button>
                      ) : null}
                      {deployment?.legal_actions.includes("ROLLBACK_MODEL_RELEASE") ? (
                        <Button danger onClick={() => setRollbackTarget(deployment)}>回滚</Button>
                      ) : null}
                    </Space>
                  );
                },
              },
            ]}
          />
        )}
      </Space>

      <Modal
        title={`发起 Shadow 部署${requestTarget ? ` · ${requestTarget.release_id}` : ""}`}
        open={Boolean(requestTarget)}
        onCancel={() => setRequestTarget(undefined)}
        onOk={() => form.submit()}
        confirmLoading={busy === `request-${requestTarget?.release_id}`}
        okText="提交期望状态"
        width={760}
      >
        <Alert
          type="warning"
          showIcon
          message="提交后不会立即改变发布状态；控制器必须确认模型与影子路由均已就绪。"
          style={{ marginBottom: 16 }}
        />
        <Form
          form={form}
          layout="vertical"
          onFinish={(values) => void request(values)}
          initialValues={{
            namespace: "industrial-models",
            gateway_name: "industrial-agent-gateway",
            hostname: "models.ops.example.com",
            route_name: "industrial-agent-model",
            stable_service_name: "industrial-agent-stable",
            service_account_name: "model-storage-reader",
            serving_runtime_name: "vllm-lora-runtime",
            artifact_uri_prefix: "s3://industrial-models",
            routing_mode: "STANDARD",
            autoscaling_mode: "FIXED",
          }}
        >
          <Form.Item name="routing_mode" label="路由模式" rules={[{ required: true }]}>
            <RoutingModeSelect />
          </Form.Item>
          {routingMode === "INFERENCE_POOL" ? (
            <Alert
              type="warning"
              showIcon
              message="InferencePool 使用失败关闭（FailClose）"
              description="Endpoint Picker、InferencePool 或目标 Gateway 引用未就绪时，控制器不会静默退回普通候选 Service；稳定版本回滚仍保持独立可用。"
              style={{ marginBottom: 16 }}
            />
          ) : null}
          <Form.Item
            name="autoscaling_mode"
            label="自动扩缩容模式"
            rules={[{ required: true }]}
          >
            <AutoscalingModeSelect />
          </Form.Item>
          {autoscalingMode === "KEDA_VLLM" ? (
            <>
              <Alert
                type="info"
                showIcon
                message="Prometheus 查询和伸缩行为由服务端冻结"
                description="页面只提交有界副本数和每副本运行中请求目标；指标名、查询范围、Prometheus 地址及 600 秒缩容窗口不可编辑。"
                style={{ marginBottom: 16 }}
              />
              <Form.Item
                name="autoscaling_min_replicas"
                label="最小 GPU 副本"
                preserve={false}
                rules={[{ required: true, message: "请输入 2～8 的最小副本" }]}
              >
                <InputNumber min={2} max={8} precision={0} style={{ width: "100%" }} />
              </Form.Item>
              <Form.Item
                name="autoscaling_max_replicas"
                label="最大 GPU 副本"
                preserve={false}
                rules={[{ required: true, message: "请输入 2～32 的最大副本" }]}
              >
                <InputNumber min={2} max={32} precision={0} style={{ width: "100%" }} />
              </Form.Item>
              <Form.Item
                name="autoscaling_target_running_requests"
                label="每副本运行中请求目标"
                preserve={false}
                rules={[{ required: true, message: "请输入 1～32 的请求目标" }]}
              >
                <InputNumber min={1} max={32} precision={0} style={{ width: "100%" }} />
              </Form.Item>
            </>
          ) : null}
          {deploymentFields.map(([name, label]) => (
            <Form.Item key={name} name={name} label={label} rules={[{ required: true }]}>
              <Input />
            </Form.Item>
          ))}
          {routingMode === "INFERENCE_POOL" ? (
            <>
              <Form.Item
                name="endpoint_picker_service_name"
                label="Endpoint Picker Service"
                preserve={false}
                rules={[{ required: true, message: "请输入同 Namespace 的 EPP Service" }]}
              >
                <Input />
              </Form.Item>
              <Form.Item
                name="endpoint_picker_service_port"
                label="Endpoint Picker 端口"
                preserve={false}
                rules={[{ required: true, message: "请输入 EPP Service 端口" }]}
              >
                <InputNumber min={1} max={65535} precision={0} style={{ width: "100%" }} />
              </Form.Item>
            </>
          ) : null}
        </Form>
      </Modal>

      <Modal
        title="回滚到稳定服务"
        open={Boolean(rollbackTarget)}
        onCancel={() => setRollbackTarget(undefined)}
        onOk={() => void rollback()}
        okButtonProps={{ danger: true, disabled: rollbackReason.trim().length < 3 }}
        confirmLoading={busy === `rollback-${rollbackTarget?.deployment_id}`}
        okText="确认回滚"
      >
        <Alert type="warning" showIcon message="控制器会将 HTTPRoute 恢复为稳定服务 100%，并保留完整迁移证据。" style={{ marginBottom: 16 }} />
        <Input
          value={rollbackReason}
          onChange={(event) => setRollbackReason(event.target.value)}
          placeholder="填写机器可读原因码，例如 latency_regression"
        />
      </Modal>

      <Drawer
        title="部署状态与观察证据"
        width={900}
        open={Boolean(selected)}
        onClose={() => setSelected(undefined)}
      >
        {selected ? (
          <Space direction="vertical" size="large" style={{ width: "100%" }}>
            <Descriptions bordered size="small" column={1}>
              <Descriptions.Item label="Deployment ID">{selected.deployment_id}</Descriptions.Item>
              <Descriptions.Item label="KServe Service">{selected.namespace}/{selected.service_name}</Descriptions.Item>
              <Descriptions.Item label="HTTPRoute">{selected.route_name}</Descriptions.Item>
              <Descriptions.Item label="路由模式">{selected.routing_mode}</Descriptions.Item>
              {selected.routing_mode === "INFERENCE_POOL" ? (
                <>
                  <Descriptions.Item label="InferencePool">
                    {selected.inference_pool_name ?? "未返回"}
                  </Descriptions.Item>
                  <Descriptions.Item label="Endpoint Picker">
                    {selected.endpoint_picker_service_name && selected.endpoint_picker_service_port
                      ? `${selected.endpoint_picker_service_name}:${selected.endpoint_picker_service_port}`
                      : "未返回"}
                  </Descriptions.Item>
                  <Descriptions.Item label="失败模式">
                    {selected.endpoint_picker_failure_mode ?? "未返回"}
                  </Descriptions.Item>
                </>
              ) : null}
              <Descriptions.Item label="自动扩缩容模式">
                {selected.autoscaling_mode ?? "FIXED"}
              </Descriptions.Item>
              {selected.autoscaling_mode === "KEDA_VLLM" ? (
                <>
                  <Descriptions.Item label="伸缩策略版本">
                    {selected.autoscaling_policy_version ?? "未返回"}
                  </Descriptions.Item>
                  <Descriptions.Item label="GPU 副本范围">
                    {selected.autoscaling_min_replicas ?? "—"} / {selected.autoscaling_max_replicas ?? "—"}
                  </Descriptions.Item>
                  <Descriptions.Item label="服务指标">
                    {selected.autoscaling_metric_name ?? "未返回"} · {selected.autoscaling_metric_backend ?? "未返回"}
                  </Descriptions.Item>
                  <Descriptions.Item label="指标服务范围">
                    {selected.autoscaling_metric_scope ?? "未返回"}
                  </Descriptions.Item>
                  <Descriptions.Item label="伸缩目标">
                    {selected.autoscaling_target_running_requests ?? "—"} 个运行中请求/副本
                  </Descriptions.Item>
                  <Descriptions.Item label="缩容稳定窗口">
                    {selected.autoscaling_scale_down_stabilization_seconds ?? "—"} 秒
                  </Descriptions.Item>
                </>
              ) : null}
              <Descriptions.Item label="端点">{selected.endpoint_url ?? "未就绪"}</Descriptions.Item>
              <Descriptions.Item label="期望 Spec hash"><Typography.Text code copyable>{selected.desired_spec_hash}</Typography.Text></Descriptions.Item>
              <Descriptions.Item label="已应用 Spec hash"><Typography.Text code copyable>{selected.applied_spec_hash ?? "未应用"}</Typography.Text></Descriptions.Item>
              <Descriptions.Item label="Provider revision">{selected.provider_revision ?? "—"}</Descriptions.Item>
            </Descriptions>
            <Table
              rowKey="observation_id"
              size="small"
              pagination={false}
              dataSource={selected.observations}
              columns={[
                { title: "阶段", dataIndex: "stage" },
                { title: "结论", dataIndex: "decision", render: (value: string) => <Tag color={decisionColor(value)}>{value}</Tag> },
                { title: "请求/关键", render: (_, row) => `${row.request_count} / ${row.critical_case_count}` },
                {
                  title: "时序模型在线证据",
                  width: 250,
                  render: (_, row) => {
                    const metrics = row.metrics;
                    if (metrics.timeseries_request_count === undefined) return "—";
                    return (
                      <Space direction="vertical" size={0}>
                        <Typography.Text>
                          Runtime 请求 {metrics.timeseries_request_count} · P95 {formatMetric(metrics.timeseries_p95_latency_ms, " ms")}
                        </Typography.Text>
                        <Typography.Text type="secondary">
                          错误 {formatRate(metrics.timeseries_error_rate)} · 规则降级 {formatRate(metrics.timeseries_fallback_rate)}
                        </Typography.Text>
                        <Typography.Text type="secondary">
                          标签 {metrics.timeseries_labeled_outcome_count} · 误报 {formatRate(metrics.timeseries_false_positive_rate)} · 漏报 {formatRate(metrics.timeseries_miss_rate)}
                        </Typography.Text>
                      </Space>
                    );
                  },
                },
                { title: "失败原因", dataIndex: "failure_reasons", render: (value: string[]) => value.length ? value.join(", ") : "—" },
                { title: "证据 hash", dataIndex: "evidence_hash", render: (value: string) => <Typography.Text code>{value.slice(0, 16)}</Typography.Text> },
                { title: "窗口结束", dataIndex: "window_end", render: (value: string) => new Date(value).toLocaleString() },
              ]}
            />
          </Space>
        ) : null}
      </Drawer>
    </Card>
  );
}

const deploymentFields: Array<[keyof DeploymentPlanValues, string]> = [
  ["namespace", "Kubernetes Namespace"],
  ["gateway_name", "Gateway 名称"],
  ["hostname", "模型服务域名"],
  ["route_name", "HTTPRoute 名称"],
  ["stable_service_name", "稳定版本 Service"],
  ["service_account_name", "模型存储 ServiceAccount"],
  ["serving_runtime_name", "ServingRuntime Profile"],
  ["artifact_uri_prefix", "Adapter S3 前缀"],
];

function RoutingModeSelect({
  value = "STANDARD",
  onChange,
}: {
  value?: ModelDeploymentPlanInput["routing_mode"];
  onChange?: (value: ModelDeploymentPlanInput["routing_mode"]) => void;
}) {
  const labels = {
    STANDARD: "Standard Service",
    INFERENCE_POOL: "InferencePool 智能路由",
  } as const;
  return (
    <Dropdown
      trigger={["click"]}
      menu={{
        selectable: true,
        selectedKeys: [value],
        items: [
          { key: "STANDARD", label: labels.STANDARD },
          { key: "INFERENCE_POOL", label: labels.INFERENCE_POOL },
        ],
        onClick: ({ key }) => onChange?.(key as ModelDeploymentPlanInput["routing_mode"]),
      }}
    >
      <Button
        block
        role="combobox"
        aria-label="路由模式"
        aria-haspopup="listbox"
        style={{ textAlign: "left" }}
      >
        {labels[value]}
      </Button>
    </Dropdown>
  );
}

function AutoscalingModeSelect({
  value = "FIXED",
  onChange,
}: {
  value?: ModelDeploymentPlanInput["autoscaling_mode"];
  onChange?: (value: ModelDeploymentPlanInput["autoscaling_mode"]) => void;
}) {
  const labels = {
    FIXED: "固定副本",
    KEDA_VLLM: "KEDA vLLM GPU 自动扩缩容",
  } as const;
  return (
    <Dropdown
      trigger={["click"]}
      menu={{
        selectable: true,
        selectedKeys: [value],
        items: [
          { key: "FIXED", label: labels.FIXED },
          { key: "KEDA_VLLM", label: labels.KEDA_VLLM },
        ],
        onClick: ({ key }) => (
          onChange?.(key as ModelDeploymentPlanInput["autoscaling_mode"])
        ),
      }}
    >
      <Button
        block
        role="combobox"
        aria-label="自动扩缩容模式"
        aria-haspopup="listbox"
        style={{ textAlign: "left" }}
      >
        {labels[value]}
      </Button>
    </Dropdown>
  );
}

function controllerColor(status: string): string {
  if (status === "READY") return "green";
  if (status === "FAILED") return "red";
  return "gold";
}

function decisionColor(decision: string): string {
  if (decision === "PASS") return "green";
  if (decision === "ROLLBACK") return "red";
  return "gold";
}

function formatRate(value: number | undefined): string {
  return value === undefined ? "—" : `${(value * 100).toFixed(2)}%`;
}

function formatMetric(value: number | undefined, unit: string): string {
  return value === undefined ? "—" : `${value.toFixed(1)}${unit}`;
}
