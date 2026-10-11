"use client";

import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  Modal,
  Progress,
  Space,
  Table,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useState } from "react";

import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  ApiClientError,
  type EnterpriseCandidateReleaseBatchProgress,
  type EnterpriseModelComponent,
  type ModelDeploymentPlanInput,
  listEnterpriseCandidateReleaseBatchDeploymentQueue,
  promoteModelRelease,
  requestEnterpriseCandidateReleaseComponentShadow,
  rollbackModelRelease,
} from "@/lib/api/client";

type ComponentProgress = EnterpriseCandidateReleaseBatchProgress["components"][number];

type ShadowTarget = {
  batch: EnterpriseCandidateReleaseBatchProgress;
  component: ComponentProgress;
};

type DeploymentSummary = {
  deployableComponents: number;
  activeDeployments: number;
  promotableComponents: number;
  rollbackAvailableComponents: number;
};

const emptySummary: DeploymentSummary = {
  deployableComponents: 0,
  activeDeployments: 0,
  promotableComponents: 0,
  rollbackAvailableComponents: 0,
};

export function EnterpriseReleaseBatchDeploymentPanel({
  onChanged,
}: {
  onChanged: () => Promise<void>;
}) {
  const [batches, setBatches] = useState<EnterpriseCandidateReleaseBatchProgress[]>();
  const [available, setAvailable] = useState(true);
  const [summary, setSummary] = useState<DeploymentSummary>(emptySummary);
  const [error, setError] = useState<unknown>();
  const [commandError, setCommandError] = useState<unknown>();
  const [busy, setBusy] = useState<string>();
  const [shadowTarget, setShadowTarget] = useState<ShadowTarget>();
  const [rollbackTarget, setRollbackTarget] = useState<ComponentProgress>();
  const [rollbackReason, setRollbackReason] = useState("operator_requested_rollback");
  const [form] = Form.useForm<ModelDeploymentPlanInput>();

  const load = useCallback(async () => {
    try {
      const result = await listEnterpriseCandidateReleaseBatchDeploymentQueue();
      setBatches(result.batches);
      setSummary({
        deployableComponents: result.deployableComponents,
        activeDeployments: result.activeDeployments,
        promotableComponents: result.promotableComponents,
        rollbackAvailableComponents: result.rollbackAvailableComponents,
      });
      setAvailable(true);
      setError(undefined);
    } catch (cause) {
      if (cause instanceof ApiClientError && cause.status === 403) {
        setAvailable(false);
        setBatches([]);
        setError(undefined);
        return;
      }
      setAvailable(true);
      setError(cause);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  function openShadow(
    batch: EnterpriseCandidateReleaseBatchProgress,
    component: ComponentProgress,
  ) {
    setShadowTarget({ batch, component });
    form.setFieldsValue(defaultPlan(component.component));
  }

  async function requestShadow(values: ModelDeploymentPlanInput) {
    if (!shadowTarget) return;
    const key = `shadow-${shadowTarget.component.release_id}`;
    setBusy(key);
    setCommandError(undefined);
    try {
      const input: ModelDeploymentPlanInput = {
        ...values,
        routing_mode: values.routing_mode ?? "STANDARD",
        autoscaling_mode: values.autoscaling_mode ?? "FIXED",
      };
      await requestEnterpriseCandidateReleaseComponentShadow(
        shadowTarget.batch.batch_key_sha256,
        shadowTarget.component.component,
        shadowTarget.batch.requested_by_subject_id,
        input,
      );
      setShadowTarget(undefined);
      form.resetFields();
      await Promise.all([load(), onChanged()]);
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function promote(component: ComponentProgress) {
    if (component.deployment_version == null) return;
    const key = `promote-${component.release_id}`;
    setBusy(key);
    setCommandError(undefined);
    try {
      await promoteModelRelease(component.release_id, component.deployment_version);
      await Promise.all([load(), onChanged()]);
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function rollback() {
    if (!rollbackTarget || rollbackTarget.deployment_version == null) return;
    const key = `rollback-${rollbackTarget.release_id}`;
    setBusy(key);
    setCommandError(undefined);
    try {
      await rollbackModelRelease(
        rollbackTarget.release_id,
        rollbackTarget.deployment_version,
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

  if (!available) return null;

  return (
    <Card
      title="三组件部署编排"
      extra={(
        <Space wrap>
          <Tag color="blue">可发起 {summary.deployableComponents}</Tag>
          <Tag color="cyan">部署中 {summary.activeDeployments}</Tag>
          <Tag color="green">可晋级 {summary.promotableComponents}</Tag>
          <Tag color="red">可回滚 {summary.rollbackAvailableComponents}</Tag>
        </Space>
      )}
    >
      <Space direction="vertical" size="middle" style={{ width: "100%" }}>
        <Alert
          showIcon
          type="info"
          message="共同批次、独立部署"
          description="只有 LLM、VLM、RUL 三项独立审批全部通过后才能发起 Shadow。每个组件使用独立 HTTPRoute 和稳定服务，并继续逐项收集观察证据、晋级或回滚。"
        />
        {commandError ? (
          <ErrorState error={commandError} onRetry={() => setCommandError(undefined)} />
        ) : null}
        {error ? (
          <ErrorState error={error} onRetry={() => void load()} />
        ) : batches === undefined ? (
          <LoadingState />
        ) : batches.length === 0 ? (
          <EmptyState description="暂无已完成三项独立审批的部署批次" />
        ) : (
          batches.map((batch) => (
            <Card
              key={`${batch.requested_by_subject_id}:${batch.batch_key_sha256}`}
              type="inner"
              title={(
                <Space wrap>
                  <Typography.Text strong>批次 {batch.batch_key_sha256.slice(0, 12)}</Typography.Text>
                  <Tag>{batch.status}</Tag>
                  <Tag color="purple">申请人 {batch.requested_by_subject_id}</Tag>
                </Space>
              )}
              extra={(
                <Space wrap>
                  <Tag>已部署 {batch.deployment_requested_count}/3</Tag>
                  <Tag color="cyan">Shadow Ready {batch.shadow_ready_count}</Tag>
                  <Tag color="gold">Canary {batch.canary_count}</Tag>
                  <Tag color="green">Production {batch.production_count}</Tag>
                </Space>
              )}
            >
              <Table
                rowKey="release_id"
                size="small"
                pagination={false}
                scroll={{ x: 1180 }}
                dataSource={batch.components}
                columns={[
                  {
                    title: "组件 / Release",
                    width: 260,
                    render: (_, component) => (
                      <Space direction="vertical" size={0}>
                        <Tag color="geekblue">{component.component}</Tag>
                        <Typography.Text code>{component.release_id}</Typography.Text>
                      </Space>
                    ),
                  },
                  {
                    title: "控制器",
                    render: (_, component) => component.deployment_status
                      ? <Tag color={controllerColor(component.deployment_status)}>{component.deployment_status}</Tag>
                      : <Tag>未申请</Tag>,
                  },
                  {
                    title: "阶段",
                    render: (_, component) => component.deployment_id
                      ? (
                          <Space>
                            <Tag>{component.current_stage}</Tag>
                            <span>→</span>
                            <Tag color="blue">{component.desired_stage}</Tag>
                          </Space>
                        )
                      : "—",
                  },
                  {
                    title: "流量",
                    width: 150,
                    render: (_, component) => (
                      <Progress
                        percent={component.observed_traffic_percent ?? 0}
                        size="small"
                        format={(value) => `${value ?? 0}%`}
                      />
                    ),
                  },
                  {
                    title: "最新观察",
                    render: (_, component) => component.latest_observation_decision
                      ? (
                          <Tag color={decisionColor(component.latest_observation_decision)}>
                            {component.latest_observation_decision}
                          </Tag>
                        )
                      : "—",
                  },
                  {
                    title: "下一动作",
                    dataIndex: "deployment_next_action",
                    render: (value: string) => <Tag>{value}</Tag>,
                  },
                  {
                    title: "失败原因",
                    render: (_, component) => component.deployment_failure_reason ?? "—",
                  },
                  {
                    title: "操作",
                    fixed: "right",
                    width: 300,
                    render: (_, component) => (
                      <Space wrap>
                        {component.deployment_legal_actions.includes(
                          "REQUEST_MODEL_DEPLOYMENT",
                        ) ? (
                          <Button
                            type="primary"
                            onClick={() => openShadow(batch, component)}
                          >
                            发起 {component.component} Shadow
                          </Button>
                        ) : null}
                        {component.deployment_legal_actions.includes(
                          "PROMOTE_MODEL_RELEASE",
                        ) ? (
                          <Button
                            type="primary"
                            loading={busy === `promote-${component.release_id}`}
                            onClick={() => void promote(component)}
                          >
                            晋级下一阶段
                          </Button>
                        ) : null}
                        {component.deployment_legal_actions.includes(
                          "ROLLBACK_MODEL_RELEASE",
                        ) ? (
                          <Button danger onClick={() => setRollbackTarget(component)}>
                            回滚
                          </Button>
                        ) : null}
                      </Space>
                    ),
                  },
                ]}
              />
            </Card>
          ))
        )}
      </Space>

      <Modal
        title={shadowTarget
          ? `发起 ${shadowTarget.component.component} Shadow`
          : "发起 Shadow"}
        open={Boolean(shadowTarget)}
        onCancel={() => {
          setShadowTarget(undefined);
          form.resetFields();
        }}
        onOk={() => form.submit()}
        confirmLoading={busy === `shadow-${shadowTarget?.component.release_id}`}
        okText="提交 Shadow 请求"
        width={720}
      >
        <Alert
          type="warning"
          showIcon
          message="该动作只提交期望状态；控制器确认 KServe 与路由就绪后才进入 Shadow。"
          style={{ marginBottom: 16 }}
        />
        <Form
          form={form}
          layout="vertical"
          onFinish={(values) => void requestShadow(values)}
        >
          {deploymentFields.map(([name, label]) => (
            <Form.Item key={name} name={name} label={label} rules={[{ required: true }]}>
              <Input />
            </Form.Item>
          ))}
        </Form>
      </Modal>

      <Modal
        title={rollbackTarget
          ? `回滚 ${rollbackTarget.component} 到稳定服务`
          : "回滚到稳定服务"}
        open={Boolean(rollbackTarget)}
        onCancel={() => setRollbackTarget(undefined)}
        onOk={() => void rollback()}
        okText="确认回滚"
        okButtonProps={{ danger: true, disabled: rollbackReason.trim().length < 3 }}
        confirmLoading={busy === `rollback-${rollbackTarget?.release_id}`}
      >
        <Alert
          type="warning"
          showIcon
          message="仅回滚当前组件；同批次其他组件保持各自状态。"
          style={{ marginBottom: 16 }}
        />
        <Input
          value={rollbackReason}
          onChange={(event) => setRollbackReason(event.target.value)}
          placeholder="填写机器可读原因码，例如 latency_regression"
        />
      </Modal>
    </Card>
  );
}

const deploymentFields: Array<[keyof ModelDeploymentPlanInput, string]> = [
  ["namespace", "Kubernetes Namespace"],
  ["gateway_name", "Gateway 名称"],
  ["hostname", "模型服务域名"],
  ["route_name", "独立 HTTPRoute 名称"],
  ["stable_service_name", "独立稳定版本 Service"],
  ["service_account_name", "模型存储 ServiceAccount"],
  ["serving_runtime_name", "ServingRuntime Profile"],
  ["artifact_uri_prefix", "Adapter S3 前缀"],
];

function defaultPlan(component: EnterpriseModelComponent): ModelDeploymentPlanInput {
  const slug = component.toLowerCase();
  return {
    namespace: "industrial-models",
    gateway_name: "industrial-agent-gateway",
    hostname: "models.ops.example.com",
    route_name: `industrial-agent-${slug}`,
    stable_service_name: `industrial-agent-${slug}-stable`,
    service_account_name: "model-storage-reader",
    serving_runtime_name: "vllm-lora-runtime",
    artifact_uri_prefix: "s3://industrial-models",
    routing_mode: "STANDARD",
    autoscaling_mode: "FIXED",
  };
}

function controllerColor(status: string) {
  if (status === "READY") return "green";
  if (status === "FAILED") return "red";
  if (status === "PENDING" || status === "APPLYING") return "blue";
  return "default";
}

function decisionColor(decision: string) {
  if (decision === "PASS") return "green";
  if (decision === "ROLLBACK") return "red";
  return "gold";
}
