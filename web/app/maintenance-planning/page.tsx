"use client";

import {
  Alert,
  Button,
  Card,
  Checkbox,
  Col,
  Descriptions,
  Drawer,
  Form,
  Input,
  List,
  Modal,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  type CreateMaintenancePlanningEvaluationSuiteInput,
  type MaintenancePlanningActivation,
  type MaintenancePlanningEvaluationRun,
  type MaintenancePlanningEvaluationSuite,
  type MaintenancePlanningCouncil,
  createMaintenancePlanningCouncil,
  createMaintenancePlanningActivation,
  createMaintenancePlanningEvaluation,
  createMaintenancePlanningEvaluationSuite,
  getMaintenancePlanningCouncil,
  decideMaintenancePlanningActivation,
  listMaintenancePlanningActivations,
  listMaintenancePlanningCouncils,
  listMaintenancePlanningEvaluations,
  listMaintenancePlanningEvaluationSuites,
  retryMaintenancePlanningCouncil,
  rollbackMaintenancePlanningActivation,
  reviewMaintenancePlanningCouncil,
  type MaintenanceReviewPolicyStatus,
  getMaintenanceReviewPolicy,
  registerMaintenanceReviewRollout,
  changeMaintenanceReviewPolicy,
} from "@/lib/api/client";

type CreateValues = { incident_id: string; diagnosis_run_id: string };
type ReviewDecision = "ACCEPT" | "REJECT";
type EvaluationSuiteValues = {
  name: string;
  suite_version: string;
  evidence_tier: "PROJECT_STAGING_GOLD" | "ENTERPRISE_GOLD";
  cases_json: string;
};
type EvaluationRunValues = { suite_id: string; council_ids: string };
type ActivationRequestValues = {
  target_environment: "PROJECT_STAGING" | "PRODUCTION";
  reason: string;
};
type ActivationAction = "APPROVE" | "REJECT" | "ROLLBACK";


const statusColor: Record<string, string> = {
  QUEUED: "default",
  RUNNING: "processing",
  REVIEW_PENDING: "gold",
  ACCEPTED: "green",
  REJECTED: "red",
  FAILED: "volcano",
};

const roleLabel: Record<string, string> = {
  SAFETY: "安全专员",
  PARTS: "备件专员",
  DISPATCH: "派工专员",
  COORDINATOR: "协调 Agent",
};

export default function MaintenancePlanningPage() {
  const [councils, setCouncils] = useState<MaintenancePlanningCouncil[]>();
  const [selected, setSelected] = useState<MaintenancePlanningCouncil>();
  const [incidentFilter, setIncidentFilter] = useState("");
  const [error, setError] = useState<unknown>();
  const [commandError, setCommandError] = useState<unknown>();
  const [busy, setBusy] = useState<string>();
  const [createOpen, setCreateOpen] = useState(false);
  const [reviewDecision, setReviewDecision] = useState<ReviewDecision>();
  const [reviewReason, setReviewReason] = useState("");
  const [createForm] = Form.useForm<CreateValues>();
  const [evaluationSuites, setEvaluationSuites] = useState<MaintenancePlanningEvaluationSuite[]>();
  const [evaluations, setEvaluations] = useState<MaintenancePlanningEvaluationRun[]>();
  const [selectedEvaluation, setSelectedEvaluation] = useState<MaintenancePlanningEvaluationRun>();
  const [evaluationError, setEvaluationError] = useState<unknown>();
  const [suiteOpen, setSuiteOpen] = useState(false);
  const [evaluationRunOpen, setEvaluationRunOpen] = useState(false);
  const [suiteForm] = Form.useForm<EvaluationSuiteValues>();
  const [evaluationRunForm] = Form.useForm<EvaluationRunValues>();
  const [activations, setActivations] = useState<MaintenancePlanningActivation[]>();
  const [activationEvaluation, setActivationEvaluation] =
    useState<MaintenancePlanningEvaluationRun>();
  const [activationAction, setActivationAction] = useState<{
    item: MaintenancePlanningActivation;
    action: ActivationAction;
  }>();
  const [activationReason, setActivationReason] = useState("");
  const [activationForm] = Form.useForm<ActivationRequestValues>();

  const load = useCallback(async () => {
    try {
      const result = await listMaintenancePlanningCouncils(
        incidentFilter.trim() || undefined,
      );
      setCouncils(result.councils);
      setSelected((current) =>
        current
          ? result.councils.find((item) => item.council_id === current.council_id) ?? current
          : current,
      );
      setError(undefined);
    } catch (cause) {
      setError(cause);
    }
  }, [incidentFilter]);

  useEffect(() => {
    void load();
  }, [load]);

  const loadEvaluations = useCallback(async () => {
    try {
      const [suiteResult, evaluationResult, activationResult] = await Promise.all([
        listMaintenancePlanningEvaluationSuites(),
        listMaintenancePlanningEvaluations(),
        listMaintenancePlanningActivations(),
      ]);
      setEvaluationSuites(suiteResult.suites);
      setEvaluations(evaluationResult.evaluations);
      setActivations(activationResult.activations);
      setSelectedEvaluation((current) =>
        current
          ? evaluationResult.evaluations.find(
              (item) => item.evaluation_id === current.evaluation_id,
            ) ?? current
          : current,
      );
      setEvaluationError(undefined);
    } catch (cause) {
      setEvaluationError(cause);
    }
  }, []);

  useEffect(() => {
    void loadEvaluations();
  }, [loadEvaluations]);

  useEffect(() => {
    if (!selected || !["QUEUED", "RUNNING"].includes(selected.status)) return;
    const timer = window.setInterval(() => {
      void getMaintenancePlanningCouncil(selected.council_id)
        .then((item) => {
          replace(item);
          if (!["QUEUED", "RUNNING"].includes(item.status)) void load();
        })
        .catch(setCommandError);
    }, 5_000);
    return () => window.clearInterval(timer);
  }, [selected?.council_id, selected?.status, load]);

  const stats = useMemo(
    () => ({
      total: councils?.length ?? 0,
      running: councils?.filter((item) => ["QUEUED", "RUNNING"].includes(item.status)).length ?? 0,
      pending: councils?.filter((item) => item.status === "REVIEW_PENDING").length ?? 0,
      accepted: councils?.filter((item) => item.status === "ACCEPTED").length ?? 0,
    }),
    [councils],
  );

  function replace(item: MaintenancePlanningCouncil) {
    setSelected(item);
    setCouncils((current) => [
      item,
      ...(current ?? []).filter((value) => value.council_id !== item.council_id),
    ]);
  }

  async function submitCreate(values: CreateValues) {
    setBusy("create");
    setCommandError(undefined);
    try {
      const item = await createMaintenancePlanningCouncil(values, crypto.randomUUID());
      replace(item);
      setCreateOpen(false);
      createForm.resetFields();
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function retry(item: MaintenancePlanningCouncil) {
    setBusy(`retry-${item.council_id}`);
    setCommandError(undefined);
    try {
      replace(await retryMaintenancePlanningCouncil(item.council_id, item.version));
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function submitReview() {
    if (!selected || !reviewDecision) return;
    setBusy("review");
    setCommandError(undefined);
    try {
      replace(
        await reviewMaintenancePlanningCouncil(selected.council_id, selected.version, {
          decision: reviewDecision,
          reason: reviewReason.trim(),
        }),
      );
      setReviewDecision(undefined);
      setReviewReason("");
      await load();
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  function replaceEvaluation(item: MaintenancePlanningEvaluationRun) {
    setSelectedEvaluation(item);
    setEvaluations((current) => [
      item,
      ...(current ?? []).filter((value) => value.evaluation_id !== item.evaluation_id),
    ]);
  }

  function replaceActivation(item: MaintenancePlanningActivation) {
    setActivations((current) => [
      item,
      ...(current ?? []).filter((value) => value.activation_id !== item.activation_id),
    ]);
  }

  function openActivationRequest(item: MaintenancePlanningEvaluationRun) {
    const target =
      item.decision === "PRODUCTION_CANDIDATE_ELIGIBLE"
        ? "PRODUCTION"
        : "PROJECT_STAGING";
    setActivationEvaluation(item);
    activationForm.setFieldsValue({ target_environment: target, reason: "" });
  }

  async function submitActivationRequest(values: ActivationRequestValues) {
    if (!activationEvaluation) return;
    setBusy("create-activation");
    setEvaluationError(undefined);
    try {
      replaceActivation(
        await createMaintenancePlanningActivation(
          {
            evaluation_id: activationEvaluation.evaluation_id,
            target_environment: values.target_environment,
            reason: values.reason,
          },
          crypto.randomUUID(),
        ),
      );
      setActivationEvaluation(undefined);
      activationForm.resetFields();
      await loadEvaluations();
    } catch (cause) {
      setEvaluationError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function submitActivationAction() {
    if (!activationAction) return;
    setBusy(`activation-${activationAction.action.toLowerCase()}`);
    setEvaluationError(undefined);
    try {
      const item =
        activationAction.action === "ROLLBACK"
          ? await rollbackMaintenancePlanningActivation(
              activationAction.item.activation_id,
              activationAction.item.version,
              { reason: activationReason.trim() },
            )
          : await decideMaintenancePlanningActivation(
              activationAction.item.activation_id,
              activationAction.item.version,
              {
                decision: activationAction.action,
                reason: activationReason.trim(),
              },
            );
      replaceActivation(item);
      setActivationAction(undefined);
      setActivationReason("");
      await loadEvaluations();
    } catch (cause) {
      setEvaluationError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function submitEvaluationSuite(values: EvaluationSuiteValues) {
    setBusy("create-evaluation-suite");
    setEvaluationError(undefined);
    try {
      const cases = JSON.parse(values.cases_json) as unknown;
      if (!Array.isArray(cases)) throw new Error("cases_json 必须是案例数组");
      const item = await createMaintenancePlanningEvaluationSuite({
        name: values.name,
        suite_version: values.suite_version,
        evidence_tier: values.evidence_tier,
        cases: cases as CreateMaintenancePlanningEvaluationSuiteInput["cases"],
      });
      setEvaluationSuites((current) => [
        item,
        ...(current ?? []).filter((value) => value.suite_id !== item.suite_id),
      ]);
      setSuiteOpen(false);
      suiteForm.resetFields();
    } catch (cause) {
      setEvaluationError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function submitEvaluationRun(values: EvaluationRunValues) {
    setBusy("create-evaluation-run");
    setEvaluationError(undefined);
    try {
      const councilIds = values.council_ids
        .split(/[\s,]+/)
        .map((item) => item.trim())
        .filter(Boolean);
      const item = await createMaintenancePlanningEvaluation(
        { suite_id: values.suite_id, council_ids: councilIds },
        crypto.randomUUID(),
      );
      replaceEvaluation(item);
      setEvaluationRunOpen(false);
      evaluationRunForm.resetFields();
    } catch (cause) {
      setEvaluationError(cause);
    } finally {
      setBusy(undefined);
    }
  }


  return (
    <AppShell>
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <div>
          <Typography.Title level={2}>跨职能维修方案会审</Typography.Title>
          <Alert type="warning" showIcon message="这是管理页面，查看资料会记录来源曝光，不能再盲评同一来源。"
            action={<Button href="/maintenance-planning/reviews">进入独立盲评页</Button>} />
          <ReviewPolicyControls />
          <Typography.Paragraph type="secondary">
            安全、备件、派工三个只读专员独立分析已完成诊断，协调 Agent 仅汇总三方意见，最终由另一位专家人工复核。
          </Typography.Paragraph>
        </div>

        <Alert
          showIcon
          type="warning"
          message="建议闭环，不是执行闭环"
          description="本页面不会创建工单、采购、库存预留、派工、通知或设备控制；即使人工接受，也只表示方案建议已复核。"
        />

        <Row gutter={16}>
          <Col span={6}><Card><Statistic title="会审总数" value={stats.total} /></Card></Col>
          <Col span={6}><Card><Statistic title="运行中" value={stats.running} /></Card></Col>
          <Col span={6}><Card><Statistic title="待复核" value={stats.pending} /></Card></Col>
          <Col span={6}><Card><Statistic title="已接受建议" value={stats.accepted} /></Card></Col>
        </Row>

        {commandError ? <ErrorState error={commandError} onRetry={() => setCommandError(undefined)} /> : null}

        <Card
          title="会审任务"
          extra={
            <Space>
              <Input
                allowClear
                placeholder="按 incident_id 过滤"
                value={incidentFilter}
                onChange={(event) => setIncidentFilter(event.target.value)}
                style={{ width: 240 }}
              />
              <Button onClick={() => void load()}>刷新</Button>
              <Button type="primary" onClick={() => setCreateOpen(true)}>发起会审</Button>
            </Space>
          }
        >
          {error ? (
            <ErrorState error={error} onRetry={() => void load()} />
          ) : councils === undefined ? (
            <LoadingState />
          ) : councils.length === 0 ? (
            <EmptyState description="暂无多 Agent 会审。选择一条已完成诊断发起首次跨职能会审。" />
          ) : (
            <Table
              rowKey="council_id"
              dataSource={councils}
              pagination={{ pageSize: 10 }}
              columns={[
                {
                  title: "会审 / 事件",
                  render: (_, item) => (
                    <Space direction="vertical" size={0}>
                      <Typography.Text strong>{item.council_id}</Typography.Text>
                      <Typography.Text type="secondary">{item.incident_id}</Typography.Text>
                    </Space>
                  ),
                },
                { title: "状态", render: (_, item) => <Tag color={statusColor[item.status]}>{item.status}</Tag> },
                { title: "阶段", dataIndex: "stage" },
                { title: "尝试", dataIndex: "attempt_count" },
                { title: "模型版本", dataIndex: "model_release_id" },
                {
                  title: "操作",
                  render: (_, item) => (
                    <Space>
                      <Button size="small" onClick={() => setSelected(item)}>查看</Button>
                      {item.legal_actions.includes("RETRY") ? (
                        <Button
                          size="small"
                          loading={busy === `retry-${item.council_id}`}
                          onClick={() => void retry(item)}
                        >
                          重试
                        </Button>
                      ) : null}
                    </Space>
                  ),
                },
              ]}
            />
          )}
        </Card>

        <Card
          title="单 Agent / 四 Agent Gold 盲评"
          extra={
            <Space>
              <Button onClick={() => void loadEvaluations()}>刷新评测</Button>
              <Button onClick={() => setSuiteOpen(true)}>冻结 Gold 套件</Button>
              <Button
                type="primary"
                disabled={!evaluationSuites?.length}
                onClick={() => setEvaluationRunOpen(true)}
              >
                创建 A/B 评测
              </Button>
            </Space>
          }
        >
          <Space direction="vertical" size="middle" style={{ width: "100%" }}>
            <Alert
              showIcon
              type="info"
              message="同一 Gold 案例、匿名随机 A/B、双人独立评分"
              description="系统复用已完成 DiagnosisRun 作为单 Agent 基线和 Council 作为四 Agent 候选；偏好冲突时自动要求第三位专家仲裁。质量、安全、备件、派工、时延、Token 与专家时间全部通过后，只形成候选启用证据，不自动修改运行配置。"
            />
            <Row gutter={16}>
              <Col span={6}>
                <Statistic title="冻结套件" value={evaluationSuites?.length ?? 0} />
              </Col>
              <Col span={6}>
                <Statistic title="评测总数" value={evaluations?.length ?? 0} />
              </Col>
              <Col span={6}>
                <Statistic
                  title="盲评中"
                  value={evaluations?.filter((item) => item.status === "JUDGING").length ?? 0}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="候选可用"
                  value={
                    evaluations?.filter((item) =>
                      item.decision.endsWith("CANDIDATE_ELIGIBLE"),
                    ).length ?? 0
                  }
                />
              </Col>
            </Row>
            {evaluationError ? (
              <ErrorState error={evaluationError} onRetry={() => void loadEvaluations()} />
            ) : evaluations === undefined ? (
              <LoadingState />
            ) : evaluations.length === 0 ? (
              <EmptyState description="尚无价值评测。先冻结 Gold 套件，再绑定同一批已完成会审。" />
            ) : (
              <Table
                rowKey="evaluation_id"
                dataSource={evaluations}
                pagination={{ pageSize: 8 }}
                columns={[
                  {
                    title: "评测 / 套件",
                    render: (_, item) => (
                      <Space direction="vertical" size={0}>
                        <Typography.Text strong>{item.evaluation_id}</Typography.Text>
                        <Typography.Text type="secondary">
                          {item.suite_name} · {item.suite_version}
                        </Typography.Text>
                      </Space>
                    ),
                  },
                  {
                    title: "状态",
                    render: (_, item) => (
                      <Tag color={item.status === "COMPLETED" ? "green" : item.status === "FAILED" ? "red" : "processing"}>
                        {item.status}
                      </Tag>
                    ),
                  },
                  {
                    title: "盲评进度",
                    render: (_, item) => {
                      const completed = item.pairs.filter(
                        (pair) => pair.judgment_count >= pair.required_judgment_count,
                      ).length;
                      return `${completed} / ${item.pairs.length}`;
                    },
                  },
                  {
                    title: "历史曝光隔离",
                    render: (_, item) => (
                      <Tag color={item.historical_exposure_status === "VERIFIED" ? "green" : "gold"}>
                        {item.historical_exposure_status === "VERIFIED" ? "新协议证据已形成"
                          : item.historical_exposure_status === "TRACKED" ? "新协议待完成" : "历史记录，未证明"}
                      </Tag>
                    ),
                  },
                  {
                    title: "门禁结论",
                    render: (_, item) => (
                      <Tag color={item.decision.endsWith("CANDIDATE_ELIGIBLE") ? "green" : "default"}>
                        {item.decision}
                      </Tag>
                    ),
                  },
                  {
                    title: "操作",
                    render: (_, item) => {
                      const eligible = item.decision.endsWith("CANDIDATE_ELIGIBLE")
                        && item.historical_exposure_status === "VERIFIED";
                      const hasOpenRequest = activations?.some(
                        (activation) =>
                          activation.evaluation_id === item.evaluation_id &&
                          ["PENDING_APPROVAL", "ACTIVE"].includes(activation.status),
                      );
                      return (
                        <Space>
                          <Button size="small" onClick={() => setSelectedEvaluation(item)}>
                            查看管理记录
                          </Button>
                          {eligible ? (
                            <Button
                              size="small"
                              type="primary"
                              disabled={hasOpenRequest}
                              onClick={() => openActivationRequest(item)}
                            >
                              {hasOpenRequest ? "已有启用流程" : "申请启用"}
                            </Button>
                          ) : null}
                        </Space>
                      );
                    },
                  },
                ]}
              />
            )}
          </Space>
        </Card>

        <Card
          title="四 Agent 受治理启用与回滚"
          extra={<Button onClick={() => void loadEvaluations()}>刷新启用状态</Button>}
        >
          <Space direction="vertical" size="middle" style={{ width: "100%" }}>
            <Alert
              showIcon
              type="warning"
              message="评测通过不等于自动启用"
              description="领域专家只能提交申请；不同主体的租户管理员复核后，项目 Staging 或生产运行时才可解析该租户的四 Agent 绑定。IOAP_MULTI_AGENT_PLANNING_ENABLED 仍是平台总熔断，ACTIVE 记录可随时回滚。"
            />
            <Row gutter={16}>
              <Col span={8}>
                <Statistic title="启用流程" value={activations?.length ?? 0} />
              </Col>
              <Col span={8}>
                <Statistic
                  title="待独立审批"
                  value={
                    activations?.filter((item) => item.status === "PENDING_APPROVAL")
                      .length ?? 0
                  }
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="当前 ACTIVE"
                  value={activations?.filter((item) => item.status === "ACTIVE").length ?? 0}
                />
              </Col>
            </Row>
            {activations === undefined ? (
              <LoadingState />
            ) : activations.length === 0 ? (
              <EmptyState description="暂无启用申请。只有已通过 Gold 净收益门禁的评测可发起。" />
            ) : (
              <Table
                rowKey="activation_id"
                dataSource={activations}
                pagination={{ pageSize: 8 }}
                columns={[
                  {
                    title: "启用记录 / 评测",
                    render: (_, item) => (
                      <Space direction="vertical" size={0}>
                        <Typography.Text strong>{item.activation_id}</Typography.Text>
                        <Typography.Text type="secondary" copyable>
                          {item.evaluation_id}
                        </Typography.Text>
                      </Space>
                    ),
                  },
                  {
                    title: "目标",
                    render: (_, item) => <Tag>{item.target_environment}</Tag>,
                  },
                  {
                    title: "状态",
                    render: (_, item) => (
                      <Tag
                        color={
                          item.status === "ACTIVE"
                            ? "green"
                            : item.status === "PENDING_APPROVAL"
                              ? "gold"
                              : item.status === "REJECTED"
                                ? "red"
                                : "default"
                        }
                      >
                        {item.status}
                      </Tag>
                    ),
                  },
                  {
                    title: "申请 / 决策主体",
                    render: (_, item) => (
                      <Space direction="vertical" size={0}>
                        <Typography.Text>{item.requested_by_subject_id}</Typography.Text>
                        <Typography.Text type="secondary">
                          {item.decided_by_subject_id ?? "待不同主体审批"}
                        </Typography.Text>
                      </Space>
                    ),
                  },
                  {
                    title: "策略摘要",
                    render: (_, item) => (
                      <Typography.Text copyable>
                        {item.activation_policy_hash.slice(0, 16)}…
                      </Typography.Text>
                    ),
                  },
                  {
                    title: "操作",
                    render: (_, item) => (
                      <Space>
                        {item.legal_actions.map((action) => (
                          <Button
                            key={action}
                            size="small"
                            type={action === "APPROVE" ? "primary" : "default"}
                            danger={action === "REJECT" || action === "ROLLBACK"}
                            onClick={() =>
                              setActivationAction({ item, action: action as ActivationAction })
                            }
                          >
                            {action === "APPROVE"
                              ? "批准"
                              : action === "REJECT"
                                ? "拒绝"
                                : "回滚"}
                          </Button>
                        ))}
                      </Space>
                    ),
                  },
                ]}
              />
            )}
          </Space>
        </Card>
      </Space>

      <Modal
        open={createOpen}
        title="发起跨职能维修方案会审"
        okText="提交到 Temporal"
        cancelText="取消"
        confirmLoading={busy === "create"}
        onCancel={() => setCreateOpen(false)}
        onOk={() => void createForm.submit()}
      >
        <Alert
          type="info"
          showIcon
          message="仅接受 status=COMPLETED 且内容未漂移的诊断"
          style={{ marginBottom: 16 }}
        />
        <Form form={createForm} layout="vertical" onFinish={(values) => void submitCreate(values)}>
          <Form.Item name="incident_id" label="incident_id" rules={[{ required: true, min: 3 }]}>
            <Input placeholder="incident-..." />
          </Form.Item>
          <Form.Item name="diagnosis_run_id" label="diagnosis_run_id" rules={[{ required: true, min: 3 }]}>
            <Input placeholder="diagnosis-..." />
          </Form.Item>
        </Form>
      </Modal>

      <Drawer
        width={860}
        open={Boolean(selected)}
        title="会审证据与人工复核"
        onClose={() => setSelected(undefined)}
      >
        {selected ? (
          <Space direction="vertical" size="large" style={{ width: "100%" }}>
            <Descriptions bordered size="small" column={2}>
              <Descriptions.Item label="状态"><Tag color={statusColor[selected.status]}>{selected.status}</Tag></Descriptions.Item>
              <Descriptions.Item label="版本">{selected.version}</Descriptions.Item>
              <Descriptions.Item label="诊断" span={2}><Typography.Text copyable>{selected.diagnosis_run_id}</Typography.Text></Descriptions.Item>
              <Descriptions.Item label="模型 Release">{selected.model_release_id}</Descriptions.Item>
              <Descriptions.Item label="Prompt">{selected.prompt_bundle_id}</Descriptions.Item>
              <Descriptions.Item label="启用记录" span={2}>
                {selected.activation_id ? (
                  <Typography.Text copyable>{selected.activation_id}</Typography.Text>
                ) : "历史兼容会审（未绑定动态启用记录）"}
              </Descriptions.Item>
              <Descriptions.Item label="启用环境">
                {selected.activation_target_environment ?? "-"}
              </Descriptions.Item>
              <Descriptions.Item label="启用策略摘要">
                {selected.activation_policy_hash ? (
                  <Typography.Text copyable>
                    {selected.activation_policy_hash}
                  </Typography.Text>
                ) : "-"}
              </Descriptions.Item>
              <Descriptions.Item label="报告摘要" span={2}><Typography.Text copyable>{selected.diagnosis_report_digest}</Typography.Text></Descriptions.Item>
              <Descriptions.Item label="失败原因" span={2}>{selected.failure_code ?? "-"}</Descriptions.Item>
            </Descriptions>

            {selected.plan ? <PlanCard title="协调 Agent 待复核方案" output={selected.plan} /> : null}

            <Card title={`Agent 意见（${selected.contributions.length}）`}>
              <Space direction="vertical" size="middle" style={{ width: "100%" }}>
                {selected.contributions.map((item) => (
                  <PlanCard
                    key={item.contribution_id}
                    title={`第 ${item.attempt_number} 次 · ${roleLabel[item.agent_role] ?? item.agent_role}`}
                    output={item.output}
                    extra={<Typography.Text type="secondary">{item.model_release_id}</Typography.Text>}
                  />
                ))}
              </Space>
            </Card>

            {selected.review_reason ? (
              <Alert
                type={selected.status === "ACCEPTED" ? "success" : "error"}
                message={`${selected.review_decision} · ${selected.reviewed_by_subject_id}`}
                description={selected.review_reason}
              />
            ) : null}

            {selected.legal_actions.includes("REVIEW_ACCEPT") ? (
              <Space>
                <Button type="primary" onClick={() => setReviewDecision("ACCEPT")}>接受建议</Button>
                <Button danger onClick={() => setReviewDecision("REJECT")}>驳回建议</Button>
              </Space>
            ) : null}
          </Space>
        ) : null}
      </Drawer>

      <Modal
        open={Boolean(reviewDecision)}
        title={reviewDecision === "ACCEPT" ? "独立复核并接受建议" : "独立复核并驳回建议"}
        okText="提交复核"
        cancelText="取消"
        confirmLoading={busy === "review"}
        okButtonProps={{ disabled: reviewReason.trim().length < 8 }}
        onCancel={() => setReviewDecision(undefined)}
        onOk={() => void submitReview()}
      >
        <Alert
          type="warning"
          showIcon
          message="复核人与发起人必须不同；接受不会触发任何企业系统写操作。"
          style={{ marginBottom: 16 }}
        />
        <Input.TextArea
          rows={5}
          maxLength={1_000}
          showCount
          placeholder="填写至少 8 个字符的独立复核理由"
          value={reviewReason}
          onChange={(event) => setReviewReason(event.target.value)}
        />
      </Modal>

      <Modal
        width={760}
        open={suiteOpen}
        title="冻结维修方案 Gold 套件"
        okText="校验并冻结"
        cancelText="取消"
        confirmLoading={busy === "create-evaluation-suite"}
        onCancel={() => setSuiteOpen(false)}
        onOk={() => void suiteForm.submit()}
      >
        <Alert
          showIcon
          type="warning"
          message="Gold 参考项由领域专家提供，冻结后不得被候选输出改写。"
          style={{ marginBottom: 16 }}
        />
        <Form
          form={suiteForm}
          layout="vertical"
          initialValues={{ evidence_tier: "PROJECT_STAGING_GOLD", suite_version: "v1" }}
          onFinish={(values) => void submitEvaluationSuite(values)}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="name" label="套件名称" rules={[{ required: true, min: 3 }]}>
                <Input placeholder="PUMP-X100 维修方案 Gold" />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item name="suite_version" label="版本" rules={[{ required: true }]}>
                <Input placeholder="2026.08" />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item name="evidence_tier" label="证据层级" rules={[{ required: true }]}>
                <Select
                  options={[
                    { value: "PROJECT_STAGING_GOLD", label: "项目 Staging Gold" },
                    { value: "ENTERPRISE_GOLD", label: "企业 Gold" },
                  ]}
                />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item
            name="cases_json"
            label="Gold 案例 JSON"
            rules={[{ required: true, min: 10 }]}
            extra="每项绑定 diagnosis_run_id，并填写期望发现、安全停点、允许备件和派工资格。项目候选至少 3 例；生产候选至少 30 例且必须是企业 Gold。"
          >
            <Input.TextArea
              rows={12}
              placeholder={JSON.stringify(
                [
                  {
                    diagnosis_run_id: "diagnosis-...",
                    expected_findings: ["过滤器压差异常"],
                    required_safety_hold_points: ["确认断电和泄压"],
                    allowed_parts: ["FILTER-X100"],
                    required_dispatch_constraints: ["泵组资质"],
                    high_risk: true,
                  },
                ],
                null,
                2,
              )}
            />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        open={evaluationRunOpen}
        title="创建匿名 A/B 价值评测"
        okText="创建评测"
        cancelText="取消"
        confirmLoading={busy === "create-evaluation-run"}
        onCancel={() => setEvaluationRunOpen(false)}
        onOk={() => void evaluationRunForm.submit()}
      >
        <Alert
          showIcon
          type="info"
          message="Council 必须与 Gold 套件中的 DiagnosisRun 一一对应。"
          style={{ marginBottom: 16 }}
        />
        <Form
          form={evaluationRunForm}
          layout="vertical"
          onFinish={(values) => void submitEvaluationRun(values)}
        >
          <Form.Item name="suite_id" label="冻结 Gold 套件" rules={[{ required: true }]}>
            <Select
              showSearch
              optionFilterProp="label"
              options={(evaluationSuites ?? [])
                .filter((item) => item.status === "FROZEN")
                .map((item) => ({
                  value: item.suite_id,
                  label: `${item.name} · ${item.suite_version} · ${item.sample_count} 例`,
                }))}
            />
          </Form.Item>
          <Form.Item
            name="council_ids"
            label="Council ID 列表"
            rules={[{ required: true, min: 3 }]}
            extra="使用逗号、空格或换行分隔；数量必须与套件案例数一致。"
          >
            <Input.TextArea rows={6} placeholder={"maintenance-council-...\nmaintenance-council-..."} />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        open={Boolean(activationEvaluation)}
        title="申请四 Agent 受治理启用"
        okText="提交独立审批"
        cancelText="取消"
        confirmLoading={busy === "create-activation"}
        onCancel={() => setActivationEvaluation(undefined)}
        onOk={() => void activationForm.submit()}
      >
        <Alert
          showIcon
          type="info"
          message="申请只创建待审批记录，不直接改变运行时"
          description={
            activationEvaluation
              ? `绑定评测 ${activationEvaluation.evaluation_id} 与报告 ${activationEvaluation.report_hash ?? "-"}`
              : undefined
          }
          style={{ marginBottom: 16 }}
        />
        <Form
          form={activationForm}
          layout="vertical"
          onFinish={(values) => void submitActivationRequest(values)}
        >
          <Form.Item
            name="target_environment"
            label="目标环境"
            rules={[{ required: true }]}
          >
            <Select
              options={[
                { value: "PROJECT_STAGING", label: "项目 Staging" },
                ...(activationEvaluation?.decision === "PRODUCTION_CANDIDATE_ELIGIBLE"
                  ? [{ value: "PRODUCTION", label: "生产" }]
                  : []),
              ]}
            />
          </Form.Item>
          <Form.Item
            name="reason"
            label="启用理由"
            rules={[{ required: true, min: 8, max: 1_000 }]}
          >
            <Input.TextArea
              rows={4}
              maxLength={1_000}
              showCount
              placeholder="说明净收益证据、使用范围和回滚责任"
            />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        open={Boolean(activationAction)}
        title={
          activationAction?.action === "APPROVE"
            ? "独立批准四 Agent 启用"
            : activationAction?.action === "REJECT"
              ? "拒绝四 Agent 启用"
              : "回滚四 Agent 启用"
        }
        okText={
          activationAction?.action === "APPROVE"
            ? "批准"
            : activationAction?.action === "REJECT"
              ? "拒绝"
              : "立即回滚"
        }
        okButtonProps={{
          danger:
            activationAction?.action === "REJECT" ||
            activationAction?.action === "ROLLBACK",
          disabled: activationReason.trim().length < 8,
        }}
        confirmLoading={busy?.startsWith("activation-")}
        onCancel={() => {
          setActivationAction(undefined);
          setActivationReason("");
        }}
        onOk={() => void submitActivationAction()}
      >
        <Alert
          showIcon
          type={activationAction?.action === "APPROVE" ? "warning" : "error"}
          message={
            activationAction?.action === "APPROVE"
              ? "审批人与申请人必须不同；批准只授权对应环境和固定评测摘要。"
              : "拒绝或回滚不会删除历史评测、审批和会审证据。"
          }
          style={{ marginBottom: 16 }}
        />
        <Input.TextArea
          rows={5}
          maxLength={1_000}
          showCount
          value={activationReason}
          onChange={(event) => setActivationReason(event.target.value)}
          placeholder="填写至少 8 个字符的决策或回滚理由"
        />
      </Modal>

      <Drawer
        width={1080}
        open={Boolean(selectedEvaluation)}
        title="多 Agent 净收益盲评"
        onClose={() => setSelectedEvaluation(undefined)}
      >
        {selectedEvaluation ? (
          <Space direction="vertical" size="large" style={{ width: "100%" }}>
            <Descriptions bordered size="small" column={2}>
              <Descriptions.Item label="状态">
                <Tag color={selectedEvaluation.status === "COMPLETED" ? "green" : "processing"}>
                  {selectedEvaluation.status}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="结论">
                <Tag color={selectedEvaluation.decision.endsWith("CANDIDATE_ELIGIBLE") ? "green" : "default"}>
                  {selectedEvaluation.decision}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Gold 套件">
                {selectedEvaluation.suite_name} · {selectedEvaluation.suite_version}
              </Descriptions.Item>
              <Descriptions.Item label="证据层级">
                {selectedEvaluation.evidence_tier}
              </Descriptions.Item>
              <Descriptions.Item label="策略哈希" span={2}>
                <Typography.Text copyable>{selectedEvaluation.policy_hash}</Typography.Text>
              </Descriptions.Item>
              <Descriptions.Item label="报告哈希" span={2}>
                {selectedEvaluation.report_hash ? (
                  <Typography.Text copyable>{selectedEvaluation.report_hash}</Typography.Text>
                ) : "待完成"}
              </Descriptions.Item>
            </Descriptions>

            {selectedEvaluation.status === "COMPLETED" ? (
              <Card title="净收益指标与硬门禁" size="small">
                <Descriptions bordered size="small" column={3}>
                  <Descriptions.Item label="质量提升">
                    {metric(selectedEvaluation.aggregate_metrics.quality_delta)}
                  </Descriptions.Item>
                  <Descriptions.Item label="95% CI 下界">
                    {metric(selectedEvaluation.aggregate_metrics.quality_ci_low)}
                  </Descriptions.Item>
                  <Descriptions.Item label="95% CI 上界">
                    {metric(selectedEvaluation.aggregate_metrics.quality_ci_high)}
                  </Descriptions.Item>
                  <Descriptions.Item label="时延倍率">
                    {metric(selectedEvaluation.aggregate_metrics.candidate_latency_ratio)}
                  </Descriptions.Item>
                  <Descriptions.Item label="Token 倍率">
                    {metric(selectedEvaluation.aggregate_metrics.candidate_token_ratio)}
                  </Descriptions.Item>
                  <Descriptions.Item label="专家节省秒数">
                    {metric(selectedEvaluation.aggregate_metrics.expert_time_savings_seconds)}
                  </Descriptions.Item>
                </Descriptions>
                <Space wrap style={{ marginTop: 16 }}>
                  {Object.entries(selectedEvaluation.gate_results).map(([name, passed]) => (
                    <Tag key={name} color={passed ? "green" : "red"}>
                      {passed ? "PASS" : "FAIL"} · {name}
                    </Tag>
                  ))}
                </Space>
              </Card>
            ) : (
              <Alert
                showIcon
                type="warning"
                message="此处包含可识别来源的管理资料；独立评分只能在专用盲评页领取后进行。"
              />
            )}

            {selectedEvaluation.pairs.map((pair) => (
              <Card
                key={pair.case_id}
                title={`${pair.case_id} · ${pair.judgment_count}/${pair.required_judgment_count} 份评分`}

              >
                <Space direction="vertical" size="middle" style={{ width: "100%" }}>
                  <Alert
                    type={pair.needs_adjudication ? "warning" : "info"}
                    message={
                      pair.needs_adjudication
                        ? "前两位专家偏好不一致，等待第三位专家仲裁"
                        : pair.judged_by_current_subject
                          ? "当前身份已提交本案例评分"
                          : "此处为管理复核资料，不用于匿名评分"
                    }
                  />
                  <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>
                    {JSON.stringify(pair.rubric, null, 2)}
                  </pre>
                  <Row gutter={16}>
                    <Col span={12}>
                      <PlanCard title="匿名方案 A" output={pair.variant_a} />
                    </Col>
                    <Col span={12}>
                      <PlanCard title="匿名方案 B" output={pair.variant_b} />
                    </Col>
                  </Row>
                  {pair.revealed_assignment ? (
                    <Alert
                      type="success"
                      message={`已揭盲：A = ${pair.revealed_assignment.A}，B = ${pair.revealed_assignment.B}`}
                    />
                  ) : null}
                </Space>
              </Card>
            ))}
          </Space>
        ) : null}
      </Drawer>


    </AppShell>
  );
}

function PlanCard({
  title,
  output,
  extra,
}: {
  title: string;
  output: Record<string, unknown>;
  extra?: ReactNode;
}) {
  const fields = [
    ["建议", "recommendations"],
    ["约束", "constraints"],
    ["缺失事实", "missing_facts"],
    ["安全停点", "safety_hold_points"],
    ["派工约束", "dispatch_constraints"],
  ] as const;
  return (
    <Card size="small" title={title} extra={extra}>
      <Space direction="vertical" style={{ width: "100%" }}>
        <Space wrap>
          <Tag>{String(output.agent_role ?? "UNKNOWN")}</Tag>
          <Tag color={output.readiness === "READY_FOR_HUMAN_REVIEW" ? "green" : "gold"}>
            {String(output.readiness ?? "UNKNOWN")}
          </Tag>
        </Space>
        <Typography.Paragraph>{String(output.summary ?? "暂无摘要")}</Typography.Paragraph>
        {fields.map(([label, key]) => {
          const values = stringArray(output[key]);
          return values.length ? (
            <div key={key}>
              <Typography.Text strong>{label}</Typography.Text>
              <List size="small" dataSource={values} renderItem={(item) => <List.Item>{item}</List.Item>} />
            </div>
          ) : null;
        })}
        {Array.isArray(output.required_parts) && output.required_parts.length ? (
          <div>
            <Typography.Text strong>备件建议（非库存事实）</Typography.Text>
            <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(output.required_parts, null, 2)}</pre>
          </div>
        ) : null}
      </Space>
    </Card>
  );
}

function ReviewPolicyControls() {
  const [status, setStatus] = useState<MaintenanceReviewPolicyStatus>();
  const [receipt, setReceipt] = useState<File>();
  const [deploymentConfirmed, setDeploymentConfirmed] = useState(false);
  const [receiptId, setReceiptId] = useState("");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>();
  const [notice, setNotice] = useState("");
  const pendingCommand = useRef<{ fingerprint: string; key: string } | null>(null);

  async function refresh() {
    const current = await getMaintenanceReviewPolicy();
    setStatus(current);
    return current;
  }

  async function perform(action: () => Promise<void>) {
    if (busy) return;
    setBusy(true); setError(undefined); setNotice("");
    try { await action(); } catch (cause) { setError(cause); }
    finally { setBusy(false); }
  }

  async function register() {
    if (!receipt || !deploymentConfirmed || reason.trim().length < 8) return;
    if (receipt.size > 262144) {
      throw new Error("部署记录须不超过256 KiB。");
    }
    const result = await registerMaintenanceReviewRollout(await receipt.text(), reason.trim());
    setReceiptId(result.receipt_id);
    setNotice("管理员部署确认已登记；接下来点击启用协议。");
    await refresh();
  }

  async function change(enabled: boolean) {
    if (!status) return;
    const fingerprint = JSON.stringify([enabled, status.version, receiptId, reason]);
    if (pendingCommand.current?.fingerprint !== fingerprint) {
      pendingCommand.current = { fingerprint, key: crypto.randomUUID() };
    }
    await changeMaintenanceReviewPolicy(enabled, status.version, reason, receiptId, pendingCommand.current.key);
    setNotice("命令已处理，下方显示服务端当前状态；停用不会删除曝光、领取或历史报告。");
    await refresh();
    pendingCommand.current = null;
  }

  const selected = status?.receipts.find((item) => item.receipt_id === receiptId);
  return <Card size="small" title="盲评协议部署与启用（租户管理员）" style={{ marginTop: 16 }}>
    <Space direction="vertical" style={{ width: "100%" }}>
      <Typography.Text type="secondary">
        由项目管理员确认实际部署后启用，不需要外部签名人或签名包。此页面不部署或重启服务；旧来源保持历史未知。
      </Typography.Text>
      <Button loading={busy} onClick={() => void perform(async () => { await refresh(); })}>查看 / 刷新协议状态</Button>
      {error ? <ErrorState error={error} /> : null}
      {notice ? <Alert showIcon type="success" message={notice} /> : null}
      {status ? <>
        <Space wrap>
          <Tag color={status.enabled && status.blockers.length === 0 ? "green" : "gold"}>
            {status.enabled ? "已记录启用" : "未启用 / 已停用"} · v{status.version}
          </Tag>
          <Typography.Text>环境：{status.environment_id ?? "未配置"}</Typography.Text>
          <Typography.Text>启用时间：{status.cutover_at ?? "—"}</Typography.Text>
        </Space>
        {status.blockers.map((item) => <Alert key={item} type="warning" showIcon message={item} />)}
        <label>原始回执 JSON <input type="file" accept="application/json,.json" disabled={busy}
          onChange={(event) => { setReceipt(event.target.files?.[0]); setDeploymentConfirmed(false); }} /></label>
        <Checkbox checked={deploymentConfirmed} disabled={busy}
          onChange={(event) => setDeploymentConfirmed(event.target.checked)}>
          我已核对全部读取服务的部署记录，确认新版本就绪且旧实例已退出（管理员确认，非独立签名验收）
        </Checkbox>
        <Input.TextArea aria-label="协议操作原因" placeholder="填写部署确认 / 启用 / 停用原因（至少8个字符）"
          value={reason} disabled={busy} onChange={(event) => setReason(event.target.value)} maxLength={1000} />
        <Button disabled={!receipt || !deploymentConfirmed || reason.trim().length < 8 || busy}
          loading={busy} onClick={() => void perform(register)}>
          确认部署并登记
        </Button>
        <Select aria-label="选择部署回执" placeholder="选择已登记的部署回执" style={{ width: "100%" }}
          value={receiptId || undefined} disabled={busy} onChange={setReceiptId}
          options={status.receipts.map((item) => ({ value: item.receipt_id,
            label: `${item.receipt_id} · ${item.blockers.length ? "暂不可启用" : "可用于启用"}` }))} />
        {selected?.blockers.map((item) => <Alert key={item} type="warning" message={item} />)}
        <Space>
          <Button type="primary" loading={busy}
            disabled={busy || status.enabled || reason.trim().length < 8 || !selected || selected.blockers.length > 0}
            onClick={() => Modal.confirm({ title: "启用此租户的盲评隔离协议？",
              content: "确认管理员登记的部署记录覆盖全部读取服务。此操作只启用协议，不修改历史数据。",
              onOk: () => perform(() => change(true)) })}>启用协议</Button>
          <Button danger loading={busy} disabled={busy || !status.enabled || reason.trim().length < 8}
            onClick={() => Modal.confirm({ title: "停用新的盲评资格？",
              content: "保留全部记录，旧领取不能继续提交；重新启用需要新的部署观察。",
              onOk: () => perform(() => change(false)) })}>停用协议（保留历史）</Button>
        </Space>
      </> : null}
    </Space>
  </Card>;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function metric(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "-";
}
