"use client";

import {
  Alert,
  Button,
  Card,
  Checkbox,
  Col,
  Form,
  Input,
  Modal,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";

import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  createProductionAcceptance,
  getAssuranceOverview,
  signProductionAcceptance,
  startSecurityExercise,
  type AttackScenario,
  type AssuranceOverview,
  type CreateProductionAcceptanceInput,
  type ProductionAcceptance,
} from "@/lib/api/client";

type ExerciseForm = {
  name: string;
  targetEnvironment: "STAGING" | "PRODUCTION";
  scenarioIds: AttackScenario[];
};

type AcceptanceForm = {
  releaseScope: string;
  securityExerciseId: string;
  modelReleaseId: string;
  stagingDynamicManifest: string;
  stagingAcceptanceAttestationEvidenceId: string;
  businessEvidenceRef: string;
  businessEvidenceDigest: string;
  rollbackEvidenceRef: string;
  rollbackEvidenceDigest: string;
};

type SignoffForm = {
  signoffRole: "BUSINESS" | "SECURITY" | "PLATFORM";
  evidenceRef: string;
  evidenceDigest: string;
};

const digestRule = /^sha256:[0-9a-f]{64}$/;
const referenceRule = /^[A-Za-z0-9][A-Za-z0-9._:/-]*$/;
const statusColors: Record<string, string> = {
  RUNNING: "blue",
  PASSED: "green",
  FAILED: "red",
  READY: "cyan",
  BLOCKED: "red",
  SIGNED_OFF: "green",
  STAGING_ACCEPTANCE_COMPLETE_PRODUCTION_REVIEW_REQUIRED: "green",
  VERIFIED: "green",
};

export function ProductionAssurancePanel() {
  const [overview, setOverview] = useState<AssuranceOverview>();
  const [legalActions, setLegalActions] = useState<string[]>([]);
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [exerciseOpen, setExerciseOpen] = useState(false);
  const [acceptanceOpen, setAcceptanceOpen] = useState(false);
  const [signoffTarget, setSignoffTarget] = useState<ProductionAcceptance>();
  const [submitting, setSubmitting] = useState(false);
  const [exerciseForm] = Form.useForm<ExerciseForm>();
  const [acceptanceForm] = Form.useForm<AcceptanceForm>();
  const [signoffForm] = Form.useForm<SignoffForm>();

  const load = useCallback(async () => {
    try {
      const result = await getAssuranceOverview();
      setOverview(result.overview);
      setLegalActions(result.legalActions);
      setRequestId(result.requestId);
      setError(undefined);
    } catch (cause) {
      setError(cause);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const signoffOptions = useMemo(() => {
    const options: Array<{ label: string; value: SignoffForm["signoffRole"] }> = [];
    if (legalActions.includes("SIGN_BUSINESS_ACCEPTANCE")) {
      options.push({ label: "业务负责人", value: "BUSINESS" });
    }
    if (legalActions.includes("SIGN_SECURITY_ACCEPTANCE")) {
      options.push({ label: "安全负责人", value: "SECURITY" });
    }
    if (legalActions.includes("SIGN_PLATFORM_ACCEPTANCE")) {
      options.push({ label: "平台负责人", value: "PLATFORM" });
    }
    return options;
  }, [legalActions]);

  const fullPassedExercises = useMemo(
    () => overview?.exercises.filter(
      (item) => item.status === "PASSED"
        && overview.required_scenarios.every((scenario) => item.scenario_ids.includes(scenario.scenario_id)),
    ) ?? [],
    [overview],
  );

  function openExercise() {
    if (!overview) return;
    exerciseForm.setFieldsValue({
      name: `full-red-team-${new Date().toISOString().slice(0, 10)}`,
      targetEnvironment: "STAGING",
      scenarioIds: overview.required_scenarios.map((item) => item.scenario_id as AttackScenario),
    });
    setExerciseOpen(true);
  }

  async function submitExercise() {
    const values = await exerciseForm.validateFields();
    setSubmitting(true);
    try {
      const result = await startSecurityExercise(
        {
          name: values.name,
          target_environment: values.targetEnvironment,
          scenario_ids: values.scenarioIds,
        },
        `security-exercise-${crypto.randomUUID()}`,
      );
      setRequestId(result.requestId);
      setExerciseOpen(false);
      await load();
    } catch (cause) {
      setError(cause);
    } finally {
      setSubmitting(false);
    }
  }

  async function submitAcceptance() {
    let values: AcceptanceForm;
    try {
      values = await acceptanceForm.validateFields();
    } catch {
      return;
    }
    const stagingManifest = JSON.parse(values.stagingDynamicManifest) as NonNullable<
      CreateProductionAcceptanceInput["staging_dynamic_manifest"]
    >;
    setSubmitting(true);
    try {
      const result = await createProductionAcceptance(
        {
          release_scope: values.releaseScope,
          target_environment: "PRODUCTION",
          security_exercise_id: values.securityExerciseId,
          model_release_id: values.modelReleaseId,
          staging_dynamic_manifest: stagingManifest,
          staging_acceptance_attestation_evidence_id:
            values.stagingAcceptanceAttestationEvidenceId,
          business_evidence_ref: values.businessEvidenceRef,
          business_evidence_digest: values.businessEvidenceDigest,
          rollback_evidence_ref: values.rollbackEvidenceRef,
          rollback_evidence_digest: values.rollbackEvidenceDigest,
        },
        `production-acceptance-${crypto.randomUUID()}`,
      );
      setRequestId(result.requestId);
      setAcceptanceOpen(false);
      await load();
    } catch (cause) {
      setError(cause);
    } finally {
      setSubmitting(false);
    }
  }

  function openSignoff(acceptance: ProductionAcceptance) {
    const signed = new Set(acceptance.signoffs.map((item) => item.signoff_role));
    const available = signoffOptions.find((item) => !signed.has(item.value));
    signoffForm.resetFields();
    if (available) signoffForm.setFieldValue("signoffRole", available.value);
    setSignoffTarget(acceptance);
  }

  async function submitSignoff() {
    if (!signoffTarget) return;
    const values = await signoffForm.validateFields();
    setSubmitting(true);
    try {
      const result = await signProductionAcceptance(
        signoffTarget.acceptance_id,
        signoffTarget.version,
        {
          signoff_role: values.signoffRole,
          evidence_ref: values.evidenceRef,
          evidence_digest: values.evidenceDigest,
        },
      );
      setRequestId(result.requestId);
      setSignoffTarget(undefined);
      await load();
    } catch (cause) {
      setError(cause);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card
      title="攻击演练与生产验收"
      extra={(
        <Space>
          {legalActions.includes("MANAGE_SECURITY_EXERCISE") ? (
            <Button onClick={openExercise}>发起攻击演练</Button>
          ) : null}
          {legalActions.includes("CREATE_PRODUCTION_ACCEPTANCE") ? (
            <Button type="primary" onClick={() => setAcceptanceOpen(true)}>创建生产验收</Button>
          ) : null}
        </Space>
      )}
    >
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <Alert
          showIcon
          type="info"
          message="攻击由隔离 Runner 执行，平台不允许人工直接填写 PASS"
          description="服务端固定攻击场景、预期结果和零越权副作用规则；失败自动生成 P0/P1 缺陷，生产验收必须同时满足 Canary、回滚、SLO、恢复门禁和三方签署。"
        />
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {!overview && !error ? <LoadingState label="正在读取安全演练与验收状态" /> : null}
        {overview ? (
          <>
            <Row gutter={[16, 16]}>
              <Col xs={24} md={8}>
                <Statistic title="必测攻击场景" value={overview.required_scenarios.length} />
              </Col>
              <Col xs={24} md={8}>
                <Statistic
                  title="开放 P0/P1 缺陷"
                  value={overview.open_defects.length}
                  valueStyle={{ color: overview.open_defects.length ? "#cf1322" : "#389e0d" }}
                />
              </Col>
              <Col xs={24} md={8}>
                <Statistic
                  title="已联合签署验收"
                  value={overview.acceptances.filter((item) => item.status === "SIGNED_OFF").length}
                />
              </Col>
            </Row>

            <Card size="small" title="攻击场景基线">
              <Space wrap>
                {overview.required_scenarios.map((item) => (
                  <Tag key={item.scenario_id} color="geekblue">
                    {item.title} · {item.expected_outcome}
                  </Tag>
                ))}
              </Space>
            </Card>

            <Card size="small" title="最近攻击演练">
              <Table
                rowKey="exercise_id"
                size="small"
                pagination={false}
                dataSource={overview.exercises}
                locale={{ emptyText: "尚未发起攻击演练" }}
                columns={[
                  {
                    title: "演练",
                    dataIndex: "name",
                    render: (value: string) => `${value} · 演练`,
                  },
                  { title: "环境", dataIndex: "target_environment", width: 120 },
                  {
                    title: "场景",
                    dataIndex: "scenario_ids",
                    width: 100,
                    render: (values: string[]) => values.length,
                  },
                  {
                    title: "状态",
                    dataIndex: "status",
                    width: 110,
                    render: (value: string) => <Tag color={statusColors[value]}>{value}</Tag>,
                  },
                  {
                    title: "阻断",
                    dataIndex: "blocker_codes",
                    render: (values: string[]) => values.length ? values.join("；") : "—",
                  },
                  {
                    title: "开始时间",
                    dataIndex: "started_at",
                    width: 190,
                    render: (value: string) => new Date(value).toLocaleString(),
                  },
                ]}
              />
            </Card>

            {overview.open_defects.length ? (
              <Alert
                showIcon
                type="error"
                message="存在阻断生产的安全缺陷"
                description={overview.open_defects
                  .map((item) => `${item.severity} ${item.scenario_id}`)
                  .join("；")}
              />
            ) : null}

            <Card
              size="small"
              title={(
                <Space>
                  <span>生产验收档案</span>
                  <Typography.Text strong>Staging 动态验收门禁</Typography.Text>
                  <Typography.Text strong>签名 Staging 证据</Typography.Text>
                </Space>
              )}
            >
              <Table
                rowKey="acceptance_id"
                size="small"
                pagination={false}
                dataSource={overview.acceptances}
                locale={{ emptyText: "尚未创建生产验收档案" }}
                scroll={{ x: 1600 }}
                columns={[
                  { title: "范围", dataIndex: "release_scope", width: 200 },
                  { title: "模型发布", dataIndex: "model_release_id", width: 210 },
                  {
                    title: "动态清单",
                    width: 360,
                    render: (_, item: ProductionAcceptance) => (
                      <Space orientation="vertical" size={2}>
                        <Tag color={statusColors[item.staging_dynamic_manifest_status ?? ""] ?? "default"}>
                          {item.staging_dynamic_manifest_status ?? "未提交"}
                        </Tag>
                        <Typography.Text code>
                          {item.staging_dynamic_manifest_id ?? "—"}
                        </Typography.Text>
                        <Typography.Text type="secondary">
                          {item.staging_dynamic_manifest_generated_at
                            ? new Date(item.staging_dynamic_manifest_generated_at).toLocaleString()
                            : "无服务端生成时间"}
                        </Typography.Text>
                      </Space>
                    ),
                  },
                  {
                    title: "证据绑定",
                    width: 520,
                    render: (_, item: ProductionAcceptance) => (
                      <Space orientation="vertical" size={2}>
                        <Tag color={statusColors[item.staging_acceptance_attestation_status ?? ""] ?? "default"}>
                          {item.staging_acceptance_attestation_status ?? "未验证"}
                        </Tag>
                        <Typography.Text code>
                          {item.staging_acceptance_attestation_evidence_id ?? "—"}
                        </Typography.Text>
                        <Typography.Text type="secondary">
                          {item.staging_acceptance_attestation_certificate_identity
                            ?? "无服务端签名身份"}
                        </Typography.Text>
                        <Typography.Text type="secondary">
                          {item.staging_acceptance_attestation_certificate_oidc_issuer
                            ?? "无服务端 OIDC Issuer"}
                        </Typography.Text>
                      </Space>
                    ),
                  },
                  {
                    title: "状态",
                    dataIndex: "status",
                    width: 120,
                    render: (value: string) => <Tag color={statusColors[value]}>{value}</Tag>,
                  },
                  {
                    title: "签署",
                    dataIndex: "signoffs",
                    width: 210,
                    render: (values: ProductionAcceptance["signoffs"]) => values.length
                      ? values.map((item) => item.signoff_role).join("、")
                      : "待业务、安全、平台签署",
                  },
                  {
                    title: "阻断项",
                    dataIndex: "blocker_codes",
                    render: (values: string[]) => values.length ? values.join("；") : "—",
                  },
                  {
                    title: "操作",
                    width: 90,
                    render: (_, item: ProductionAcceptance) => {
                      const signed = new Set(item.signoffs.map((value) => value.signoff_role));
                      const canSign = item.status !== "SIGNED_OFF"
                        && signoffOptions.some((option) => !signed.has(option.value));
                      return canSign ? <Button type="link" onClick={() => openSignoff(item)}>签署</Button> : "—";
                    },
                  },
                ]}
              />
            </Card>
            <Typography.Text type="secondary">
              策略 {overview.policy_version} · 请求 ID {requestId}
            </Typography.Text>
          </>
        ) : null}
      </Space>

      <Modal
        title="发起隔离攻击演练"
        open={exerciseOpen}
        okText="创建演练"
        cancelText="取消"
        confirmLoading={submitting}
        onOk={() => void submitExercise()}
        onCancel={() => setExerciseOpen(false)}
        width={760}
      >
        <Form form={exerciseForm} layout="vertical">
          <Form.Item
            name="name"
            label="演练标识"
            rules={[{ required: true }, { pattern: referenceRule }]}
          >
            <Input />
          </Form.Item>
          <Form.Item name="targetEnvironment" label="目标环境" rules={[{ required: true }]}>
            <Select options={[
              { label: "Staging", value: "STAGING" },
              { label: "Production（仅已批准隔离目标）", value: "PRODUCTION" },
            ]} />
          </Form.Item>
          <Form.Item name="scenarioIds" label="攻击场景" rules={[{ required: true }]}>
            <Checkbox.Group
              options={overview?.required_scenarios.map((item) => ({
                label: item.title,
                value: item.scenario_id,
              }))}
            />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="创建生产验收档案"
        open={acceptanceOpen}
        okText="服务端评估"
        cancelText="取消"
        confirmLoading={submitting}
        onOk={() => void submitAcceptance()}
        onCancel={() => setAcceptanceOpen(false)}
        width={720}
      >
        <Alert
          showIcon
          type="warning"
          message="创建档案不等于通过验收"
          description="先在受控 Runner 中用可信 CLI 验证并登记签名清单，再提交返回的证据 ID。服务端会绑定签名身份，并继续核验清单时效、攻击演练、Canary、回滚、SLO 与恢复门禁；浏览器不自行验证 Bundle。"
          style={{ marginBottom: 16 }}
        />
        <Form form={acceptanceForm} layout="vertical">
          <Form.Item name="releaseScope" label="首个生产范围" rules={[{ required: true }, { pattern: referenceRule }]}>
            <Input placeholder="first-production-scope" />
          </Form.Item>
          <Form.Item name="securityExerciseId" label="完整攻击演练" rules={[{ required: true }]}>
            <Select options={fullPassedExercises.map((item) => ({ label: item.name, value: item.exercise_id }))} />
          </Form.Item>
          <Form.Item name="modelReleaseId" label="生产模型 Release ID" rules={[{ required: true }, { pattern: referenceRule }]}>
            <Input />
          </Form.Item>
          <Form.Item
            name="stagingDynamicManifest"
            label="Staging 动态验收清单 JSON"
            rules={[
              { required: true, message: "请粘贴 aggregate 产出的动态验收清单" },
              {
                validator: async (_, value: string | undefined) => {
                  if (!value) return;
                  try {
                    JSON.parse(value);
                  } catch {
                    throw new Error("请输入合法 JSON");
                  }
                },
              },
            ]}
          >
            <Input.TextArea
              rows={8}
              placeholder='{"schema_version":1,"manifest_id":"sha256:..."}'
            />
          </Form.Item>
          <Form.Item
            name="stagingAcceptanceAttestationEvidenceId"
            label="Staging 签名证据 ID"
            rules={[
              { required: true, message: "请输入可信验证 CLI 返回的 evidence_id" },
              { pattern: referenceRule },
            ]}
          >
            <Input placeholder="staging-attestation-..." />
          </Form.Item>
          <EvidenceFields prefix="business" label="真实业务端到端验收" />
          <EvidenceFields prefix="rollback" label="Manifest 回滚演练" />
        </Form>
      </Modal>

      <Modal
        title="签署生产验收"
        open={Boolean(signoffTarget)}
        okText="绑定证据并签署"
        cancelText="取消"
        confirmLoading={submitting}
        onOk={() => void submitSignoff()}
        onCancel={() => setSignoffTarget(undefined)}
      >
        <Alert
          showIcon
          type="info"
          message="签署前服务端会重新核验就绪快照"
          description="如果 SLO、恢复、缺陷或发布状态发生变化，本次不会写入签署，而会把档案重新标记为 BLOCKED。"
          style={{ marginBottom: 16 }}
        />
        <Form form={signoffForm} layout="vertical">
          <Form.Item name="signoffRole" label="签署职责" rules={[{ required: true }]}>
            <Select options={signoffOptions} />
          </Form.Item>
          <Form.Item name="evidenceRef" label="签署记录引用" rules={[{ required: true }, { pattern: referenceRule }]}>
            <Input placeholder="ticket:acceptance-signoff" />
          </Form.Item>
          <Form.Item name="evidenceDigest" label="签署记录 SHA-256" rules={[{ required: true }, { pattern: digestRule }]}>
            <Input placeholder="sha256:..." />
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
}

function EvidenceFields({ prefix, label }: { prefix: "business" | "rollback"; label: string }) {
  const referenceName = `${prefix}EvidenceRef` as const;
  const digestName = `${prefix}EvidenceDigest` as const;
  return (
    <Row gutter={12}>
      <Col span={12}>
        <Form.Item name={referenceName} label={`${label}引用`} rules={[{ required: true }, { pattern: referenceRule }]}>
          <Input placeholder="report:..." />
        </Form.Item>
      </Col>
      <Col span={12}>
        <Form.Item name={digestName} label={`${label} SHA-256`} rules={[{ required: true }, { pattern: digestRule }]}>
          <Input placeholder="sha256:..." />
        </Form.Item>
      </Col>
    </Row>
  );
}
