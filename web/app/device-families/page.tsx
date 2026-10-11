"use client";

import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Drawer,
  Form,
  Input,
  InputNumber,
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
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  type CreateDeviceFamilyInput,
  type DeviceFamilyProfile,
  createDeviceFamily,
  getDeviceFamily,
  listDeviceFamilies,
  revalidateDeviceFamily,
  reviewDeviceFamily,
} from "@/lib/api/client";

type CreateValues = CreateDeviceFamilyInput;
type ReviewDecision = "ACCEPT" | "REJECT";

const statusColor: Record<string, string> = {
  BLOCKED: "volcano",
  READY: "gold",
  ACTIVE: "green",
  REJECTED: "red",
  RETIRED: "default",
};

const blockerLabels: Record<string, string> = {
  knowledge_release_not_active: "知识发布不是当前活动版本",
  published_family_knowledge_missing: "缺少已发布且适用于该设备族/型号的知识",
  gold_evaluation_suite_not_frozen: "Gold Evaluation Suite 未冻结",
  device_family_gold_slice_insufficient: "设备族 Gold 样本不足",
  device_family_evaluation_not_passed: "绑定评测未通过全部硬门禁",
  authoritative_asset_model_missing: "EAM 中没有匹配型号的设备实例",
  authoritative_model_mapping_stale: "型号权威来源已过期或时间异常",
};

export default function DeviceFamiliesPage() {
  const [profiles, setProfiles] = useState<DeviceFamilyProfile[]>();
  const [selected, setSelected] = useState<DeviceFamilyProfile>();
  const [statusFilter, setStatusFilter] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [commandError, setCommandError] = useState<unknown>();
  const [busy, setBusy] = useState<string>();
  const [createOpen, setCreateOpen] = useState(false);
  const [reviewDecision, setReviewDecision] = useState<ReviewDecision>();
  const [reviewReason, setReviewReason] = useState("");
  const [form] = Form.useForm<CreateValues>();

  const load = useCallback(async () => {
    try {
      const result = await listDeviceFamilies(statusFilter);
      setProfiles(result.profiles);
      setSelected((current) =>
        current
          ? result.profiles.find((item) => item.profile_id === current.profile_id) ?? current
          : current,
      );
      setError(undefined);
    } catch (cause) {
      setError(cause);
    }
  }, [statusFilter]);

  useEffect(() => {
    void load();
  }, [load]);

  const stats = useMemo(
    () => ({
      total: profiles?.length ?? 0,
      active: profiles?.filter((item) => item.status === "ACTIVE").length ?? 0,
      ready: profiles?.filter((item) => item.status === "READY").length ?? 0,
      blocked: profiles?.filter((item) => item.status === "BLOCKED").length ?? 0,
    }),
    [profiles],
  );

  function replace(item: DeviceFamilyProfile) {
    setSelected(item);
    setProfiles((current) => [
      item,
      ...(current ?? []).filter((value) => value.profile_id !== item.profile_id),
    ]);
  }

  async function submitCreate(values: CreateValues) {
    setBusy("create");
    setCommandError(undefined);
    try {
      const item = await createDeviceFamily(values, crypto.randomUUID());
      replace(item);
      setCreateOpen(false);
      form.resetFields();
      await load();
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function revalidate(item: DeviceFamilyProfile) {
    setBusy(`revalidate-${item.profile_id}`);
    setCommandError(undefined);
    try {
      replace(await revalidateDeviceFamily(item.profile_id, item.version));
      await load();
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
        await reviewDeviceFamily(selected.profile_id, selected.version, {
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

  const blockers = selected ? readinessBlockers(selected) : [];

  return (
    <AppShell>
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <div>
          <Typography.Title level={2}>设备族准入与诊断路由</Typography.Title>
          <Typography.Paragraph type="secondary">
            将 EAM 型号映射、已发布知识、冻结 Gold 切片和独立评测绑定为一个版本；不同人员复核激活后，诊断才使用新设备族路由。
          </Typography.Paragraph>
        </div>

        <Alert
          showIcon
          type="warning"
          message="激活代表允许诊断检索，不代表设备控制授权"
          description="本功能不会创建设备、工单、采购、库存、派工或控制动作。知识发布改变后，既有设备族绑定会在诊断入口失败关闭，必须建立新版本。"
        />

        <Row gutter={16}>
          <Col span={6}><Card><Statistic title="配置版本" value={stats.total} /></Card></Col>
          <Col span={6}><Card><Statistic title="已激活" value={stats.active} /></Card></Col>
          <Col span={6}><Card><Statistic title="待复核" value={stats.ready} /></Card></Col>
          <Col span={6}><Card><Statistic title="证据阻断" value={stats.blocked} /></Card></Col>
        </Row>

        {commandError ? (
          <ErrorState error={commandError} onRetry={() => setCommandError(undefined)} />
        ) : null}

        <Card
          title="设备族版本"
          extra={
            <Space>
              <Select
                allowClear
                placeholder="按状态过滤"
                value={statusFilter}
                onChange={setStatusFilter}
                options={["BLOCKED", "READY", "ACTIVE", "REJECTED", "RETIRED"].map(
                  (value) => ({ value, label: value }),
                )}
                style={{ width: 180 }}
              />
              <Button onClick={() => void load()}>刷新</Button>
              <Button
                type="primary"
                onClick={() => {
                  form.setFieldsValue({
                    risk_class: "HIGH",
                    minimum_gold_samples: 30,
                    source_as_of: new Date().toISOString(),
                  });
                  setCreateOpen(true);
                }}
              >
                新建设备族版本
              </Button>
            </Space>
          }
        >
          {error ? (
            <ErrorState error={error} onRetry={() => void load()} />
          ) : profiles === undefined ? (
            <LoadingState />
          ) : profiles.length === 0 ? (
            <EmptyState description="暂无设备族准入记录。请从一组真实 EAM 型号和对应 Gold 评测开始。" />
          ) : (
            <Table
              rowKey="profile_id"
              dataSource={profiles}
              pagination={{ pageSize: 10 }}
              columns={[
                {
                  title: "设备族",
                  render: (_, item) => (
                    <Space direction="vertical" size={0}>
                      <Typography.Text strong>{item.display_name}</Typography.Text>
                      <Typography.Text type="secondary">
                        {item.family_code} · v{item.family_version}
                      </Typography.Text>
                    </Space>
                  ),
                },
                { title: "状态", render: (_, item) => <Tag color={statusColor[item.status]}>{item.status}</Tag> },
                { title: "风险", dataIndex: "risk_class" },
                { title: "型号数", render: (_, item) => item.model_codes.length },
                { title: "阻断项", render: (_, item) => readinessBlockers(item).length },
                {
                  title: "操作",
                  render: (_, item) => (
                    <Space>
                      <Button size="small" onClick={() => setSelected(item)}>查看</Button>
                      {item.legal_actions.includes("REVALIDATE") ? (
                        <Button
                          size="small"
                          loading={busy === `revalidate-${item.profile_id}`}
                          onClick={() => void revalidate(item)}
                        >
                          重新核验
                        </Button>
                      ) : null}
                    </Space>
                  ),
                },
              ]}
            />
          )}
        </Card>
      </Space>

      <Modal
        open={createOpen}
        title="新建设备族准入版本"
        okText="核验证据并创建"
        cancelText="取消"
        confirmLoading={busy === "create"}
        onCancel={() => setCreateOpen(false)}
        onOk={() => void form.submit()}
        width={760}
      >
        <Form form={form} layout="vertical" onFinish={(values) => void submitCreate(values)}>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="family_code" label="稳定设备族代码" rules={[{ required: true }]}>
                <Input placeholder="compressor" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="display_name" label="显示名称" rules={[{ required: true, min: 2 }]}>
                <Input placeholder="工业压缩机" />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="risk_class" label="风险等级" rules={[{ required: true }]}>
                <Select options={["LOW", "MEDIUM", "HIGH", "CRITICAL"].map((value) => ({ value }))} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="minimum_gold_samples" label="最低设备族 Gold 样本数" rules={[{ required: true }]}>
                <InputNumber min={1} max={100_000} style={{ width: "100%" }} />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item name="model_codes" label="EAM 型号代码" rules={[{ required: true }]}>
            <Select mode="tags" tokenSeparators={[","]} placeholder="COMP-X200, COMP-X300" />
          </Form.Item>
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="source_system" label="型号权威系统" rules={[{ required: true }]}>
                <Input placeholder="enterprise-eam" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="source_record_id" label="来源记录 ID" rules={[{ required: true }]}>
                <Input placeholder="family-COMP-2026" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="source_as_of" label="来源时间（ISO 8601）" rules={[{ required: true }]}>
                <Input />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item name="knowledge_release_id" label="当前活动知识 Release ID" rules={[{ required: true }]}>
            <Input placeholder="index-release-..." />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="evaluation_suite_id" label="冻结 Gold Suite ID" rules={[{ required: true }]}>
                <Input placeholder="eval-suite-..." />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="evaluation_id" label="通过的评测 ID" rules={[{ required: true }]}>
                <Input placeholder="evaluation-..." />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>

      <Drawer
        width={860}
        open={Boolean(selected)}
        title="设备族准入证据"
        onClose={() => setSelected(undefined)}
      >
        {selected ? (
          <Space direction="vertical" size="large" style={{ width: "100%" }}>
            <Descriptions bordered size="small" column={2}>
              <Descriptions.Item label="状态"><Tag color={statusColor[selected.status]}>{selected.status}</Tag></Descriptions.Item>
              <Descriptions.Item label="记录版本">{selected.version}</Descriptions.Item>
              <Descriptions.Item label="设备族">{selected.family_code} / v{selected.family_version}</Descriptions.Item>
              <Descriptions.Item label="风险">{selected.risk_class}</Descriptions.Item>
              <Descriptions.Item label="型号" span={2}>{selected.model_codes.join(", ")}</Descriptions.Item>
              <Descriptions.Item label="知识 Release" span={2}><Typography.Text copyable>{selected.knowledge_release_id}</Typography.Text></Descriptions.Item>
              <Descriptions.Item label="Gold Suite"><Typography.Text copyable>{selected.evaluation_suite_id}</Typography.Text></Descriptions.Item>
              <Descriptions.Item label="评测"><Typography.Text copyable>{selected.evaluation_id}</Typography.Text></Descriptions.Item>
              <Descriptions.Item label="来源">{selected.source_system} / {selected.source_record_id}</Descriptions.Item>
              <Descriptions.Item label="来源时间">{selected.source_as_of}</Descriptions.Item>
              <Descriptions.Item label="证据摘要" span={2}><Typography.Text copyable>{selected.evidence_digest}</Typography.Text></Descriptions.Item>
            </Descriptions>

            <Card title={`证据阻断项（${blockers.length}）`}>
              {blockers.length ? (
                <List
                  dataSource={blockers}
                  renderItem={(item) => (
                    <List.Item><Tag color="red">BLOCKED</Tag>{blockerLabels[item] ?? item}</List.Item>
                  )}
                />
              ) : (
                <Alert type="success" showIcon message="全部自动门禁通过，可由不同主体复核。" />
              )}
            </Card>

            <Card title="冻结就绪证据">
              <Typography.Paragraph>
                <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>
                  {JSON.stringify(selected.readiness, null, 2)}
                </pre>
              </Typography.Paragraph>
            </Card>

            {selected.review_reason ? (
              <Alert
                type={selected.status === "ACTIVE" ? "success" : "error"}
                message={`${selected.review_decision} · ${selected.reviewed_by_subject_id}`}
                description={selected.review_reason}
              />
            ) : null}

            {selected.legal_actions.includes("ACCEPT") || selected.legal_actions.includes("REJECT") ? (
              <Space>
                {selected.legal_actions.includes("ACCEPT") ? (
                  <Button type="primary" onClick={() => setReviewDecision("ACCEPT")}>
                    激活诊断路由
                  </Button>
                ) : null}
                {selected.legal_actions.includes("REJECT") ? (
                  <Button danger onClick={() => setReviewDecision("REJECT")}>驳回版本</Button>
                ) : null}
              </Space>
            ) : null}
          </Space>
        ) : null}
      </Drawer>

      <Modal
        open={Boolean(reviewDecision)}
        title={reviewDecision === "ACCEPT" ? "复核并激活设备族" : "驳回设备族版本"}
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
          message="复核人与申请人必须不同；接受前服务端会重新读取全部证据。"
          style={{ marginBottom: 16 }}
        />
        <Input.TextArea
          rows={5}
          maxLength={2_000}
          showCount
          value={reviewReason}
          onChange={(event) => setReviewReason(event.target.value)}
          placeholder="填写至少 8 个字符的独立复核理由"
        />
      </Modal>
    </AppShell>
  );
}

function readinessBlockers(profile: DeviceFamilyProfile): string[] {
  const value = profile.readiness.blockers;
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}
