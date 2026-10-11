"use client";

import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Form,
  Input,
  InputNumber,
  List,
  Modal,
  Row,
  Select,
  Space,
  Statistic,
  Switch,
  Table,
  Tag,
  Typography,
  message,
} from "antd";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ErrorState, LoadingState } from "@/components/RequestState";
import {
  concludeExternalSearch,
  getExternalSearchPolicy,
  listExternalSearches,
  runExternalSearch,
  updateExternalSearchPolicy,
  type ExternalSearchConclusionInput,
  type ExternalSearchInput,
  type ExternalSearchPolicy,
  type ExternalSearchQuery,
} from "@/lib/api/client";

type PolicyFormValues = {
  enabled: boolean;
  allowed_domains: string;
  official_domains: string;
  max_results: number;
};

const USE_CASES: Array<{ value: ExternalSearchInput["use_case"]; label: string }> = [
  { value: "GENERAL_REFERENCE", label: "一般公开参考（允许域名）" },
  { value: "REPAIR_REFERENCE", label: "维修参考（仅官方域名）" },
  { value: "SAFETY_REFERENCE", label: "安全参考（仅官方域名）" },
  { value: "WARRANTY_REFERENCE", label: "质保参考（仅官方域名）" },
];

const CONCLUSIONS: Array<{
  value: ExternalSearchConclusionInput["conclusion"];
  label: string;
}> = [
  { value: "NOT_USED", label: "未采用" },
  { value: "REFERENCE_ONLY", label: "仅作外部线索" },
  { value: "ESCALATED_TO_KNOWLEDGE_REVIEW", label: "转交企业知识审核" },
];

const STATUS_COLORS: Record<string, string> = {
  RUNNING: "processing",
  SUCCEEDED: "green",
  NO_RESULTS: "default",
  BLOCKED: "orange",
  FAILED: "red",
};

const STATUS_LABELS: Record<string, string> = {
  RUNNING: "执行中",
  SUCCEEDED: "已完成",
  NO_RESULTS: "无可用结果",
  BLOCKED: "已被安全策略阻断",
  FAILED: "执行失败",
};

const CONCLUSION_LABELS: Record<string, string> = {
  PENDING_REVIEW: "待登记使用结论",
  NOT_USED: "未采用",
  REFERENCE_ONLY: "仅作外部线索",
  ESCALATED_TO_KNOWLEDGE_REVIEW: "已转交知识审核",
};

export default function ExternalReferenceWorkspacePage() {
  const [policy, setPolicy] = useState<ExternalSearchPolicy>();
  const [searches, setSearches] = useState<ExternalSearchQuery[]>([]);
  const [requestId, setRequestId] = useState<string>();
  const [error, setError] = useState<unknown>();
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searching, setSearching] = useState(false);
  const [saving, setSaving] = useState(false);
  const [policyOpen, setPolicyOpen] = useState(false);
  const [conclusionTarget, setConclusionTarget] = useState<ExternalSearchQuery>();
  const [policyForm] = Form.useForm<PolicyFormValues>();
  const [searchForm] = Form.useForm<ExternalSearchInput>();
  const [conclusionForm] = Form.useForm<ExternalSearchConclusionInput>();
  const [messageApi, messageContext] = message.useMessage();

  const load = useCallback(async (refresh = false) => {
    if (refresh) setRefreshing(true);
    else setLoading(true);
    try {
      const [policyResult, historyResult] = await Promise.all([
        getExternalSearchPolicy(),
        listExternalSearches(),
      ]);
      setPolicy(policyResult);
      setSearches(historyResult.searches);
      setRequestId(historyResult.requestId);
      setError(undefined);
    } catch (cause) {
      setError(cause);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const metrics = useMemo(() => ({
    completed: searches.filter((item) => item.status === "SUCCEEDED").length,
    blocked: searches.reduce((total, item) => total + item.blocked_result_count, 0),
    pending: searches.filter((item) => item.usage_conclusion === "PENDING_REVIEW").length,
    escalated: searches.filter(
      (item) => item.usage_conclusion === "ESCALATED_TO_KNOWLEDGE_REVIEW",
    ).length,
  }), [searches]);

  const canSearch = policy?.legal_actions.includes("RUN_SEARCH") ?? false;
  const searchReady = Boolean(canSearch && policy?.enabled && policy.runtime_available);

  function openPolicy() {
    if (!policy) return;
    policyForm.setFieldsValue({
      enabled: policy.enabled,
      allowed_domains: policy.allowed_domains.join("\n"),
      official_domains: policy.official_domains.join("\n"),
      max_results: policy.max_results,
    });
    setPolicyOpen(true);
  }

  async function savePolicy() {
    if (!policy) return;
    try {
      const values = await policyForm.validateFields();
      setSaving(true);
      const changed = await updateExternalSearchPolicy(
        {
          enabled: values.enabled,
          allowed_domains: parseDomains(values.allowed_domains),
          official_domains: parseDomains(values.official_domains),
          max_results: values.max_results,
        },
        policy.version,
      );
      setPolicy(changed);
      setPolicyOpen(false);
      messageApi.success("租户外部参考策略已更新");
    } catch (cause) {
      if (isFormError(cause)) return;
      messageApi.error(errorMessage(cause, "策略更新失败"));
    } finally {
      setSaving(false);
    }
  }

  async function executeSearch(values: ExternalSearchInput) {
    if (!searchReady) return;
    setSearching(true);
    try {
      const created = await runExternalSearch(
        values,
        `external-search-${crypto.randomUUID()}`,
      );
      setSearches((current) => [
        created,
        ...current.filter((item) => item.external_search_id !== created.external_search_id),
      ]);
      messageApi.success(searchOutcomeMessage(created));
    } catch (cause) {
      messageApi.error(errorMessage(cause, "外部参考查询失败"));
    } finally {
      setSearching(false);
    }
  }

  function openConclusion(item: ExternalSearchQuery) {
    conclusionForm.setFieldsValue({
      conclusion: "REFERENCE_ONLY",
      reason: "该结果仅作为外部线索保留，不直接进入企业知识、诊断或训练数据。",
    });
    setConclusionTarget(item);
  }

  async function saveConclusion() {
    if (!conclusionTarget) return;
    try {
      const values = await conclusionForm.validateFields();
      setSaving(true);
      const changed = await concludeExternalSearch(
        conclusionTarget.external_search_id,
        conclusionTarget.version,
        values,
      );
      setSearches((current) => current.map((item) =>
        item.external_search_id === changed.external_search_id ? changed : item));
      setConclusionTarget(undefined);
      messageApi.success("外部参考使用结论已登记");
    } catch (cause) {
      if (isFormError(cause)) return;
      messageApi.error(errorMessage(cause, "使用结论登记失败"));
    } finally {
      setSaving(false);
    }
  }

  return (
    <AppShell>
      {messageContext}
      <div className="page-stack">
        <Space align="start" style={{ justifyContent: "space-between", width: "100%" }}>
          <div>
            <Typography.Title level={2}>受控外部参考</Typography.Title>
            <Typography.Paragraph type="secondary">
              在企业知识不足时查询租户允许的公开来源，并完整登记来源、拦截结果和使用结论。
            </Typography.Paragraph>
          </div>
          <Space>
            <Link href="/ai/knowledge">返回企业知识</Link>
            <Button loading={refreshing} onClick={() => void load(true)}>刷新</Button>
          </Space>
        </Space>

        <Alert
          showIcon
          type="warning"
          message="外部网页始终是不可信参考"
          description="OFFICIAL 只表示域名命中租户官方来源清单，不代表内容已被企业审核。查询结果不会自动进入诊断 Prompt、企业知识、维修指令、数据集或训练语料；需要复用时必须重新走知识入库和独立审核。"
        />

        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {loading && !error ? <LoadingState label="正在读取外部参考策略和审计历史" /> : null}

        {!loading && !error && policy ? (
          <>
            {!policy.enabled ? (
              <Alert showIcon type="info" message="租户策略当前关闭外部搜索" />
            ) : null}
            {policy.enabled && !policy.runtime_available ? (
              <Alert
                showIcon
                type="warning"
                message="租户已允许，但外部搜索 Runtime 未配置"
                description="平台不会发起外部网络调用。需要由运维通过 Secret Provider 配置并启用受控 Tavily Runtime。"
              />
            ) : null}
            {!canSearch ? (
              <Alert
                showIcon
                type="info"
                message="当前身份只有策略和历史审计权限"
                description="服务端未授予 RUN_SEARCH，本页面不会提供搜索执行入口。"
              />
            ) : null}

            <Card
              title="租户来源策略"
              extra={policy.legal_actions.includes("UPDATE_POLICY") ? (
                <Button onClick={openPolicy}>编辑租户策略</Button>
              ) : null}
            >
              <Descriptions column={{ xs: 1, md: 2 }} size="small">
                <Descriptions.Item label="租户开关">
                  <Tag color={policy.enabled ? "green" : "default"}>
                    {policy.enabled ? "已启用" : "已关闭"}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="Runtime">
                  <Tag color={policy.runtime_available ? "green" : "orange"}>
                    {policy.runtime_available ? "已配置" : "不可用"}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="每次结果上限">{policy.max_results}</Descriptions.Item>
                <Descriptions.Item label="策略版本">v{policy.version}</Descriptions.Item>
                <Descriptions.Item label="允许域名" span={2}>
                  <DomainTags domains={policy.allowed_domains} empty="尚未配置允许域名" />
                </Descriptions.Item>
                <Descriptions.Item label="官方域名" span={2}>
                  <DomainTags domains={policy.official_domains} empty="尚未配置官方域名" color="blue" />
                </Descriptions.Item>
                <Descriptions.Item label="最近修改人">{policy.updated_by_subject_id ?? "—"}</Descriptions.Item>
                <Descriptions.Item label="最近更新时间">{formatTimestamp(policy.updated_at)}</Descriptions.Item>
              </Descriptions>
            </Card>

            <Row gutter={[16, 16]}>
              <Col xs={24} sm={12} xl={6}><Card><Statistic title="成功查询" value={metrics.completed} /></Card></Col>
              <Col xs={24} sm={12} xl={6}><Card><Statistic title="被拦截来源" value={metrics.blocked} /></Card></Col>
              <Col xs={24} sm={12} xl={6}><Card><Statistic title="待登记结论" value={metrics.pending} /></Card></Col>
              <Col xs={24} sm={12} xl={6}><Card><Statistic title="转知识审核" value={metrics.escalated} /></Card></Col>
            </Row>

            {canSearch ? (
              <Card title="执行受控搜索">
                <Form<ExternalSearchInput>
                  form={searchForm}
                  layout="vertical"
                  initialValues={{ use_case: "GENERAL_REFERENCE" }}
                  onFinish={(values) => void executeSearch(values)}
                >
                  <Row gutter={16}>
                    <Col xs={24} md={8}>
                      <Form.Item name="use_case" label="业务用途" rules={[{ required: true }]}>
                        <Select options={USE_CASES} />
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={16}>
                      <Form.Item
                        name="query_text"
                        label="公开资料查询"
                        rules={[
                          { required: true, message: "请输入查询内容" },
                          { min: 3, max: 500 },
                        ]}
                      >
                        <Input placeholder="例如：PUMP-X100 E77 官方安全停机公告" />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Space wrap>
                    <Button type="primary" htmlType="submit" loading={searching} disabled={!searchReady}>
                      查询允许来源
                    </Button>
                    <Typography.Text type="secondary">
                      维修、安全和质保用途只查询策略中的官方域名。
                    </Typography.Text>
                  </Space>
                </Form>
              </Card>
            ) : null}

            <Card title="查询历史与使用审计">
              <Table<ExternalSearchQuery>
                rowKey="external_search_id"
                dataSource={searches}
                scroll={{ x: 1180 }}
                pagination={{ pageSize: 20, showSizeChanger: true }}
                locale={{ emptyText: "尚无外部参考查询记录" }}
                expandable={{ expandedRowRender: (item) => <SearchDetail item={item} /> }}
                columns={[
                  {
                    title: "状态",
                    dataIndex: "status",
                    width: 170,
                    render: (value: string) => (
                      <Tag color={STATUS_COLORS[value]}>{STATUS_LABELS[value] ?? value}</Tag>
                    ),
                  },
                  {
                    title: "查询与用途",
                    width: 330,
                    render: (_, item) => (
                      <Space direction="vertical" size={0}>
                        <Typography.Text>{item.query_text}</Typography.Text>
                        <Typography.Text type="secondary">{useCaseLabel(item.use_case)}</Typography.Text>
                      </Space>
                    ),
                  },
                  { title: "可用结果", dataIndex: "result_count", width: 100 },
                  { title: "拦截", dataIndex: "blocked_result_count", width: 80 },
                  {
                    title: "使用结论",
                    dataIndex: "usage_conclusion",
                    width: 190,
                    render: (value: string) => CONCLUSION_LABELS[value] ?? value,
                  },
                  {
                    title: "完成时间",
                    dataIndex: "completed_at",
                    width: 190,
                    render: formatTimestamp,
                  },
                  {
                    title: "操作",
                    fixed: "right",
                    width: 130,
                    render: (_, item) => item.legal_actions.includes("CONCLUDE") ? (
                      <Button size="small" onClick={() => openConclusion(item)}>登记结论</Button>
                    ) : "—",
                  },
                ]}
              />
            </Card>

            {requestId ? <Typography.Text className="request-id">request_id: {requestId}</Typography.Text> : null}
          </>
        ) : null}
      </div>

      <Modal
        title="编辑租户外部参考策略"
        open={policyOpen}
        okText="保存策略"
        cancelText="取消"
        confirmLoading={saving}
        onOk={() => void savePolicy()}
        onCancel={() => setPolicyOpen(false)}
      >
        <Alert
          showIcon
          type="info"
          style={{ marginBottom: 16 }}
          message="仅填写公共 DNS 域名"
          description="每行或逗号分隔；不接受 URL、通配符、IP、端口、路径或单标签内网主机。官方域名必须同时包含在允许域名中。"
        />
        <Form<PolicyFormValues> form={policyForm} layout="vertical">
          <Form.Item name="enabled" label="启用租户外部参考" valuePropName="checked">
            <Switch />
          </Form.Item>
          <Form.Item
            name="allowed_domains"
            label="允许域名"
            rules={[{ required: true, message: "启用策略时必须配置允许域名" }]}
          >
            <Input.TextArea rows={4} placeholder="vendor.example.com\nindustry.example.org" />
          </Form.Item>
          <Form.Item name="official_domains" label="官方域名">
            <Input.TextArea rows={3} placeholder="vendor.example.com" />
          </Form.Item>
          <Form.Item name="max_results" label="单次最大结果数" rules={[{ required: true }]}>
            <InputNumber min={1} max={10} style={{ width: "100%" }} />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="登记外部参考使用结论"
        open={Boolean(conclusionTarget)}
        okText="保存结论"
        cancelText="取消"
        confirmLoading={saving}
        onOk={() => void saveConclusion()}
        onCancel={() => setConclusionTarget(undefined)}
      >
        <Form<ExternalSearchConclusionInput> form={conclusionForm} layout="vertical">
          <Form.Item name="conclusion" label="使用结论" rules={[{ required: true }]}>
            <Select options={CONCLUSIONS} />
          </Form.Item>
          <Form.Item
            name="reason"
            label="结论依据"
            rules={[
              { required: true, message: "请记录结论依据" },
              { min: 3, max: 1000 },
            ]}
          >
            <Input.TextArea rows={4} />
          </Form.Item>
        </Form>
      </Modal>
    </AppShell>
  );
}

function SearchDetail({ item }: { item: ExternalSearchQuery }) {
  return (
    <div className="page-stack" style={{ gap: 12 }}>
      <Descriptions column={{ xs: 1, md: 2 }} size="small">
        <Descriptions.Item label="查询 ID">{item.external_search_id}</Descriptions.Item>
        <Descriptions.Item label="查询版本">v{item.version}</Descriptions.Item>
        <Descriptions.Item label="Provider">{item.provider}</Descriptions.Item>
        <Descriptions.Item label="Provider Request">{item.provider_request_id ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="执行人">{item.requested_by_subject_id}</Descriptions.Item>
        <Descriptions.Item label="用量 Credits">{item.usage_credits ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="失败原因">{item.failure_code ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="结论登记人">{item.concluded_by_subject_id ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="结论依据" span={2}>{item.conclusion_reason ?? "—"}</Descriptions.Item>
      </Descriptions>
      <List
        header={<Typography.Text strong>允许展示的外部参考摘要</Typography.Text>}
        dataSource={item.references}
        locale={{ emptyText: "没有通过来源与安全门禁的参考" }}
        renderItem={(reference) => (
          <List.Item>
            <List.Item.Meta
              title={(
                <Space wrap>
                  <Tag color={reference.trust_level === "OFFICIAL" ? "blue" : "default"}>
                    {reference.trust_level === "OFFICIAL" ? "官方域名" : "允许域名"}
                  </Tag>
                  <a href={reference.url} target="_blank" rel="noopener noreferrer nofollow">
                    {reference.title}
                  </a>
                </Space>
              )}
              description={(
                <Space direction="vertical" size={2}>
                  <Typography.Text>{reference.summary}</Typography.Text>
                  <Typography.Text type="secondary">
                    {reference.domain} · 相关度 {Math.round(reference.relevance_score * 100)}% · 抓取 {formatTimestamp(reference.fetched_at)}
                  </Typography.Text>
                  <Typography.Text type="secondary" code>{reference.content_hash}</Typography.Text>
                </Space>
              )}
            />
          </List.Item>
        )}
      />
    </div>
  );
}

function DomainTags({
  domains,
  empty,
  color,
}: {
  domains: string[];
  empty: string;
  color?: string;
}) {
  if (domains.length === 0) return <Typography.Text type="secondary">{empty}</Typography.Text>;
  return <Space wrap>{domains.map((domain) => <Tag key={domain} color={color}>{domain}</Tag>)}</Space>;
}

function parseDomains(value: string): string[] {
  return [...new Set(value.split(/[\s,]+/).map((item) => item.trim()).filter(Boolean))];
}

function useCaseLabel(value: string): string {
  return USE_CASES.find((item) => item.value === value)?.label ?? value;
}

function formatTimestamp(value: string | null | undefined): string {
  return value ? new Date(value).toLocaleString("zh-CN") : "—";
}

function searchOutcomeMessage(item: ExternalSearchQuery): string {
  if (item.status === "SUCCEEDED") return `查询完成，保留 ${item.result_count} 条受控参考`;
  if (item.status === "BLOCKED") return "查询已由安全策略阻断并记录审计";
  if (item.status === "NO_RESULTS") return "查询完成，但没有通过治理门禁的参考";
  return `查询已记录为${STATUS_LABELS[item.status] ?? item.status}`;
}

function isFormError(value: unknown): boolean {
  return Boolean(value && typeof value === "object" && "errorFields" in value);
}

function errorMessage(value: unknown, fallback: string): string {
  return value instanceof Error ? value.message : fallback;
}
