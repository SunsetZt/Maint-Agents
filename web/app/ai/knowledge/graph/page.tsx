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
  InputNumber,
  Modal,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from "antd";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import {
  type CreateKnowledgeGraphEdgeInput,
  type CreateKnowledgeGraphNodeInput,
  type KnowledgeGraphExtraction,
  type KnowledgeGraphPath,
  type KnowledgeGraphRelease,
  type KnowledgeRelease,
  type KnowledgeSearchProfile,
  type KnowledgeSearchResult,
  activateKnowledgeGraph,
  addKnowledgeGraphEdge,
  addKnowledgeGraphNode,
  createKnowledgeGraph,
  evaluateKnowledgeGraph,
  listKnowledgeGraphs,
  listKnowledgeGraphExtractions,
  listKnowledgeReleases,
  listKnowledgeSearchProfiles,
  queryKnowledgeGraph,
  queryKnowledgeSearchProfile,
  requestKnowledgeGraphExtraction,
  retryKnowledgeGraphExtraction,
  reviewKnowledgeGraphExtraction,
  syncKnowledgeGraph,
} from "@/lib/api/client";

const nodeTypes = [
  "COMPONENT",
  "FAILURE_MODE",
  "SYMPTOM",
  "CAUSE",
  "ACTION",
  "MATERIAL",
] as const;

const relationTypes = [
  "DEPENDS_ON",
  "CAUSES",
  "MANIFESTS_AS",
  "MITIGATED_BY",
  "COMPATIBLE_WITH",
  "INCOMPATIBLE_WITH",
  "PART_OF",
] as const;

type CreateValues = { source_index_release_id: string; name: string };
type NodeValues = Omit<CreateKnowledgeGraphNodeInput, "metadata"> & {
  metadata_json?: string;
};
type EdgeValues = Omit<CreateKnowledgeGraphEdgeInput, "metadata"> & {
  metadata_json?: string;
};
type EvaluationValues = { cases_json: string };
type ExtractionReviewDecision = "ACCEPT" | "REJECT";
type ExtractionCandidateBundle = {
  nodes: Array<{
    node_key: string;
    node_type: string;
    display_name: string;
    citation_id: string;
  }>;
  edges: Array<{
    source_node_key: string;
    target_node_key: string;
    relation_type: string;
    confidence: number;
    citation_id: string;
  }>;
};
type QueryValues = {
  graph_name: string;
  start_node_key: string;
  target_node_type?: (typeof nodeTypes)[number];
  relation_types: (typeof relationTypes)[number][];
  max_hops: number;
  limit: number;
};

const exampleCases = JSON.stringify(
  [
    {
      start_node_key: "pump.body",
      target_node_type: "FAILURE_MODE",
      relation_types: ["DEPENDS_ON", "CAUSES"],
      max_hops: 3,
      expected_target_node_key: "failure.overheat",
    },
    {
      start_node_key: "alarm.e77",
      target_node_type: "CAUSE",
      relation_types: ["MANIFESTS_AS", "CAUSES"],
      max_hops: 3,
      expected_target_node_key: "cause.low-pressure",
    },
    {
      start_node_key: "bearing.main",
      target_node_type: "ACTION",
      relation_types: ["CAUSES", "MITIGATED_BY"],
      max_hops: 4,
      expected_target_node_key: "action.replace-bearing",
    },
  ],
  null,
  2,
);

export default function KnowledgeGraphPage() {
  const [graphs, setGraphs] = useState<KnowledgeGraphRelease[]>();
  const [indexReleases, setIndexReleases] = useState<KnowledgeRelease[]>([]);
  const [searchProfiles, setSearchProfiles] = useState<KnowledgeSearchProfile[]>([]);
  const [extractions, setExtractions] = useState<KnowledgeGraphExtraction[]>([]);
  const [selected, setSelected] = useState<KnowledgeGraphRelease>();
  const [paths, setPaths] = useState<KnowledgeGraphPath[]>([]);
  const [error, setError] = useState<unknown>();
  const [commandError, setCommandError] = useState<unknown>();
  const [busy, setBusy] = useState<string>();
  const [createOpen, setCreateOpen] = useState(false);
  const [nodeOpen, setNodeOpen] = useState(false);
  const [edgeOpen, setEdgeOpen] = useState(false);
  const [evaluationOpen, setEvaluationOpen] = useState(false);
  const [extractionOpen, setExtractionOpen] = useState(false);
  const [citationQuery, setCitationQuery] = useState("");
  const [citationDeviceModel, setCitationDeviceModel] = useState("");
  const [citationResults, setCitationResults] = useState<
    KnowledgeSearchResult["evidence"]
  >([]);
  const [selectedCitationIds, setSelectedCitationIds] = useState<string[]>([]);
  const [reviewJob, setReviewJob] = useState<KnowledgeGraphExtraction>();
  const [reviewDecision, setReviewDecision] = useState<ExtractionReviewDecision>("ACCEPT");
  const [reviewReason, setReviewReason] = useState("");
  const [createForm] = Form.useForm<CreateValues>();
  const [nodeForm] = Form.useForm<NodeValues>();
  const [edgeForm] = Form.useForm<EdgeValues>();
  const [evaluationForm] = Form.useForm<EvaluationValues>();
  const [queryForm] = Form.useForm<QueryValues>();

  const load = useCallback(async () => {
    const [graphResult, releaseResult, searchProfileResult] = await Promise.allSettled([
      listKnowledgeGraphs(),
      listKnowledgeReleases("PUBLISHED"),
      listKnowledgeSearchProfiles(),
    ]);
    if (graphResult.status === "fulfilled") {
      setGraphs(graphResult.value.releases);
      setSelected((current) =>
        current
          ? graphResult.value.releases.find(
              (item) => item.graph_release_id === current.graph_release_id,
            ) ?? current
          : current,
      );
      setError(undefined);
    } else {
      setError(graphResult.reason);
    }
    if (releaseResult.status === "fulfilled") {
      setIndexReleases(releaseResult.value.releases);
    }
    if (searchProfileResult.status === "fulfilled") {
      setSearchProfiles(searchProfileResult.value.profiles);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!selected) {
      setExtractions([]);
      return;
    }
    let cancelled = false;
    void listKnowledgeGraphExtractions(selected.graph_release_id)
      .then((result) => {
        if (!cancelled) setExtractions(result.extractions);
      })
      .catch((cause) => {
        if (!cancelled) setCommandError(cause);
      });
    return () => {
      cancelled = true;
    };
  }, [selected?.graph_release_id]);

  const run = useCallback(
    async (key: string, command: () => Promise<KnowledgeGraphRelease>) => {
      setBusy(key);
      setCommandError(undefined);
      try {
        const result = await command();
        setSelected(result);
        await load();
        return result;
      } catch (cause) {
        setCommandError(cause);
        return undefined;
      } finally {
        setBusy(undefined);
      }
    },
    [load],
  );

  const activeGraphs = useMemo(
    () => graphs?.filter((item) => item.status === "ACTIVE") ?? [],
    [graphs],
  );
  const totalNodes = useMemo(
    () => graphs?.reduce((sum, item) => sum + item.nodes.length, 0) ?? 0,
    [graphs],
  );
  const totalEdges = useMemo(
    () => graphs?.reduce((sum, item) => sum + item.edges.length, 0) ?? 0,
    [graphs],
  );
  const extractionSearchProfile = useMemo(
    () =>
      searchProfiles.find(
        (profile) =>
          profile.source_index_release_id === selected?.source_index_release_id &&
          profile.status === "ACTIVE",
      ) ??
      searchProfiles.find(
        (profile) => profile.source_index_release_id === selected?.source_index_release_id,
      ),
    [searchProfiles, selected?.source_index_release_id],
  );

  function replaceExtraction(result: KnowledgeGraphExtraction) {
    setExtractions((current) => [
      result,
      ...current.filter((item) => item.extraction_job_id !== result.extraction_job_id),
    ]);
  }

  function openExtractionRequest() {
    setCitationQuery("");
    setCitationDeviceModel("");
    setCitationResults([]);
    setSelectedCitationIds([]);
    setExtractionOpen(true);
  }

  async function searchCitations() {
    if (!extractionSearchProfile || !citationQuery.trim() || !citationDeviceModel.trim()) {
      setCommandError(new Error("请选择可用检索配置，并填写检索内容与设备型号"));
      return;
    }
    setBusy("citation-search");
    setCommandError(undefined);
    try {
      const result = await queryKnowledgeSearchProfile({
        profile_name: extractionSearchProfile.name,
        query_text: citationQuery.trim(),
        device_model: citationDeviceModel.trim(),
        limit: 12,
      });
      setCitationResults(result.evidence);
      setSelectedCitationIds([]);
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function submitExtraction() {
    if (!selected || selectedCitationIds.length === 0) return;
    setBusy("request-extraction");
    setCommandError(undefined);
    try {
      const result = await requestKnowledgeGraphExtraction(
        selected.graph_release_id,
        selected.state_version,
        selectedCitationIds,
        crypto.randomUUID(),
      );
      replaceExtraction(result);
      setExtractionOpen(false);
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  async function retryExtraction(item: KnowledgeGraphExtraction) {
    setBusy(`retry-${item.extraction_job_id}`);
    setCommandError(undefined);
    try {
      replaceExtraction(
        await retryKnowledgeGraphExtraction(item.extraction_job_id, item.version),
      );
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  function openReview(item: KnowledgeGraphExtraction, decision: ExtractionReviewDecision) {
    setReviewJob(item);
    setReviewDecision(decision);
    setReviewReason("");
  }

  async function submitReview() {
    if (!reviewJob || reviewReason.trim().length < 8) return;
    setBusy(`review-${reviewJob.extraction_job_id}`);
    setCommandError(undefined);
    try {
      replaceExtraction(
        await reviewKnowledgeGraphExtraction(
          reviewJob.extraction_job_id,
          reviewJob.version,
          reviewDecision,
          reviewReason.trim(),
        ),
      );
      setReviewJob(undefined);
    } catch (cause) {
      setCommandError(cause);
    } finally {
      setBusy(undefined);
    }
  }

  if (!graphs && !error) {
    return <AppShell><LoadingState label="正在加载 GraphRAG 因果证据" /></AppShell>;
  }

  return (
    <AppShell>
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <Space align="start" style={{ justifyContent: "space-between", width: "100%" }}>
          <div>
            <Typography.Title level={2}>GraphRAG 故障因果证据</Typography.Title>
            <Typography.Paragraph type="secondary">
              只投影部件依赖、故障传播、兼容性和处置链路。每个节点与边都绑定已发布
              CitationAnchor；未通过独立评测的候选图不会进入可查询活动面。
            </Typography.Paragraph>
          </div>
          <Space>
            <Link href="/ai/knowledge">返回知识治理</Link>
            <Button type="primary" onClick={() => setCreateOpen(true)}>创建候选图</Button>
          </Space>
        </Space>

        <Alert
          type="info"
          showIcon
          message="旁路证据，不替代全文检索与 pgvector"
          description="查询仅接受图名、节点键、关系白名单、目标类型、1–4 跳和结果上限；后端不开放任意 Cypher，并在返回前再次校验租户、ACL、设备型号、有效期和引用来源。"
        />
        {commandError ? <ErrorState error={commandError} /> : null}
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}

        <Row gutter={16}>
          <Col span={6}><Card><Statistic title="候选/历史发布" value={graphs?.length ?? 0} /></Card></Col>
          <Col span={6}><Card><Statistic title="活动图" value={activeGraphs.length} /></Card></Col>
          <Col span={6}><Card><Statistic title="引用节点" value={totalNodes} /></Card></Col>
          <Col span={6}><Card><Statistic title="因果关系" value={totalEdges} /></Card></Col>
        </Row>

        <Card title="图谱发布流水线">
          {!graphs?.length ? (
            <EmptyState description="尚无候选图；请从一个已发布知识索引创建首个因果图候选。" />
          ) : (
            <Table
              rowKey="graph_release_id"
              dataSource={graphs}
              pagination={false}
              onRow={(record) => ({ onClick: () => setSelected(record) })}
              columns={[
                { title: "图谱", dataIndex: "name", render: (value, row) => <Space><Typography.Text strong>{value}</Typography.Text><Tag>v{row.version}</Tag></Space> },
                { title: "状态", dataIndex: "status", render: (value) => <Tag color={statusColor(value)}>{value}</Tag> },
                { title: "来源索引", dataIndex: "source_index_release_id", ellipsis: true },
                { title: "节点 / 边", render: (_, row) => `${row.nodes.length} / ${row.edges.length}` },
                { title: "路径召回", render: (_, row) => formatMetric(row.metrics.expected_path_recall) },
                { title: "操作", render: (_, row) => <Button size="small" onClick={(event) => { event.stopPropagation(); setSelected(row); }}>查看与推进</Button> },
              ]}
            />
          )}
        </Card>

        <Card title="活动图多跳查询">
          <Form<QueryValues>
            form={queryForm}
            layout="vertical"
            initialValues={{ max_hops: 3, limit: 10, relation_types: ["DEPENDS_ON", "CAUSES", "MANIFESTS_AS", "MITIGATED_BY"] }}
            onFinish={async (values) => {
              setBusy("query");
              setCommandError(undefined);
              try {
                const result = await queryKnowledgeGraph(values);
                setPaths(result.paths);
              } catch (cause) {
                setCommandError(cause);
              } finally {
                setBusy(undefined);
              }
            }}
          >
            <Row gutter={16}>
              <Col span={6}><Form.Item name="graph_name" label="活动图" rules={[{ required: true }]}><Select options={activeGraphs.map((item) => ({ value: item.name, label: `${item.name} v${item.version}` }))} /></Form.Item></Col>
              <Col span={6}><Form.Item name="start_node_key" label="起点节点键" rules={[{ required: true }]}><Input placeholder="pump.body" /></Form.Item></Col>
              <Col span={4}><Form.Item name="target_node_type" label="目标类型"><Select allowClear options={nodeTypes.map(option)} /></Form.Item></Col>
              <Col span={4}><Form.Item name="max_hops" label="最大跳数"><InputNumber min={1} max={4} style={{ width: "100%" }} /></Form.Item></Col>
              <Col span={4}><Form.Item name="limit" label="结果上限"><InputNumber min={1} max={20} style={{ width: "100%" }} /></Form.Item></Col>
            </Row>
            <Form.Item name="relation_types" label="允许关系" rules={[{ required: true }]}><Select mode="multiple" options={relationTypes.map(option)} /></Form.Item>
            <Button type="primary" htmlType="submit" loading={busy === "query"}>查询可解释路径</Button>
          </Form>
          <Space direction="vertical" size="middle" style={{ width: "100%", marginTop: 20 }}>
            {paths.map((path, index) => (
              <Card key={`${path.nodes.map((node) => node.graph_node_id).join("-")}-${index}`} size="small" title={`证据路径 ${index + 1} · ${path.edges.length} 跳`}>
                <Space wrap>
                  {path.nodes.map((node, nodeIndex) => (
                    <Space key={node.graph_node_id}>
                      <Tag color="blue">{node.node_type}</Tag>
                      <Typography.Text strong>{node.display_name}</Typography.Text>
                      <Typography.Text code>{node.node_key}</Typography.Text>
                      <Typography.Text type="secondary">引用 {node.citation_id}</Typography.Text>
                      {nodeIndex < path.edges.length ? <Tag color="purple">→ {path.edges[nodeIndex].relation_type} →</Tag> : null}
                    </Space>
                  ))}
                </Space>
              </Card>
            ))}
          </Space>
        </Card>
      </Space>

      <Modal
        title="创建 GraphRAG 候选发布"
        open={createOpen}
        onCancel={() => setCreateOpen(false)}
        onOk={() => createForm.submit()}
        confirmLoading={busy === "create"}
      >
        <Form form={createForm} layout="vertical" onFinish={async (values) => {
          const result = await run("create", () => createKnowledgeGraph(values, crypto.randomUUID()));
          if (result) { setCreateOpen(false); createForm.resetFields(); }
        }}>
          <Form.Item name="source_index_release_id" label="已发布知识索引" rules={[{ required: true }]}><Select options={indexReleases.map((item) => ({ value: item.release_id, label: `${item.name} v${item.version} · ${item.release_id}` }))} /></Form.Item>
          <Form.Item name="name" label="图谱名称" rules={[{ required: true }]}><Input placeholder="pump-causal-graph" /></Form.Item>
        </Form>
      </Modal>

      <Modal
        title="添加引用节点"
        open={nodeOpen}
        onCancel={() => setNodeOpen(false)}
        onOk={() => nodeForm.submit()}
        confirmLoading={busy === "node"}
      >
        <Form form={nodeForm} layout="vertical" initialValues={{ node_type: "COMPONENT", metadata_json: "{}" }} onFinish={async (values) => {
          if (!selected) return;
          let metadata: Record<string, unknown>;
          try { metadata = JSON.parse(values.metadata_json || "{}"); } catch { setCommandError(new Error("节点 metadata 必须是 JSON 对象")); return; }
          const result = await run("node", () => addKnowledgeGraphNode(selected.graph_release_id, selected.state_version, { node_key: values.node_key, node_type: values.node_type, display_name: values.display_name, citation_id: values.citation_id, metadata }));
          if (result) { setNodeOpen(false); nodeForm.resetFields(); }
        }}>
          <Form.Item name="node_key" label="稳定节点键" rules={[{ required: true }]}><Input placeholder="bearing.main" /></Form.Item>
          <Form.Item name="node_type" label="节点类型" rules={[{ required: true }]}><Select options={nodeTypes.map(option)} /></Form.Item>
          <Form.Item name="display_name" label="显示名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="citation_id" label="CitationAnchor ID" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="metadata_json" label="附加元数据 JSON"><Input.TextArea rows={3} /></Form.Item>
        </Form>
      </Modal>

      <Modal
        title="添加有向因果关系"
        open={edgeOpen}
        onCancel={() => setEdgeOpen(false)}
        onOk={() => edgeForm.submit()}
        confirmLoading={busy === "edge"}
      >
        <Form form={edgeForm} layout="vertical" initialValues={{ relation_type: "CAUSES", confidence: 1, metadata_json: "{}" }} onFinish={async (values) => {
          if (!selected) return;
          let metadata: Record<string, unknown>;
          try { metadata = JSON.parse(values.metadata_json || "{}"); } catch { setCommandError(new Error("关系 metadata 必须是 JSON 对象")); return; }
          const result = await run("edge", () => addKnowledgeGraphEdge(selected.graph_release_id, selected.state_version, { source_node_id: values.source_node_id, target_node_id: values.target_node_id, relation_type: values.relation_type, confidence: values.confidence, citation_id: values.citation_id, metadata }));
          if (result) { setEdgeOpen(false); edgeForm.resetFields(); }
        }}>
          <Form.Item name="source_node_id" label="起点" rules={[{ required: true }]}><Select options={selected?.nodes.map(nodeOption)} /></Form.Item>
          <Form.Item name="target_node_id" label="终点" rules={[{ required: true }]}><Select options={selected?.nodes.map(nodeOption)} /></Form.Item>
          <Form.Item name="relation_type" label="关系" rules={[{ required: true }]}><Select options={relationTypes.map(option)} /></Form.Item>
          <Form.Item name="confidence" label="人工确认置信度" rules={[{ required: true }]}><InputNumber min={0} max={1} step={0.01} style={{ width: "100%" }} /></Form.Item>
          <Form.Item name="citation_id" label="关系 CitationAnchor ID" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="metadata_json" label="附加元数据 JSON"><Input.TextArea rows={3} /></Form.Item>
        </Form>
      </Modal>

      <Modal
        title="提交独立多跳金标评测"
        open={evaluationOpen}
        width={760}
        onCancel={() => setEvaluationOpen(false)}
        onOk={() => evaluationForm.submit()}
        confirmLoading={busy === "evaluate"}
      >
        <Alert type="warning" showIcon message="至少 3 个金标案例，评测人不能是候选图创建人；预期路径召回需达到 0.8。" />
        <Form form={evaluationForm} layout="vertical" initialValues={{ cases_json: exampleCases }} onFinish={async (values) => {
          if (!selected) return;
          try {
            const cases = JSON.parse(values.cases_json);
            const result = await run("evaluate", () => evaluateKnowledgeGraph(selected.graph_release_id, selected.state_version, { cases }));
            if (result) setEvaluationOpen(false);
          } catch (cause) { setCommandError(cause); }
        }}>
          <Form.Item name="cases_json" label="评测案例 JSON" rules={[{ required: true }]}><Input.TextArea rows={18} style={{ fontFamily: "monospace" }} /></Form.Item>
        </Form>
      </Modal>

      <Modal
        title="从授权引用发起模型因果抽取"
        open={extractionOpen}
        width={780}
        onCancel={() => setExtractionOpen(false)}
        footer={[
          <Button key="cancel" onClick={() => setExtractionOpen(false)}>取消</Button>,
          <Button
            key="submit"
            type="primary"
            disabled={selectedCitationIds.length === 0}
            loading={busy === "request-extraction"}
            onClick={() => void submitExtraction()}
          >
            提交模型抽取
          </Button>,
        ]}
      >
        <Space direction="vertical" size="middle" style={{ width: "100%" }}>
          <Alert
            type="info"
            showIcon
            message="页面只提交服务端已授权的 CitationAnchor ID"
            description="Prompt、模型版本、推理参数和输出 Schema 均由后端治理配置锁定，操作者不能在此绕过。"
          />
          <Typography.Text type="secondary">
            检索配置：{extractionSearchProfile?.name ?? "当前来源索引没有可用检索配置"}
          </Typography.Text>
          <Row gutter={12}>
            <Col span={10}>
              <Input
                placeholder="故障、部件或现象"
                value={citationQuery}
                onChange={(event) => setCitationQuery(event.target.value)}
              />
            </Col>
            <Col span={8}>
              <Input
                placeholder="例如 PUMP-X100"
                value={citationDeviceModel}
                onChange={(event) => setCitationDeviceModel(event.target.value)}
              />
            </Col>
            <Col span={6}>
              <Button
                block
                disabled={!extractionSearchProfile}
                loading={busy === "citation-search"}
                onClick={() => void searchCitations()}
              >
                检索可选引用
              </Button>
            </Col>
          </Row>
          {citationResults.map((evidence) => (
            <Card key={evidence.citation_id} size="small">
              <Checkbox
                checked={selectedCitationIds.includes(evidence.citation_id)}
                onChange={(event) =>
                  setSelectedCitationIds((current) =>
                    event.target.checked
                      ? [...current, evidence.citation_id]
                      : current.filter((citationId) => citationId !== evidence.citation_id),
                  )
                }
              >
                <Space direction="vertical" size={2}>
                  <Typography.Text strong>{evidence.title}</Typography.Text>
                  <Typography.Text code>{evidence.citation_id}</Typography.Text>
                  <Typography.Text type="secondary">
                    {evidence.device_model} · 第 {evidence.page_number ?? "—"} 页
                  </Typography.Text>
                  <Typography.Paragraph style={{ marginBottom: 0 }}>
                    {evidence.content}
                  </Typography.Paragraph>
                </Space>
              </Checkbox>
            </Card>
          ))}
        </Space>
      </Modal>

      <Modal
        title={reviewDecision === "ACCEPT" ? "接受整包候选" : "拒绝整包候选"}
        open={Boolean(reviewJob)}
        okText={reviewDecision === "ACCEPT" ? "确认接受" : "确认拒绝"}
        okButtonProps={{ disabled: reviewReason.trim().length < 8 }}
        confirmLoading={Boolean(reviewJob && busy === `review-${reviewJob.extraction_job_id}`)}
        onCancel={() => setReviewJob(undefined)}
        onOk={() => void submitReview()}
      >
        <Alert
          type={reviewDecision === "ACCEPT" ? "warning" : "info"}
          showIcon
          message={
            reviewDecision === "ACCEPT"
              ? "接受后只写入当前 DRAFT；不会自动同步、评测或激活"
              : "拒绝不会写入任何图节点或关系"
          }
          style={{ marginBottom: 16 }}
        />
        <Input.TextArea
          rows={4}
          placeholder="填写复核理由"
          value={reviewReason}
          onChange={(event) => setReviewReason(event.target.value)}
        />
      </Modal>

      <Drawer
        title={selected ? `${selected.name} v${selected.version}` : "图谱详情"}
        width={860}
        open={Boolean(selected)}
        onClose={() => setSelected(undefined)}
        extra={selected ? (
          <Space>
            {selected.legal_actions.includes("REQUEST_EXTRACTION") ? <Button type="primary" onClick={openExtractionRequest}>发起模型抽取</Button> : null}
            {selected.legal_actions.includes("ADD_NODE") ? <Button onClick={() => setNodeOpen(true)}>添加节点</Button> : null}
            {selected.legal_actions.includes("ADD_EDGE") ? <Button onClick={() => setEdgeOpen(true)}>添加关系</Button> : null}
            {selected.legal_actions.includes("SYNC") ? <Button loading={busy === "sync"} onClick={() => void run("sync", () => syncKnowledgeGraph(selected.graph_release_id, selected.state_version))}>同步 Neo4j</Button> : null}
            {selected.legal_actions.includes("EVALUATE") ? <Button onClick={() => setEvaluationOpen(true)}>独立评测</Button> : null}
            {selected.legal_actions.includes("ACTIVATE") ? <Button type="primary" loading={busy === "activate"} onClick={() => void run("activate", () => activateKnowledgeGraph(selected.graph_release_id, selected.state_version))}>激活旁路证据</Button> : null}
          </Space>
        ) : null}
      >
        {selected ? (
          <Space direction="vertical" size="large" style={{ width: "100%" }}>
            <GraphDetail release={selected} />
            <ExtractionPanel
              extractions={extractions}
              busy={busy}
              onRetry={(item) => void retryExtraction(item)}
              onReview={openReview}
            />
          </Space>
        ) : null}
      </Drawer>
    </AppShell>
  );
}

function GraphDetail({ release }: { release: KnowledgeGraphRelease }) {
  return (
    <Space direction="vertical" size="large" style={{ width: "100%" }}>
      <Descriptions bordered size="small" column={2}>
        <Descriptions.Item label="状态"><Tag color={statusColor(release.status)}>{release.status}</Tag></Descriptions.Item>
        <Descriptions.Item label="状态版本">{release.state_version}</Descriptions.Item>
        <Descriptions.Item label="来源索引" span={2}>{release.source_index_release_id}</Descriptions.Item>
        <Descriptions.Item label="Schema">{release.schema_version}</Descriptions.Item>
        <Descriptions.Item label="Manifest">{release.manifest_hash ?? "尚未同步"}</Descriptions.Item>
        <Descriptions.Item label="创建人">{release.created_by_subject_id}</Descriptions.Item>
        <Descriptions.Item label="评测人">{release.evaluated_by_subject_id ?? "待评测"}</Descriptions.Item>
      </Descriptions>
      {release.failure_codes.length ? <Alert type="error" message="门禁未通过" description={release.failure_codes.join("、")} /> : null}
      <Card size="small" title={`节点（${release.nodes.length}）`}>
        <Table size="small" pagination={false} rowKey="graph_node_id" dataSource={release.nodes} columns={[
          { title: "类型", dataIndex: "node_type", render: (value) => <Tag>{value}</Tag> },
          { title: "节点", render: (_, row) => <Space direction="vertical" size={0}><Typography.Text strong>{row.display_name}</Typography.Text><Typography.Text code>{row.node_key}</Typography.Text></Space> },
          { title: "设备型号", dataIndex: "device_models", render: (value: string[]) => value.join(", ") || "通用" },
          { title: "引用", dataIndex: "citation_id", ellipsis: true },
        ]} />
      </Card>
      <Card size="small" title={`关系（${release.edges.length}）`}>
        <Table size="small" pagination={false} rowKey="graph_edge_id" dataSource={release.edges} columns={[
          { title: "起点", dataIndex: "source_node_id", render: (value) => release.nodes.find((node) => node.graph_node_id === value)?.display_name ?? value },
          { title: "关系", dataIndex: "relation_type", render: (value) => <Tag color="purple">{value}</Tag> },
          { title: "终点", dataIndex: "target_node_id", render: (value) => release.nodes.find((node) => node.graph_node_id === value)?.display_name ?? value },
          { title: "置信度", dataIndex: "confidence" },
          { title: "引用", dataIndex: "citation_id", ellipsis: true },
        ]} />
      </Card>
      <Card size="small" title="评测与激活门禁">
        {!release.evaluation ? <Typography.Text type="secondary">尚未执行独立评测</Typography.Text> : (
          <Descriptions bordered size="small" column={2}>
            <Descriptions.Item label="结果"><Tag color={release.evaluation.status === "PASSED" ? "green" : "red"}>{release.evaluation.status}</Tag></Descriptions.Item>
            <Descriptions.Item label="策略">{release.evaluation.policy_version}</Descriptions.Item>
            <Descriptions.Item label="金标案例">{String(release.metrics.gold_case_count ?? 0)}</Descriptions.Item>
            <Descriptions.Item label="预期路径召回">{formatMetric(release.metrics.expected_path_recall)}</Descriptions.Item>
            <Descriptions.Item label="门禁" span={2}><Space wrap>{Object.entries(release.gate_results).map(([name, passed]) => <Tag key={name} color={passed ? "green" : "red"}>{name}: {passed ? "PASS" : "FAIL"}</Tag>)}</Space></Descriptions.Item>
          </Descriptions>
        )}
      </Card>
    </Space>
  );
}

function ExtractionPanel({
  extractions,
  busy,
  onRetry,
  onReview,
}: {
  extractions: KnowledgeGraphExtraction[];
  busy?: string;
  onRetry: (item: KnowledgeGraphExtraction) => void;
  onReview: (item: KnowledgeGraphExtraction, decision: ExtractionReviewDecision) => void;
}) {
  return (
    <Card size="small" title="模型因果抽取与独立复核">
      <Alert
        type="info"
        showIcon
        message="候选生成与领域复核职责分离"
        description="模型只生成绑定来源、Prompt、模型和结果哈希的整包候选；请求人不能复核自己的任务，接受后也只进入 DRAFT。"
        style={{ marginBottom: 16 }}
      />
      {!extractions.length ? (
        <EmptyState description="当前图谱尚无模型抽取任务。" />
      ) : (
        <Space direction="vertical" size="middle" style={{ width: "100%" }}>
          {extractions.map((item) => {
            const candidate = extractionCandidate(item.candidate_bundle);
            return (
              <Card
                key={item.extraction_job_id}
                size="small"
                title={
                  <Space wrap>
                    <Tag color={extractionStatusColor(item.status)}>
                      {extractionStatusLabel(item.status)}
                    </Tag>
                    <Typography.Text code>{item.extraction_job_id}</Typography.Text>
                    <Typography.Text type="secondary">尝试 {item.attempt_count}</Typography.Text>
                  </Space>
                }
                extra={
                  <Space>
                    {item.legal_actions.includes("RETRY") ? (
                      <Button
                        size="small"
                        loading={busy === `retry-${item.extraction_job_id}`}
                        onClick={() => onRetry(item)}
                      >
                        重试抽取
                      </Button>
                    ) : null}
                    {item.legal_actions.includes("REVIEW_REJECT") ? (
                      <Button size="small" danger onClick={() => onReview(item, "REJECT")}>
                        拒绝整包候选
                      </Button>
                    ) : null}
                    {item.legal_actions.includes("REVIEW_ACCEPT") ? (
                      <Button size="small" type="primary" onClick={() => onReview(item, "ACCEPT")}>
                        接受整包候选
                      </Button>
                    ) : null}
                  </Space>
                }
              >
                <Descriptions bordered size="small" column={2}>
                  <Descriptions.Item label="模型别名">{item.model_alias}</Descriptions.Item>
                  <Descriptions.Item label="模型发布">{item.model_release_id}</Descriptions.Item>
                  <Descriptions.Item label="模型清单哈希" span={2}>{item.model_manifest_hash}</Descriptions.Item>
                  <Descriptions.Item label="Prompt Bundle">{item.prompt_bundle_id}</Descriptions.Item>
                  <Descriptions.Item label="Prompt 哈希">{item.prompt_bundle_hash}</Descriptions.Item>
                  <Descriptions.Item label="来源绑定哈希">{item.source_binding_hash}</Descriptions.Item>
                  <Descriptions.Item label="上下文哈希">{item.context_hash ?? "待推理"}</Descriptions.Item>
                  <Descriptions.Item label="结果哈希">{item.result_hash ?? "待生成"}</Descriptions.Item>
                  <Descriptions.Item label="输出 Schema">{item.response_schema_version}</Descriptions.Item>
                  <Descriptions.Item label="请求人">{item.requested_by_subject_id}</Descriptions.Item>
                  <Descriptions.Item label="复核人">{item.reviewed_by_subject_id ?? "待独立复核"}</Descriptions.Item>
                  {item.failure_code ? (
                    <Descriptions.Item label="失败代码" span={2}>
                      <Typography.Text type="danger">{item.failure_code}</Typography.Text>
                    </Descriptions.Item>
                  ) : null}
                </Descriptions>
                {candidate ? (
                  <Row gutter={12} style={{ marginTop: 16 }}>
                    <Col span={12}>
                      <Card size="small" title={`候选节点（${candidate.nodes.length}）`}>
                        <Space direction="vertical" style={{ width: "100%" }}>
                          {candidate.nodes.map((node) => (
                            <div key={node.node_key}>
                              <Space wrap>
                                <Tag>{node.node_type}</Tag>
                                <Typography.Text strong>{node.display_name}</Typography.Text>
                                <Typography.Text code>{node.node_key}</Typography.Text>
                              </Space>
                              <Typography.Text type="secondary">引用 {node.citation_id}</Typography.Text>
                            </div>
                          ))}
                        </Space>
                      </Card>
                    </Col>
                    <Col span={12}>
                      <Card size="small" title={`候选关系（${candidate.edges.length}）`}>
                        <Space direction="vertical" style={{ width: "100%" }}>
                          {candidate.edges.map((edge, index) => (
                            <div key={`${edge.source_node_key}-${edge.target_node_key}-${index}`}>
                              <Space wrap>
                                <Typography.Text code>{edge.source_node_key}</Typography.Text>
                                <Tag color="purple">{edge.relation_type}</Tag>
                                <Typography.Text code>{edge.target_node_key}</Typography.Text>
                              </Space>
                              <Typography.Text type="secondary">
                                置信度 {edge.confidence} · 引用 {edge.citation_id}
                              </Typography.Text>
                            </div>
                          ))}
                        </Space>
                      </Card>
                    </Col>
                  </Row>
                ) : null}
              </Card>
            );
          })}
        </Space>
      )}
    </Card>
  );
}

function extractionCandidate(
  value: KnowledgeGraphExtraction["candidate_bundle"],
): ExtractionCandidateBundle | undefined {
  if (!value || !Array.isArray(value.nodes) || !Array.isArray(value.edges)) return undefined;
  return value as unknown as ExtractionCandidateBundle;
}

function extractionStatusLabel(status: string) {
  const labels: Record<string, string> = {
    QUEUED: "排队中",
    RUNNING: "抽取中",
    REVIEW_PENDING: "待领域复核",
    ACCEPTED: "已写入 DRAFT，仍需同步、评测和激活",
    REJECTED: "已拒绝",
    FAILED: "抽取失败",
  };
  return labels[status] ?? status;
}

function extractionStatusColor(status: string) {
  if (status === "ACCEPTED") return "green";
  if (status === "REJECTED" || status === "FAILED") return "red";
  if (status === "REVIEW_PENDING") return "blue";
  return "gold";
}

function option(value: string) {
  return { value, label: value };
}

function nodeOption(node: KnowledgeGraphRelease["nodes"][number]) {
  return { value: node.graph_node_id, label: `${node.display_name} · ${node.node_key}` };
}

function statusColor(status: string) {
  if (status === "ACTIVE" || status === "EVALUATED") return "green";
  if (status === "REJECTED" || status === "REVOKED") return "red";
  if (status === "SYNCED") return "blue";
  if (status === "RETIRED") return "default";
  return "gold";
}

function formatMetric(value: unknown) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";
}
