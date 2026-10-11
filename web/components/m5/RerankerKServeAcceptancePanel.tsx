"use client";

import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Row,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useState } from "react";

import { LoadingState } from "@/components/RequestState";
import {
  type EnterpriseRerankerKServeAcceptance,
  getEnterpriseRerankerKServeAcceptance,
} from "@/lib/api/client";

type AcceptanceResult = Awaited<ReturnType<typeof getEnterpriseRerankerKServeAcceptance>>;
type RolloutStage = EnterpriseRerankerKServeAcceptance["stages"][number];

export function RerankerKServeAcceptancePanel() {
  const [result, setResult] = useState<AcceptanceResult>();
  const [error, setError] = useState<unknown>();
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setResult(await getEnterpriseRerankerKServeAcceptance());
      setError(undefined);
    } catch (reason) {
      setResult(undefined);
      setError(reason);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading && !result) return <LoadingState />;
  if (!result) {
    return (
      <Alert
        showIcon
        type="warning"
        message="Reranker KServe 验收证据暂不可用"
        description={(
          <Space direction="vertical">
            <Typography.Text type="secondary">
              {error instanceof Error ? error.message : "无法读取已验证的发布证据。"}
            </Typography.Text>
            <Button size="small" onClick={() => void load()}>重新读取</Button>
          </Space>
        )}
      />
    );
  }

  const { acceptance } = result;
  const cleanupPassed = Object.values(acceptance.cleanup).every(Boolean);
  return (
    <Card
      title="Reranker 真实 GPU / KServe 发布验收"
      extra={(
        <Space wrap>
          <Tag color="green">四阶段通过</Tag>
          <Tag color="blue">{acceptance.classification}</Tag>
        </Space>
      )}
    >
      <Alert
        showIcon
        type="success"
        message="真实 GPU、Shadow、Canary 与回滚证据已闭环"
        description="最终 ROLLED_BACK 是主动回滚演练的预期状态，不是发布失败；该证据属于项目内部本地 Staging，不声明外部企业生产上线。"
        style={{ marginBottom: 16 }}
      />
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={12} lg={6}>
          <Statistic title="治理请求" value={result.governedRequests} />
        </Col>
        <Col xs={12} lg={6}>
          <Statistic title="发布阶段" value={result.stages} suffix="/ 4" />
        </Col>
        <Col xs={12} lg={6}>
          <Statistic
            title="质量分增益"
            value={acceptance.quality.candidate_score_delta}
            precision={3}
          />
        </Col>
        <Col xs={12} lg={6}>
          <Statistic
            title="最终候选流量"
            value={acceptance.model_release.final_traffic_percent}
            suffix="%"
          />
        </Col>
      </Row>
      <Descriptions size="small" bordered column={{ xs: 1, md: 2 }}>
        <Descriptions.Item label="GPU 执行">
          <Tag color={result.actualGpuExecution ? "green" : "red"}>
            {result.actualGpuExecution ? "真实 GPU 已执行" : "未执行"}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item label="模型运行模拟">
          {acceptance.runtime.model_execution_simulated ? "是" : "否"}
        </Descriptions.Item>
        <Descriptions.Item label="ModelRelease">
          <Typography.Text code>{acceptance.model_release.release_id}</Typography.Text>
        </Descriptions.Item>
        <Descriptions.Item label="最终状态">
          <Tag color="purple">{acceptance.model_release.final_release_status}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="运行镜像" span={2}>
          <Typography.Text code copyable={{ text: acceptance.runtime.image_digest }}>
            {acceptance.runtime.image_repository}:{acceptance.runtime.image_tag}
          </Typography.Text>
        </Descriptions.Item>
        <Descriptions.Item label="KServe 控制器">
          {acceptance.kserve.controller_image}
        </Descriptions.Item>
        <Descriptions.Item label="资源清理">
          <Tag color={cleanupPassed ? "green" : "red"}>
            {cleanupPassed ? "全部停止并清理" : "未闭环"}
          </Tag>
        </Descriptions.Item>
      </Descriptions>
      <Table<RolloutStage>
        style={{ marginTop: 16 }}
        rowKey="stage"
        size="small"
        pagination={false}
        dataSource={acceptance.stages}
        scroll={{ x: 900 }}
        columns={[
          { title: "阶段", dataIndex: "stage", render: (value: string) => <Tag>{value}</Tag> },
          { title: "请求数", dataIndex: "request_count" },
          { title: "Stable", dataIndex: "stable_response_count" },
          { title: "Candidate", dataIndex: "candidate_response_count" },
          {
            title: "候选比例",
            dataIndex: "candidate_response_ratio",
            render: (value: number) => `${(value * 100).toFixed(1)}%`,
          },
          {
            title: "P95",
            dataIndex: "p95_latency_ms",
            render: (value: number) => `${value.toFixed(1)} ms`,
          },
          {
            title: "FSM 决策",
            dataIndex: "model_release_observation_decision",
            render: (value: string) => <Tag color="green">{value}</Tag>,
          },
        ]}
      />
      <Typography.Paragraph type="secondary" style={{ marginTop: 16, marginBottom: 0 }}>
        证据链：{" "}
        <Typography.Text code copyable={{ text: result.evidenceChainSha256 }}>
          {result.evidenceChainSha256}
        </Typography.Text>
      </Typography.Paragraph>
    </Card>
  );
}
