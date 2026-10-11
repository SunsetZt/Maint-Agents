"use client";

import {
  Alert,
  Button,
  Card,
  Descriptions,
  Form,
  Input,
  List,
  Radio,
  Select,
  Space,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useState } from "react";

import {
  type DiagnosisQualityFeedback,
  type DiagnosisQualityFeedbackInput,
  createDiagnosisFeedback,
  listDiagnosisFeedback,
} from "@/lib/api/client";

export type DiagnosisFeedbackPanelProps = {
  diagnosisRunId: string;
};

type FeedbackValues = Pick<
  DiagnosisQualityFeedbackInput,
  "verdict" | "issue_codes" | "comment"
>;

const ISSUE_OPTIONS: Array<{
  value: NonNullable<FeedbackValues["issue_codes"]>[number];
  label: string;
}> = [
  { value: "INCORRECT_ROOT_CAUSE", label: "根因不正确" },
  { value: "MISSING_EVIDENCE", label: "缺少关键证据" },
  { value: "WRONG_CITATION", label: "引用错误" },
  { value: "OUTDATED_KNOWLEDGE", label: "知识已过期" },
  { value: "TOOL_FACT_MISMATCH", label: "工具事实不一致" },
  { value: "INCOMPLETE_NEXT_STEPS", label: "后续步骤不完整" },
  { value: "UNSAFE_RECOMMENDATION", label: "建议不安全" },
  { value: "OTHER", label: "其他" },
];

const VERDICT_LABELS: Record<string, string> = {
  HELPFUL: "有帮助",
  PARTIALLY_HELPFUL: "部分有帮助",
  NOT_HELPFUL: "无帮助",
  UNSAFE: "存在安全风险",
};

export function DiagnosisFeedbackPanel({ diagnosisRunId }: DiagnosisFeedbackPanelProps) {
  const [form] = Form.useForm<FeedbackValues>();
  const [feedback, setFeedback] = useState<DiagnosisQualityFeedback[]>();
  const [canSubmit, setCanSubmit] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>();
  const [receipt, setReceipt] = useState<string>();

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const result = await listDiagnosisFeedback(diagnosisRunId);
      setFeedback(result.feedback);
      setCanSubmit(result.canSubmit);
    } catch (caught) {
      setError(caught);
    }
  }, [diagnosisRunId]);

  useEffect(() => { void load(); }, [load]);

  async function submit(values: FeedbackValues) {
    setLoading(true);
    setError(undefined);
    setReceipt(undefined);
    try {
      const result = await createDiagnosisFeedback(
        {
          diagnosis_run_id: diagnosisRunId,
          verdict: values.verdict,
          issue_codes: values.issue_codes ?? [],
          comment: values.comment?.trim() || null,
        },
        crypto.randomUUID(),
      );
      setFeedback((current) => {
        const remaining = (current ?? []).filter(
          (item) => item.feedback_id !== result.feedback.feedback_id,
        );
        return [...remaining, result.feedback];
      });
      setCanSubmit(false);
      setReceipt("反馈已由服务端绑定并保存");
      form.resetFields();
    } catch (caught) {
      setError(caught);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card title="诊断质量反馈" data-diagnosis-run-id={diagnosisRunId}>
      <div className="page-stack">
        <Alert
          type="info"
          showIcon
          message="反馈是追加式质量事实"
          description="评价和说明不会自动重新诊断，也不会覆盖 AI 原始报告；需要改变正式结论时，请继续使用专家接管与专家修订流程。"
        />
        {receipt ? <Alert type="success" showIcon message={receipt} /> : null}
        {error ? (
          <Alert
            type="error"
            showIcon
            message="诊断质量反馈请求失败"
            description={error instanceof Error ? error.message : "请稍后重试"}
            action={<Button size="small" onClick={() => void load()}>重试</Button>}
          />
        ) : null}
        {feedback === undefined ? (
          <Typography.Text type="secondary">正在读取服务端反馈事实……</Typography.Text>
        ) : feedback.length === 0 ? (
          <Typography.Text type="secondary">当前诊断尚无质量反馈。</Typography.Text>
        ) : (
          <List
            dataSource={feedback}
            renderItem={(item) => (
              <List.Item>
                <Card
                  size="small"
                  title={<Space wrap><Tag>{VERDICT_LABELS[item.verdict] ?? item.verdict}</Tag><Typography.Text code>{item.feedback_id}</Typography.Text></Space>}
                  style={{ width: "100%" }}
                >
                  <div className="page-stack">
                    {item.comment ? <Typography.Paragraph>{item.comment}</Typography.Paragraph> : null}
                    {item.issue_codes.length > 0 ? (
                      <Space wrap>{item.issue_codes.map((code) => <Tag key={code}>{code}</Tag>)}</Space>
                    ) : null}
                    <Descriptions
                      bordered
                      size="small"
                      column={1}
                      items={[
                        { key: "diagnosis", label: "诊断运行", children: item.diagnosis_run_id },
                        { key: "agent", label: "Agent 运行", children: item.agent_run_id },
                        { key: "report", label: "服务端报告摘要", children: <Typography.Text code copyable>{item.report_digest}</Typography.Text> },
                        { key: "manifest", label: "服务端 Manifest 摘要", children: <Typography.Text code copyable>{item.manifest_digest}</Typography.Text> },
                        { key: "model", label: "服务端模型版本", children: item.model_release_id ?? "—" },
                        { key: "prompt", label: "服务端提示词版本", children: item.prompt_bundle_id ?? "—" },
                        { key: "index", label: "服务端知识索引版本", children: item.index_release_id ?? "—" },
                        { key: "author", label: "提交主体", children: `${item.submitted_by_subject_id} · ${item.submitted_by_role}` },
                        { key: "created", label: "提交时间", children: item.created_at },
                      ]}
                    />
                  </div>
                </Card>
              </List.Item>
            )}
          />
        )}
        {canSubmit ? (
          <Card size="small" title="评价本次 AI 诊断">
            <Form<FeedbackValues>
              form={form}
              layout="vertical"
              initialValues={{ verdict: "HELPFUL", issue_codes: [] }}
              onFinish={(values) => void submit(values)}
            >
              <Form.Item name="verdict" label="评价" rules={[{ required: true }]}>
                <Radio.Group>
                  <Radio value="HELPFUL">有帮助</Radio>
                  <Radio value="PARTIALLY_HELPFUL">部分有帮助</Radio>
                  <Radio value="NOT_HELPFUL">无帮助</Radio>
                  <Radio value="UNSAFE">存在安全风险</Radio>
                </Radio.Group>
              </Form.Item>
              <Form.Item
                noStyle
                shouldUpdate={(previous, current) => previous.verdict !== current.verdict}
              >
                {({ getFieldValue }) => {
                  const verdict = getFieldValue("verdict") as FeedbackValues["verdict"];
                  const negative = verdict === "NOT_HELPFUL" || verdict === "UNSAFE";
                  return (
                    <>
                      <Form.Item
                        name="issue_codes"
                        label="问题类型"
                        rules={negative ? [{ required: true, message: "负面或安全评价必须选择问题类型" }] : []}
                      >
                        <Select mode="multiple" options={ISSUE_OPTIONS} placeholder="选择可复核的问题类型" />
                      </Form.Item>
                      <Form.Item
                        name="comment"
                        label="补充说明"
                        rules={negative ? [{ required: true, min: 8, message: "请提供至少 8 个字符的可复核依据" }] : []}
                      >
                        <Input.TextArea rows={3} maxLength={4000} showCount placeholder="说明与现场事实、证据或安全要求的对应关系" />
                      </Form.Item>
                    </>
                  );
                }}
              </Form.Item>
              <Button type="primary" htmlType="submit" loading={loading}>
                提交质量反馈
              </Button>
            </Form>
          </Card>
        ) : feedback !== undefined ? (
          <Typography.Text type="secondary">当前主体不能再次提交；既有反馈保持不可覆盖。</Typography.Text>
        ) : null}
      </div>
    </Card>
  );
}
