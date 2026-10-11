"use client";

import { Alert, Button, Card, Form, Input, Modal, Select, Space, Typography } from "antd";
import Link from "next/link";
import { useEffect, useState } from "react";

import { ErrorState } from "@/components/RequestState";
import {
  type CustomerNotificationProposal,
  type CustomerNotificationProposalInput,
  type CustomerNotificationOption,
  listCustomerNotificationOptions,
  proposeCustomerNotification,
} from "@/lib/api/client";

type NotificationValues = Omit<CustomerNotificationProposalInput,
  "incident_version" | "channel" | "contact_point_id"> & { delivery_option: string };

const optionKey = (option: CustomerNotificationOption) =>
  `${option.channel}:${option.contact_point_id ?? "portal"}`;

function optionLabel(option: CustomerNotificationOption) {
  const prefix = { PORTAL: "", PORTAL_AND_EMAIL: "邮件：", PORTAL_AND_SMS: "短信：" }[
    option.channel
  ];
  return `${prefix}${option.masked_destination}`;
}

export function CustomerNotificationProposalPanel({
  incidentId,
  incidentVersion,
  incidentStatus,
}: {
  incidentId: string;
  incidentVersion: number;
  incidentStatus: string;
}) {
  const [form] = Form.useForm<NotificationValues>();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>();
  const [proposal, setProposal] = useState<CustomerNotificationProposal>();
  const [options, setOptions] = useState<CustomerNotificationOption[]>();
  const [externalUnavailableReason, setExternalUnavailableReason] = useState<string | null>();
  const terminal = ["CLOSED", "CANCELLED"].includes(incidentStatus);

  useEffect(() => {
    if (terminal) return;
    let active = true;
    void Promise.resolve().then(() => listCustomerNotificationOptions(incidentId))
      .then((result) => {
        if (!active) return;
        setOptions(result.options);
        setExternalUnavailableReason(result.externalUnavailableReason);
      })
      .catch((caught) => {
        if (active) setError(caught);
      });
    return () => { active = false; };
  }, [incidentId, terminal]);

  async function submit(values: NotificationValues) {
    const { delivery_option, ...content } = values;
    const selected = options?.find((option) => optionKey(option) === delivery_option);
    if (!selected) return;
    setBusy(true);
    setError(undefined);
    try {
      const result = await proposeCustomerNotification(incidentId, {
        ...content,
        incident_version: incidentVersion,
        channel: selected.channel,
        contact_point_id: selected.contact_point_id,
      });
      setProposal(result.proposal);
      setOpen(false);
      form.resetFields();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="客户通知提案（T2）">
      <div className="page-stack">
        <Alert
          type="info"
          showIcon
          message="受控模板、固定接收范围、人工复核后发布"
          description="通知先发布到客户门户；只有服务端返回的已验证、已同意联系点才能同时发送邮件或短信。标题、正文、渠道与脱敏联系点会整体绑定到审批。"
        />
        {options ? (
          <Space wrap aria-label="可用通知渠道">
            {options.map((option) => (
              <Typography.Text key={optionKey(option)} code={option.channel !== "PORTAL"}>{option.masked_destination}</Typography.Text>
            ))}
          </Space>
        ) : null}
        {externalUnavailableReason ? (
          <Alert type="warning" showIcon message="站外通知当前不可用，仍可使用客户门户" description={externalUnavailableReason} />
        ) : null}
        {error ? <ErrorState error={error} /> : null}
        {proposal ? (
          <Alert
            type="success"
            showIcon
            message={`提案 ${proposal.proposal_id} 已进入审批箱`}
            description={(
              <Space wrap>
                <Typography.Text>当前状态：{proposal.status}</Typography.Text>
                <Link href="/approvals">前往审批与发布</Link>
              </Space>
            )}
          />
        ) : null}
        <Button type="primary" disabled={terminal || !options?.length} onClick={() => setOpen(true)}>
          创建客户通知提案
        </Button>
        {terminal ? (
          <Typography.Text type="secondary">终态故障单不能再创建客户通知。</Typography.Text>
        ) : null}
      </div>

      <Modal
        open={open}
        title="创建客户通知提案"
        okText="提交审批"
        cancelText="取消"
        confirmLoading={busy}
        onCancel={() => setOpen(false)}
        onOk={() => form.submit()}
        destroyOnHidden
      >
        <Form<NotificationValues>
          form={form}
          layout="vertical"
          initialValues={{ template_id: "DIAGNOSIS_SUMMARY", delivery_option: options?.[0] ? optionKey(options[0]) : undefined }}
          onFinish={(values) => void submit(values)}
        >
          <Form.Item name="delivery_option" label="通知渠道" rules={[{ required: true }]}>
            <Select options={options?.map((option) => ({
              value: optionKey(option), label: optionLabel(option),
            }))} />
          </Form.Item>
          <Form.Item name="template_id" label="通知模板" rules={[{ required: true }]}>
            <Select
              options={[
                { value: "DIAGNOSIS_SUMMARY", label: "诊断结论摘要" },
                { value: "INFORMATION_REQUEST", label: "资料补充请求" },
                { value: "SERVICE_PROGRESS", label: "服务进展" },
              ]}
            />
          </Form.Item>
          <Form.Item
            name="headline"
            label="客户可见标题"
            rules={[{ required: true, min: 3, max: 120 }]}
          >
            <Input showCount maxLength={120} />
          </Form.Item>
          <Form.Item
            name="detail"
            label="客户可见正文"
            rules={[{ required: true, min: 3, max: 1000 }]}
          >
            <Input.TextArea rows={5} showCount maxLength={1000} />
          </Form.Item>
          <Form.Item
            name="next_step"
            label="下一步"
            rules={[{ required: true, min: 3, max: 500 }]}
          >
            <Input.TextArea rows={3} showCount maxLength={500} />
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
}
