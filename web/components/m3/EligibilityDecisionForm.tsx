"use client";

import { Alert, Button, Form, Input, Select } from "antd";

import type { CandidateEligibilityInput } from "@/lib/api/client";

export type EligibilityDecisionValues = {
  decision: "ALLOW" | "DENY";
  purpose: string;
  decision_basis: string;
  license_status: "APPROVED" | "REJECTED" | "UNKNOWN";
  retention_until?: string;
  policy_version: string;
};

type Props = {
  submitting: boolean;
  onSubmit: (input: CandidateEligibilityInput) => void;
};

export function EligibilityDecisionForm({ submitting, onSubmit }: Props) {
  const [form] = Form.useForm<EligibilityDecisionValues>();
  const decision = Form.useWatch("decision", form) ?? "ALLOW";
  const allowsTraining = decision === "ALLOW";

  return (
    <Form<EligibilityDecisionValues>
      form={form}
      layout="vertical"
      initialValues={{
        decision: "ALLOW",
        purpose: "industrial_model_improvement",
        decision_basis: "enterprise-maintenance-training-contract",
        license_status: "APPROVED",
        retention_until: localDateTime(365),
        policy_version: "industrial-data-eligibility-v1",
      }}
      onValuesChange={(changed) => {
        if (changed.decision === "ALLOW") {
          form.setFieldValue("license_status", "APPROVED");
        } else if (changed.decision === "DENY") {
          form.setFieldsValue({ license_status: "UNKNOWN", retention_until: undefined });
        }
      }}
      onFinish={(values) => onSubmit(buildEligibilityDecisionInput(values))}
    >
      <Form.Item name="decision" label="治理决定" rules={[{ required: true }]}>
        <Select
          options={[
            { value: "ALLOW", label: "批准用于指定用途" },
            { value: "DENY", label: "拒绝或暂不授权" },
          ]}
        />
      </Form.Item>
      <Form.Item name="purpose" label="限定用途" rules={[{ required: true, whitespace: true }]}>
        <Input />
      </Form.Item>
      <Form.Item
        name="decision_basis"
        label={allowsTraining ? "授权依据" : "拒绝或暂缓依据"}
        rules={[{ required: true, whitespace: true }]}
      >
        <Input.TextArea rows={2} />
      </Form.Item>
      <Form.Item name="license_status" label="许可证状态" rules={[{ required: true }]}>
        <Select
          disabled={allowsTraining}
          options={[
            { value: "APPROVED", label: "已批准" },
            { value: "REJECTED", label: "已拒绝" },
            { value: "UNKNOWN", label: "尚未确认" },
          ]}
        />
      </Form.Item>
      {allowsTraining ? (
        <Form.Item
          name="retention_until"
          label="授权保留截止时间"
          rules={[{ required: true }]}
        >
          <Input type="datetime-local" />
        </Form.Item>
      ) : null}
      <Form.Item name="policy_version" label="治理策略版本" rules={[{ required: true, whitespace: true }]}>
        <Input />
      </Form.Item>
      <Alert
        showIcon
        type={allowsTraining ? "info" : "warning"}
        title={allowsTraining ? "批准后仍需通过 DLP 与人工复核" : "拒绝后候选不会进入策展或标注"}
        description={
          allowsTraining
            ? "该决定只授予指定用途资格，不会绕过脱敏、许可证、保留期和质量门禁。"
            : "决定会追加保存；在尚未产生 DLP 派生数据前，可依据新的合规材料重新决策。"
        }
        style={{ marginBottom: 16 }}
      />
      <Button
        htmlType="submit"
        type="primary"
        danger={!allowsTraining}
        loading={submitting}
      >
        {allowsTraining ? "确认批准训练用途" : "确认拒绝训练用途"}
      </Button>
    </Form>
  );
}

export function buildEligibilityDecisionInput(
  values: EligibilityDecisionValues,
): CandidateEligibilityInput {
  const allowsTraining = values.decision === "ALLOW";
  return {
    purpose: values.purpose.trim(),
    allow_training: allowsTraining,
    consent_basis: values.decision_basis.trim(),
    license_status: allowsTraining ? "APPROVED" : values.license_status,
    retention_until: allowsTraining && values.retention_until
      ? new Date(values.retention_until).toISOString()
      : null,
    policy_version: values.policy_version.trim(),
  };
}

function localDateTime(daysFromNow: number): string {
  const value = new Date(Date.now() + daysFromNow * 24 * 60 * 60 * 1000);
  value.setMinutes(value.getMinutes() - value.getTimezoneOffset());
  return value.toISOString().slice(0, 16);
}
