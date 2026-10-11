"use client";

import { Button, Input, List, Space, Tag, Typography } from "antd";
import type { Dispatch, SetStateAction } from "react";

import type { EvidenceBundle, FieldQrCodeDecision } from "@/lib/api/client";

export function QrCodeReviewItem({
  candidate,
  evidence,
  qrCodeDecisions,
  setQrCodeDecisions,
  variant,
}: {
  candidate: NonNullable<EvidenceBundle["qr_codes"]>[number];
  evidence: EvidenceBundle;
  qrCodeDecisions: Record<string, FieldQrCodeDecision>;
  setQrCodeDecisions: Dispatch<SetStateAction<Record<string, FieldQrCodeDecision>>>;
  variant: "field" | "incident";
}) {
  const keyPrefix = variant === "field" ? "field-" : "";
  const CorrectionInput = variant === "field" ? Input : Input.TextArea;
  const decision = qrCodeDecisions[candidate.candidate_id];
  const unsafe = candidate.security_findings.length > 0;
  return (
    <List.Item
      actions={[
        ...(!unsafe ? [
          <Button
            aria-label={`接受二维码 ${candidate.candidate_id}`}
            key={`accept-${keyPrefix}qr-${candidate.candidate_id}`}
            type={decision?.disposition === "ACCEPTED" ? "primary" : "default"}
            disabled={evidence.status === "CONFIRMED"}
            onClick={() => setQrCodeDecisions((current) => ({
              ...current,
              [candidate.candidate_id]: {
                disposition: "ACCEPTED",
                corrected_text: null,
              },
            }))}
          >接受</Button>,
        ] : []),
        <Button
          aria-label={`拒绝二维码 ${candidate.candidate_id}`}
          key={`reject-${keyPrefix}qr-${candidate.candidate_id}`}
          danger
          type={decision?.disposition === "REJECTED" ? "primary" : "default"}
          disabled={evidence.status === "CONFIRMED"}
          onClick={() => setQrCodeDecisions((current) => ({
            ...current,
            [candidate.candidate_id]: {
              disposition: "REJECTED",
              corrected_text: null,
            },
          }))}
        >拒绝</Button>,
        <Button
          aria-label={`安全修正二维码 ${candidate.candidate_id}`}
          key={`correct-${keyPrefix}qr-${candidate.candidate_id}`}
          type={decision?.disposition === "CORRECTED" ? "primary" : "default"}
          disabled={evidence.status === "CONFIRMED"}
          onClick={() => setQrCodeDecisions((current) => ({
            ...current,
            [candidate.candidate_id]: {
              disposition: "CORRECTED",
              corrected_text: current[candidate.candidate_id]?.corrected_text ?? "",
            },
          }))}
        >安全修正</Button>,
      ]}
    >
      <div className="page-stack" style={{ width: "100%" }}>
        <Space wrap>
          <Tag color="purple">二维码</Tag>
          <Tag>{candidate.payload_kind}</Tag>
          {variant === "incident" ? unsafe ? <Tag color="red">内容安全命中</Tag> : <Tag color="green">未命中安全规则</Tag> : null}
          {variant === "field" ? <Tag>{evidence.processor_versions.qr ?? "unknown"}</Tag> : null}
          {candidate.source_frame_id ? <Tag>{`帧 ${candidate.source_frame_id}`}</Tag> : null}
          {variant === "field" ? unsafe ? <Tag color="red">内容安全命中</Tag> : <Tag color="green">未命中安全规则</Tag> : null}
        </Space>
        <Typography.Text code>{candidate.text}</Typography.Text>
        <Typography.Text type="secondary">
          {`区域 x=${candidate.bbox.x.toFixed(2)}, y=${candidate.bbox.y.toFixed(2)}, w=${candidate.bbox.width.toFixed(2)}, h=${candidate.bbox.height.toFixed(2)}`}
        </Typography.Text>
        {candidate.security_findings.map((finding) => (
          <Typography.Text key={`${candidate.candidate_id}:${finding.pattern_id}`} type="danger">
            {`${finding.pattern_id} · ${finding.category} · ${finding.policy_version}`}
          </Typography.Text>
        ))}
        {decision?.disposition === "CORRECTED" ? (
          <CorrectionInput
            aria-label={`二维码安全修正文 ${candidate.candidate_id}`}
            value={decision.corrected_text ?? ""}
            placeholder={variant === "field" ? "输入人工核实后的安全参考文本" : "只填写人工核实后的安全参考文本"}
            onChange={(event) => setQrCodeDecisions((current) => ({
              ...current,
              [candidate.candidate_id]: {
                disposition: "CORRECTED",
                corrected_text: event.target.value,
              },
            }))}
          />
        ) : null}
      </div>
    </List.Item>
  );
}
