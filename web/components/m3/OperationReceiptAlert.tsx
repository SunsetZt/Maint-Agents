"use client";

import { Alert, Space, Typography } from "antd";

export type OperationReceipt = {
  title: string;
  resourceId: string;
  status: string;
  requestId?: string;
};

export function OperationReceiptAlert({
  receipt,
  onClose,
}: {
  receipt: OperationReceipt;
  onClose: () => void;
}) {
  return (
    <Alert
      showIcon
      closable
      type="success"
      title={receipt.title}
      onClose={onClose}
      description={(
        <Space direction="vertical" size={2}>
          <Typography.Text>
            资源：<Typography.Text code>{receipt.resourceId}</Typography.Text>
          </Typography.Text>
          <Typography.Text>结果：{receipt.status}</Typography.Text>
          {receipt.requestId ? (
            <Typography.Text type="secondary" className="request-id">
              请求标识：{receipt.requestId}
            </Typography.Text>
          ) : null}
        </Space>
      )}
    />
  );
}
