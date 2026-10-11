import { Descriptions, Tag } from "antd";

const COLORS: Record<string, string> = {
  FAILED: "red",
  REJECTED: "red",
  EXPIRED: "orange",
  PENDING: "gold",
  QUEUED: "blue",
  RUNNING: "processing",
  IN_PROGRESS: "processing",
  APPROVED: "green",
  COMPLETED: "green",
  VERIFIED: "cyan",
  CLOSED: "default",
  SUCCEEDED: "green",
};

export function StatusTag({ status, version }: { status: string; version?: number }) {
  return (
    <Tag color={COLORS[status] ?? "blue"}>
      {status}{version === undefined ? "" : ` · v${version}`}
    </Tag>
  );
}

export function FactGrid({ facts }: { facts: Array<[string, unknown]> }) {
  return (
    <Descriptions bordered size="small" column={{ xs: 1, sm: 2 }}>
      {facts.map(([label, value]) => (
        <Descriptions.Item key={label} label={label}>
          {value === null || value === undefined || value === "" ? "—" : String(value)}
        </Descriptions.Item>
      ))}
    </Descriptions>
  );
}
