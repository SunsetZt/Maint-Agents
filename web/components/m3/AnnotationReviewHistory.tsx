"use client";

import { Alert, Card, Descriptions, List, Space, Tag, Typography } from "antd";

import type { AnnotationTaskHistory } from "@/lib/api/client";

export function AnnotationReviewHistory({
  tasks,
}: {
  tasks: AnnotationTaskHistory[];
}) {
  return (
    <Card title={`标注与复核历史 (${tasks.length})`} size="small">
      <List
        dataSource={tasks}
        locale={{ emptyText: "尚未创建标注复核任务" }}
        renderItem={(task, index) => {
          const reviewsByRevision = new Map(
            task.reviews.map((review) => [review.revision_id, review]),
          );
          return (
            <List.Item>
              <div style={{ width: "100%" }}>
                <Space wrap style={{ marginBottom: 8 }}>
                  <Typography.Text strong>任务 {index + 1}</Typography.Text>
                  <Tag color={task.status === "APPROVED" ? "green" : task.status === "CONFLICT" ? "red" : "orange"}>
                    {task.status}
                  </Tag>
                  <Tag>{task.risk_level}</Tag>
                  <Tag>复核 {task.reviews.length}/{task.required_reviews}</Tag>
                </Space>
                <Descriptions
                  size="small"
                  column={1}
                  items={[
                    { key: "schema", label: "标注模式", children: task.schema_version },
                    { key: "project", label: "Label Studio 项目", children: task.project_id },
                    { key: "external", label: "外部任务版本", children: `${task.external_task_id} · ${task.external_version}` },
                    { key: "consistency", label: "一致性结论", children: task.consistency_status },
                    { key: "payload", label: "任务 Payload 哈希", children: <Typography.Text code copyable>{task.payload_hash}</Typography.Text> },
                    { key: "creator", label: "创建主体", children: task.created_by_subject_id },
                    { key: "created", label: "创建时间", children: new Date(task.created_at).toLocaleString("zh-CN") },
                    { key: "id", label: "任务标识", children: <Typography.Text code copyable>{task.task_id}</Typography.Text> },
                  ]}
                />
                <List
                  size="small"
                  header="不可变复核证据"
                  dataSource={task.revisions}
                  locale={{ emptyText: "等待专家提交复核" }}
                  renderItem={(revision) => {
                    const review = reviewsByRevision.get(revision.revision_id);
                    return (
                      <List.Item>
                        <Space direction="vertical" size={2} style={{ width: "100%" }}>
                          <Space wrap>
                            <Tag>{revision.reviewer_subject_id}</Tag>
                            <Tag color={review?.outcome === "SUBMITTED" ? "green" : "orange"}>
                              {review?.outcome ?? "尚未复核"}
                            </Tag>
                            <Typography.Text type="secondary">
                              外部版本 {revision.external_version}
                            </Typography.Text>
                          </Space>
                          <Typography.Text code copyable>{revision.labels_digest}</Typography.Text>
                          <Typography.Text type="secondary">
                            {new Date(revision.submitted_at).toLocaleString("zh-CN")}
                          </Typography.Text>
                        </Space>
                      </List.Item>
                    );
                  }}
                />
                {task.status === "CONFLICT" || task.status === "REVIEW_REQUIRED" ? (
                  <Alert
                    showIcon
                    type="error"
                    title={task.status === "CONFLICT" ? "专家标签不一致，需要仲裁" : "外部任务发生漂移，需要重新复核"}
                    style={{ marginTop: 12 }}
                  />
                ) : null}
              </div>
            </List.Item>
          );
        }}
      />
    </Card>
  );
}
