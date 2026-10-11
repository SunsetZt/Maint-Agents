"use client";

import { Card, Col, Row, Tag, Typography } from "antd";
import Link from "next/link";

import { AppShell } from "@/components/AppShell";

type WorkspaceDashboardProps = {
  tenantId?: string;
  roles: string[];
};

export function WorkspaceDashboard({ tenantId, roles }: WorkspaceDashboardProps) {
  return (
    <AppShell>
      <div className="page-stack">
        <div>
          <Typography.Title level={2}>售后故障工作台</Typography.Title>
          <Typography.Text type="secondary">
            租户上下文：{tenantId ?? "等待身份映射"}
          </Typography.Text>
          <div>
            {roles.map((role) => (
              <Tag key={role}>{role}</Tag>
            ))}
          </div>
        </div>
        <Row gutter={[16, 16]}>
          <Col xs={24} md={12}>
            <Card title="治理数据与训练候选">
              <p>用途授权、DLP、标注复核、Parquet 快照和 OpenLineage 全链路追踪。</p>
              <Link href="/ai/datasets">进入治理数据工作台</Link>
            </Card>
          </Col>
          <Col xs={24} md={12}>
            <Card title="新建故障草稿">
              <p>从服务端授权设备开始，提交文字并上传待扫描媒体。</p>
              <Link href="/incidents/new">进入报障旅程</Link>
            </Card>
          </Col>
          <Col xs={24} md={12}>
            <Card title="设备选择">
              <p>只显示角色与设备范围交集内的设备及其事实来源。</p>
              <Link href="/assets/select">查看授权设备</Link>
            </Card>
          </Col>
          <Col xs={24} md={12}>
            <Card title="高风险操作审批">
              <p>职责分离审批、执行时风险复核、幂等零件预留与售后工单创建。</p>
              <Link href="/approvals">进入审批箱</Link>
            </Card>
          </Col>
          <Col xs={24} md={12}>
            <Card title="正式业务闭环">
              <p>识别、证据确认、Incident、诊断、可续传事件、引用与工单均使用真实 API。</p>
              <Typography.Text type="secondary">
                从故障草稿进入多模态识别。
              </Typography.Text>
            </Card>
          </Col>
        </Row>
      </div>
    </AppShell>
  );
}
