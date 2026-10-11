"use client";

import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  InputNumber,
  List,
  Modal,
  Select,
  Space,
  Switch,
  Tabs,
  Tag,
  Typography,
} from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/RequestState";
import { FactGrid, StatusTag } from "@/components/m2/BusinessState";
import {
  type TenantAdministrationRecord,
  type TenantAssetOption,
  type TenantMember,
  type TenantOverview,
  type TenantRetentionPolicy,
  type TenantSiteOption,
  createTenantMember,
  getTenantAdministrationOverview,
  listTenantAdministrationRecords,
  listTenantMembers,
  updateTenantMember,
  updateTenantRetentionPolicy,
} from "@/lib/api/client";

type MemberFormValues = {
  subject_id?: string;
  oidc_issuer?: string;
  oidc_subject?: string;
  display_name?: string;
  email?: string;
  status: "ACTIVE" | "SUSPENDED";
  roles: string[];
  asset_ids: string[];
  site_ids: string[];
  reason: string;
};

type RetentionFormValues = {
  incident_days: number;
  media_days: number;
  knowledge_days: number;
  training_data_days: number;
  audit_days: number;
  legal_hold: boolean;
  reason: string;
};

const DEFAULT_RETENTION: RetentionFormValues = {
  incident_days: 1095,
  media_days: 365,
  knowledge_days: 1825,
  training_data_days: 1095,
  audit_days: 2555,
  legal_hold: false,
  reason: "建立租户数据保留基线",
};

export default function TenantAdministrationPage() {
  const [tenant, setTenant] = useState<TenantOverview>();
  const [policy, setPolicy] = useState<TenantRetentionPolicy>();
  const [members, setMembers] = useState<TenantMember[]>();
  const [records, setRecords] = useState<TenantAdministrationRecord[]>();
  const [roles, setRoles] = useState<string[]>([]);
  const [assets, setAssets] = useState<TenantAssetOption[]>([]);
  const [sites, setSites] = useState<TenantSiteOption[]>([]);
  const [memberTotal, setMemberTotal] = useState(0);
  const [requestId, setRequestId] = useState<string>();
  const [editingMember, setEditingMember] = useState<TenantMember | null>();
  const [memberModalOpen, setMemberModalOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>();
  const [memberForm] = Form.useForm<MemberFormValues>();
  const [retentionForm] = Form.useForm<RetentionFormValues>();

  const load = useCallback(async () => {
    setError(undefined);
    try {
      const [overview, memberResult, auditResult] = await Promise.all([
        getTenantAdministrationOverview(),
        listTenantMembers({ limit: 100 }),
        listTenantAdministrationRecords({ limit: 50 }),
      ]);
      setTenant(overview.tenant);
      setPolicy(overview.retentionPolicy);
      setRoles(overview.assignableRoles);
      setAssets(overview.assets);
      setSites(overview.sites);
      setMembers(memberResult.members);
      setMemberTotal(memberResult.total);
      setRecords(auditResult.records);
      setRequestId(overview.requestId);
      const current = overview.retentionPolicy;
      retentionForm.setFieldsValue(current.configured ? {
        incident_days: current.incident_days!,
        media_days: current.media_days!,
        knowledge_days: current.knowledge_days!,
        training_data_days: current.training_data_days!,
        audit_days: current.audit_days!,
        legal_hold: current.legal_hold,
        reason: "调整租户数据保留策略",
      } : DEFAULT_RETENTION);
    } catch (caught) {
      setError(caught);
    }
  }, [retentionForm]);

  useEffect(() => { void load(); }, [load]);

  const roleOptions = useMemo(() => roles.map((role) => ({
    value: role,
    label: roleLabel(role),
  })), [roles]);
  const assetOptions = useMemo(() => assets.map((asset) => ({
    value: asset.asset_id,
    label: `${asset.display_name ?? asset.asset_id}${asset.model_code ? ` · ${asset.model_code}` : ""}`,
  })), [assets]);
  const siteOptions = useMemo(() => sites.map((site) => ({
    value: site.site_id,
    label: `${site.site_name} · ${site.site_id}`,
  })), [sites]);

  function openCreateMember() {
    setEditingMember(null);
    memberForm.resetFields();
    memberForm.setFieldsValue({
      oidc_issuer: "http://localhost:8080/realms/industrial-ops",
      status: "ACTIVE",
      roles: [],
      asset_ids: [],
      site_ids: [],
    });
    setMemberModalOpen(true);
  }

  function openEditMember(member: TenantMember) {
    setEditingMember(member);
    memberForm.setFieldsValue({
      subject_id: member.subject_id,
      oidc_issuer: member.oidc_issuer,
      oidc_subject: member.oidc_subject,
      display_name: member.display_name ?? undefined,
      email: member.email ?? undefined,
      status: member.status as "ACTIVE" | "SUSPENDED",
      roles: member.roles,
      asset_ids: member.asset_ids,
      site_ids: member.site_ids,
      reason: "调整成员角色或资源范围",
    });
    setMemberModalOpen(true);
  }

  async function submitMember(values: MemberFormValues) {
    setBusy(true);
    setError(undefined);
    try {
      if (editingMember) {
        await updateTenantMember(
          editingMember.subject_id,
          editingMember.version,
          {
            display_name: values.display_name ?? null,
            email: values.email ?? null,
            status: values.status,
            roles: values.roles,
            asset_ids: values.asset_ids,
            site_ids: values.site_ids,
            reason: values.reason,
          },
          crypto.randomUUID(),
        );
      } else {
        await createTenantMember({
          subject_id: values.subject_id!,
          oidc_issuer: values.oidc_issuer!,
          oidc_subject: values.oidc_subject!,
          display_name: values.display_name ?? null,
          email: values.email ?? null,
          roles: values.roles,
          asset_ids: values.asset_ids,
          site_ids: values.site_ids,
          reason: values.reason,
        }, crypto.randomUUID());
      }
      setMemberModalOpen(false);
      await load();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  async function submitRetention(values: RetentionFormValues) {
    if (!policy) return;
    setBusy(true);
    setError(undefined);
    try {
      await updateTenantRetentionPolicy(
        policy.version,
        values,
        crypto.randomUUID(),
      );
      await load();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  if (!tenant && !error) {
    return <AppShell><LoadingState label="正在读取租户治理配置" /></AppShell>;
  }

  return (
    <AppShell>
      <div className="page-stack">
        <Typography.Title level={2}>租户治理中心</Typography.Title>
        <Alert
          type="info"
          showIcon
          message="企业 OIDC 负责登录身份，平台负责租户内最小权限"
          description="受管理成员的状态、岗位角色和设备/站点范围在每次 API 请求时重新从平台目录投影；暂停与撤权不依赖旧 Token 自然过期。外部 IdP 账号生命周期仍由企业身份系统管理。"
        />
        {error ? <ErrorState error={error} onRetry={() => void load()} /> : null}
        {tenant ? (
          <Card title={tenant.display_name ?? tenant.tenant_id} extra={<StatusTag status={tenant.status} version={tenant.version} />}>
            <FactGrid facts={[
              ["租户 ID", tenant.tenant_id],
              ["成员总数", memberTotal],
              ["有效成员", tenant.active_member_count],
              ["设备数量", tenant.asset_count],
              ["站点数量", tenant.site_count],
              ["请求标识", requestId],
            ]} />
          </Card>
        ) : null}

        <Tabs items={[
          {
            key: "members",
            label: "成员与权限",
            children: (
              <Card title="租户成员" extra={<Button type="primary" onClick={openCreateMember}>登记成员</Button>}>
                {members?.length === 0 ? <EmptyState description="尚未登记受管理成员" /> : null}
                <List
                  grid={{ gutter: 16, xs: 1, xl: 2 }}
                  dataSource={members}
                  renderItem={(member) => (
                    <List.Item>
                      <Card
                        size="small"
                        title={member.display_name ?? member.subject_id}
                        extra={<StatusTag status={member.status} version={member.version} />}
                      >
                        <div className="page-stack">
                          <Typography.Text type="secondary">{member.email ?? member.subject_id}</Typography.Text>
                          <Space wrap>{member.roles.map((role) => <Tag key={role} color="blue">{roleLabel(role)}</Tag>)}</Space>
                          <FactGrid facts={[
                            ["Subject ID", member.subject_id],
                            ["设备范围", member.asset_ids.length ? member.asset_ids.join("、") : "未授权"],
                            ["站点范围", member.site_ids.length ? member.site_ids.join("、") : "未授权"],
                            ["更新时间", formatTimestamp(member.updated_at)],
                          ]} />
                          <Button onClick={() => openEditMember(member)}>调整角色与范围</Button>
                        </div>
                      </Card>
                    </List.Item>
                  )}
                />
              </Card>
            ),
          },
          {
            key: "retention",
            label: "数据保留",
            children: (
              <Card title="租户数据保留策略" extra={policy ? <Tag color={policy.configured ? "green" : "orange"}>{policy.configured ? `v${policy.version}` : "未配置"}</Tag> : null}>
                <Alert
                  type="warning"
                  showIcon
                  message="期限由合同和法规确定"
                  description="这里保存租户级生命周期基线，供到期扫描、删除工作流和治理门禁读取；修改不会绕过已有 Legal Hold，也不会直接同步删除历史数据。"
                  style={{ marginBottom: 16 }}
                />
                <Form form={retentionForm} layout="vertical" onFinish={(values) => void submitRetention(values)}>
                  <Space wrap align="start">
                    <RetentionDays name="incident_days" label="故障与工单（天）" />
                    <RetentionDays name="media_days" label="媒体证据（天）" />
                    <RetentionDays name="knowledge_days" label="知识文档（天）" />
                    <RetentionDays name="training_data_days" label="训练数据（天）" />
                    <RetentionDays name="audit_days" label="安全审计（天）" />
                  </Space>
                  <Form.Item name="legal_hold" label="全租户 Legal Hold" valuePropName="checked">
                    <Switch checkedChildren="冻结删除" unCheckedChildren="正常到期" />
                  </Form.Item>
                  <Form.Item name="reason" label="变更原因" rules={[{ required: true, min: 2 }]}>
                    <Input.TextArea rows={3} />
                  </Form.Item>
                  <Button type="primary" htmlType="submit" loading={busy}>保存保留策略</Button>
                </Form>
              </Card>
            ),
          },
          {
            key: "audit",
            label: "管理审计",
            children: (
              <Card title="追加式管理记录">
                {records?.length === 0 ? <EmptyState description="尚无租户管理变更" /> : null}
                <List
                  dataSource={records}
                  renderItem={(record) => (
                    <List.Item>
                      <Card size="small" style={{ width: "100%" }}>
                        <Space wrap>
                          <Tag color="purple">{commandLabel(record.command_type)}</Tag>
                          <Typography.Text>{record.target_type} · {record.target_id}</Typography.Text>
                          <Typography.Text type="secondary">v{record.previous_version} → v{record.target_version}</Typography.Text>
                        </Space>
                        <Typography.Paragraph style={{ marginTop: 12 }}>{record.reason}</Typography.Paragraph>
                        <Typography.Text type="secondary">{record.actor_subject_id} · {formatTimestamp(record.occurred_at)}</Typography.Text>
                      </Card>
                    </List.Item>
                  )}
                />
              </Card>
            ),
          },
        ]} />
      </div>

      <Modal
        open={memberModalOpen}
        title={editingMember ? "调整成员权限" : "登记 OIDC 成员"}
        width={760}
        okText="确认保存"
        cancelText="取消"
        confirmLoading={busy}
        onCancel={() => setMemberModalOpen(false)}
        onOk={() => memberForm.submit()}
        destroyOnHidden
      >
        <Form form={memberForm} layout="vertical" onFinish={(values) => void submitMember(values)}>
          <Space wrap align="start" style={{ width: "100%" }}>
            <Form.Item name="subject_id" label="平台 Subject ID" rules={[{ required: true }]}>
              <Input disabled={Boolean(editingMember)} style={{ width: 320 }} />
            </Form.Item>
            <Form.Item name="display_name" label="成员姓名">
              <Input style={{ width: 280 }} />
            </Form.Item>
            <Form.Item name="email" label="企业邮箱">
              <Input type="email" style={{ width: 320 }} />
            </Form.Item>
            {editingMember ? (
              <Form.Item name="status" label="成员状态" rules={[{ required: true }]}>
                <Select style={{ width: 180 }} options={[
                  { value: "ACTIVE", label: "有效" },
                  { value: "SUSPENDED", label: "暂停" },
                ]} />
              </Form.Item>
            ) : null}
          </Space>
          {!editingMember ? (
            <>
              <Form.Item name="oidc_issuer" label="OIDC Issuer" rules={[{ required: true }]}>
                <Input />
              </Form.Item>
              <Form.Item name="oidc_subject" label="OIDC Subject" rules={[{ required: true }]}>
                <Input placeholder="企业 IdP 中不可变的用户 sub" />
              </Form.Item>
            </>
          ) : null}
          <Form.Item name="roles" label="岗位角色" rules={[{ required: true }]}>
            <Select mode="multiple" options={roleOptions} placeholder="至少选择一个角色" />
          </Form.Item>
          <Form.Item name="asset_ids" label="设备授权范围">
            <Select mode="multiple" options={assetOptions} placeholder="按设备授权；可与站点范围组合" />
          </Form.Item>
          <Form.Item name="site_ids" label="站点授权范围">
            <Select mode="multiple" options={siteOptions} placeholder="按站点授权" />
          </Form.Item>
          <Form.Item name="reason" label="管理原因" rules={[{ required: true, min: 2 }]}>
            <Input.TextArea rows={3} placeholder="写入不可覆盖的租户管理记录" />
          </Form.Item>
        </Form>
      </Modal>
    </AppShell>
  );
}

function RetentionDays({ name, label }: { name: keyof RetentionFormValues; label: string }) {
  return (
    <Form.Item name={name} label={label} rules={[{ required: true }]}>
      <InputNumber min={1} max={36_500} style={{ width: 180 }} />
    </Form.Item>
  );
}

function roleLabel(role: string): string {
  return {
    customer_contact: "客户联系人",
    field_engineer: "现场工程师",
    after_sales_engineer: "售后工程师",
    domain_expert: "领域专家",
    tenant_admin: "租户管理员",
    security_auditor: "安全审计员",
    data_steward: "数据负责人",
    annotation_admin: "标注管理员",
    model_engineer: "模型工程师",
    model_evaluator: "模型评测员",
    model_release_approver: "模型发布审批人",
    model_release_operator: "模型发布操作员",
    platform_operator: "平台运维",
  }[role] ?? role;
}

function commandLabel(command: string): string {
  return {
    CREATE_MEMBER: "登记成员",
    UPDATE_MEMBER: "调整成员权限",
    UPDATE_RETENTION: "修改保留策略",
  }[command] ?? command;
}

function formatTimestamp(value: string): string {
  const timestamp = new Date(value);
  return Number.isNaN(timestamp.getTime())
    ? value
    : timestamp.toLocaleString("zh-CN", { hour12: false });
}
