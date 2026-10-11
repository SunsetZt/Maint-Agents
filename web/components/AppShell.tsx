"use client";

import { Button, Layout, Menu, Typography } from "antd";
import type { MenuProps } from "antd";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { signOut } from "next-auth/react";
import type { ReactNode } from "react";

const { Header, Content } = Layout;

const navigationItems: MenuProps["items"] = [
  {
    key: "/workspace",
    label: <Link href="/workspace">工作台</Link>,
  },
  {
    key: "after-sales",
    label: "售后业务",
    children: [
      { key: "/incidents", label: <Link href="/incidents">故障受理</Link> },
      { key: "/field", label: <Link href="/field">现场服务</Link> },
      {
        key: "/expert/collaborations",
        label: <Link href="/expert/collaborations">专家协作</Link>,
      },
      { key: "/portal", label: <Link href="/portal">客户门户</Link> },
      {
        key: "/entitlements",
        label: <Link href="/entitlements">服务权益</Link>,
      },
      { key: "/parts", label: <Link href="/parts">备件查询</Link> },
      { key: "/approvals", label: <Link href="/approvals">审批箱</Link> },
      {
        key: "/reconciliations",
        label: <Link href="/reconciliations">副作用对账</Link>,
      },
      { key: "/dispatch", label: <Link href="/dispatch">调度中心</Link> },
      {
        key: "/work-orders",
        label: <Link href="/work-orders">工单运营</Link>,
      },
    ],
  },
  {
    key: "ai-platform",
    label: "AI 中台",
    children: [
      {
        key: "/ai/datasets",
        label: <Link href="/ai/datasets">治理数据</Link>,
      },
      {
        key: "/ai/knowledge",
        label: <Link href="/ai/knowledge">企业知识</Link>,
      },
      {
        key: "/ai/knowledge/graph",
        label: <Link href="/ai/knowledge/graph">因果图谱</Link>,
      },
      {
        key: "/ai/knowledge/search",
        label: <Link href="/ai/knowledge/search">企业搜索</Link>,
      },
      {
        key: "/ai/knowledge/external",
        label: <Link href="/ai/knowledge/external">外部参考</Link>,
      },
      {
        key: "/ai/experiments",
        label: <Link href="/ai/experiments">训练评测</Link>,
      },
      {
        key: "/ai/releases",
        label: <Link href="/ai/releases">模型发布</Link>,
      },
      {
        key: "/ai/enterprise-assets",
        label: <Link href="/ai/enterprise-assets">企业资产</Link>,
      },
      {
        key: "/ai/prompts",
        label: <Link href="/ai/prompts">Prompt 治理</Link>,
      },
      {
        key: "/ai/memories",
        label: <Link href="/ai/memories">Memory 治理</Link>,
      },
    ],
  },
  {
    key: "collaborative-intelligence",
    label: "协作智能",
    children: [
      {
        key: "/collaboration",
        label: <Link href="/collaboration">供应商协作</Link>,
      },
      {
        key: "/maintenance-planning",
        label: <Link href="/maintenance-planning">多 Agent 会审</Link>,
      },
      {
        key: "/device-families",
        label: <Link href="/device-families">设备族准入</Link>,
      },
      {
        key: "/predictive-maintenance",
        label: <Link href="/predictive-maintenance">预测维护</Link>,
      },
    ],
  },
  {
    key: "platform-governance",
    label: "平台治理",
    children: [
      { key: "/ops", label: <Link href="/ops">运行运维</Link> },
      {
        key: "/ops/service-performance",
        label: <Link href="/ops/service-performance">服务绩效</Link>,
      },
      { key: "/ops/gpu", label: <Link href="/ops/gpu">GPU 运维</Link> },
      { key: "/ops/cost", label: <Link href="/ops/cost">成本治理</Link> },
      {
        key: "/ops/traces",
        label: <Link href="/ops/traces">链路观测</Link>,
      },
      { key: "/ops/tools", label: <Link href="/ops/tools">工具治理</Link> },
      {
        key: "/security",
        label: <Link href="/security">安全审计</Link>,
      },
      { key: "/tenant", label: <Link href="/tenant">租户治理</Link> },
    ],
  },
];

const navigationPaths = [
  "/workspace",
  "/incidents",
  "/field",
  "/expert/collaborations",
  "/portal",
  "/entitlements",
  "/parts",
  "/approvals",
  "/reconciliations",
  "/dispatch",
  "/work-orders",
  "/ai/datasets",
  "/ai/knowledge/graph",
  "/ai/knowledge/search",
  "/ai/knowledge/external",
  "/ai/knowledge",
  "/ai/experiments",
  "/ai/releases",
  "/ai/enterprise-assets",
  "/ai/prompts",
  "/ai/memories",
  "/collaboration",
  "/maintenance-planning",
  "/device-families",
  "/predictive-maintenance",
  "/ops/service-performance",
  "/ops/gpu",
  "/ops/cost",
  "/ops/traces",
  "/ops/tools",
  "/ops",
  "/security",
  "/tenant",
] as const;

function selectedNavigationPath(pathname: string) {
  return (
    navigationPaths.find(
      (path) => pathname === path || pathname.startsWith(`${path}/`),
    ) ?? "/workspace"
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();

  return (
    <Layout className="app-shell">
      <Header className="app-header">
        <Typography.Text className="app-brand">
          <span className="app-brand-full">
            工业设备智能运维与售后 Agent 平台
          </span>
          <span className="app-brand-short">工业运维 Agent</span>
        </Typography.Text>
        <Menu
          aria-label="主导航"
          className="app-navigation"
          items={navigationItems}
          mode="horizontal"
          overflowedIndicator="更多"
          selectedKeys={[selectedNavigationPath(pathname)]}
          theme="dark"
          triggerSubMenuAction="click"
        />
        <Button
          aria-label="退出当前企业账号"
          className="app-signout"
          type="text"
          onClick={() => void signOut({ callbackUrl: "/login" })}
        >
          退出登录
        </Button>
      </Header>
      <Content className="app-content">{children}</Content>
    </Layout>
  );
}
