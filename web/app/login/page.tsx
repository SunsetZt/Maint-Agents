"use client";

import { Alert, Button, Card, Typography } from "antd";
import { signIn } from "next-auth/react";
import { useEffect, useState } from "react";

const AUTH_ERROR_MESSAGES: Record<string, string> = {
  AccessDenied: "当前企业账号没有访问此平台的权限。",
  Configuration: "企业身份服务配置异常，请联系平台管理员。",
  OAuthCallback: "企业身份服务拒绝了登录回调，请稍后重试。",
  OAuthSignin: "无法连接企业身份服务，请稍后重试。",
  SessionExpired: "企业登录会话已过期，请重新登录。",
};

export default function LoginPage() {
  const [authError, setAuthError] = useState<string>();
  const [signingIn, setSigningIn] = useState(false);

  useEffect(() => {
    const errorCode = new URLSearchParams(window.location.search).get("error");
    if (errorCode) {
      setAuthError(
        AUTH_ERROR_MESSAGES[errorCode] ?? "企业身份登录失败，请稍后重试。",
      );
    }
  }, []);

  async function handleSignIn() {
    setAuthError(undefined);
    setSigningIn(true);
    try {
      const requestedCallback = new URLSearchParams(window.location.search).get(
        "callbackUrl",
      );
      const callbackUrl =
        requestedCallback?.startsWith("/") && !requestedCallback.startsWith("//")
          ? requestedCallback
          : "/workspace";
      await signIn("keycloak", { callbackUrl }, { prompt: "login" });
    } catch {
      setAuthError("无法发起企业身份登录，请检查网络后重试。");
      setSigningIn(false);
    }
  }

  return (
    <main className="app-content">
      <Card title="企业身份登录" style={{ maxWidth: 520, margin: "10vh auto" }}>
        <div className="page-stack">
          <Typography.Paragraph>
            使用企业 OIDC 身份进入授权设备工作区。平台不会在浏览器中保存 API Token。
          </Typography.Paragraph>
          {authError ? (
            <Alert
              type="error"
              showIcon
              message="企业身份登录失败"
              description={authError}
            />
          ) : null}
          <Button
            type="primary"
            size="large"
            loading={signingIn}
            onClick={handleSignIn}
          >
            使用企业账号登录
          </Button>
        </div>
      </Card>
    </main>
  );
}
