"use client";

import { Alert, Button, Card, Space, Tag, Typography } from "antd";
import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

export type FieldPwaStatus =
  | "unsupported"
  | "online-only"
  | "installable"
  | "installed"
  | "update-ready";

interface InstallChoice {
  outcome: "accepted" | "dismissed";
  platform: string;
}

interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<void>;
  readonly userChoice: Promise<InstallChoice>;
}

export interface FieldPwaContextValue {
  status: FieldPwaStatus;
  requestInstall(): Promise<boolean>;
  applyUpdate(): boolean;
}

const safeDefault: FieldPwaContextValue = {
  status: "online-only",
  requestInstall: async () => false,
  applyUpdate: () => false,
};

const FieldPwaContext = createContext<FieldPwaContextValue>(safeDefault);

export function FieldPwaProvider({
  children,
  reloadPage = () => window.location.reload(),
}: {
  children: ReactNode;
  reloadPage?: () => void;
}) {
  const [status, setStatus] = useState<FieldPwaStatus>("online-only");
  const installPromptRef = useRef<BeforeInstallPromptEvent | undefined>(undefined);
  const waitingWorkerRef = useRef<ServiceWorker | undefined>(undefined);
  const reloadRequestedRef = useRef(false);
  const reloadPerformedRef = useRef(false);
  const reloadPageRef = useRef(reloadPage);

  useEffect(() => {
    const standalone = isStandalone();
    if (!navigator.serviceWorker) {
      setStatus(standalone ? "installed" : "unsupported");
      return;
    }

    let disposed = false;
    let registration: ServiceWorkerRegistration | undefined;
    let installingWorker: ServiceWorker | null = null;

    const showWaitingWorker = (worker: ServiceWorker | null | undefined) => {
      if (!worker || disposed) return;
      waitingWorkerRef.current = worker;
      setStatus("update-ready");
    };
    const handleInstallingState = () => {
      if (installingWorker?.state === "installed") {
        showWaitingWorker(registration?.waiting ?? installingWorker);
      }
    };
    const handleUpdateFound = () => {
      if (installingWorker) {
        installingWorker.removeEventListener("statechange", handleInstallingState);
      }
      installingWorker = registration?.installing ?? null;
      installingWorker?.addEventListener("statechange", handleInstallingState);
    };
    const handleInstallPrompt = (event: Event) => {
      const prompt = event as BeforeInstallPromptEvent;
      event.preventDefault();
      installPromptRef.current = prompt;
      if (!isStandalone() && !waitingWorkerRef.current) setStatus("installable");
    };
    const handleInstalled = () => {
      installPromptRef.current = undefined;
      if (!waitingWorkerRef.current) setStatus("installed");
    };
    const handleControllerChange = () => {
      if (!reloadRequestedRef.current || reloadPerformedRef.current) return;
      reloadPerformedRef.current = true;
      reloadPageRef.current();
    };

    setStatus(standalone ? "installed" : "online-only");
    window.addEventListener("beforeinstallprompt", handleInstallPrompt);
    window.addEventListener("appinstalled", handleInstalled);
    navigator.serviceWorker.addEventListener("controllerchange", handleControllerChange);

    void navigator.serviceWorker.register("/field-sw.js", {
      scope: "/field/",
      updateViaCache: "none",
    }).then((registered) => {
      if (disposed) return;
      registration = registered;
      if (registered.waiting) showWaitingWorker(registered.waiting);
      registered.addEventListener("updatefound", handleUpdateFound);
    }).catch(() => {
      if (!disposed && !standalone) setStatus("online-only");
    });

    return () => {
      disposed = true;
      window.removeEventListener("beforeinstallprompt", handleInstallPrompt);
      window.removeEventListener("appinstalled", handleInstalled);
      navigator.serviceWorker.removeEventListener(
        "controllerchange",
        handleControllerChange,
      );
      registration?.removeEventListener("updatefound", handleUpdateFound);
      installingWorker?.removeEventListener("statechange", handleInstallingState);
      installPromptRef.current = undefined;
      waitingWorkerRef.current = undefined;
    };
  }, []);

  const requestInstall = useCallback(async () => {
    const prompt = installPromptRef.current;
    if (!prompt) return false;
    installPromptRef.current = undefined;
    await prompt.prompt();
    const choice = await prompt.userChoice;
    if (choice.outcome === "accepted") {
      setStatus("installed");
      return true;
    }
    setStatus("online-only");
    return false;
  }, []);

  const applyUpdate = useCallback(() => {
    const waiting = waitingWorkerRef.current;
    if (!waiting) return false;
    waitingWorkerRef.current = undefined;
    reloadRequestedRef.current = true;
    waiting.postMessage({ type: "SKIP_WAITING" });
    return true;
  }, []);

  return (
    <FieldPwaContext.Provider value={{ status, requestInstall, applyUpdate }}>
      {children}
    </FieldPwaContext.Provider>
  );
}

export function useFieldPwa(): FieldPwaContextValue {
  return useContext(FieldPwaContext);
}

export function FieldPwaStatusPanel() {
  const { status, requestInstall, applyUpdate } = useFieldPwa();
  const [submitting, setSubmitting] = useState(false);
  const presentation = STATUS_PRESENTATION[status];

  const install = async () => {
    setSubmitting(true);
    try {
      await requestInstall();
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Card size="small" title="现场应用安装与更新">
      <div className="page-stack">
        <Alert
          type={presentation.type}
          showIcon
          title={presentation.message}
          description={presentation.description}
          action={status === "installable" ? (
            <Button loading={submitting} onClick={() => void install()}>
              安装到设备
            </Button>
          ) : status === "update-ready" ? (
            <Button type="primary" onClick={applyUpdate}>应用更新</Button>
          ) : undefined}
        />
        <Space wrap>
          <Tag color="blue">作用域 /field/</Tag>
          <Tag>只缓存静态安全壳</Tag>
          <Tag>不启用后台同步</Tag>
        </Space>
        <Typography.Text type="secondary">
          PWA 只提供可安装的现场入口和断网安全页，不授予离线身份或业务权限；
          工单、工作包和待同步事实仍由现有完整性检查与服务端复检控制。
        </Typography.Text>
      </div>
    </Card>
  );
}

function isStandalone(): boolean {
  if (typeof window === "undefined") return false;
  const iosNavigator = navigator as Navigator & { standalone?: boolean };
  return window.matchMedia("(display-mode: standalone)").matches
    || iosNavigator.standalone === true;
}

const STATUS_PRESENTATION: Record<
  FieldPwaStatus,
  { type: "success" | "info" | "warning"; message: string; description: string }
> = {
  unsupported: {
    type: "warning",
    message: "当前浏览器不支持现场应用安装",
    description: "在线工单、受治理离线工作包和手动同步能力仍可正常使用。",
  },
  "online-only": {
    type: "info",
    message: "现场工作台当前以浏览器模式运行",
    description: "浏览器提供安装提示后，页面才会显示安装按钮，不会伪造安装成功。",
  },
  installable: {
    type: "info",
    message: "现场工作台可以安装到当前设备",
    description: "安装必须由你显式确认，只添加安全应用壳，不复制工单或身份数据。",
  },
  installed: {
    type: "success",
    message: "现场应用壳已安装",
    description: "断网重新打开时只显示无业务数据的安全离线页，恢复网络后重新鉴权。",
  },
  "update-ready": {
    type: "warning",
    message: "现场应用有等待中的更新",
    description: "更新不会自动刷新当前操作；确认后才切换版本并仅重载一次页面。",
  },
};
