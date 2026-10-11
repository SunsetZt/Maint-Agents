import type { Metadata } from "next";
import type { ReactNode } from "react";

import { FieldPwaProvider } from "@/components/m3/FieldPwaProvider";

export const metadata: Metadata = {
  title: "现场工程师工作台",
  manifest: "/manifest.webmanifest",
  themeColor: "#0b5cad",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "现场运维",
  },
};

export default function FieldLayout({ children }: { children: ReactNode }) {
  return <FieldPwaProvider>{children}</FieldPwaProvider>;
}
