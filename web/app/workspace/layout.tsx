import type { ReactNode } from "react";

import { requireSession } from "@/lib/auth/require-session";

export default async function WorkspaceLayout({ children }: { children: ReactNode }) {
  await requireSession();
  return children;
}
