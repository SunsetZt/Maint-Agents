import { redirect } from "next/navigation";

import { auth } from "@/auth";
import { WorkspaceDashboard } from "@/components/WorkspaceDashboard";

export default async function WorkspacePage() {
  const session = await auth();
  if (!session?.user) redirect("/login");

  return (
    <WorkspaceDashboard
      tenantId={session.user.tenantId}
      roles={session.user.roles ?? []}
    />
  );
}
