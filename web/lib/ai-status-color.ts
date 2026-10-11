export function experimentStatusColor(status: string) {
  if (status === "COMPLETED") return "green";
  if (status === "RUNNING") return "processing";
  if (status.includes("FAILED")) return "red";
  return "default";
}

export function releaseStatusColor(status: string): string {
  if (["PRODUCTION", "CANDIDATE"].includes(status)) return "green";
  if (["REJECTED", "ROLLED_BACK"].includes(status)) return "red";
  if (["APPROVAL_PENDING", "VALIDATING", "CANARY", "SHADOW"].includes(status)) return "gold";
  return "blue";
}
