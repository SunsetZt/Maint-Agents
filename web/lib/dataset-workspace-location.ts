import { isStableResourceId } from "@/lib/value-guards";

export type DatasetWorkspaceTarget = {
  kind: "candidate" | "run" | "snapshot";
  id: string;
};

const TARGET_PARAMETERS = {
  candidate: "candidate_id",
  run: "run_id",
  snapshot: "snapshot_id",
} as const;

export function readDatasetWorkspaceTarget(
  search: string | URLSearchParams,
): DatasetWorkspaceTarget | undefined {
  const parameters = typeof search === "string" ? new URLSearchParams(search) : search;
  if (Object.values(TARGET_PARAMETERS).some((parameter) =>
    parameters.getAll(parameter).length > 1)) return undefined;
  const targets = Object.entries(TARGET_PARAMETERS)
    .map(([kind, parameter]) => ({
      kind: kind as DatasetWorkspaceTarget["kind"],
      id: parameters.get(parameter)?.trim() ?? "",
    }))
    .filter((target) => target.id.length > 0);
  if (targets.length !== 1 || !isStableResourceId(targets[0].id)) return undefined;
  return targets[0];
}

export function buildDatasetWorkspaceUrl(
  currentHref: string,
  target?: DatasetWorkspaceTarget,
): string {
  const url = new URL(currentHref);
  for (const parameter of Object.values(TARGET_PARAMETERS)) {
    url.searchParams.delete(parameter);
  }
  if (target) url.searchParams.set(TARGET_PARAMETERS[target.kind], target.id);
  return `${url.pathname}${url.search}${url.hash}`;
}

export function datasetWorkspaceTargetKey(target?: DatasetWorkspaceTarget): string {
  return target ? `${target.kind}:${target.id}` : "none";
}
