import { isStableResourceId } from "@/lib/value-guards";

export type ExperimentCreateTarget = {
  datasetSnapshotId: string;
};

const CREATE_PARAMETER = "create_experiment";
const SNAPSHOT_PARAMETER = "dataset_snapshot_id";
const DETAIL_PARAMETER = "experiment_id";
const EVALUATION_DETAIL_PARAMETER = "evaluation_id";

export function readExperimentCreateTarget(
  search: string | URLSearchParams,
): ExperimentCreateTarget | undefined {
  const parameters = typeof search === "string" ? new URLSearchParams(search) : search;
  if (parameters.getAll(CREATE_PARAMETER).length !== 1) return undefined;
  if (parameters.getAll(SNAPSHOT_PARAMETER).length !== 1) return undefined;
  if (parameters.get(CREATE_PARAMETER) !== "1") return undefined;
  const datasetSnapshotId = parameters.get(SNAPSHOT_PARAMETER)?.trim() ?? "";
  if (!isStableResourceId(datasetSnapshotId)) return undefined;
  return { datasetSnapshotId };
}

export function buildExperimentCreateUrl(datasetSnapshotId: string): string {
  if (!isStableResourceId(datasetSnapshotId)) {
    throw new Error("Dataset snapshot ID is invalid");
  }
  const parameters = new URLSearchParams({
    [CREATE_PARAMETER]: "1",
    [SNAPSHOT_PARAMETER]: datasetSnapshotId,
  });
  return `/ai/experiments?${parameters.toString()}`;
}

export function removeExperimentCreateTarget(currentHref: string): string {
  const url = new URL(currentHref);
  url.searchParams.delete(CREATE_PARAMETER);
  url.searchParams.delete(SNAPSHOT_PARAMETER);
  return `${url.pathname}${url.search}${url.hash}`;
}

export function readExperimentDetailTarget(
  search: string | URLSearchParams,
): string | undefined {
  const parameters = typeof search === "string" ? new URLSearchParams(search) : search;
  if (parameters.getAll(DETAIL_PARAMETER).length !== 1) return undefined;
  if (parameters.has(CREATE_PARAMETER) ||
      parameters.has(SNAPSHOT_PARAMETER) ||
      parameters.has(EVALUATION_DETAIL_PARAMETER)) return undefined;
  const experimentId = parameters.get(DETAIL_PARAMETER)?.trim() ?? "";
  return isStableResourceId(experimentId) ? experimentId : undefined;
}

export function buildExperimentDetailUrl(
  currentHref: string,
  experimentId?: string,
): string {
  if (experimentId !== undefined && !isStableResourceId(experimentId)) {
    throw new Error("Training experiment ID is invalid");
  }
  const url = new URL(currentHref);
  url.searchParams.delete(CREATE_PARAMETER);
  url.searchParams.delete(SNAPSHOT_PARAMETER);
  url.searchParams.delete(DETAIL_PARAMETER);
  url.searchParams.delete(EVALUATION_DETAIL_PARAMETER);
  if (experimentId) url.searchParams.set(DETAIL_PARAMETER, experimentId);
  return `${url.pathname}${url.search}${url.hash}`;
}

export function readEvaluationDetailTarget(
  search: string | URLSearchParams,
): string | undefined {
  const parameters = typeof search === "string" ? new URLSearchParams(search) : search;
  if (parameters.getAll(EVALUATION_DETAIL_PARAMETER).length !== 1) return undefined;
  if (parameters.has(CREATE_PARAMETER) ||
      parameters.has(SNAPSHOT_PARAMETER) ||
      parameters.has(DETAIL_PARAMETER)) return undefined;
  const evaluationId = parameters.get(EVALUATION_DETAIL_PARAMETER)?.trim() ?? "";
  return isStableResourceId(evaluationId) ? evaluationId : undefined;
}

export function buildEvaluationDetailUrl(
  currentHref: string,
  evaluationId?: string,
): string {
  if (evaluationId !== undefined && !isStableResourceId(evaluationId)) {
    throw new Error("Model evaluation ID is invalid");
  }
  const url = new URL(currentHref);
  url.searchParams.delete(CREATE_PARAMETER);
  url.searchParams.delete(SNAPSHOT_PARAMETER);
  url.searchParams.delete(DETAIL_PARAMETER);
  url.searchParams.delete(EVALUATION_DETAIL_PARAMETER);
  if (evaluationId) url.searchParams.set(EVALUATION_DETAIL_PARAMETER, evaluationId);
  return `${url.pathname}${url.search}${url.hash}`;
}
