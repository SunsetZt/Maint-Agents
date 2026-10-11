export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function isStableResourceId(value: string): boolean {
  return /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value);
}

// Input normalization only; callers keep their existing validation rules.
export function normalizeStableId(value: string): string | undefined {
  return value.trim() || undefined;
}
