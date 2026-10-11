import {
  appendFieldWorkOrderEntry,
  type FieldEntryInput,
} from "@/lib/api/client";

const STORAGE_KEY = "industrial-ops.field-entry-queue.v1";

export type PendingFieldEntry = {
  workOrderId: string;
  input: FieldEntryInput;
  queuedAt: string;
};

export function createFieldOperationId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `field-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function pendingFieldEntries(workOrderId?: string): PendingFieldEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const value: unknown = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "[]");
    if (!Array.isArray(value)) return [];
    return value
      .filter(isPendingFieldEntry)
      .filter((item) => !workOrderId || item.workOrderId === workOrderId);
  } catch {
    return [];
  }
}

export function queueFieldEntry(workOrderId: string, input: FieldEntryInput): void {
  if (typeof window === "undefined") return;
  const current = pendingFieldEntries();
  const duplicate = current.some((item) =>
    item.workOrderId === workOrderId
    && item.input.client_operation_id === input.client_operation_id);
  if (duplicate) return;
  current.push({ workOrderId, input, queuedAt: new Date().toISOString() });
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(current));
}

export async function syncFieldEntries(
  workOrderId?: string,
  send: typeof appendFieldWorkOrderEntry = appendFieldWorkOrderEntry,
): Promise<{ synced: number; remaining: number; error?: unknown }> {
  const all = pendingFieldEntries();
  const selected = all.filter((item) => !workOrderId || item.workOrderId === workOrderId);
  let synced = 0;
  for (const item of selected) {
    try {
      await send(item.workOrderId, item.input);
    } catch (error) {
      return {
        synced,
        remaining: pendingFieldEntries(workOrderId).length,
        error,
      };
    }
    removeFieldEntry(item.workOrderId, item.input.client_operation_id);
    synced += 1;
  }
  return { synced, remaining: pendingFieldEntries(workOrderId).length };
}

function removeFieldEntry(workOrderId: string, clientOperationId: string): void {
  if (typeof window === "undefined") return;
  const remaining = pendingFieldEntries().filter((item) => !(
    item.workOrderId === workOrderId
    && item.input.client_operation_id === clientOperationId
  ));
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(remaining));
}

function isPendingFieldEntry(value: unknown): value is PendingFieldEntry {
  if (!value || typeof value !== "object") return false;
  const item = value as Partial<PendingFieldEntry>;
  return Boolean(
    typeof item.workOrderId === "string"
    && item.input
    && typeof item.input === "object"
    && typeof item.input.client_operation_id === "string"
    && typeof item.queuedAt === "string",
  );
}
