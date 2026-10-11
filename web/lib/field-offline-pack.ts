import type { WorkOrderOfflinePack } from "@/lib/api/client";
import { isRecord } from "@/lib/value-guards";

const DATABASE_NAME = "industrial-ops-field-offline";
const DATABASE_VERSION = 1;
const STORE_NAME = "work-order-packs";
const SCHEMA_VERSION = "field-offline-pack-v1";

export interface FieldOfflinePackStorage {
  get(key: string): Promise<unknown>;
  put(key: string, value: WorkOrderOfflinePack): Promise<void>;
  delete(key: string): Promise<void>;
}

class IndexedDbFieldOfflinePackStorage implements FieldOfflinePackStorage {
  async get(key: string): Promise<unknown> {
    return this.run("readonly", (store) => store.get(key));
  }

  async put(key: string, value: WorkOrderOfflinePack): Promise<void> {
    await this.run("readwrite", (store) => store.put(value, key));
  }

  async delete(key: string): Promise<void> {
    await this.run("readwrite", (store) => store.delete(key));
  }

  private async run<T>(
    mode: IDBTransactionMode,
    operation: (store: IDBObjectStore) => IDBRequest<T>,
  ): Promise<T> {
    const database = await openDatabase();
    try {
      const transaction = database.transaction(STORE_NAME, mode);
      const request = operation(transaction.objectStore(STORE_NAME));
      const result = await requestResult(request);
      await transactionResult(transaction);
      return result;
    } finally {
      database.close();
    }
  }
}

const browserStorage = new IndexedDbFieldOfflinePackStorage();

export async function saveFieldOfflinePack(
  pack: WorkOrderOfflinePack,
  storage: FieldOfflinePackStorage = browserStorage,
): Promise<void> {
  if (
    !isPackShape(pack)
    || !isPersistablePack(pack, new Date())
    || !await contentHashMatches(pack)
  ) {
    throw new Error("offline_pack_integrity_failed");
  }
  await storage.put(storageKey(pack.subject_id, pack.work_order_id), pack);
}

export async function loadFieldOfflinePack(
  subjectId: string,
  workOrderId: string,
  storage: FieldOfflinePackStorage = browserStorage,
  now: Date = new Date(),
): Promise<WorkOrderOfflinePack | undefined> {
  const key = storageKey(subjectId, workOrderId);
  const candidate = await storage.get(key);
  if (
    !isPackShape(candidate)
    || candidate.subject_id !== subjectId
    || candidate.work_order_id !== workOrderId
    || !isPersistablePack(candidate, now)
    || !await contentHashMatches(candidate)
  ) {
    await storage.delete(key);
    return undefined;
  }
  return candidate;
}

export async function removeFieldOfflinePack(
  subjectId: string,
  workOrderId: string,
  storage: FieldOfflinePackStorage = browserStorage,
): Promise<void> {
  await storage.delete(storageKey(subjectId, workOrderId));
}

async function contentHashMatches(pack: WorkOrderOfflinePack): Promise<boolean> {
  if (!/^sha256:[0-9a-f]{64}$/.test(pack.content_hash)) return false;
  const bytes = new TextEncoder().encode(canonicalJson(pack.snapshot));
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  const actual = Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")).join("");
  return pack.content_hash === `sha256:${actual}`;
}

function canonicalJson(value: unknown, key?: string): string {
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) =>
      `${JSON.stringify(key)}:${canonicalJson(record[key], key)}`).join(",")}}`;
  }
  if (typeof value === "number" && key === "confidence") {
    return pythonFloatJson(value);
  }
  return JSON.stringify(value);
}

function pythonFloatJson(value: number): string {
  if (Object.is(value, -0)) return "-0.0";
  if (Number.isInteger(value)) return `${value}.0`;
  if (Math.abs(value) >= 0.0001) return String(value);
  const [coefficient, exponentText] = value.toExponential().split("e");
  const exponent = Number(exponentText);
  const sign = exponent < 0 ? "-" : "+";
  return `${coefficient}e${sign}${Math.abs(exponent).toString().padStart(2, "0")}`;
}

function isPackShape(value: unknown): value is WorkOrderOfflinePack {
  if (!isRecord(value)) return false;
  const pack = value;
  return Boolean(
    pack.schema_version === SCHEMA_VERSION
    && typeof pack.pack_id === "string"
    && (pack.status === "ACTIVE" || pack.status === "SUPERSEDED" || pack.status === "REVOKED")
    && typeof pack.subject_id === "string"
    && pack.subject_id.trim().length > 0
    && typeof pack.work_order_id === "string"
    && pack.work_order_id.trim().length > 0
    && isInteger(pack.work_order_version)
    && typeof pack.asset_id === "string"
    && isInteger(pack.asset_version)
    && typeof pack.assignment_id === "string"
    && typeof pack.assignment_assigned_at === "string"
    && typeof pack.issued_at === "string"
    && typeof pack.expires_at === "string"
    && typeof pack.content_hash === "string"
    && (pack.revoked_at === null || typeof pack.revoked_at === "string")
    && isInteger(pack.version)
    && isSnapshotShape(pack.snapshot),
  );
}

function isSnapshotShape(value: unknown): value is WorkOrderOfflinePack["snapshot"] {
  if (!isRecord(value)) return false;
  const workOrder = value.work_order;
  const asset = value.asset;
  const incident = value.incident;
  const site = value.site;
  const diagnosis = value.diagnosis;
  const citations = value.citations;
  return Boolean(
    isRecord(workOrder)
    && typeof workOrder.work_order_id === "string"
    && typeof workOrder.incident_id === "string"
    && typeof workOrder.status === "string"
    && typeof workOrder.priority === "string"
    && isNullableString(workOrder.sla_due_at)
    && isNullableString(workOrder.service_window_start)
    && isNullableString(workOrder.service_window_end)
    && isInteger(workOrder.version)
    && isRecord(asset)
    && typeof asset.asset_id === "string"
    && isNullableString(asset.display_name)
    && isNullableString(asset.model_code)
    && isNullableString(asset.serial_number)
    && isNullableString(asset.lifecycle_status)
    && isInteger(asset.version)
    && (site === null || (
      isRecord(site)
      && typeof site.site_id === "string"
      && typeof site.site_name === "string"
    ))
    && isRecord(incident)
    && typeof incident.incident_id === "string"
    && isNullableString(incident.description)
    && isNullableString(incident.severity)
    && isNullableString(incident.category)
    && (diagnosis === null || isDiagnosisShape(diagnosis))
    && Array.isArray(citations)
    && citations.every(isCitationShape)
    && Array.isArray(value.legal_actions)
    && value.legal_actions.length === 1
    && value.legal_actions[0] === "READ_OFFLINE_SNAPSHOT"
  );
}

function isDiagnosisShape(value: unknown): boolean {
  return Boolean(
    isRecord(value)
    && typeof value.diagnosis_run_id === "string"
    && typeof value.status === "string"
    && isInteger(value.version)
    && isNullableString(value.conclusion)
    && isStringArray(value.next_checks)
    && isStringArray(value.recommended_actions)
    && (value.confidence === null || (
      typeof value.confidence === "number"
      && Number.isFinite(value.confidence)
      && value.confidence >= 0
      && value.confidence <= 1
    ))
    && isStringArray(value.citation_ids)
  );
}

function isCitationShape(value: unknown): boolean {
  return Boolean(
    isRecord(value)
    && typeof value.citation_id === "string"
    && typeof value.anchor_kind === "string"
    && (value.page_number === null || isInteger(value.page_number))
    && isNullableString(value.excerpt)
    && typeof value.excerpt_checksum === "string"
  );
}

function isNullableString(value: unknown): boolean {
  return value === null || typeof value === "string";
}

function isStringArray(value: unknown): boolean {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isInteger(value: unknown): boolean {
  return typeof value === "number" && Number.isInteger(value);
}

function isPersistablePack(pack: WorkOrderOfflinePack, now: Date): boolean {
  const issuedAt = Date.parse(pack.issued_at);
  const expiresAt = Date.parse(pack.expires_at);
  return Boolean(
    pack.status === "ACTIVE"
    && pack.revoked_at === null
    && Number.isFinite(issuedAt)
    && Number.isFinite(expiresAt)
    && issuedAt < expiresAt
    && expiresAt > now.getTime()
    && pack.snapshot.work_order.work_order_id === pack.work_order_id
    && pack.snapshot.work_order.version === pack.work_order_version
    && pack.snapshot.asset.asset_id === pack.asset_id
    && pack.snapshot.asset.version === pack.asset_version
  );
}

function storageKey(subjectId: string, workOrderId: string): string {
  if (subjectId.includes("\u0000") || workOrderId.includes("\u0000")) {
    throw new Error("offline_pack_binding_invalid");
  }
  return `${subjectId}\u0000${workOrderId}`;
}

function openDatabase(): Promise<IDBDatabase> {
  if (!globalThis.indexedDB) {
    return Promise.reject(new Error("indexeddb_unavailable"));
  }
  return new Promise((resolve, reject) => {
    const request = globalThis.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) {
        request.result.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("indexeddb_open_failed"));
    request.onblocked = () => reject(new Error("indexeddb_open_blocked"));
  });
}

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("indexeddb_request_failed"));
  });
}

function transactionResult(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(
      transaction.error ?? new Error("indexeddb_transaction_aborted"),
    );
    transaction.onerror = () => reject(
      transaction.error ?? new Error("indexeddb_transaction_failed"),
    );
  });
}
