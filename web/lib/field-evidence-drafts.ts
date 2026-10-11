import { uploadFieldEvidence } from "@/lib/api/client";
import { isRecord } from "@/lib/value-guards";

const DATABASE_NAME = "industrial-ops-field-evidence";
const DATABASE_VERSION = 1;
const STORE_NAME = "drafts";
const SCHEMA_VERSION = "field-evidence-draft-v1" as const;
const MAX_BYTES = 10 * 1024 * 1024;
const RETENTION_MS = 7 * 24 * 60 * 60 * 1000;
const FUTURE_SKEW_MS = 5 * 60 * 1000;
const SUPPORTED_MIME_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "video/mp4",
  "video/webm",
]);

export type FieldEvidenceDraft = {
  schema_version: typeof SCHEMA_VERSION;
  subject_id: string;
  work_order_id: string;
  client_operation_id: string;
  created_at: string;
  declared_mime: string;
  size_bytes: number;
  content_hash: string;
  blob: Blob;
};

export type FieldEvidenceDraftStorageEntry = {
  key: string;
  value: unknown;
};

export interface FieldEvidenceDraftStorage {
  list(): Promise<FieldEvidenceDraftStorageEntry[]>;
  put(key: string, value: FieldEvidenceDraft): Promise<void>;
  delete(key: string): Promise<void>;
}

export type SaveFieldEvidenceDraftInput = {
  subjectId: string;
  workOrderId: string;
  clientOperationId: string;
  createdAt?: Date;
  file: File;
};

export type FieldEvidenceDraftSender = (
  workOrderId: string,
  workOrderVersion: number,
  clientOperationId: string,
  contentHash: string,
  declaredMime: string,
  content: Blob,
) => Promise<unknown>;

export type SyncFieldEvidenceDraftsInput = {
  subjectId: string;
  workOrderId: string;
  workOrderVersion: number;
  online: boolean;
  authoritativeWorkLoaded: boolean;
  storage?: FieldEvidenceDraftStorage;
  send?: FieldEvidenceDraftSender;
  now?: Date;
};

export type SyncFieldEvidenceDraftsResult = {
  synced: number;
  remaining: number;
  blockedReason?: "offline" | "authoritative_work_order_required";
  error?: unknown;
};

const browserStorage: FieldEvidenceDraftStorage = {
  async list() {
    const database = await openDatabase();
    try {
      const transaction = database.transaction(STORE_NAME, "readonly");
      const completed = transactionResult(transaction);
      const store = transaction.objectStore(STORE_NAME);
      const [keys, values] = await Promise.all([
        requestResult(store.getAllKeys()),
        requestResult(store.getAll()),
      ]);
      await completed;
      return keys.map((key, index) => ({ key: String(key), value: values[index] }));
    } finally {
      database.close();
    }
  },
  async put(key, value) {
    const database = await openDatabase();
    try {
      const transaction = database.transaction(STORE_NAME, "readwrite");
      const completed = transactionResult(transaction);
      transaction.objectStore(STORE_NAME).put(value, key);
      await completed;
    } finally {
      database.close();
    }
  },
  async delete(key) {
    const database = await openDatabase();
    try {
      const transaction = database.transaction(STORE_NAME, "readwrite");
      const completed = transactionResult(transaction);
      transaction.objectStore(STORE_NAME).delete(key);
      await completed;
    } finally {
      database.close();
    }
  },
};

export async function saveFieldEvidenceDraft(
  input: SaveFieldEvidenceDraftInput,
  storage: FieldEvidenceDraftStorage = browserStorage,
): Promise<FieldEvidenceDraft> {
  const subjectId = input.subjectId.trim();
  const workOrderId = input.workOrderId.trim();
  const operationId = input.clientOperationId.trim();
  const declaredMime = normalizeMime(input.file.type);
  const createdAt = input.createdAt ?? new Date();
  if (!subjectId || !workOrderId || !operationId || !Number.isFinite(createdAt.getTime())) {
    throw new Error("field_evidence_draft_binding_invalid");
  }
  if (
    !SUPPORTED_MIME_TYPES.has(declaredMime)
    || input.file.size < 1
    || input.file.size > MAX_BYTES
  ) {
    throw new Error("field_evidence_draft_media_invalid");
  }
  const blob = input.file.slice(0, input.file.size, declaredMime);
  const draft: FieldEvidenceDraft = {
    schema_version: SCHEMA_VERSION,
    subject_id: subjectId,
    work_order_id: workOrderId,
    client_operation_id: operationId,
    created_at: createdAt.toISOString(),
    declared_mime: declaredMime,
    size_bytes: blob.size,
    content_hash: await contentHash(blob),
    blob,
  };
  const duplicate = (await loadFieldEvidenceDrafts(
    subjectId,
    workOrderId,
    storage,
    createdAt,
  )).find((existing) => (
    existing.content_hash === draft.content_hash
    && existing.declared_mime === draft.declared_mime
    && existing.size_bytes === draft.size_bytes
  ));
  if (duplicate) return duplicate;

  await storage.put(storageKey(draft), draft);
  return draft;
}

export async function loadFieldEvidenceDrafts(
  subjectId: string,
  workOrderId: string,
  storage: FieldEvidenceDraftStorage = browserStorage,
  now: Date = new Date(),
): Promise<FieldEvidenceDraft[]> {
  const prefix = storagePrefix(subjectId, workOrderId);
  const entries = await storage.list();
  const valid: FieldEvidenceDraft[] = [];
  for (const entry of entries) {
    if (!entry.key.startsWith(prefix)) continue;
    const candidate = entry.value;
    if (
      !isDraftShape(candidate)
      || candidate.subject_id !== subjectId
      || candidate.work_order_id !== workOrderId
      || entry.key !== storageKey(candidate)
      || !isCurrent(candidate, now)
      || !await contentMatches(candidate)
    ) {
      await storage.delete(entry.key);
      continue;
    }
    valid.push(candidate);
  }
  return valid.sort((left, right) => (
    Date.parse(left.created_at) - Date.parse(right.created_at)
    || left.client_operation_id.localeCompare(right.client_operation_id)
  ));
}

export async function removeFieldEvidenceDraft(
  draft: FieldEvidenceDraft,
  storage: FieldEvidenceDraftStorage = browserStorage,
): Promise<void> {
  await storage.delete(storageKey(draft));
}

export async function syncFieldEvidenceDrafts(
  input: SyncFieldEvidenceDraftsInput,
): Promise<SyncFieldEvidenceDraftsResult> {
  const storage = input.storage ?? browserStorage;
  const send = input.send ?? uploadFieldEvidence;
  const now = input.now ?? new Date();
  const drafts = await loadFieldEvidenceDrafts(
    input.subjectId,
    input.workOrderId,
    storage,
    now,
  );
  if (!input.online) {
    return { synced: 0, remaining: drafts.length, blockedReason: "offline" };
  }
  if (!input.authoritativeWorkLoaded) {
    return {
      synced: 0,
      remaining: drafts.length,
      blockedReason: "authoritative_work_order_required",
    };
  }

  let synced = 0;
  for (const draft of drafts) {
    try {
      await send(
        input.workOrderId,
        input.workOrderVersion,
        draft.client_operation_id,
        draft.content_hash,
        draft.declared_mime,
        draft.blob,
      );
      await removeFieldEvidenceDraft(draft, storage);
      synced += 1;
    } catch (error) {
      return {
        synced,
        remaining: drafts.length - synced,
        error,
      };
    }
  }
  return { synced, remaining: 0 };
}

async function contentMatches(draft: FieldEvidenceDraft): Promise<boolean> {
  if (draft.blob.size !== draft.size_bytes) return false;
  if (normalizeMime(draft.blob.type) !== draft.declared_mime) return false;
  return await contentHash(draft.blob) === draft.content_hash;
}

async function contentHash(blob: Blob): Promise<string> {
  const bytes = typeof blob.arrayBuffer === "function"
    ? await blob.arrayBuffer()
    : await new Response(blob).arrayBuffer();
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  const hex = Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")).join("");
  return `sha256:${hex}`;
}

function isDraftShape(value: unknown): value is FieldEvidenceDraft {
  if (!isRecord(value)) return false;
  return Boolean(
    value.schema_version === SCHEMA_VERSION
    && typeof value.subject_id === "string"
    && value.subject_id.length > 0
    && typeof value.work_order_id === "string"
    && value.work_order_id.length > 0
    && typeof value.client_operation_id === "string"
    && value.client_operation_id.length > 0
    && typeof value.created_at === "string"
    && typeof value.declared_mime === "string"
    && SUPPORTED_MIME_TYPES.has(value.declared_mime)
    && typeof value.size_bytes === "number"
    && Number.isInteger(value.size_bytes)
    && value.size_bytes >= 1
    && value.size_bytes <= MAX_BYTES
    && typeof value.content_hash === "string"
    && /^sha256:[0-9a-f]{64}$/.test(value.content_hash)
    && value.blob instanceof Blob
  );
}

function isCurrent(draft: FieldEvidenceDraft, now: Date): boolean {
  const createdAt = Date.parse(draft.created_at);
  return Number.isFinite(createdAt)
    && createdAt <= now.getTime() + FUTURE_SKEW_MS
    && now.getTime() - createdAt <= RETENTION_MS;
}

function normalizeMime(value: string): string {
  return value.split(";", 1)[0].trim().toLowerCase();
}

function storageKey(draft: Pick<
  FieldEvidenceDraft,
  "subject_id" | "work_order_id" | "client_operation_id"
>): string {
  return `${storagePrefix(draft.subject_id, draft.work_order_id)}${encodeURIComponent(draft.client_operation_id)}`;
}

function storagePrefix(subjectId: string, workOrderId: string): string {
  return `${encodeURIComponent(subjectId)}\u0000${encodeURIComponent(workOrderId)}\u0000`;
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
