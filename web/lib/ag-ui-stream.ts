export interface AgUiTransportEvent {
  id?: string;
  type: string;
  data: Record<string, unknown>;
}

/** Read JSON AG-UI frames, accepting a complete final JSON frame at EOF. */
export async function* readAgUiEvents(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<AgUiTransportEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      pending += done ? decoder.decode() : decoder.decode(value, { stream: true });
      const frames = pending.split(/\r?\n\r?\n/);
      pending = frames.pop() ?? "";
      if (done && pending) frames.push(pending);
      for (const frame of frames) {
        const parsed = parseAgUiTransportFrame(frame);
        if (parsed) yield parsed;
      }
      if (done) return;
    }
  } finally {
    reader.releaseLock();
  }
}

function parseAgUiTransportFrame(frame: string): AgUiTransportEvent | undefined {
  const fields = new Map<string, string>();
  for (const line of frame.split("\n")) {
    const separator = line.indexOf(":");
    if (separator < 0) continue;
    fields.set(line.slice(0, separator), line.slice(separator + 1).trimStart());
  }
  const data = fields.get("data");
  if (!data) return undefined;
  try {
    const parsed = JSON.parse(data) as Record<string, unknown>;
    if (typeof parsed.type !== "string") return undefined;
    const id = fields.get("id");
    return {
      ...(id === undefined ? {} : { id }),
      type: parsed.type,
      data: parsed,
    };
  } catch {
    return undefined;
  }
}
