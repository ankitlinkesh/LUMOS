// Typed client for the TRIAD-RAG demo API (triad/api/app.py). Shapes here must
// stay byte-for-byte in sync with triad/api/serialize.py.

export interface Meta {
  service: "fake" | "real";
  data_source: string;
  note: string;
}

export interface Tenant {
  id: string;
  label: string;
  n_docs: number;
}

export interface Taint {
  quarantined: boolean;
  flags: string[];
  score: number;
  reasons: string[];
}

export interface ScoredChunk {
  id: string;
  tenant: string;
  score: number;
  preview: string;
  source_type: string;
  data_source: "real" | "synthetic";
  taint: Taint;
}

export interface TraceEvent {
  chunk_id: string;
  stage: "ingest" | "retrieve" | "prompt" | "egress";
  event: string;
  detail: string;
}

export interface AskResponse {
  answer: string;
  declined: boolean;
  decline_reason: string | null;
  leak_mode: boolean;
  latency_ms: number;
  chunks: ScoredChunk[];
  trace: TraceEvent[];
  cached: boolean;
  data_source: "real" | "mixed";
}

export interface QuarantineItem {
  id: string;
  tenant: string;
  preview: string;
  score: number;
  flags: string[];
  reasons: string[];
}

export interface ProbeSide {
  leaked: boolean;
  n_foreign: number;
  declined: boolean;
  chunks: ScoredChunk[];
}

export interface ProbeResponse {
  secure: ProbeSide;
  leaky: ProbeSide;
  property_test: { passed: number; total: number; fake: boolean } | null;
}

export interface ChunkTraceStep {
  stage: "ingest" | "retrieve" | "prompt" | "egress";
  status: string;
  detail: string;
  at: string;
}

export interface ResultRow {
  attack: string;
  asr_before: number;
  asr_after: number;
  clean_accuracy: number;
  added_latency_ms: number;
  fpr: number;
  n: number;
  data_source: string;
  fake: boolean;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // ignore, keep statusText
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  meta: () => req<Meta>("/meta"),
  tenants: () => req<Tenant[]>("/tenants"),
  ask: (tenant: string, question: string, defense: boolean) =>
    req<AskResponse>("/ask", {
      method: "POST",
      body: JSON.stringify({ tenant, question, defense }),
    }),
  quarantine: () => req<QuarantineItem[]>("/quarantine"),
  release: (id: string) =>
    req<{ released: boolean }>(`/quarantine/${encodeURIComponent(id)}/release`, { method: "POST" }),
  probe: (as_tenant: string, target_tenant: string) =>
    req<ProbeResponse>("/probe", {
      method: "POST",
      body: JSON.stringify({ as_tenant, target_tenant }),
    }),
  trace: (chunkId: string) => req<ChunkTraceStep[]>(`/trace/${encodeURIComponent(chunkId)}`),
  results: () => req<ResultRow[]>("/results"),
};
