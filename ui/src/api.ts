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
  // Corpus-level, conservative label: could a synthetic chunk have been
  // retrieved at all (see the README's Demo UI section for why "mixed" and
  // "real" coexist). n_chunks_real/n_chunks_synthetic is the per-response
  // fact -- of the chunks THIS call actually retrieved, how many really
  // are which.
  data_source: "real" | "mixed";
  n_chunks_real: number;
  n_chunks_synthetic: number;
}

export interface QuarantineItem {
  id: string;
  tenant: string;
  preview: string;
  score: number;
  flags: string[];
  reasons: string[];
}

// The CEO's view of /api/quarantine: counts only, enforced server-side (see
// triad/api/app.py's get_quarantine and the README's "Roles and access
// control" section) -- no id/preview/reason ever reaches this shape.
export interface QuarantineAggregate {
  total: number;
  by_tenant: Record<string, number>;
  by_flag: Record<string, number>;
}

function isQuarantineAggregate(x: QuarantineItem[] | QuarantineAggregate): x is QuarantineAggregate {
  return !Array.isArray(x);
}

export type Role = "employee" | "dbmanager" | "securityhead" | "ceo";

export interface Me {
  username: string;
  role: Role;
  tenant: string | null;
  permissions: string[];
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
  // The query the probe actually ran, always derived from target_tenant's
  // own content -- never a fixed generic string. gold_leaked is the strong
  // claim: the SPECIFIC chunk this query was built to be answerable from
  // came back on the leaky side for a different requester.
  query: string;
  target_gold_chunk_id: string | null;
  gold_leaked: boolean;
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

// Thrown by req() on any non-2xx response. Carries the real HTTP status so
// callers can tell "session expired" (401) apart from "wrong role" (403)
// apart from an ordinary application error, instead of parsing it back out
// of a message string.
export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    // Explicit, though "same-origin" is fetch's own default: the UI and API
    // are always same-origin here (one FastAPI app serves both, see
    // app.py), and the session cookie is HttpOnly/SameSite=Strict so it
    // only ever rides along on same-origin requests anyway.
    credentials: "same-origin",
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
    if (res.status === 401) {
      unauthorizedListener?.();
    }
    throw new ApiError(res.status, detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// Fired whenever any req() call gets a 401 -- the one signal that covers
// both "never logged in" and "session expired mid-use." App.tsx is the only
// subscriber: it flips back to the login screen. A bare module-level
// listener (rather than routing every component through a context) keeps
// every existing section component's fetch/catch logic unchanged.
type UnauthorizedListener = () => void;
let unauthorizedListener: UnauthorizedListener | null = null;
export function setUnauthorizedListener(fn: UnauthorizedListener | null) {
  unauthorizedListener = fn;
}

export const api = {
  meta: () => req<Meta>("/meta"),
  me: () => req<Me>("/me"),
  login: (username: string, password: string) =>
    req<{ username: string; role: Role; tenant: string | null }>("/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  logout: () => req<{ logged_out: boolean }>("/logout", { method: "POST" }),
  tenants: () => req<Tenant[]>("/tenants"),
  ask: (tenant: string, question: string, defense: boolean) =>
    req<AskResponse>("/ask", {
      method: "POST",
      body: JSON.stringify({ tenant, question, defense }),
    }),
  quarantine: () => req<QuarantineItem[] | QuarantineAggregate>("/quarantine"),
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

export { isQuarantineAggregate };
