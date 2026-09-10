import { useEffect, useState } from "react";
import { api, type Meta, type Tenant } from "./api";
import AskSection from "./components/AskSection";
import QuarantineSection from "./components/QuarantineSection";
import TraceSection from "./components/TraceSection";
import ProbeSection from "./components/ProbeSection";
import ResultsSection from "./components/ResultsSection";
import DemoModeBanner from "./components/DemoModeBanner";
import { LoadingBox, ErrorBox } from "./components/StatusBox";

const NAV = [
  { id: "ask", label: "Defense OFF/ON" },
  { id: "quarantine", label: "Quarantine" },
  { id: "trace", label: "Taint trace" },
  { id: "probe", label: "Cross-tenant probe" },
  { id: "results", label: "Results" },
];

// Demo hint ids so the trace viewer's autocomplete has something to offer
// before a judge has clicked a chunk elsewhere on the page.
const HINT_CHUNK_IDS = ["doc-a-budget", "doc-x-poison", "doc-y-poison", "doc-b-1", "doc-c-1"];

type TenantsState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ok"; tenants: Tenant[] };

export default function App() {
  const [tenantsState, setTenantsState] = useState<TenantsState>({ status: "loading" });
  const [selectedChunkId, setSelectedChunkId] = useState("");
  const [meta, setMeta] = useState<Meta | null>(null);

  function loadTenants() {
    setTenantsState({ status: "loading" });
    api
      .tenants()
      .then((tenants) => setTenantsState({ status: "ok", tenants }))
      .catch((e: Error) => setTenantsState({ status: "error", message: e.message }));
  }

  useEffect(loadTenants, []);
  useEffect(() => {
    api.meta().then(setMeta).catch(() => {});
  }, []);

  // Fail-safe direction: the banner is HIDDEN only once /api/meta has
  // positively confirmed service === "real". While /api/meta is still
  // loading, or if it fails outright, we do not know this isn't the fake
  // service, so the warning stays up rather than risking fake data reading
  // as real during that window.
  const showDemoBanner = meta?.service !== "real";

  return (
    <div className="min-h-screen bg-white text-slate-900">
      {showDemoBanner && <DemoModeBanner note={meta?.note} />}
      <header className="border-b border-slate-200 bg-slate-900 text-white">
        <div className="mx-auto max-w-6xl px-6 py-8">
          <p className="text-sm font-semibold uppercase tracking-widest text-slate-400">TRIAD-RAG</p>
          <p className="mt-2 max-w-4xl text-2xl font-bold leading-snug sm:text-3xl">
            &ldquo;A retrieved document may supply facts. It may never widen who sees what, trigger
            an action, or send data out.&rdquo;
          </p>
        </div>
        <nav className="border-t border-slate-800 bg-slate-950/40">
          <div className="mx-auto flex max-w-6xl flex-wrap gap-1 px-6 py-2">
            {NAV.map((n) => (
              <a
                key={n.id}
                href={`#${n.id}`}
                className="rounded-md px-3 py-1.5 text-sm font-medium text-slate-300 hover:bg-slate-800 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-white"
              >
                {n.label}
              </a>
            ))}
          </div>
        </nav>
      </header>

      <main className="mx-auto max-w-6xl px-6 pb-24">
        {tenantsState.status === "loading" && (
          <div className="pt-10">
            <LoadingBox label="Loading tenants..." />
          </div>
        )}
        {tenantsState.status === "error" && (
          <div className="pt-10">
            <ErrorBox message={tenantsState.message} onRetry={loadTenants} />
          </div>
        )}
        {tenantsState.status === "ok" && (
          <>
            <AskSection tenants={tenantsState.tenants} onInspectChunk={setSelectedChunkId} />
            <QuarantineSection onInspectChunk={setSelectedChunkId} />
            <TraceSection
              chunkId={selectedChunkId}
              onChunkIdChange={setSelectedChunkId}
              knownChunkIds={HINT_CHUNK_IDS}
            />
            <ProbeSection tenants={tenantsState.tenants} onInspectChunk={setSelectedChunkId} />
            <ResultsSection />
          </>
        )}
      </main>

      <footer className="border-t border-slate-200 bg-slate-50 py-6 text-center text-sm text-slate-500">
        TRIAD-RAG demo &middot; Stage 3 egress checks are paused in this build
      </footer>
    </div>
  );
}
