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
    <div className="lumo-shell min-h-screen">
      {showDemoBanner && <DemoModeBanner note={meta?.note} />}
      <header className="lumo-header">
        <div className="lumo-topbar">
          <a href="#top" className="lumo-brand" aria-label="TRIAD-RAG home">
            <span className="lumo-brand-mark" aria-hidden="true">✦</span>
            <span>TRIAD<span className="lumo-brand-accent">/</span>RAG</span>
          </a>
          <div className="lumo-topbar-meta">
            <span className={`lumo-status-dot ${meta?.service === "real" ? "is-live" : ""}`} />
            <span>{meta?.service === "real" ? "LIVE PIPELINE" : "DEMO SERVICE"}</span>
            <span className="lumo-divider" />
            <span>SECURITY CONSOLE</span>
          </div>
          <span className="lumo-version">STAGE 1 — 3</span>
        </div>
        <div id="top" className="lumo-hero">
          <div>
            <p className="lumo-kicker">RETRIEVAL SECURITY OPERATIONS</p>
            <h1>Context stays scoped.<br /><span>Instructions stay contained.</span></h1>
            <p className="lumo-hero-copy">
              Observe how tainted documents are quarantined, tenant boundaries are enforced,
              and egress is evaluated across the full retrieval path.
            </p>
          </div>
          <div className="lumo-hero-orbit" aria-hidden="true">
            <div className="lumo-orbit-ring ring-one" /><div className="lumo-orbit-ring ring-two" />
            <div className="lumo-orbit-core">TRIAD<br /><small>GUARD</small></div>
          </div>
        </div>
        <nav className="lumo-nav" aria-label="Security console sections">
          {NAV.map((n, i) => (
            <a key={n.id} href={`#${n.id}`}><span>{String(i + 1).padStart(2, "0")}</span>{n.label}</a>
          ))}
        </nav>
      </header>

      <div className="lumo-overview-strip">
        <div><span className="lumo-stat-label">SERVICE</span><strong>{meta?.service === "real" ? "REAL PIPELINE" : "SYNTHETIC DEMO"}</strong></div>
        <div><span className="lumo-stat-label">TENANTS IN SCOPE</span><strong>{tenantsState.status === "ok" ? tenantsState.tenants.length : "—"}</strong></div>
        <div><span className="lumo-stat-label">ACTIVE CONTROLS</span><strong>INGEST · RETRIEVE · EGRESS</strong></div>
        <div><span className="lumo-stat-label">DATA INTEGRITY</span><strong className="lumo-accent-text">TRACEABLE</strong></div>
      </div>

      <main className="lumo-main mx-auto max-w-6xl px-6 pb-24">
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

      <footer className="lumo-footer">
        {/* Stage 3 is implemented and measured (see the README's "Stage 3 --
            output and egress" section) but whether it is switched ON for
            THIS build depends on how the server was started -- never assert
            a specific state here that this static string can't verify.
            The Ask/Trace panels' egress step shows the real, current
            enabled/disabled fact for this build. */}
        TRIAD/RAG &middot; SECURE RETRIEVAL OBSERVABILITY &middot; Stage 3 egress checks: see the Ask/Trace panels
      </footer>
    </div>
  );
}
