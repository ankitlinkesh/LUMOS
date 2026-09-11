import { useEffect, useState } from "react";
import { api, setUnauthorizedListener, type Me, type Meta, type Role, type Tenant } from "./api";
import AskSection from "./components/AskSection";
import TenantsSection from "./components/TenantsSection";
import QuarantineSection from "./components/QuarantineSection";
import TraceSection from "./components/TraceSection";
import ProbeSection from "./components/ProbeSection";
import ResultsSection from "./components/ResultsSection";
import DemoModeBanner from "./components/DemoModeBanner";
import LoginScreen from "./components/LoginScreen";
import { LoadingBox, ErrorBox } from "./components/StatusBox";

// Demo hint ids so the trace viewer's autocomplete has something to offer
// before a judge has clicked a chunk elsewhere on the page. (An employee
// account will get a 403 tracing a foreign one of these -- that's the point,
// see the README's "Roles and access control" section.)
const HINT_CHUNK_IDS = ["doc-a-budget", "doc-x-poison", "doc-y-poison", "doc-b-1", "doc-c-1"];

// Which panel ids a role's account can reach, in display order. This is
// cosmetic sequencing only -- every id here maps to a route the SERVER also
// allows for that role (see triad/api/auth.py's POLICY); hiding a section a
// role can't use is a convenience, not the control.
const SECTIONS_FOR_ROLE: Record<Role, { id: string; label: string }[]> = {
  employee: [
    { id: "ask", label: "Ask" },
    { id: "trace", label: "Taint trace" },
  ],
  dbmanager: [
    { id: "tenants", label: "Tenants" },
    { id: "quarantine", label: "Quarantine (read-only)" },
    { id: "trace", label: "Taint trace" },
  ],
  securityhead: [
    { id: "tenants", label: "Tenants" },
    { id: "quarantine", label: "Quarantine" },
    { id: "trace", label: "Taint trace" },
    { id: "probe", label: "Cross-tenant probe" },
    { id: "results", label: "Results" },
  ],
  ceo: [
    { id: "quarantine", label: "Quarantine (aggregate)" },
    { id: "results", label: "Results" },
  ],
};

type AuthState =
  | { status: "loading" }
  | { status: "anonymous" }
  | { status: "authed"; me: Me };

type TenantsState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ok"; tenants: Tenant[] };

export default function App() {
  const [auth, setAuth] = useState<AuthState>({ status: "loading" });
  const [meta, setMeta] = useState<Meta | null>(null);
  const [tenantsState, setTenantsState] = useState<TenantsState>({ status: "loading" });
  const [selectedChunkId, setSelectedChunkId] = useState("");

  function checkSession() {
    api
      .me()
      .then((me) => setAuth({ status: "authed", me }))
      .catch(() => setAuth({ status: "anonymous" }));
  }

  // /api/meta is public -- fetched unconditionally so the DEMO MODE banner
  // can render even on the login screen, before anyone has authenticated.
  useEffect(() => {
    api.meta().then(setMeta).catch(() => {});
  }, []);

  useEffect(checkSession, []);

  // A 401 on ANY api call (not just /api/me) means "not logged in right
  // now" -- covers both the initial check above and a session that expired
  // mid-use. Either way, the fix is the same: show the login screen again.
  useEffect(() => {
    setUnauthorizedListener(() => setAuth({ status: "anonymous" }));
    return () => setUnauthorizedListener(null);
  }, []);

  const role: Role | null = auth.status === "authed" ? auth.me.role : null;
  const needsTenantsList = role === "dbmanager" || role === "securityhead";

  function loadTenants() {
    setTenantsState({ status: "loading" });
    api
      .tenants()
      .then((tenants) => setTenantsState({ status: "ok", tenants }))
      .catch((e: Error) => setTenantsState({ status: "error", message: e.message }));
  }

  useEffect(() => {
    if (needsTenantsList) loadTenants();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [needsTenantsList]);

  async function logout() {
    try {
      await api.logout();
    } finally {
      setAuth({ status: "anonymous" });
      setSelectedChunkId("");
    }
  }

  // Fail-safe direction: the banner is HIDDEN only once /api/meta has
  // positively confirmed service === "real". While /api/meta is still
  // loading, or if it fails outright, we do not know this isn't the fake
  // service, so the warning stays up rather than risking fake data reading
  // as real during that window.
  const showDemoBanner = meta?.service !== "real";

  if (auth.status === "loading") {
    return (
      <div className="lumo-shell flex min-h-screen items-center justify-center">
        <LoadingBox label="Checking session..." />
      </div>
    );
  }

  if (auth.status === "anonymous") {
    return (
      <>
        {showDemoBanner && <DemoModeBanner note={meta?.note} />}
        <LoginScreen onLoggedIn={checkSession} />
      </>
    );
  }

  const me = auth.me;
  const sections = SECTIONS_FOR_ROLE[me.role];
  const numberOf = (id: string) => sections.findIndex((s) => s.id === id) + 1;
  const has = (id: string) => sections.some((s) => s.id === id);

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
          <div className="flex items-center gap-3">
            <span className="lumo-version hidden sm:inline">STAGE 1 — 3</span>
            <div className="flex items-center gap-2 rounded-full border border-purple-400/25 bg-white/5 px-3 py-1.5">
              <span className="text-xs font-semibold text-white">{me.username}</span>
              <span className="rounded-full bg-purple-500/30 px-2 py-0.5 text-[0.65rem] font-bold uppercase tracking-wide text-purple-100">
                {me.role}
              </span>
              {me.tenant && (
                <span className="hidden font-mono text-[0.65rem] text-purple-200/70 sm:inline">tenant: {me.tenant}</span>
              )}
              <button
                onClick={logout}
                className="ml-1 rounded-full border border-white/20 px-2.5 py-1 text-xs font-semibold text-white hover:bg-white/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-purple-300"
              >
                Log out
              </button>
            </div>
          </div>
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
          {sections.map((n, i) => (
            <a key={n.id} href={`#${n.id}`}><span>{String(i + 1).padStart(2, "0")}</span>{n.label}</a>
          ))}
        </nav>
      </header>

      <div className="lumo-overview-strip">
        <div><span className="lumo-stat-label">SERVICE</span><strong>{meta?.service === "real" ? "REAL PIPELINE" : "SYNTHETIC DEMO"}</strong></div>
        <div><span className="lumo-stat-label">SIGNED IN AS</span><strong>{me.role.toUpperCase()}</strong></div>
        <div><span className="lumo-stat-label">ACTIVE CONTROLS</span><strong>INGEST · RETRIEVE · EGRESS</strong></div>
        <div><span className="lumo-stat-label">DATA INTEGRITY</span><strong className="lumo-accent-text">TRACEABLE</strong></div>
      </div>

      <main className="lumo-main mx-auto max-w-6xl px-6 pb-24">
        {has("ask") && me.tenant && (
          <AskSection tenant={me.tenant} onInspectChunk={setSelectedChunkId} />
        )}

        {has("tenants") && (
          <>
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
              <TenantsSection number={numberOf("tenants")} tenants={tenantsState.tenants} />
            )}
          </>
        )}

        {has("quarantine") && (
          <QuarantineSection number={numberOf("quarantine")} role={me.role} onInspectChunk={setSelectedChunkId} />
        )}

        {has("trace") && (
          <TraceSection
            number={numberOf("trace")}
            chunkId={selectedChunkId}
            onChunkIdChange={setSelectedChunkId}
            knownChunkIds={HINT_CHUNK_IDS}
          />
        )}

        {has("probe") && tenantsState.status === "ok" && (
          <ProbeSection number={numberOf("probe")} tenants={tenantsState.tenants} onInspectChunk={setSelectedChunkId} />
        )}

        {has("results") && <ResultsSection number={numberOf("results")} />}
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
