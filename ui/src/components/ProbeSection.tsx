import { useState } from "react";
import { api, type ProbeResponse, type ProbeSide, type Tenant } from "../api";
import Panel from "./Panel";
import Badge from "./Badge";
import ChunkCard from "./ChunkCard";
import { LoadingBox, ErrorBox } from "./StatusBox";

type State =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ok"; data: ProbeResponse };

function ProbeSideCard({
  label,
  side,
  outcomeVariant,
  onInspectChunk,
  targetTenant,
}: {
  label: string;
  side: ProbeSide;
  outcomeVariant: "red" | "green";
  onInspectChunk: (id: string) => void;
  targetTenant: string;
}) {
  const border = outcomeVariant === "red" ? "border-red-300 bg-red-50" : "border-emerald-300 bg-emerald-50";
  const text = outcomeVariant === "red" ? "text-red-900" : "text-emerald-900";
  return (
    <div className="flex-1 rounded-xl border border-slate-200 bg-white">
      <div className={`flex items-center justify-between rounded-t-xl border-b px-4 py-3 ${border}`}>
        <span className={`text-lg font-bold ${text}`}>{label}</span>
        <Badge variant={outcomeVariant}>{side.leaked ? `${side.n_foreign} foreign leaked` : "0 foreign"}</Badge>
      </div>
      <div className="p-4">
        {side.declined && (
          <p className="mb-3 rounded-lg border border-emerald-300 bg-emerald-50 p-3 text-emerald-900">
            Declined: scope check refused to widen the search.
          </p>
        )}
        <ul className="space-y-3">
          {side.chunks.map((c) => (
            <ChunkCard key={c.id} chunk={c} foreign={c.tenant === targetTenant} onInspect={onInspectChunk} />
          ))}
          {side.chunks.length === 0 && <p className="text-base text-slate-500">No chunks returned.</p>}
        </ul>
      </div>
    </div>
  );
}

export default function ProbeSection({
  tenants,
  onInspectChunk,
}: {
  tenants: Tenant[];
  onInspectChunk: (id: string) => void;
}) {
  const [asTenant, setAsTenant] = useState("tenant-a");
  const [targetTenant, setTargetTenant] = useState("tenant-b");
  const [state, setState] = useState<State>({ status: "idle" });

  async function fire() {
    setState({ status: "loading" });
    try {
      const data = await api.probe(asTenant, targetTenant);
      setState({ status: "ok", data });
    } catch (e) {
      setState({ status: "error", message: (e as Error).message });
    }
  }

  return (
    <Panel
      id="probe"
      number={4}
      title="Cross-tenant probe"
      description="Fire the same probe as the defended retriever (secure) and the deliberately vulnerable baseline (leaky), side by side."
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          fire();
        }}
        className="flex flex-wrap items-end gap-4 rounded-xl border border-slate-200 bg-slate-50 p-4"
      >
        <label className="flex flex-col gap-1">
          <span className="text-sm font-semibold text-slate-700">As tenant</span>
          <select
            value={asTenant}
            onChange={(e) => setAsTenant(e.target.value)}
            className="rounded-md border border-slate-300 bg-white px-3 py-2 text-base focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
          >
            {tenants.map((t) => (
              <option key={t.id} value={t.id}>
                {t.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm font-semibold text-slate-700">Target tenant</span>
          <select
            value={targetTenant}
            onChange={(e) => setTargetTenant(e.target.value)}
            className="rounded-md border border-slate-300 bg-white px-3 py-2 text-base focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
          >
            {tenants.map((t) => (
              <option key={t.id} value={t.id}>
                {t.label}
              </option>
            ))}
          </select>
        </label>
        <button
          type="submit"
          className="rounded-md bg-slate-900 px-5 py-2.5 text-base font-semibold text-white hover:bg-slate-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
        >
          Fire probe
        </button>
      </form>

      <div className="mt-6">
        {state.status === "idle" && <p className="text-base text-slate-400">Pick two tenants and fire the probe.</p>}
        {state.status === "loading" && <LoadingBox label="Probing..." />}
        {state.status === "error" && <ErrorBox message={state.message} onRetry={fire} />}
        {state.status === "ok" && (
          <>
            <div className="mb-4 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
              <span className="font-semibold">Query (derived from {targetTenant}'s own content):</span>{" "}
              <span className="italic">&ldquo;{state.data.query}&rdquo;</span>
              {state.data.gold_leaked && (
                <span className="ml-2">
                  <Badge variant="red" title="The exact chunk this query was built to be answerable from came back for a different requester">
                    gold chunk leaked
                  </Badge>
                </span>
              )}
            </div>
            <div className="flex flex-col gap-6 lg:flex-row">
              <ProbeSideCard
                label="Secure (defended)"
                side={state.data.secure}
                outcomeVariant="green"
                onInspectChunk={onInspectChunk}
                targetTenant={targetTenant}
              />
              <ProbeSideCard
                label="Leaky (vulnerable baseline)"
                side={state.data.leaky}
                outcomeVariant="red"
                onInspectChunk={onInspectChunk}
                targetTenant={targetTenant}
              />
            </div>
            <div className="mt-4">
              {state.data.property_test ? (
                state.data.property_test.fake ? (
                  <Badge variant="amber" title="This pass count is fabricated for the demo, not measured">
                    FAKE property test: {state.data.property_test.passed}/{state.data.property_test.total}{" "}
                    (not measured)
                  </Badge>
                ) : (
                  <Badge variant="green">
                    Stage 2 property test: {state.data.property_test.passed}/{state.data.property_test.total} passed
                  </Badge>
                )
              ) : (
                <Badge variant="neutral">Property test not run (same tenant on both sides)</Badge>
              )}
            </div>
          </>
        )}
      </div>
    </Panel>
  );
}
