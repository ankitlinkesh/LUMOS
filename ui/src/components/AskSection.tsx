import { useState } from "react";
import { api, type AskResponse } from "../api";
import Panel from "./Panel";
import Badge from "./Badge";
import ChunkCard from "./ChunkCard";
import TraceList from "./TraceList";
import { LoadingBox, ErrorBox } from "./StatusBox";

// Only an employee can reach POST /api/ask, and the server forces defense=ON
// for every employee request regardless of what's sent (see app.py's
// post_ask) -- the deliberately-leaky OFF path is a securityhead-only tool
// now (fired from the Cross-tenant probe panel's "leaky" side). So this is a
// single defended answer, not an OFF/ON comparison; there is nothing for a
// toggle to control.
type PaneState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ok"; data: AskResponse };

export default function AskSection({
  tenant,
  onInspectChunk,
}: {
  tenant: string;
  onInspectChunk: (id: string) => void;
}) {
  const [question, setQuestion] = useState("What is the Q3 budget?");
  const [state, setState] = useState<PaneState>({ status: "idle" });

  async function run(e: React.FormEvent) {
    e.preventDefault();
    setState({ status: "loading" });
    try {
      // defense is forced true here for clarity -- the server ignores this
      // field for an employee and forces it true either way.
      const data = await api.ask(tenant, question, true);
      setState({ status: "ok", data });
    } catch (err) {
      setState({ status: "error", message: (err as Error).message });
    }
  }

  return (
    <Panel
      id="ask"
      number={1}
      title="Ask your tenant's documents"
      description={`Scoped to your account's tenant (${tenant}) and always run through the defended pipeline -- the vulnerable baseline isn't reachable from this account.`}
    >
      <form
        onSubmit={run}
        className="flex flex-wrap items-end gap-4 rounded-xl border border-slate-200 bg-slate-50 p-4"
      >
        <label className="flex min-w-[20rem] flex-1 flex-col gap-1">
          <span className="text-sm font-semibold text-slate-700">Question</span>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            className="rounded-md border border-slate-300 bg-white px-3 py-2 text-base focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
          />
        </label>
        <button
          type="submit"
          className="rounded-md bg-slate-900 px-5 py-2.5 text-base font-semibold text-white hover:bg-slate-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
        >
          Ask
        </button>
      </form>

      <div className="mt-6 rounded-xl border border-emerald-300 bg-white">
        <div className="flex items-center justify-between rounded-t-xl border-b border-emerald-300 bg-emerald-50 px-4 py-3">
          <span className="text-lg font-bold text-emerald-900">Defended answer</span>
          {state.status === "ok" && (
            <span className="text-sm text-slate-600">{state.data.latency_ms.toFixed(0)} ms</span>
          )}
        </div>
        <div className="p-4">
          {state.status === "idle" && <p className="text-base text-slate-400">Ask a question to see the answer here.</p>}
          {state.status === "loading" && <LoadingBox label="Asking..." />}
          {state.status === "error" && <ErrorBox message={state.message} />}
          {state.status === "ok" && (
            <>
              {state.data.declined ? (
                <div className="rounded-lg border border-emerald-300 bg-emerald-50 p-4">
                  <p className="font-semibold text-emerald-900">Declined</p>
                  <p className="mt-1 text-slate-700">{state.data.decline_reason}</p>
                </div>
              ) : (
                <p className="text-lg leading-relaxed text-slate-900">{state.data.answer}</p>
              )}
              <div className="mt-3 flex flex-wrap gap-2">
                {state.data.cached && <Badge variant="neutral">cached</Badge>}
                <Badge
                  variant="neutral"
                  title="Corpus-level label: could a synthetic chunk have been retrieved at all (see README)"
                >
                  data: {state.data.data_source}
                </Badge>
                <Badge
                  variant="neutral"
                  title="Per-response fact: of the chunks THIS call actually retrieved, how many are real vs synthetic"
                >
                  this answer: {state.data.n_chunks_real} real / {state.data.n_chunks_synthetic} synthetic
                </Badge>
              </div>

              <h3 className="mt-5 text-base font-semibold text-slate-700">
                Retrieved chunks ({state.data.chunks.length})
              </h3>
              <ul className="mt-2 space-y-3">
                {state.data.chunks.map((c) => (
                  <ChunkCard key={c.id} chunk={c} onInspect={onInspectChunk} />
                ))}
              </ul>

              <h3 className="mt-5 text-base font-semibold text-slate-700">Trace</h3>
              <TraceList trace={state.data.trace} />
            </>
          )}
        </div>
      </div>
    </Panel>
  );
}
