import { useState } from "react";
import { api, type AskResponse, type Tenant } from "../api";
import Panel from "./Panel";
import Badge from "./Badge";
import ChunkCard from "./ChunkCard";
import TraceList from "./TraceList";
import { LoadingBox, ErrorBox } from "./StatusBox";

type PaneState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ok"; data: AskResponse };

function AnswerPane({
  label,
  variant,
  state,
  onInspectChunk,
}: {
  label: string;
  variant: "off" | "on";
  state: PaneState;
  onInspectChunk: (id: string) => void;
}) {
  const headerColor = variant === "off" ? "border-red-300 bg-red-50" : "border-emerald-300 bg-emerald-50";
  const headerText = variant === "off" ? "text-red-900" : "text-emerald-900";

  return (
    <div className="flex-1 rounded-xl border border-slate-200 bg-white">
      <div className={`flex items-center justify-between rounded-t-xl border-b px-4 py-3 ${headerColor}`}>
        <span className={`text-lg font-bold ${headerText}`}>{label}</span>
        {state.status === "ok" && (
          <span className="text-sm text-slate-600">{state.data.latency_ms.toFixed(0)} ms</span>
        )}
      </div>
      <div className="p-4">
        {state.status === "idle" && <p className="text-base text-slate-400">Run a question to see the answer here.</p>}
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
              {state.data.leak_mode && <Badge variant="red">cross-tenant leak</Badge>}
              {state.data.cached && <Badge variant="neutral">cached</Badge>}
              <Badge variant="neutral">data: {state.data.data_source}</Badge>
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
  );
}

export default function AskSection({
  tenants,
  onInspectChunk,
}: {
  tenants: Tenant[];
  onInspectChunk: (id: string) => void;
}) {
  const [tenant, setTenant] = useState("tenant-a");
  const [question, setQuestion] = useState("What is the Q3 budget?");
  const [off, setOff] = useState<PaneState>({ status: "idle" });
  const [on, setOn] = useState<PaneState>({ status: "idle" });

  async function run() {
    setOff({ status: "loading" });
    setOn({ status: "loading" });
    await Promise.all([
      api
        .ask(tenant, question, false)
        .then((data) => setOff({ status: "ok", data }))
        .catch((e: Error) => setOff({ status: "error", message: e.message })),
      api
        .ask(tenant, question, true)
        .then((data) => setOn({ status: "ok", data }))
        .catch((e: Error) => setOn({ status: "error", message: e.message })),
    ]);
  }

  return (
    <Panel
      id="ask"
      number={1}
      title="Defense OFF vs. ON"
      description="Same tenant, same question, run through the vulnerable baseline and the defended pipeline side by side."
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          run();
        }}
        className="flex flex-wrap items-end gap-4 rounded-xl border border-slate-200 bg-slate-50 p-4"
      >
        <label className="flex flex-col gap-1">
          <span className="text-sm font-semibold text-slate-700">Tenant</span>
          <select
            value={tenant}
            onChange={(e) => setTenant(e.target.value)}
            className="rounded-md border border-slate-300 bg-white px-3 py-2 text-base focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
          >
            {tenants.map((t) => (
              <option key={t.id} value={t.id}>
                {t.label}
              </option>
            ))}
          </select>
        </label>
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
          Ask both
        </button>
      </form>

      <div className="mt-6 flex flex-col gap-6 lg:flex-row">
        <AnswerPane label="Defense OFF" variant="off" state={off} onInspectChunk={onInspectChunk} />
        <AnswerPane label="Defense ON" variant="on" state={on} onInspectChunk={onInspectChunk} />
      </div>
    </Panel>
  );
}
