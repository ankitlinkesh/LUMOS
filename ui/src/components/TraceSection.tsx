import { useEffect, useState } from "react";
import { api, type ChunkTraceStep } from "../api";
import Panel from "./Panel";
import Badge from "./Badge";
import { LoadingBox, ErrorBox } from "./StatusBox";

type FetchState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ok"; steps: ChunkTraceStep[] };

const STAGE_LABEL: Record<ChunkTraceStep["stage"], string> = {
  ingest: "Ingest",
  retrieve: "Retrieve",
  prompt: "Prompt",
  egress: "Egress",
};

function statusVariant(status: string): "neutral" | "red" | "green" | "amber" {
  if (["quarantined", "blocked", "skipped"].includes(status)) return "red";
  if (["clean", "eligible", "used", "included", "enabled"].includes(status)) return "green";
  if (status === "disabled") return "amber";
  return "neutral";
}

export default function TraceSection({
  number,
  chunkId,
  onChunkIdChange,
  knownChunkIds,
}: {
  number: number;
  chunkId: string;
  onChunkIdChange: (id: string) => void;
  knownChunkIds: string[];
}) {
  const [state, setState] = useState<FetchState>({ status: "idle" });
  const [draft, setDraft] = useState(chunkId);

  useEffect(() => setDraft(chunkId), [chunkId]);

  useEffect(() => {
    if (!chunkId) {
      setState({ status: "idle" });
      return;
    }
    setState({ status: "loading" });
    api
      .trace(chunkId)
      .then((steps) => setState({ status: "ok", steps }))
      .catch((e: Error) => setState({ status: "error", message: e.message }));
  }, [chunkId]);

  return (
    <Panel
      id="trace"
      number={number}
      title="Taint-trace viewer"
      description="The life of one document: ingest, retrieve, prompt, egress. Click 'View trace' on any chunk above, or type a chunk id."
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onChunkIdChange(draft.trim());
        }}
        className="flex flex-wrap items-end gap-4 rounded-xl border border-slate-200 bg-slate-50 p-4"
      >
        <label className="flex min-w-[16rem] flex-1 flex-col gap-1">
          <span className="text-sm font-semibold text-slate-700">Chunk id</span>
          <input
            list="known-chunk-ids"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="e.g. doc-x-poison"
            className="rounded-md border border-slate-300 bg-white px-3 py-2 font-mono text-base focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
          />
          <datalist id="known-chunk-ids">
            {knownChunkIds.map((id) => (
              <option key={id} value={id} />
            ))}
          </datalist>
        </label>
        <button
          type="submit"
          className="rounded-md bg-slate-900 px-5 py-2.5 text-base font-semibold text-white hover:bg-slate-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
        >
          Trace
        </button>
      </form>

      <div className="mt-6">
        {state.status === "idle" && (
          <p className="text-base text-slate-400">No chunk selected yet.</p>
        )}
        {state.status === "loading" && <LoadingBox label={`Tracing ${chunkId}...`} />}
        {state.status === "error" && <ErrorBox message={state.message} />}
        {state.status === "ok" && (
          <ol className="flex flex-col gap-3 sm:flex-row sm:items-stretch">
            {state.steps.map((s, i) => (
              <li key={i} className="flex flex-1 flex-col gap-2 rounded-xl border border-slate-200 bg-white p-4">
                <div className="flex items-center justify-between">
                  <span className="text-lg font-bold text-slate-900">{STAGE_LABEL[s.stage]}</span>
                  <Badge variant={statusVariant(s.status)}>{s.status}</Badge>
                </div>
                <p className="text-sm text-slate-600">{s.detail}</p>
                <p className="mt-auto text-xs text-slate-400">{s.at}</p>
              </li>
            ))}
          </ol>
        )}
      </div>
    </Panel>
  );
}
