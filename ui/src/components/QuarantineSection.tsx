import { useEffect, useState } from "react";
import { api, type QuarantineItem } from "../api";
import Panel from "./Panel";
import Badge from "./Badge";
import { LoadingBox, ErrorBox } from "./StatusBox";

type ListState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ok"; items: QuarantineItem[] };

export default function QuarantineSection({ onInspectChunk }: { onInspectChunk: (id: string) => void }) {
  const [state, setState] = useState<ListState>({ status: "loading" });
  const [releasing, setReleasing] = useState<string | null>(null);

  function load() {
    setState({ status: "loading" });
    api
      .quarantine()
      .then((items) => setState({ status: "ok", items }))
      .catch((e: Error) => setState({ status: "error", message: e.message }));
  }

  useEffect(load, []);

  async function release(id: string) {
    setReleasing(id);
    try {
      await api.release(id);
      load();
    } catch (e) {
      setState({ status: "error", message: (e as Error).message });
    } finally {
      setReleasing(null);
    }
  }

  return (
    <Panel
      id="quarantine"
      number={2}
      title="Quarantine queue"
      description="Documents blocked at ingestion before they could ever be retrieved, with the score and reasons Stage 1 flagged them for."
    >
      {state.status === "loading" && <LoadingBox label="Loading quarantine queue..." />}
      {state.status === "error" && <ErrorBox message={state.message} onRetry={load} />}
      {state.status === "ok" && (
        <>
          {state.items.length === 0 ? (
            <p className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-base text-slate-600">
              Quarantine queue is empty.
            </p>
          ) : (
            <ul className="space-y-3">
              {state.items.map((q) => (
                <li key={q.id} className="rounded-lg border border-red-300 bg-red-50 p-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-sm text-slate-500">{q.id}</span>
                    <Badge variant="red">tenant: {q.tenant}</Badge>
                    <Badge variant="red">score {q.score.toFixed(2)}</Badge>
                    {q.flags.map((f) => (
                      <Badge key={f} variant="amber">
                        {f}
                      </Badge>
                    ))}
                    <div className="ml-auto flex gap-2">
                      <button
                        onClick={() => onInspectChunk(q.id)}
                        className="rounded-md border border-slate-300 bg-white px-2.5 py-1 text-sm font-semibold text-slate-700 hover:bg-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
                      >
                        View trace
                      </button>
                      <button
                        onClick={() => release(q.id)}
                        disabled={releasing === q.id}
                        className="rounded-md bg-emerald-700 px-2.5 py-1 text-sm font-semibold text-white hover:bg-emerald-800 disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-600"
                      >
                        {releasing === q.id ? "Releasing..." : "Release"}
                      </button>
                    </div>
                  </div>
                  <p className="mt-2 text-base text-slate-800">{q.preview}</p>
                  <ul className="mt-2 list-inside list-disc text-sm text-red-800">
                    {q.reasons.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                  </ul>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </Panel>
  );
}
