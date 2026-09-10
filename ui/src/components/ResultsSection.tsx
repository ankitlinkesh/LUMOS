import { useEffect, useState } from "react";
import { api, type ResultRow } from "../api";
import Panel from "./Panel";
import Badge from "./Badge";
import { LoadingBox, ErrorBox } from "./StatusBox";

type State =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ok"; rows: ResultRow[] };

function pct(x: number): string {
  return `${(x * 100).toFixed(0)}%`;
}

export default function ResultsSection() {
  const [state, setState] = useState<State>({ status: "loading" });

  function load() {
    setState({ status: "loading" });
    api
      .results()
      .then((rows) => setState({ status: "ok", rows }))
      .catch((e: Error) => setState({ status: "error", message: e.message }));
  }

  useEffect(load, []);

  return (
    <Panel
      id="results"
      number={5}
      title="Results"
      description="Attack success rate before/after the defense, clean accuracy, added latency, and false-positive rate — never invented, only measured."
    >
      {state.status === "loading" && <LoadingBox label="Loading results..." />}
      {state.status === "error" && <ErrorBox message={state.message} onRetry={load} />}
      {state.status === "ok" && (
        <>
          {state.rows.length === 0 ? (
            <p className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-base text-slate-600">
              No results yet — the eval harness hasn't produced numbers for this build.
            </p>
          ) : (
            <div className="overflow-x-auto rounded-xl border border-slate-200">
              <table className="w-full min-w-[720px] text-left text-base">
                <thead className="bg-slate-100 text-sm uppercase tracking-wide text-slate-600">
                  <tr>
                    <th className="px-4 py-3">Attack</th>
                    <th className="px-4 py-3">ASR before</th>
                    <th className="px-4 py-3">ASR after</th>
                    <th className="px-4 py-3">Clean accuracy</th>
                    <th className="px-4 py-3">Added latency</th>
                    <th className="px-4 py-3">FPR</th>
                    <th className="px-4 py-3">n</th>
                    <th className="px-4 py-3">Data</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 bg-white">
                  {state.rows.map((r) => (
                    <tr key={r.attack}>
                      <td className="px-4 py-3 font-semibold text-slate-900">
                        <div className="flex flex-wrap items-center gap-2">
                          {r.attack}
                          {r.fake && <Badge variant="amber">FAKE DATA</Badge>}
                        </div>
                      </td>
                      <td className="px-4 py-3 text-red-800">{pct(r.asr_before)}</td>
                      <td className="px-4 py-3 text-emerald-800">{pct(r.asr_after)}</td>
                      <td className="px-4 py-3">{pct(r.clean_accuracy)}</td>
                      <td className="px-4 py-3">{r.added_latency_ms.toFixed(0)} ms</td>
                      <td className="px-4 py-3">{pct(r.fpr)}</td>
                      <td className="px-4 py-3">{r.n}</td>
                      <td className="px-4 py-3 text-slate-500">{r.data_source}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </Panel>
  );
}
