import type { TraceEvent } from "../api";
import Badge from "./Badge";

const STAGE_VARIANT: Record<TraceEvent["stage"], "neutral" | "blue"> = {
  ingest: "neutral",
  retrieve: "neutral",
  prompt: "blue",
  egress: "neutral",
};

export default function TraceList({ trace }: { trace: TraceEvent[] }) {
  if (trace.length === 0) return null;
  return (
    <ol className="mt-4 space-y-2 border-l-2 border-slate-200 pl-4">
      {trace.map((e, i) => (
        <li key={i} className="text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={STAGE_VARIANT[e.stage]}>{e.stage}</Badge>
            <span className="font-semibold text-slate-800">{e.event}</span>
            {e.chunk_id !== "-" && <span className="font-mono text-slate-400">{e.chunk_id}</span>}
          </div>
          <p className="mt-0.5 text-slate-600">{e.detail}</p>
        </li>
      ))}
    </ol>
  );
}
