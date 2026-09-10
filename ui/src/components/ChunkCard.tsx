import type { ScoredChunk } from "../api";
import Badge from "./Badge";

export default function ChunkCard({
  chunk,
  foreign = false,
  onInspect,
}: {
  chunk: ScoredChunk;
  foreign?: boolean;
  onInspect?: (chunkId: string) => void;
}) {
  const flagged = chunk.taint.quarantined || chunk.taint.flags.length > 0 || foreign;

  return (
    <li
      className={`rounded-lg border p-4 ${
        flagged ? "border-red-300 bg-red-50" : "border-slate-200 bg-white"
      }`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm text-slate-500">{chunk.id}</span>
        <Badge variant={foreign ? "red" : "neutral"}>tenant: {chunk.tenant}</Badge>
        <Badge variant="neutral">score {chunk.score.toFixed(2)}</Badge>
        <Badge variant="neutral">{chunk.source_type}</Badge>
        <Badge variant={chunk.data_source === "real" ? "blue" : "neutral"}>{chunk.data_source}</Badge>
        {foreign && <Badge variant="red">foreign tenant</Badge>}
        {chunk.taint.quarantined && <Badge variant="red">quarantined</Badge>}
        {chunk.taint.flags.map((f) => (
          <Badge key={f} variant="amber">
            {f}
          </Badge>
        ))}
        {onInspect && (
          <button
            onClick={() => onInspect(chunk.id)}
            className="ml-auto rounded-md border border-slate-300 px-2.5 py-1 text-sm font-semibold text-slate-700 hover:bg-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
          >
            View trace &rarr;
          </button>
        )}
      </div>
      <p className="mt-2 text-base text-slate-800">{chunk.preview}</p>
      {chunk.taint.reasons.length > 0 && (
        <ul className="mt-2 list-inside list-disc text-sm text-red-800">
          {chunk.taint.reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      )}
    </li>
  );
}
