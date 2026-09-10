// Persistent, unmissable "this is all fake" banner. Sticky at the very top of
// the page so it stays visible on every section a judge scrolls to — the
// single page IS every view, so one pinned banner covers all of them.
// Amber, deliberately not red or green: those colors are reserved for
// attacks and defenses respectively (see design brief), and this banner is
// neither -- it's a data-provenance warning that must not be confused with
// either judgment.
export default function DemoModeBanner({ note }: { note?: string }) {
  return (
    <div
      role="alert"
      className="sticky top-0 z-50 flex flex-wrap items-center justify-center gap-x-3 gap-y-1 bg-amber-400 px-4 py-2 text-center text-sm font-bold uppercase tracking-wide text-amber-950 shadow-md sm:text-base"
    >
      <span aria-hidden="true">&#9888;</span>
      <span>Demo mode &mdash; fake data, not measured results</span>
      <span className="hidden font-normal normal-case tracking-normal opacity-80 sm:inline">
        {note ?? "Confirming data source..."}
      </span>
    </div>
  );
}
