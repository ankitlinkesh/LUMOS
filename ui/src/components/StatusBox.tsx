export function LoadingBox({ label = "Loading..." }: { label?: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-6 text-slate-600"
    >
      <span
        aria-hidden="true"
        className="h-5 w-5 flex-none animate-spin rounded-full border-2 border-slate-300 border-t-slate-600"
      />
      <span className="text-base">{label}</span>
    </div>
  );
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-amber-300 bg-amber-50 px-4 py-4 text-amber-900">
      <span className="text-base font-medium">Request failed: {message}</span>
      {onRetry && (
        <button
          onClick={onRetry}
          className="rounded-md border border-amber-400 bg-white px-3 py-1.5 text-sm font-semibold text-amber-900 hover:bg-amber-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500"
        >
          Retry
        </button>
      )}
    </div>
  );
}
