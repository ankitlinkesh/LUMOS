import { useState } from "react";
import { api, ApiError } from "../api";

export default function LoginScreen({ onLoggedIn }: { onLoggedIn: () => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.login(username, password);
      onLoggedIn();
    } catch (e) {
      if (e instanceof ApiError) {
        const retryAfter = e.message; // server always sends the same generic message
        setError(retryAfter || "Invalid username or password.");
      } else {
        setError("Could not reach the server. Try again.");
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="lumo-shell flex min-h-screen items-center justify-center px-6">
      <div className="w-full max-w-sm rounded-2xl border border-purple-400/20 bg-black/30 p-8 shadow-2xl backdrop-blur">
        <div className="mb-6 flex items-center gap-2">
          <span aria-hidden="true" className="grid h-8 w-8 place-items-center rounded-lg border border-purple-300 text-purple-200">✦</span>
          <span className="text-sm font-bold tracking-[0.18em] text-white">
            TRIAD<span className="text-purple-400">/</span>RAG
          </span>
        </div>
        <h1 className="text-xl font-bold text-white">Sign in</h1>
        <p className="mt-1 text-sm text-purple-200/70">
          Security console access is role-scoped and enforced on the server.
        </p>

        <form onSubmit={submit} className="mt-6 flex flex-col gap-4">
          <label className="flex flex-col gap-1">
            <span className="text-xs font-semibold uppercase tracking-wide text-purple-200/70">Username</span>
            <input
              autoFocus
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="rounded-md border border-purple-400/30 bg-white/5 px-3 py-2 text-base text-white placeholder:text-purple-200/40 focus:outline-none focus-visible:ring-2 focus-visible:ring-purple-400"
              placeholder="e.g. allen-p"
              autoComplete="username"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs font-semibold uppercase tracking-wide text-purple-200/70">Password</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="rounded-md border border-purple-400/30 bg-white/5 px-3 py-2 text-base text-white placeholder:text-purple-200/40 focus:outline-none focus-visible:ring-2 focus-visible:ring-purple-400"
              autoComplete="current-password"
            />
          </label>

          {error && (
            <p role="alert" className="rounded-lg border border-red-400/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={busy || !username || !password}
            className="mt-2 rounded-md bg-purple-500 px-4 py-2.5 text-base font-semibold text-white hover:bg-purple-400 disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-purple-300"
          >
            {busy ? "Signing in..." : "Sign in"}
          </button>
        </form>

        <p className="mt-6 text-xs text-purple-200/50">
          Demo credentials are documented in the README's "Demo credentials" section.
        </p>
      </div>
    </div>
  );
}
