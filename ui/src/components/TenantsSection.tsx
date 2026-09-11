import type { Tenant } from "../api";
import Panel from "./Panel";

// A plain read-only listing -- dbmanager/securityhead/ceo can all reach
// GET /api/tenants (see the README's "Roles and access control" table); this
// is the view for it. Tenants are already fetched once by App.tsx to
// populate other forms, so this just renders what it's given.
export default function TenantsSection({ number, tenants }: { number: number; tenants: Tenant[] }) {
  return (
    <Panel
      id="tenants"
      number={number}
      title="Tenants"
      description="Every tenant this deployment currently isolates retrieval for."
    >
      <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {tenants.map((t) => (
          <li key={t.id} className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="font-mono text-sm text-slate-500">{t.id}</p>
            <p className="mt-1 text-lg font-semibold text-slate-900">{t.label}</p>
            <p className="mt-1 text-sm text-slate-600">{t.n_docs} document{t.n_docs === 1 ? "" : "s"}</p>
          </li>
        ))}
      </ul>
    </Panel>
  );
}
