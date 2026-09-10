"""Stage 2's two retrievers: the real one, and the deliberately broken baseline
it exists to be measured against.

``SecureRetriever`` fails closed: any exception, or too few of the caller's own
documents, becomes a declined result -- never an unfiltered search and never a
top-up from someone else's tenant. ``LeakyRetriever`` reproduces the exact
published bug (arXiv:2605.05287): a global similarity search, filtered by tenant
only *after* the fact, with a fallback that ignores tenant entirely when too few
survive the filter. It exists only so the leak rate can be measured and shown to
be zero for the secure path and nonzero for the naive one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from triad.contract import RetrievalResult
from triad.embed.base import Embedder
from triad.retrieval.scope import Scope
from triad.retrieval.store import TenantStore

_DECLINE_NOT_ENOUGH = "not enough of your documents"


def _tenant_label(scope: Scope) -> str:
    """A single display string for ``RetrievalResult.tenant``. The result field
    is documented as "the principal the search ran as"; when the scope holds
    more than one tenant (e.g. a role that can see several), the label lists
    them all rather than arbitrarily picking one."""
    tenants = sorted(scope.tenants)
    return "+".join(tenants)


@dataclass
class SecureRetriever:
    """Search that never widens its own scope. Every result's ``foreign_chunks()``
    is empty by construction -- both because the underlying ``TenantStore.search``
    pushes the tenant predicate into Chroma, and because this class re-checks
    every returned chunk against the scope before returning (defense in depth:
    the invariant holds even if the store implementation is ever swapped for one
    that filters less carefully)."""

    store: TenantStore
    embedder: Embedder

    def retrieve(
        self,
        query: str,
        scope: Scope,
        k: int = 5,
        min_results: int = 1,
        min_score: float | None = None,
    ) -> RetrievalResult:
        start = time.perf_counter()
        scope_applied = tuple(sorted(scope.tenants))
        tenant_field = _tenant_label(scope)
        try:
            chunks = self._search_in_scope(query, scope, k, min_score)
        except Exception as exc:  # fail closed: any failure declines, never falls through to an unfiltered search
            latency_ms = (time.perf_counter() - start) * 1000
            return RetrievalResult(
                query=query,
                tenant=tenant_field,
                chunks=(),
                declined=True,
                decline_reason=f"retrieval error: {exc}",
                scope_applied=scope_applied,
                latency_ms=latency_ms,
            )

        latency_ms = (time.perf_counter() - start) * 1000
        if len(chunks) < min_results:
            # Too few of the caller's OWN documents matched. Return what did
            # match (never zero it out) but never top up from another tenant.
            return RetrievalResult(
                query=query,
                tenant=tenant_field,
                chunks=chunks,
                declined=True,
                decline_reason=_DECLINE_NOT_ENOUGH,
                scope_applied=scope_applied,
                latency_ms=latency_ms,
            )
        return RetrievalResult(
            query=query,
            tenant=tenant_field,
            chunks=chunks,
            scope_applied=scope_applied,
            latency_ms=latency_ms,
        )

    def _search_in_scope(self, query: str, scope: Scope, k: int, min_score: float | None) -> tuple:
        if not scope:  # an empty scope searches nothing -- fail closed, not "everything"
            return ()
        query_vec = self.embedder.embed_query(query)
        scored = self.store.search(query_vec, scope.tenants, k)
        if min_score is not None:
            scored = [sc for sc in scored if sc.score >= min_score]
        # Re-check against scope even though store.search already scoped the
        # `where` clause: the invariant must hold no matter what the store did.
        scored = [sc for sc in scored if sc.chunk.tenant in scope.tenants]
        return tuple(scored)


@dataclass
class LeakyRetriever:
    """The deliberately vulnerable baseline. Exists ONLY to demonstrate, and let
    ``leak_rate`` measure, the bug SecureRetriever fixes: enterprise RAG systems
    that filter by tenant AFTER a global similarity search, and fall back to
    tenant-blind results when too few survive the filter. Never wire this into
    anything that serves real queries; ``leak_mode=True`` is stamped on every
    result it returns so downstream code can refuse to trust it.
    """

    store: TenantStore
    embedder: Embedder
    overfetch_multiplier: int = 3  # the "k·m" global overfetch before filtering

    def retrieve(
        self,
        query: str,
        scope: Scope,
        k: int = 5,
        min_results: int = 1,
        min_score: float | None = None,
    ) -> RetrievalResult:
        start = time.perf_counter()
        query_vec = self.embedder.embed_query(query)
        # THE BUG, step 1: search globally, with no tenant predicate at all.
        overfetched = self.store.search_unscoped(query_vec, k=k * self.overfetch_multiplier)
        if min_score is not None:
            overfetched = [sc for sc in overfetched if sc.score >= min_score]

        # THE BUG, step 2: filter by tenant only now, after the search already ran.
        in_tenant = [sc for sc in overfetched if sc.chunk.tenant in scope.tenants][:k]

        if len(in_tenant) < min_results:
            # THE BUG, step 3: too few survived the filter, so fall back to the
            # raw global top-k -- tenant boundary ignored entirely.
            chunks = tuple(overfetched[:k])
        else:
            chunks = tuple(in_tenant)

        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            query=query,
            tenant=_tenant_label(scope),
            chunks=chunks,
            scope_applied=tuple(sorted(scope.tenants)),
            latency_ms=latency_ms,
            leak_mode=True,
        )


def leak_rate(retriever, probes: Sequence[tuple]) -> float:
    """Fraction of ``probes`` whose result has any ``foreign_chunks()``.

    Each probe is ``(query, scope)`` or ``(query, scope, k)``; ``k`` defaults to
    5 when omitted. Returns 0.0 for an empty probe set (vacuously no leaks)."""
    if not probes:
        return 0.0
    leaked = 0
    for probe in probes:
        if len(probe) == 3:
            query, scope, k = probe
        else:
            query, scope = probe
            k = 5
        result = retriever.retrieve(query, scope, k=k)
        if result.foreign_chunks():
            leaked += 1
    return leaked / len(probes)
