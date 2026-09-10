"""Stage 2: authorization-gated retrieval.

The problem this stage exists to close (arXiv:2605.05287): filtering by tenant
AFTER a similarity search leaks cross-tenant data in 98-100% of probes when the
system falls back to nearest-neighbours regardless of tenant; gating INSIDE the
search (the tenant predicate pushed into the vector index's own query) drops
that to 0%. ``SecureRetriever`` does the latter; ``LeakyRetriever`` reproduces
the former so the difference can be measured, not just asserted.
"""

from triad.retrieval.retriever import LeakyRetriever, SecureRetriever, leak_rate
from triad.retrieval.scope import Scope
from triad.retrieval.store import TenantStore

__all__ = ["Scope", "TenantStore", "SecureRetriever", "LeakyRetriever", "leak_rate"]
