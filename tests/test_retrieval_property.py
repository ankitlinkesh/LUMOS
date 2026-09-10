"""Hypothesis property tests: the cross-tenant-leak invariant must hold for
*every* combination of tenants, doc counts (including empty tenants), scope,
k, min_results and min_score -- not just the handful of cases the example-based
tests happen to cover. This is the same invariant the H1 spike in
``test_retrieval_store.py`` demonstrates once; this file fuzzes it.

Kept fast deliberately: a small fixed vocabulary (no per-doc string generation),
an in-memory ephemeral Chroma client per example, and ``deadline=None`` since
Chroma's own setup cost varies enough to trip Hypothesis's default deadline on a
slow CI box.
"""

from __future__ import annotations

import chromadb
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from triad.contract import Chunk, Provenance
from triad.embed import HashEmbedder
from triad.retrieval.retriever import SecureRetriever
from triad.retrieval.scope import Scope
from triad.retrieval.store import TenantStore

_TENANT_NAMES = ["t0", "t1", "t2", "t3", "t4", "t5"]
_VOCAB = [
    "quarterly earnings report finance numbers",
    "purple giraffe skateboard umbrella",
    "authorization gating tenant boundary",
    "cross tenant leakage incident report",
    "random unrelated gardening notes",
]


@st.composite
def _retrieval_scenario(draw):
    n_tenants = draw(st.integers(min_value=2, max_value=6))
    tenants = _TENANT_NAMES[:n_tenants]
    docs_per_tenant = {t: draw(st.integers(min_value=0, max_value=20)) for t in tenants}
    query = draw(st.sampled_from(_VOCAB))
    k = draw(st.integers(min_value=1, max_value=10))
    min_results = draw(st.integers(min_value=0, max_value=5))
    min_score = draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False)))
    scope_tenants = draw(st.sets(st.sampled_from(tenants), min_size=0, max_size=n_tenants))
    return tenants, docs_per_tenant, query, k, min_results, min_score, scope_tenants


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(_retrieval_scenario())
def test_secure_retriever_never_leaks_across_scope(scenario):
    tenants, docs_per_tenant, query, k, min_results, min_score, scope_tenants = scenario

    embedder = HashEmbedder(dim=64)
    store = TenantStore(client=chromadb.EphemeralClient(), space=embedder.space)

    chunks = []
    for tenant in tenants:
        for i in range(docs_per_tenant[tenant]):
            # Deterministic (not hypothesis-random) so this stays stable under
            # shrinking; just needs some variety across docs and tenants.
            vocab_idx = (i + _TENANT_NAMES.index(tenant)) % len(_VOCAB)
            chunks.append(
                Chunk(
                    id=f"{tenant}-{i}",
                    text=_VOCAB[vocab_idx],
                    tenant=tenant,
                    source_type="passage",
                    provenance=Provenance("prop-test", f"{tenant}-{i}", "synthetic"),
                )
            )
    if chunks:
        store.add(chunks, embedder.embed_documents([c.text for c in chunks]))

    scope = Scope(frozenset(scope_tenants))
    retriever = SecureRetriever(store=store, embedder=embedder)

    result = retriever.retrieve(query, scope, k=k, min_results=min_results, min_score=min_score)

    # The core invariant, true for the successful path AND the declined path
    # (and, per the dedicated fail-closed unit test, the error path too).
    assert result.foreign_chunks() == ()
    # scope_applied must never exceed what the caller actually authorized.
    assert set(result.scope_applied) <= scope.tenants
    if result.declined:
        assert result.decline_reason


@given(
    st.sets(st.sampled_from(_TENANT_NAMES)),
    st.sets(st.sampled_from(_TENANT_NAMES)),
)
def test_scope_narrow_never_grows(a_tenants, b_tenants):
    a = Scope(frozenset(a_tenants))
    b = Scope(frozenset(b_tenants))
    result = a.narrow(b)
    assert result.tenants <= a.tenants
    assert result.tenants <= b.tenants
