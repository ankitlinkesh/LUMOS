"""SecureRetriever vs. LeakyRetriever: declines, fail-closed on error, and the
leak-rate measurement the whole module exists to make possible."""

from __future__ import annotations

import chromadb
import pytest

from triad.contract import Chunk, Provenance, TaintVerdict
from triad.embed import HashEmbedder
from triad.retrieval.retriever import LeakyRetriever, SecureRetriever, leak_rate
from triad.retrieval.scope import Scope
from triad.retrieval.store import TenantStore


def make_chunk(cid, tenant, text, quarantined=False):
    return Chunk(
        id=cid,
        text=text,
        tenant=tenant,
        source_type="email",
        provenance=Provenance("test-ds", cid, "synthetic"),
        taint=TaintVerdict(quarantined=quarantined),
    )


@pytest.fixture
def embedder():
    return HashEmbedder(dim=128)


@pytest.fixture
def store(embedder):
    return TenantStore(client=chromadb.EphemeralClient(), space=embedder.space)


def seed(store, embedder, chunks):
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))


def test_secure_retriever_returns_only_in_scope_chunks(store, embedder):
    seed(store, embedder, [
        make_chunk("b1", "B", "quarterly earnings report finance numbers"),
        make_chunk("a1", "A", "quarterly earnings report finance numbers alpha"),
        make_chunk("a2", "A", "quarterly earnings report finance summary"),
    ])
    retriever = SecureRetriever(store=store, embedder=embedder)

    result = retriever.retrieve("quarterly earnings report finance numbers", Scope.of("A"), k=5)

    assert result.declined is False
    assert {sc.chunk.tenant for sc in result.chunks} == {"A"}
    assert result.foreign_chunks() == ()
    assert result.scope_applied == ("A",)
    assert result.latency_ms >= 0.0


def test_secure_retriever_declines_when_too_few_results(store, embedder):
    seed(store, embedder, [make_chunk("a1", "A", "quarterly earnings report")])
    retriever = SecureRetriever(store=store, embedder=embedder)

    result = retriever.retrieve("quarterly earnings report", Scope.of("A"), k=5, min_results=3)

    assert result.declined is True
    assert result.decline_reason == "not enough of your documents"
    # the one valid in-scope match is still returned, not zeroed out
    assert len(result.chunks) == 1
    assert result.chunks[0].chunk.tenant == "A"


def test_secure_retriever_never_tops_up_from_another_tenant(store, embedder):
    """Even with plenty of near-identical docs in tenant B, a scope of just A
    with too few A docs must decline rather than quietly widening."""
    seed(store, embedder, [
        make_chunk("a1", "A", "quarterly earnings report finance numbers"),
        make_chunk("b1", "B", "quarterly earnings report finance numbers"),
        make_chunk("b2", "B", "quarterly earnings report finance summary"),
        make_chunk("b3", "B", "quarterly earnings report finance figures"),
    ])
    retriever = SecureRetriever(store=store, embedder=embedder)

    result = retriever.retrieve("quarterly earnings report finance numbers", Scope.of("A"), k=5, min_results=3)

    assert result.declined is True
    assert result.foreign_chunks() == ()
    assert all(sc.chunk.tenant == "A" for sc in result.chunks)


def test_secure_retriever_empty_scope_declines(store, embedder):
    seed(store, embedder, [make_chunk("a1", "A", "hello world")])
    retriever = SecureRetriever(store=store, embedder=embedder)

    result = retriever.retrieve("hello world", Scope(frozenset()), k=5)

    assert result.declined is True
    assert result.chunks == ()


def test_secure_retriever_min_score_filters_weak_matches(store, embedder):
    seed(store, embedder, [make_chunk("a1", "A", "purple giraffe skateboard umbrella")])
    retriever = SecureRetriever(store=store, embedder=embedder)

    result = retriever.retrieve("quarterly earnings report finance numbers", Scope.of("A"), k=5, min_score=0.5)

    assert result.declined is True
    assert result.chunks == ()


def test_secure_retriever_fails_closed_on_embedder_error(store):
    class BrokenEmbedder:
        name = "broken"
        dim = 8
        space = "cosine"

        def embed_documents(self, texts):
            raise RuntimeError("boom")

        def embed_query(self, text):
            raise RuntimeError("boom")

    retriever = SecureRetriever(store=store, embedder=BrokenEmbedder())
    result = retriever.retrieve("anything", Scope.of("A"), k=5)

    assert result.declined is True
    assert "boom" in result.decline_reason
    assert result.chunks == ()


def test_quarantined_chunks_never_returned_by_secure_retriever(store, embedder):
    seed(store, embedder, [
        make_chunk("a1", "A", "quarterly earnings report", quarantined=True),
        make_chunk("a2", "A", "quarterly earnings figures"),
    ])
    retriever = SecureRetriever(store=store, embedder=embedder)
    result = retriever.retrieve("quarterly earnings report", Scope.of("A"), k=5, min_results=1)
    assert all(sc.chunk.id != "a1" for sc in result.chunks)


def _leak_probe_store(embedder):
    """A store crafted so LeakyRetriever's fallback triggers: within the
    overfetched global window (k * overfetch_multiplier = 1 * 3 = 3), every
    match is tenant B -- tenant A's one document is a weak match that never
    makes it into that window. Post-filtering to A therefore finds nothing, so
    the leaky fallback kicks in and returns the global top-k, tenant-blind."""
    store = TenantStore(client=chromadb.EphemeralClient(), space=embedder.space)
    seed(store, embedder, [
        make_chunk("b1", "B", "quarterly earnings report finance numbers"),
        make_chunk("b2", "B", "quarterly earnings report finance summary"),
        make_chunk("b3", "B", "quarterly earnings report finance figures"),
        make_chunk("a1", "A", "totally unrelated content about gardening"),
    ])
    return store


def test_leaky_retriever_leaks_on_the_fallback_case(embedder):
    store = _leak_probe_store(embedder)
    retriever = LeakyRetriever(store=store, embedder=embedder, overfetch_multiplier=3)

    result = retriever.retrieve("quarterly earnings report finance numbers", Scope.of("A"), k=1, min_results=1)

    assert result.leak_mode is True
    assert result.foreign_chunks() != ()
    assert any(sc.chunk.tenant == "B" for sc in result.chunks)


def test_leak_rate_zero_for_secure_nonzero_for_leaky(embedder):
    store = _leak_probe_store(embedder)
    secure = SecureRetriever(store=store, embedder=embedder)
    leaky = LeakyRetriever(store=store, embedder=embedder, overfetch_multiplier=3)

    probes = [("quarterly earnings report finance numbers", Scope.of("A"), 1)] * 5

    assert leak_rate(secure, probes) == 0.0
    assert leak_rate(leaky, probes) > 0.0


def test_leak_rate_of_empty_probes_is_zero(embedder, store):
    secure = SecureRetriever(store=store, embedder=embedder)
    assert leak_rate(secure, []) == 0.0
