"""Pure, fast pieces of triad.eval.poisonedrag: the paper's own ASR/clean-
accuracy metric (reproduced verbatim) and the poison-chunk construction.
No network, no model loading -- importing the module is safe (heavy clients
are constructed lazily inside main(), never at import time)."""

from __future__ import annotations

import chromadb
import pytest

from triad.contract import Chunk, Provenance
from triad.embed import HashEmbedder
from triad.eval import _common
from triad.eval.poisonedrag import (
    assert_all_conditions_retrieved,
    assert_stores_healthy,
    ask_and_score,
    attack_succeeded,
    clean_correct,
    clean_str,
    gold_ids_by_target,
    poison_chunks_for_targets,
)
from triad.llm.client import EmptyResponse
from triad.pipeline import DefenseConfig, Pipeline
from triad.retrieval.scope import Scope
from triad.retrieval.store import TenantStore


def test_clean_str_matches_paper_recipe():
    # data/raw/PoisonedRAG/src/utils.py:113-120
    assert clean_str("Paris.") == "paris"
    assert clean_str("  Paris  ") == "paris"
    assert clean_str("PARIS") == "paris"
    assert clean_str("a") == "a"  # len<=1: trailing "." never stripped from a 1-char string
    assert clean_str(".") == "."


def test_attack_succeeded_is_substring_match_after_clean_str():
    assert attack_succeeded("Paris", "I believe the answer is Paris, based on the context.")
    assert not attack_succeeded("Paris", "The answer is London.")
    # case/period insensitive, matching the paper's own clean_str on both sides
    assert attack_succeeded("PARIS.", "the city is paris and it is lovely")


def test_clean_correct_same_semantics_as_attack_succeeded():
    assert clean_correct("42", "The answer is 42.")
    assert not clean_correct("42", "The answer is 43.")


class _FakeTarget:
    def __init__(self, tid, corpus, question, adv_texts):
        self.id = tid
        self.corpus = corpus
        self.question = question
        self.adv_texts = adv_texts


def test_poison_chunks_for_targets_builds_5_chunks_per_target_real_provenance():
    targets = [
        _FakeTarget("test1", "nq", "who is the CEO?", (
            "who is the CEO?.A", "who is the CEO?.B", "who is the CEO?.C",
            "who is the CEO?.D", "who is the CEO?.E",
        )),
    ]
    chunks = poison_chunks_for_targets(targets)
    assert len(chunks) == 5
    ids = {c.id for c in chunks}
    assert len(ids) == 5  # all unique
    for c in chunks:
        assert c.tenant == "public"
        assert c.source_type == "passage"
        assert c.provenance.data_source == "real"  # the paper's OWN released text, not fabricated
        assert c.provenance.dataset == "poisonedrag:nq"
        assert c.text.startswith("who is the CEO?.")


def test_poison_chunks_for_multiple_targets_ids_never_collide():
    targets = [
        _FakeTarget("test1", "nq", "q1", tuple(f"q1.{i}" for i in range(5))),
        _FakeTarget("test2", "nq", "q2", tuple(f"q2.{i}" for i in range(5))),
    ]
    chunks = poison_chunks_for_targets(targets)
    assert len(chunks) == 10
    assert len({c.id for c in chunks}) == 10


# -- the two eval-level guards (the most important deliverable of the ------
# -- silent-empty-retrieval postmortem: a condition that retrieves nothing --
# -- must fail the run loudly, never get scored) ---------------------------

def _make_chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, text=text, tenant="public", source_type="passage",
                 provenance=Provenance("test-ds", cid, "synthetic"))


def _pipeline(embedder, chunks) -> Pipeline:
    store = TenantStore(client=chromadb.EphemeralClient(), space=embedder.space)
    if chunks:
        store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    defense_off = DefenseConfig(stage1a=False, stage1b=False, secure_retrieval=False, collapse_topk=False, stage3_enabled=False)
    return Pipeline(store=store, embedder=embedder, llm=None, defense=defense_off)


def test_assert_stores_healthy_passes_when_all_four_stores_retrieve_something():
    embedder = HashEmbedder(dim=64)
    chunks = [_make_chunk(f"c{i}", "quarterly earnings report finance numbers") for i in range(5)]
    pipelines = {name: _pipeline(embedder, chunks) for name in
                 ("off_clean", "off_poisoned", "on_clean", "on_poisoned")}

    assert_stores_healthy(pipelines, "quarterly earnings report finance numbers", Scope.of("public"))  # must not raise


def test_assert_stores_healthy_raises_on_an_empty_store():
    embedder = HashEmbedder(dim=64)
    healthy_chunks = [_make_chunk(f"c{i}", "quarterly earnings report finance numbers") for i in range(5)]
    pipelines = {
        "off_clean": _pipeline(embedder, []),  # never populated -- the exact failure mode observed
        "off_poisoned": _pipeline(embedder, healthy_chunks),
        "on_clean": _pipeline(embedder, healthy_chunks),
        "on_poisoned": _pipeline(embedder, healthy_chunks),
    }

    with pytest.raises(SystemExit, match="off_clean"):
        assert_stores_healthy(pipelines, "quarterly earnings report finance numbers", Scope.of("public"))


class _FakeStore:
    """Reports a healthy ``count()`` -- the exact confirmed failure signature
    is a store whose metadata says it's populated while retrieval still comes
    back empty, which ``count()`` alone can never catch."""

    def count(self):
        return 10_117  # matches the real incident's clean-corpus size


class _FakeRetrieval:
    chunks = ()


class _FakeRetriever:
    def retrieve(self, query, scope, k=5):
        return _FakeRetrieval()


class _FakePipeline:
    """Duck-types just what ``assert_stores_healthy`` touches: ``.store`` and
    ``._retriever()``. Used instead of a real ``Pipeline`` here because
    ``LeakyRetriever`` always returns *something* from a non-empty store
    (unscoped top-k has no relevance threshold) -- the real incident's
    "populated but empty" signature can only be produced at the Chroma
    query layer (covered separately in test_retrieval_store.py's
    mismatched-length regression test), so this test stubs the retriever
    directly to prove ``assert_stores_healthy`` itself reacts correctly to
    that signature, independent of what produces it."""

    store = _FakeStore()

    def _retriever(self):
        return _FakeRetriever()


def test_assert_stores_healthy_raises_when_a_populated_store_retrieves_nothing():
    """Simulates the confirmed failure signature directly: ``store.count() >
    0`` but a query returns 0 chunks -- the guard must catch this even though
    it is NOT just a ``count() == 0`` check."""
    embedder = HashEmbedder(dim=64)
    healthy_chunks = [_make_chunk(f"c{i}", "quarterly earnings report finance numbers") for i in range(5)]
    pipelines = {
        "off_clean": _FakePipeline(),
        "off_poisoned": _pipeline(embedder, healthy_chunks),
        "on_clean": _pipeline(embedder, healthy_chunks),
        "on_poisoned": _pipeline(embedder, healthy_chunks),
    }

    with pytest.raises(SystemExit, match="off_clean"):
        assert_stores_healthy(pipelines, "quarterly earnings report finance numbers", Scope.of("public"))


def _result(n_retrieved, empty_response_failure=False):
    return {"n_retrieved": n_retrieved, "empty_response_failure": empty_response_failure}


def test_assert_all_conditions_retrieved_passes_when_every_probe_retrieved_something():
    per_target = {
        "asr_off": [_result(5), _result(5)],
        "asr_on": [_result(5), _result(5)],
        "clean_off": [_result(5), _result(5)],
        "clean_on": [_result(5), _result(5)],
    }
    assert_all_conditions_retrieved(per_target)  # must not raise


def test_assert_all_conditions_retrieved_raises_on_the_confirmed_bug_signature():
    per_target = {
        "asr_off": [_result(5), _result(5)],
        "asr_on": [_result(5), _result(5)],
        "clean_off": [_result(0), _result(0)],  # the exact signature this whole guard exists to catch
        "clean_on": [_result(5), _result(5)],
    }
    with pytest.raises(RuntimeError, match="clean_off"):
        assert_all_conditions_retrieved(per_target)


def test_assert_all_conditions_retrieved_ignores_empty_response_failures():
    """An LLM call that failed after retries (``n_retrieved: None``) is a
    scored failure, not a silent-empty-retrieval bug -- the guard must not
    conflate the two."""
    per_target = {
        "asr_off": [_result(None, empty_response_failure=True), _result(5)],
        "asr_on": [_result(5)],
        "clean_off": [_result(5)],
        "clean_on": [_result(5)],
    }
    assert_all_conditions_retrieved(per_target)  # must not raise


# -- the (A) vs (B) diagnostic fields: n_poison_in_topk / n_gold_in_topk / --
# -- retrieved_ids, added to every per-target row (record-only, no ---------
# -- scoring/threshold/retrieval/prompt change) ----------------------------

def test_gold_ids_by_target_keyed_per_target_not_flattened(tmp_path):
    """Unlike ``beir.gold_ids_for_queries`` (one combined tuple for ALL
    requested ids -- what corpus construction needs), this keeps gold ids
    keyed by target id, since scoring needs to know which gold id belongs to
    which question."""
    root = tmp_path
    (root / "qrels").mkdir()
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        "test1\tdoc6\t1\n"
        "test1\tdoc7\t1\n"
        "test107\tdoc3859\t1\n",
        encoding="utf-8",
    )
    result = gold_ids_by_target(["test1", "test107", "test999"], root=root)
    assert result["test1"] == ("doc6", "doc7")
    assert result["test107"] == ("doc3859",)
    assert result["test999"] == ()  # requested but absent from qrels: empty tuple, not a missing key


def test_gold_ids_by_target_empty_input_returns_empty_dict():
    assert gold_ids_by_target([]) == {}


class _FakeAskTarget:
    def __init__(self, tid, question, incorrect_answer, correct_answer):
        self.id = tid
        self.question = question
        self.incorrect_answer = incorrect_answer
        self.correct_answer = correct_answer


class _FakeChatLLM:
    """Duck-types the ``llm.chat`` surface ``Pipeline.ask`` calls -- same
    shape as ``test_pipeline_core.FakeLLM``."""

    def __init__(self, text="42"):
        self.text = text

    def chat(self, messages, *, principal, scope=(), model=None, max_tokens=None, temperature=None):
        from types import SimpleNamespace
        return SimpleNamespace(text=self.text, cached=False)


class _AlwaysEmptyLLM:
    def chat(self, messages, *, principal, scope=(), model=None, max_tokens=None, temperature=None):
        raise EmptyResponse("no usable text")


def test_ask_and_score_records_poison_and_gold_counts_and_retrieved_ids():
    embedder = HashEmbedder(dim=64)
    gold_chunk = _make_chunk("doc6", "quarterly earnings report finance numbers")
    poison_chunk = _make_chunk("poison:nq:test1:0", "quarterly earnings report finance numbers poison")
    filler_chunk = _make_chunk("filler1", "quarterly earnings report finance numbers filler")
    # k == store size: all 3 chunks are retrieved deterministically regardless of score.
    pipeline = _pipeline(embedder, [gold_chunk, poison_chunk, filler_chunk])
    target = _FakeAskTarget("test1", "what were the quarterly earnings?", "wrong", "42")
    cache_stats = _common.CacheStats()

    results = ask_and_score(
        pipeline, _FakeChatLLM(text="42"), [target], scored_field="clean", k=3,
        cache_stats=cache_stats, gold_ids_by_target={"test1": ("doc6",)},
    )

    assert len(results) == 1
    r = results[0]
    assert set(r["retrieved_ids"]) == {"doc6", "poison:nq:test1:0", "filler1"}
    assert r["n_poison_in_topk"] == 1
    assert r["n_gold_in_topk"] == 1
    assert r["success"] is True


def test_ask_and_score_n_gold_in_topk_is_none_without_a_gold_map():
    """The task brief: if gold ids aren't available, skip the field rather
    than guessing -- ``None`` (not 0) records "unknown", distinct from
    "known and zero"."""
    embedder = HashEmbedder(dim=64)
    poison_chunk = _make_chunk("poison:nq:test1:0", "quarterly earnings report finance numbers poison")
    filler_chunk = _make_chunk("filler1", "quarterly earnings report finance numbers filler")
    pipeline = _pipeline(embedder, [poison_chunk, filler_chunk])
    target = _FakeAskTarget("test1", "what were the quarterly earnings?", "wrong", "42")

    results = ask_and_score(
        pipeline, _FakeChatLLM(text="42"), [target], scored_field="clean", k=2,
        cache_stats=_common.CacheStats(),  # no gold_ids_by_target passed
    )

    assert results[0]["n_gold_in_topk"] is None
    assert results[0]["n_poison_in_topk"] == 1  # unaffected by the gold map's absence


def test_ask_and_score_empty_response_failure_leaves_diagnostic_fields_none():
    embedder = HashEmbedder(dim=64)
    filler_chunk = _make_chunk("filler1", "quarterly earnings report finance numbers filler")
    pipeline = _pipeline(embedder, [filler_chunk])
    target = _FakeAskTarget("test1", "what were the quarterly earnings?", "wrong", "42")

    results = ask_and_score(
        pipeline, _AlwaysEmptyLLM(), [target], scored_field="clean", k=1,
        cache_stats=_common.CacheStats(), gold_ids_by_target={"test1": ("doc6",)},
    )

    r = results[0]
    assert r["empty_response_failure"] is True
    assert r["retrieved_ids"] is None
    assert r["n_poison_in_topk"] is None
    assert r["n_gold_in_topk"] is None
