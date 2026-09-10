"""Pure, fast pieces of triad.eval.poisonedrag: the paper's own ASR/clean-
accuracy metric (reproduced verbatim) and the poison-chunk construction.
No network, no model loading -- importing the module is safe (heavy clients
are constructed lazily inside main(), never at import time)."""

from __future__ import annotations

from triad.eval.poisonedrag import (
    attack_succeeded,
    clean_correct,
    clean_str,
    poison_chunks_for_targets,
)


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
