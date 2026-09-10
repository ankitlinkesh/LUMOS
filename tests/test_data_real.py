"""Slow tests against the REAL downloaded files (run with `-m slow`). They pin the
counts measured on 2026-09-10 so a broken or partial download is caught before it
silently shrinks an experiment."""

import pytest

from triad.data import bipia, enronqa, llmail, poisonedrag

pytestmark = pytest.mark.slow


def test_enronqa_is_deduplicated_to_73772_emails_across_150_tenants():
    emails = enronqa.load_emails()
    assert len(emails) == 73_772
    assert len({c.tenant for c in emails}) == 150
    assert len({c.id for c in emails}) == len(emails)
    assert all(c.provenance.data_source == "real" for c in emails)


@pytest.mark.parametrize("corpus", ["nq", "hotpotqa", "msmarco"])
def test_poisonedrag_targets_use_the_papers_question_prefix(corpus):
    targets = poisonedrag.load_targets(corpus=corpus)
    assert len(targets) == 100
    for t in targets:
        assert len(t.adv_texts) == 5
        assert all(a == t.question + "." + raw for a, raw in zip(t.adv_texts, t.raw_adv_texts))


@pytest.mark.parametrize("phase,attempts,unique", [(1, 3_018, 1_551), (2, 306, 222)])
def test_llmail_all_objectives_counts(phase, attempts, unique):
    assert len(llmail.load_attacks(phase=phase, dedupe=False)) == attempts
    assert len(llmail.load_attacks(phase=phase)) == unique


def test_llmail_benign_false_positive_set():
    assert len(llmail.load_benign()) == 203


@pytest.mark.parametrize("task,n", [("email", 50), ("table", 100), ("code", 50)])
def test_bipia_tasks_present(task, n):
    assert len(bipia.load(task=task)) == n
