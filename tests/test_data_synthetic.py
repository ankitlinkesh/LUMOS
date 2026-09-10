"""Synthetic data: determinism, schema-identity with the real loaders, and the
"impossible to mistake for real" invariant (data_source + dataset prefix)."""

from __future__ import annotations

from triad.data import synthetic
from triad.data.types import DatasetHandle


def test_generate_emails_deterministic_for_same_seed():
    a = synthetic.generate_emails(seed=42)
    b = synthetic.generate_emails(seed=42)
    assert a == b


def test_generate_emails_every_chunk_has_a_tenant_and_is_tagged_synthetic():
    emails = synthetic.generate_emails(seed=1)
    assert emails
    for c in emails:
        assert c.tenant and c.tenant.strip()
        assert c.provenance.data_source == "synthetic"
        assert c.provenance.dataset.startswith("synthetic:")


def test_generate_emails_covers_all_tenants_with_overlapping_topics():
    emails = synthetic.generate_emails(seed=1)
    assert {c.tenant for c in emails} == set(synthetic.TENANTS)
    subjects_by_tenant = {}
    for c in emails:
        topic = c.metadata["subject"].split(" - update")[0]
        subjects_by_tenant.setdefault(c.tenant, set()).add(topic)
    # At least two tenants must share at least one topic, or cross-tenant probes
    # have nothing to be tempted by.
    all_topic_sets = list(subjects_by_tenant.values())
    overlap_found = any(
        all_topic_sets[i] & all_topic_sets[j]
        for i in range(len(all_topic_sets))
        for j in range(i + 1, len(all_topic_sets))
    )
    assert overlap_found


def test_generate_qa_gold_and_incorrect_answers_present_and_distinct():
    qa = synthetic.generate_qa(seed=1)
    assert qa
    for r in qa:
        assert r.gold_answers
        assert r.incorrect_answers
        assert r.gold_answers[0] not in r.incorrect_answers
        assert r.tenant
        assert r.provenance.data_source == "synthetic"


def test_generate_poison_targets_use_the_same_prefix_recipe_as_the_real_loader():
    targets = synthetic.generate_poison_targets(seed=1)
    assert targets
    for t in targets:
        prefix = t.question + "."
        for raw, prefixed in zip(t.raw_adv_texts, t.adv_texts):
            assert prefixed == prefix + raw
        assert t.provenance.data_source == "synthetic"


def test_generate_llmail_hidden_payload_techniques_present():
    attacks = synthetic.generate_llmail(seed=1)
    bodies = " ".join(a.body for a in attacks)
    assert "​" in bodies            # zero-width space
    assert "<!--" in bodies              # HTML comment
    assert "color:#ffffff" in bodies     # white-on-white
    assert "<|user|>" in bodies          # fake turn marker
    assert "![status](" in bodies        # data-carrying markdown image link
    for a in attacks:
        assert a.all_objectives_met()
        assert a.provenance.data_source == "synthetic"


def test_generate_benign_tagged_synthetic():
    benign = synthetic.generate_benign(seed=1)
    assert benign
    for c in benign:
        assert c.provenance.data_source == "synthetic"


def test_dataset_handle_requires_a_reason_for_synthetic_records():
    emails = synthetic.generate_emails(seed=1)
    import pytest
    with pytest.raises(ValueError):
        DatasetHandle(name="x", records=tuple(emails))  # synthetic records, no mix_reason
    h = DatasetHandle(name="x", records=tuple(emails), mix_reason="test")
    assert h.data_source == "synthetic" and h.n_real == 0


def test_cli_writes_jsonl_files(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--seed", "7", "--out-dir", str(tmp_path)])
    synthetic.main()
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == sorted(synthetic._ALL_GENERATORS.keys())
    import json
    with open(tmp_path / "emails.jsonl", encoding="utf-8") as fh:
        first = json.loads(fh.readline())
    assert first["provenance"]["data_source"] == "synthetic"
