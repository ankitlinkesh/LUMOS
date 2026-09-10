"""get_dataset: real data first, always. Synthetic records are only MIXED INTO a real
dataset on explicit request (with a reason), never substituted when a loader fails,
and TRIAD_REQUIRE_REAL=1 forbids mixing for reported numbers."""

from __future__ import annotations

import json

import pytest

from triad.contract import Chunk, Provenance
from triad.data import registry
from triad.data.errors import DataUnavailable
from triad.data.types import DatasetHandle


def _poison_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    entries = {
        f"test{i}": {"id": f"test{i}", "question": f"question {i}", "correct answer": "a",
                     "incorrect answer": "b", "adv_texts": ["x", "y", "z", "u", "v"]}
        for i in range(2)
    }
    (root / "nq.json").write_text(json.dumps(entries), encoding="utf-8")
    return root


def _real_emails(**_):
    return [Chunk(id=f"alice/inbox/{i}.", text=f"real mail {i}", tenant="alice", source_type="email",
                  provenance=Provenance("enronqa", f"alice/inbox/{i}.", "real")) for i in range(3)]


def test_every_real_dataset_has_a_synthetic_generator_to_mix_with():
    assert set(registry.REAL_LOADERS) <= set(registry.SYNTHETIC_GENERATORS)


def test_unknown_dataset_name_raises_value_error():
    with pytest.raises(ValueError):
        registry.get_dataset("not-a-real-dataset")


def test_real_only_by_default(tmp_path, capsys):
    h = registry.get_dataset("poisonedrag.targets", root=_poison_fixture(tmp_path), corpus="nq")
    assert h.data_source == "real" and (h.n_real, h.n_synthetic) == (2, 0)
    assert h.mix_reason is None and "MIXED" not in capsys.readouterr().err


def test_missing_real_file_raises_and_never_substitutes(tmp_path, monkeypatch):
    monkeypatch.delenv("TRIAD_REQUIRE_REAL", raising=False)
    with pytest.raises(DataUnavailable):
        registry.get_dataset("enronqa.emails", root=tmp_path / "missing")


def test_broken_real_file_raises_data_unavailable(tmp_path):
    (tmp_path / "nq.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DataUnavailable):
        registry.get_dataset("poisonedrag.targets", root=tmp_path, corpus="nq")


def test_mixing_adds_to_real_records_and_reports_composition(tmp_path, capsys):
    h = registry.get_dataset("poisonedrag.targets", root=_poison_fixture(tmp_path), corpus="nq",
                             synthetic_n=3, mix_reason="need more target questions")
    assert h.data_source == "mixed"
    assert h.n_real == 2 and 1 <= h.n_synthetic <= 3
    assert sum(r.provenance.data_source == "real" for r in h.records) == 2
    assert "2 real +" in h.describe() and "need more target questions" in h.describe()
    assert "MIXED DATASET" in capsys.readouterr().err


def test_mixing_requires_a_reason(tmp_path):
    with pytest.raises(ValueError):
        registry.get_dataset("poisonedrag.targets", root=_poison_fixture(tmp_path), corpus="nq", synthetic_n=2)


def test_require_real_forbids_mixing(tmp_path, monkeypatch):
    root = _poison_fixture(tmp_path)
    monkeypatch.setenv("TRIAD_REQUIRE_REAL", "1")
    assert registry.get_dataset("poisonedrag.targets", root=root, corpus="nq").data_source == "real"
    with pytest.raises(DataUnavailable):
        registry.get_dataset("poisonedrag.targets", root=root, corpus="nq", synthetic_n=2, mix_reason="x")


def test_require_real_only_on_literal_one(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIAD_REQUIRE_REAL", "true")
    h = registry.get_dataset("poisonedrag.targets", root=_poison_fixture(tmp_path), corpus="nq",
                             synthetic_n=1, mix_reason="dev")
    assert h.data_source == "mixed"


def test_synthetic_records_planted_into_real_tenants(monkeypatch):
    monkeypatch.setitem(registry.REAL_LOADERS, "enronqa.emails", _real_emails)
    h = registry.get_dataset("enronqa.emails", synthetic_n=4, mix_reason="hidden-payload coverage",
                             into_tenants=("alice",))
    synth = [r for r in h.records if r.provenance.data_source == "synthetic"]
    assert len(synth) == 4 and {r.tenant for r in synth} == {"alice"}
    assert all(r.id.startswith("synthetic:") for r in synth)
    assert len({r.id for r in h.records}) == len(h.records)  # no id collisions with real mail


def test_handle_counts_composition_from_records_not_from_caller():
    real = _real_emails()
    with pytest.raises(ValueError):
        DatasetHandle(name="x", records=tuple(real), mix_reason="claims a mix that isn't there")
    assert DatasetHandle(name="x", records=tuple(real)).describe() == "x: 3 real"
