"""Eval-harness record types that are NOT part of the frozen cross-team contract.

`triad.contract` freezes only the types every stage touches (Chunk, RetrievalResult,
GuardDecision, ...) -- its own docstring scopes "these types" to that set. QA pairs,
PoisonedRAG targets and LLMail attacks are inputs to the eval harness, not things a
stage passes to another stage, so they live here instead of widening the frozen file.

All frozen; all carry a `Provenance` so `data_source` travels with every record, the
same invariant `Chunk` enforces -- a synthetic QA pair or poison target must be as
hard to mistake for real as a synthetic Chunk is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from triad.contract import Provenance


@dataclass(frozen=True)
class QARecord:
    """One question over one EnronQA email. `incorrect_answers` is this question's
    own list (EnronQA aligns questions[i] / gold_answers[i] / incorrect_answers[i]
    by index within a row) -- never a pool shared across a whole inbox."""

    question: str
    gold_answers: tuple[str, ...]
    incorrect_answers: tuple[str, ...]
    email_path: str               # links back to the Chunk.id from enronqa.load_emails
    tenant: str
    provenance: Provenance

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("QARecord needs a non-empty question")
        if not self.tenant.strip():
            raise ValueError(f"QARecord {self.question!r} has no tenant")


@dataclass(frozen=True)
class PoisonTarget:
    """One PoisonedRAG target question, with the released `adv_texts` both raw (as
    shipped in the JSON) and prefixed exactly per the paper's own attack code -- see
    `triad.data.poisonedrag` for the cited line."""

    id: str
    corpus: str                   # "nq" | "hotpotqa" | "msmarco"
    question: str
    correct_answer: str
    incorrect_answer: str
    raw_adv_texts: tuple[str, ...]     # as released, no question prefix
    adv_texts: tuple[str, ...]         # question + "." + raw_adv_texts[i], attack.py's recipe
    provenance: Provenance

    def __post_init__(self) -> None:
        if len(self.raw_adv_texts) != len(self.adv_texts):
            raise ValueError(f"PoisonTarget {self.id!r}: raw/prefixed adv_texts count mismatch")


@dataclass(frozen=True)
class InjectionAttack:
    """One LLMail-Inject submission (or a synthetic stand-in shaped like one)."""

    id: str
    subject: str
    body: str
    scenario: str
    objectives: Mapping[str, bool]   # the five email.retrieved / defense.undetected / exfil.* keys
    provenance: Provenance

    def all_objectives_met(self) -> bool:
        keys = ("email.retrieved", "defense.undetected", "exfil.sent", "exfil.destination", "exfil.content")
        return all(self.objectives.get(k) for k in keys)


@dataclass(frozen=True)
class BipiaRecord:
    """One BIPIA context + question (EmailQA / TableQA / CodeQA)."""

    task: str                     # "email" | "table" | "code"
    context: str
    question: str
    ideal: str
    provenance: Provenance


DATASET_RECORD = QARecord | PoisonTarget | InjectionAttack | BipiaRecord


def _record_data_source(record: object) -> str | None:
    """Read the `data_source` a record carries, whether it's a Chunk (`.provenance`)
    or one of the types above (also `.provenance`). Returns None for anything that
    carries no provenance at all, so `DatasetHandle` can skip -- not crash on -- a
    record type nobody has wired provenance into yet."""

    prov = getattr(record, "provenance", None)
    return getattr(prov, "data_source", None)


@dataclass(frozen=True)
class DatasetHandle:
    """What `triad.data.registry.get_dataset` returns: real data by default, with
    synthetic records only ever MIXED IN on explicit request, never substituted.

    The composition is COUNTED from each record's own `provenance.data_source`, not
    declared by the caller, so a handle cannot misreport how much of it is real.
    Every number reported from a mixed dataset must carry `describe()`."""

    name: str
    records: tuple[object, ...]
    mix_reason: str | None = None          # required whenever any synthetic record is present
    n_real: int = field(init=False)
    n_synthetic: int = field(init=False)

    def __post_init__(self) -> None:
        untagged = [r for r in self.records if _record_data_source(r) not in ("real", "synthetic")]
        if untagged:
            raise ValueError(f"{self.name}: {len(untagged)} record(s) carry no real/synthetic provenance")
        n_syn = sum(1 for r in self.records if _record_data_source(r) == "synthetic")
        object.__setattr__(self, "n_synthetic", n_syn)
        object.__setattr__(self, "n_real", len(self.records) - n_syn)
        if n_syn and not self.mix_reason:
            raise ValueError(f"{self.name}: synthetic records present, so a mix_reason is required")
        if not n_syn and self.mix_reason:
            raise ValueError(f"{self.name}: mix_reason given but no synthetic records were mixed in")

    @property
    def data_source(self) -> str:
        """'real', 'mixed', or 'synthetic' (only when every record is synthetic)."""
        if not self.n_synthetic:
            return "real"
        return "synthetic" if not self.n_real else "mixed"

    def describe(self) -> str:
        base = f"{self.name}: {self.n_real:,} real"
        if not self.n_synthetic:
            return base
        return f"{base} + {self.n_synthetic:,} synthetic ({self.mix_reason})"
