"""The frozen contract every TRIAD-RAG stage codes against.

Three stages, one invariant: a retrieved chunk may supply facts, but it may never
widen its own retrieval scope, authorize an action, or cause an egress. These types
carry the evidence each stage needs to enforce that, from ingestion to the answer.

Change these types only by agreement: every stage, the eval harness and the UI
depend on them. All are frozen (immutable) so a stage cannot quietly rewrite the
provenance or tenant of something it was handed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

DATA_SOURCES = frozenset({"real", "synthetic"})
STAGES = frozenset({"ingest", "retrieve", "generate", "egress"})


@dataclass(frozen=True)
class Provenance:
    """Where a chunk came from. ``data_source`` is mandatory so a synthetic
    fallback record can never be mistaken for real data in a reported number."""

    dataset: str          # e.g. "enronqa", "poisonedrag:nq", "llmail:phase2", "synthetic:enron"
    record_id: str        # stable id within that dataset (EnronQA uses the email `path`)
    data_source: str      # "real" | "synthetic"

    def __post_init__(self) -> None:
        if self.data_source not in DATA_SOURCES:
            raise ValueError(f"data_source must be one of {sorted(DATA_SOURCES)}, got {self.data_source!r}")
        if not self.dataset or not self.record_id:
            raise ValueError("provenance needs a dataset and a record_id")


@dataclass(frozen=True)
class TaintVerdict:
    """Stage 1's judgement of a chunk. Retrieved content is always untrusted DATA;
    ``quarantined`` means it must never become searchable."""

    untrusted: bool = True
    quarantined: bool = False
    flags: tuple[str, ...] = ()           # e.g. ("hidden_text", "query_echo")
    score: float = 0.0                    # 0 = clean .. 1 = certainly malicious
    reasons: tuple[str, ...] = ()         # human-readable, shown in the quarantine queue


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    tenant: str                           # the owning principal; never empty
    source_type: str                      # "email" | "passage" | "document"
    provenance: Provenance
    taint: TaintVerdict = field(default_factory=TaintVerdict)
    metadata: Mapping[str, str] = field(default_factory=dict)   # subject, sender, ...

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("chunk id must be non-empty")
        if not self.tenant or not self.tenant.strip():
            raise ValueError(f"chunk {self.id!r} has no tenant: every chunk must be owned")


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float                          # similarity; higher = closer


@dataclass(frozen=True)
class RetrievalResult:
    """What Stage 2 returns. ``declined`` is a first-class outcome: when too few of
    the caller's own documents match, the answer is "not enough of YOUR documents",
    never a widened search."""

    query: str
    tenant: str                           # the principal the search ran as
    chunks: tuple[ScoredChunk, ...]
    declined: bool = False
    decline_reason: str | None = None
    scope_applied: tuple[str, ...] = ()   # tenants the search was allowed to see
    latency_ms: float = 0.0
    leak_mode: bool = False               # True only for the deliberately vulnerable baseline

    def __post_init__(self) -> None:
        if self.declined and not self.decline_reason:
            raise ValueError("a declined retrieval must say why")

    def foreign_chunks(self) -> tuple[ScoredChunk, ...]:
        """Chunks owned by a tenant outside the applied scope: must always be empty
        unless ``leak_mode`` is on. The Stage 2 property test asserts on this."""
        allowed = set(self.scope_applied) or {self.tenant}
        return tuple(sc for sc in self.chunks if sc.chunk.tenant not in allowed)


@dataclass(frozen=True)
class GuardDecision:
    """Every stage's verdict. ``escalate`` means a human must confirm; it never
    coexists with ``allow``. ``rewritten`` holds content the guard changed (for
    example an answer with a data-carrying link removed)."""

    stage: str
    allow: bool
    escalate: bool = False
    reasons: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    rewritten: str | None = None
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {sorted(STAGES)}, got {self.stage!r}")
        if self.allow and self.escalate:
            raise ValueError("a decision cannot both allow and escalate")
        if not self.allow and not self.reasons:
            raise ValueError("a blocking or escalating decision must give a reason")

    @classmethod
    def ok(cls, stage: str, **kw: Any) -> "GuardDecision":
        return cls(stage=stage, allow=True, **kw)

    @classmethod
    def block(cls, stage: str, *reasons: str, **kw: Any) -> "GuardDecision":
        return cls(stage=stage, allow=False, reasons=tuple(reasons), **kw)

    @classmethod
    def needs_human(cls, stage: str, *reasons: str, **kw: Any) -> "GuardDecision":
        return cls(stage=stage, allow=False, escalate=True, reasons=tuple(reasons), **kw)
