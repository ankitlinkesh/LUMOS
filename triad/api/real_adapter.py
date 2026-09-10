"""Adapter skeleton: wraps a real pipeline object so it satisfies ``DemoService``.

``triad.pipeline.Pipeline`` does not exist yet (the real pipeline is being built
separately). This module is intentionally a stub with the right shape — the
orchestrator finishes each method once ``Pipeline`` lands. Every method is typed
against the same ``DemoService`` Protocol as ``FakeDemoService`` so swapping one
for the other in ``__main__.py`` requires no other changes.
"""

from __future__ import annotations

from typing import Any

from triad.api.service import (
    AskResult,
    ChunkTraceStep,
    ProbeResult,
    QuarantineItem,
    ResultRow,
    ServiceMeta,
    TenantInfo,
)

__all__ = ["RealDemoService"]


class RealDemoService:
    """Implements ``DemoService`` by delegating to a real ``Pipeline`` instance.

    TODO (orchestrator): every method below needs a real implementation once
    ``triad.pipeline.Pipeline`` exists. Suggested mapping, based on the frozen
    contract types (see ``triad/contract.py``):
      - list_tenants: enumerate the tenants the pipeline's store knows about.
      - ask: run Stage 2 retrieval (RetrievalResult) with defense on/off,
        then Stage 3/generation; wrap chunks as ScoredChunk (already the
        contract type, no reshaping needed) and each stage's GuardDecision as
        a TraceEvent.
      - list_quarantine / release_quarantine: read/mutate Stage 1's quarantine
        store, keyed by Chunk.id.
      - probe: run retrieval twice (leak_mode=True baseline vs. defended) and
        report RetrievalResult.foreign_chunks(); property_test can surface
        the Stage 2 property-test pass/fail count if the harness exposes it.
      - trace_for_chunk: walk the chunk's recorded GuardDecisions per stage.
      - results_table: read the eval harness's persisted attack-success-rate
        table; only ever return real measured numbers, never fabricate them
        (ResultRow.fake must stay False here, or the row must not be returned
        until real numbers exist).
    """

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    def meta(self) -> ServiceMeta:
        # Safe to implement now, unlike the methods below: it needs no Pipeline
        # internals. Every judge-facing view keys off this to decide whether to
        # show the "DEMO MODE — FAKE DATA" banner, so it must say "real" the
        # moment RealDemoService is actually in use, even before every method
        # below is wired up.
        return ServiceMeta(service="real", data_source="real", note="Live pipeline output.")

    def list_tenants(self) -> list[TenantInfo]:
        raise NotImplementedError("RealDemoService.list_tenants: wire to Pipeline once it exists")

    def ask(self, tenant: str, question: str, defense: bool) -> AskResult:
        raise NotImplementedError("RealDemoService.ask: wire to Pipeline once it exists")

    def list_quarantine(self) -> list[QuarantineItem]:
        raise NotImplementedError("RealDemoService.list_quarantine: wire to Pipeline once it exists")

    def release_quarantine(self, item_id: str) -> bool:
        raise NotImplementedError("RealDemoService.release_quarantine: wire to Pipeline once it exists")

    def probe(self, as_tenant: str, target_tenant: str) -> ProbeResult:
        raise NotImplementedError("RealDemoService.probe: wire to Pipeline once it exists")

    def trace_for_chunk(self, chunk_id: str) -> list[ChunkTraceStep] | None:
        raise NotImplementedError("RealDemoService.trace_for_chunk: wire to Pipeline once it exists")

    def results_table(self) -> list[ResultRow]:
        raise NotImplementedError("RealDemoService.results_table: wire to Pipeline once it exists")
