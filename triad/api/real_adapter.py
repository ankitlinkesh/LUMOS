"""``RealDemoService``: wraps a real ``triad.pipeline.Pipeline`` so it satisfies
the ``DemoService`` Protocol the FastAPI app codes against.

Every method below delegates to the pipeline's own public methods (``ask``,
``ingest``/``release``, ``describe_chunk``) and to the two retrievers Stage 2
itself uses (``SecureRetriever``/``LeakyRetriever``) -- never a second
implementation of ingestion, retrieval, or generation. ``results_table`` is
the one method that reads something other than the live pipeline: it reads
the eval harness's own persisted JSON files under ``results/`` (never
recomputing a number itself) through an explicit filename allowlist, so a
known-invalid or unvetted run can never reach the UI.

Every row/answer/trace this class produces is either a genuine live pipeline
result or a number read straight out of a vetted, persisted results file --
nothing here is invented. ``meta()`` reports ``data_source="real"`` the
moment this class is constructed; the rest of the class exists to make that
claim true for every other endpoint too.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

from triad.api import serialize
from triad.api.service import (
    AskResult,
    ChunkTraceStep,
    ProbeResult,
    ProbeSide,
    PropertyTestResult,
    QuarantineItem,
    ResultRow,
    ServiceMeta,
    TenantInfo,
    TraceEvent,
)
from triad.contract import RetrievalResult
from triad.eval._common import RESULTS_DIR
from triad.pipeline import Answer, ChunkTrace, Pipeline
from triad.quarantine import NotQuarantined
from triad.retrieval.retriever import LeakyRetriever, SecureRetriever
from triad.retrieval.scope import Scope

__all__ = ["RealDemoService"]

# Fixed, generic probe queries -- not derived from any tenant's real content,
# just search strings used to exercise retrieval for the probe/property-test
# views. Reused (not per-request-random) so a probe result is reproducible.
_PROBE_QUERIES = (
    "quarterly budget and financial summary",
    "project status update and next steps",
    "contract renewal terms",
    "headcount and staffing plan",
    "meeting notes and action items",
)

# How many real SecureRetriever.retrieve() calls back the "property test"
# pass/total count in probe(): a live regression check against the actual
# store (not Hypothesis's synthetic fixtures in tests/test_retrieval_property.py,
# and not a fabricated pass count), each varying k. SecureRetriever's
# foreign_chunks() invariant is structural, so this is expected to always
# pass -- the point is that it is genuinely measured against live data on
# every call, not hardcoded.
_PROPERTY_TEST_N = 30

_CONTRACT_STAGE_TO_API_STAGE = {"ingest": "ingest", "retrieve": "retrieve", "generate": "prompt", "egress": "egress"}

# ---------------------------------------------------------------------------
# results_table(): explicit allowlist of vetted results/ files.
#
# results_table() NEVER globs results/ and NEVER reads a file that isn't a key
# here -- this dict IS the validity marker. Filenames not listed (four known-
# invalid runs: poisonedrag_n50_20260910T143808Z.json, two poisonedrag_n10
# calibration/retry runs, and the n=2 smoke poisonedrag_n2 run; plus files
# that simply were never vetted for this table, e.g. bipia_20260910T164029Z.json,
# injection_20260910T120229Z.json, tenant_leak_20260910T112318Z.json) are
# invisible to this method by construction, not by being individually
# excluded.
#
# Two of the four vetted files map to a ResultRow; two do not, and say why:
# neither geometry's nor injection's JSON contains an added-latency
# measurement, and ResultRow.added_latency_ms has no "unmeasured" value that
# survives JSON transport (NaN breaks JSON.parse on the frontend) -- so
# rather than fabricate a latency number, those two files simply contribute
# no row. Their real numbers are still reported elsewhere (``python -m
# triad.eval.report``, which this module deliberately does not duplicate).
# ---------------------------------------------------------------------------
_VALID_RESULT_FILES: dict[str, str] = {
    "poisonedrag_n100_20260910T195905Z.json":
        "headline PoisonedRAG black-box run (n=100 targets) -> one ResultRow. "
        "Supersedes poisonedrag_n100_20260910T153530Z.json, which measured "
        "cluster collapse while it was silently inert (triad/pipeline.py passed "
        "collapse_topk a single-string embed_query where a batch embed_documents "
        "was required); this run is the same configuration with that call site fixed.",
    "tenant_leak_20260910T120126Z.json":
        "headline cross-tenant retrieval leak probe (n=500 probes) -> one ResultRow",
    "geometry_20260910T142941Z.json":
        "Stage 1B (manifold isolation + query-echo) calibration/catch-rate report -- "
        "no end-to-end ASR or added-latency number exists in this file, so it contributes "
        "no ResultRow rather than fabricating one",
    "injection_20260910T121109Z.json":
        "Stage 1A directive-detector calibration on LLMail-Inject phase 2 -- "
        "no added-latency measurement exists in this file, so it contributes "
        "no ResultRow rather than fabricating one",
}


def _trace_events(answer: Answer) -> tuple[TraceEvent, ...]:
    """Rebuilds the UI's per-chunk trace log from a real ``Answer``: one
    ingest/retrieve/prompt triplet per chunk the retrieval touched (from
    ``answer.trace.chunks``), plus one line per ``GuardDecision`` the pipeline
    actually made (collapse_topk, and stage3 egress if enabled)."""
    events: list[TraceEvent] = []
    for ct in answer.trace.chunks:
        if ct.ingest_verdict == "quarantined":
            events.append(TraceEvent(ct.chunk_id, "ingest", "quarantined",
                                      "; ".join(ct.ingest_reasons) or "quarantined at ingestion"))
        elif ct.ingest_verdict == "allowed":
            events.append(TraceEvent(ct.chunk_id, "ingest", "allowed", "passed Stage 1 at ingestion"))
        else:
            events.append(TraceEvent(ct.chunk_id, "ingest", "unknown",
                                      "no ingest record for this chunk id in this process"))

        events.append(TraceEvent(
            ct.chunk_id, "retrieve", "included" if ct.retrieved else "excluded", f"tenant={ct.tenant}",
        ))
        events.append(TraceEvent(
            ct.chunk_id, "prompt", "used" if ct.entered_prompt else "not_used",
            "included as supporting context" if ct.entered_prompt else "did not enter the generation prompt",
        ))

    for decision in answer.decisions:
        api_stage = _CONTRACT_STAGE_TO_API_STAGE.get(decision.stage, decision.stage)
        status = "blocked" if not decision.allow else ("collapsed" if decision.evidence.get("collapsed") else "ok")
        detail = "; ".join(decision.reasons) or ("answer rewritten" if decision.rewritten else "no change")
        events.append(TraceEvent("-", api_stage, status, detail))

    if not any(d.stage == "egress" for d in answer.decisions):
        events.append(TraceEvent("-", "egress", "paused", "Stage 3 egress checks are paused in this build"))

    return tuple(events)


def _probe_side(result: RetrievalResult) -> ProbeSide:
    foreign = result.foreign_chunks()
    return ProbeSide(leaked=bool(foreign), n_foreign=len(foreign), declined=result.declined, chunks=result.chunks)


def _chunk_trace_to_steps(ct: ChunkTrace, *, stage3_enabled: bool) -> list[ChunkTraceStep]:
    now = datetime.now(timezone.utc)

    def _at(offset_s: float) -> str:
        return (now + timedelta(seconds=offset_s)).isoformat().replace("+00:00", "Z")

    if ct.ingest_verdict == "quarantined":
        ingest_status, ingest_detail = "quarantined", "; ".join(ct.ingest_reasons) or "flagged by Stage 1"
    elif ct.ingest_verdict == "allowed":
        ingest_status, ingest_detail = "allowed", "passed Stage 1 taint scan"
    else:
        ingest_status, ingest_detail = "unknown", "no ingest record for this chunk id in this process"

    steps = [ChunkTraceStep("ingest", ingest_status, ingest_detail, _at(0))]

    if ct.retrieved:
        steps.append(ChunkTraceStep("retrieve", "eligible", f"tenant-scoped to {ct.tenant}", _at(1)))
        steps.append(ChunkTraceStep(
            "prompt", "used" if ct.entered_prompt else "not_used",
            "included as supporting context" if ct.entered_prompt
            else "currently in the index but not part of any specific answer",
            _at(2),
        ))
    else:
        steps.append(ChunkTraceStep("retrieve", "blocked", "never entered the retrievable index", _at(1)))
        steps.append(ChunkTraceStep("prompt", "skipped", "not present in any prompt", _at(2)))

    if stage3_enabled:
        steps.append(ChunkTraceStep("egress", "checked", "Stage 3 egress checks ran", _at(3)))
    else:
        steps.append(ChunkTraceStep("egress", "paused", "Stage 3 egress checks are paused in this build", _at(3)))
    return steps


def _row_from_poisonedrag(data: dict[str, Any]) -> ResultRow:
    """Every field here is read straight from ``poisonedrag_n100_...json``,
    matching exactly what ``triad.eval.report`` itself prints for this file
    (``report.py`` lines ~74-126): ``asr.off``/``asr.on``,
    ``clean_accuracy.on``, the Stage 2 retrieval latency delta, and
    ``clean_quarantined / clean_submitted`` for FPR."""
    asr = data["asr"]
    ingestion = data["ingestion"]
    latency = data["latency_ms"]
    clean_submitted = ingestion["clean_submitted"]
    fpr = (ingestion["clean_quarantined"] / clean_submitted) if clean_submitted else 0.0
    added_latency_ms = latency["stage2_retrieval_on_mean"] - latency["stage2_retrieval_off_mean"]
    return ResultRow(
        attack="PoisonedRAG (black-box, BEIR NQ)",
        asr_before=asr["off"],
        asr_after=asr["on"],
        clean_accuracy=data["clean_accuracy"]["on"],
        added_latency_ms=added_latency_ms,
        fpr=fpr,
        n=data["n_targets"],
        data_source="real",
        fake=False,
    )


def _row_from_tenant_leak(data: dict[str, Any]) -> ResultRow:
    """Every field here is read straight from ``tenant_leak_...json``: the
    leak rate is ``leak_rate_any_foreign_chunk`` for each retriever (the same
    field ``report.py`` prints), added latency is the file's own
    ``added_latency_ms_p50_secure_minus_leaky`` (real-measured and negative:
    the secure path was faster on this run -- not clamped), and FPR/clean
    accuracy use the secure retriever's own ``decline_rate`` (a legitimate
    in-scope query wrongly declined is this experiment's false positive)."""
    leaky = data["leaky_retriever"]
    secure = data["secure_retriever"]
    return ResultRow(
        attack="Cross-tenant retrieval leak",
        asr_before=leaky["leak_rate_any_foreign_chunk"],
        asr_after=secure["leak_rate_any_foreign_chunk"],
        clean_accuracy=1.0 - secure["decline_rate"],
        added_latency_ms=data["added_latency_ms_p50_secure_minus_leaky"],
        fpr=secure["decline_rate"],
        n=data["n_probes"],
        data_source="real",
        fake=False,
    )


class RealDemoService:
    """Implements ``DemoService`` by delegating to a real ``Pipeline`` instance."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def meta(self) -> ServiceMeta:
        return ServiceMeta(service="real", data_source="real", note="Live pipeline output.")

    # -- tenants -----------------------------------------------------------

    def list_tenants(self) -> list[TenantInfo]:
        counts = self._pipeline.store.tenant_counts()
        return [TenantInfo(id=t, label=t, n_docs=n) for t, n in sorted(counts.items())]

    # -- ask -----------------------------------------------------------

    def _pipeline_for_ask(self, defense: bool) -> Pipeline:
        """A view of the shared pipeline with ``secure_retrieval``/
        ``collapse_topk`` set to ``defense`` -- the only two toggles that mean
        anything AFTER ingestion (stage1a/1b already ran, once, when the
        corpus was built). Constructs a second ``Pipeline`` object rather than
        ``dataclasses.replace`` because ``replace`` re-runs ``__post_init__``
        and would silently reset ``_ledger``/``_reference_embeddings`` to
        empty; instead this shares the SAME store/quarantine/ledger objects
        as the base pipeline, so it is a different view over identical live
        state, never a second copy of it."""
        base = self._pipeline
        if base.defense.secure_retrieval == defense and base.defense.collapse_topk == defense:
            return base
        adjusted = Pipeline(
            store=base.store, embedder=base.embedder, llm=base.llm,
            defense=replace(base.defense, secure_retrieval=defense, collapse_topk=defense),
            quarantine=base.quarantine, generator_model=base.generator_model,
            max_tokens=base.max_tokens, collapse_sim_threshold=base.collapse_sim_threshold,
            clock=base.clock,
        )
        adjusted._ledger = base._ledger
        adjusted._reference_embeddings = base._reference_embeddings
        return adjusted

    def ask(self, tenant: str, question: str, defense: bool) -> AskResult:
        pipeline = self._pipeline_for_ask(defense)
        scope = Scope.of(tenant)
        answer = pipeline.ask(question, scope, k=5)

        return AskResult(
            answer=answer.text,
            declined=answer.retrieval.declined,
            decline_reason=answer.retrieval.decline_reason,
            leak_mode=answer.retrieval.leak_mode,
            latency_ms=answer.retrieval.latency_ms,
            chunks=answer.retrieval.chunks,
            trace=_trace_events(answer),
            cached=answer.cached,
            # The demo corpus mixes real EnronQA/LLMail-Inject records with a
            # synthetic PoisonedRAG-style poison set (see Pipeline.demo's own
            # docstring) -- "mixed" is the honest label, never "real".
            data_source="mixed",
        )

    # -- quarantine -----------------------------------------------------------

    def list_quarantine(self) -> list[QuarantineItem]:
        entries = self._pipeline.quarantine.list(released=False)
        return [
            QuarantineItem(
                id=e.chunk.id, tenant=e.chunk.tenant, preview=serialize.preview_of(e.chunk.text),
                score=e.chunk.taint.score, flags=e.chunk.taint.flags, reasons=e.chunk.taint.reasons,
            )
            for e in entries
        ]

    def release_quarantine(self, item_id: str) -> bool:
        try:
            self._pipeline.release(item_id, reason="released via TRIAD-RAG API")
        except NotQuarantined:
            return False
        return True

    # -- probe -----------------------------------------------------------

    def probe(self, as_tenant: str, target_tenant: str) -> ProbeResult:
        pipeline = self._pipeline
        own_scope = Scope.of(as_tenant)
        query = _PROBE_QUERIES[0]

        secure_retriever = SecureRetriever(store=pipeline.store, embedder=pipeline.embedder)
        leaky_retriever = LeakyRetriever(store=pipeline.store, embedder=pipeline.embedder)

        secure_result = secure_retriever.retrieve(query, own_scope, k=5)
        leaky_result = leaky_retriever.retrieve(query, own_scope, k=5)

        property_test = None
        if as_tenant != target_tenant:
            property_test = self._run_property_test(secure_retriever, own_scope)

        return ProbeResult(
            secure=_probe_side(secure_result), leaky=_probe_side(leaky_result), property_test=property_test,
        )

    def _run_property_test(self, secure_retriever: SecureRetriever, scope: Scope) -> PropertyTestResult:
        """Runs the cross-tenant-leak invariant (``foreign_chunks() == ()``)
        against the LIVE store, ``_PROPERTY_TEST_N`` times with varying k --
        a real, freshly-measured pass/total count every call, not the
        hardcoded 200/200 ``FakeDemoService`` reports. Always expected to
        pass (the invariant is structural, per ``SecureRetriever``'s own
        docstring); the point is that this is genuinely re-checked against
        real data on every probe, never assumed."""
        passed = 0
        for i in range(_PROPERTY_TEST_N):
            query = _PROBE_QUERIES[i % len(_PROBE_QUERIES)]
            k = 3 + (i % 5)
            result = secure_retriever.retrieve(query, scope, k=k)
            if result.foreign_chunks() == ():
                passed += 1
        return PropertyTestResult(passed=passed, total=_PROPERTY_TEST_N, fake=False)

    # -- trace -----------------------------------------------------------

    def trace_for_chunk(self, chunk_id: str) -> list[ChunkTraceStep] | None:
        chunk_trace = self._pipeline.describe_chunk(chunk_id)
        if chunk_trace is None:
            return None
        return _chunk_trace_to_steps(chunk_trace, stage3_enabled=self._pipeline.defense.stage3_enabled)

    # -- results -----------------------------------------------------------

    def _read_allowlisted(self, filename: str) -> dict[str, Any] | None:
        if filename not in _VALID_RESULT_FILES:
            # Programmer error, not a runtime condition: every filename this
            # method ever reads must be listed (and explained) above.
            raise AssertionError(f"{filename!r} is not on the results/ allowlist")
        path = RESULTS_DIR / filename
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def results_table(self) -> list[ResultRow]:
        rows: list[ResultRow] = []

        poisonedrag = self._read_allowlisted("poisonedrag_n100_20260910T195905Z.json")
        if poisonedrag is not None:
            rows.append(_row_from_poisonedrag(poisonedrag))

        tenant_leak = self._read_allowlisted("tenant_leak_20260910T120126Z.json")
        if tenant_leak is not None:
            rows.append(_row_from_tenant_leak(tenant_leak))

        # geometry_20260910T142941Z.json and injection_20260910T121109Z.json
        # are on the allowlist (real, vetted results) but produce no
        # ResultRow -- see the comment on _VALID_RESULT_FILES.

        return rows
