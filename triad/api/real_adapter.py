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
    NoUsableTargetDocument,
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
from triad.data.enronqa import load_qa
from triad.data.types import QARecord
from triad.eval._common import RESULTS_DIR
from triad.pipeline import Answer, ChunkTrace, Pipeline
from triad.quarantine import NotQuarantined
from triad.retrieval.retriever import LeakyRetriever, SecureRetriever
from triad.retrieval.scope import Scope

__all__ = ["RealDemoService"]

# Fixed, generic probe queries used ONLY by the property test (which re-checks
# SecureRetriever's structural no-leak invariant against the caller's OWN
# scope and therefore has no "target" to be specific about). The probe's
# actual secure-vs-leaky comparison no longer uses these -- see
# _default_qa_loader / _target_probe_query below, and the comment on
# probe() -- because a fixed generic query never tests what target_tenant
# shapes; it only tests how well as_tenant's own mail matches a generic
# string.
_PROBE_QUERIES = (
    "quarterly budget and financial summary",
    "project status update and next steps",
    "contract renewal terms",
    "headcount and staffing plan",
    "meeting notes and action items",
)


def _default_qa_loader(target_tenant: str) -> list[QARecord]:
    """The real question source: EnronQA's own test-split QA records for
    ``target_tenant``, the exact same source and split
    ``triad.eval.tenant_leak`` uses to build its cross-tenant probes.
    Deliberately NOT shared code with tenant_leak.py (see probe()'s
    docstring) -- this reproduces its *predicate*, not its module, since
    tenant_leak.py additionally does corpus-building/centroid-pairing that a
    single already-chosen (as_tenant, target_tenant) probe has no use for."""
    return load_qa(split="test", users=[target_tenant])

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
    # Supersedes poisonedrag_n100_20260910T195905Z.json (kept on disk, no
    # longer surfaced by the UI): 195905Z was the corrected pre-retune run,
    # ASR 62%->47%; this run (224217Z) is the current headline retuned run,
    # ASR 62%->12%. See README's "Attack success rate -- historical baseline
    # and current rerun" section for the full lineage of all six
    # poisonedrag_n100_* files.
    "poisonedrag_n100_20260910T224217Z.json":
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
        if decision.evidence.get("decisions"):
            blocked = decision.evidence.get("blocked", ())
            redacted = decision.evidence.get("redacted", ())
            detail = f"context policy: blocked={len(blocked)}, redacted={len(redacted)}"
        else:
            detail = "; ".join(decision.reasons) or ("answer rewritten" if decision.rewritten else "no change")
        events.append(TraceEvent("-", api_stage, status, detail))

    if not any(d.stage == "egress" for d in answer.decisions):
        # Reached only when the pipeline that produced this Answer had
        # stage3_enabled=False -- a real configuration fact (Stage 3 itself
        # is implemented and measured; see the README's "Stage 3 -- output
        # and egress" section), not "paused"/unimplemented.
        events.append(TraceEvent(
            "-", "egress", "disabled", "Stage 3 egress checks are disabled by configuration in this build",
        ))

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

    # This view describes CONFIGURATION, not a specific answer's outcome --
    # /api/trace/{id} isn't tied to any particular /api/ask call, so it
    # cannot honestly claim a check "ran" on this chunk (see _trace_events
    # above for the per-answer version, which reports a real per-call
    # GuardDecision). Stage 3 itself is implemented and measured either way
    # (README's "Stage 3 -- output and egress" section); this step only
    # says whether it is switched on for this build.
    if stage3_enabled:
        steps.append(ChunkTraceStep(
            "egress", "enabled", "Stage 3 egress checks are enabled in this build", _at(3),
        ))
    else:
        steps.append(ChunkTraceStep(
            "egress", "disabled", "Stage 3 egress checks are disabled by configuration in this build", _at(3),
        ))
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

    def __init__(self, pipeline: Pipeline, *, qa_loader=_default_qa_loader) -> None:
        self._pipeline = pipeline
        # Injectable so tests can supply fabricated QARecords instead of
        # reading real EnronQA parquet files -- same DI pattern as `store`/
        # `embedder`/`llm` on Pipeline itself. Defaults to the real loader in
        # every non-test path (including --real).
        self._qa_loader = qa_loader
        self._qa_cache: dict[str, tuple[QARecord, ...]] = {}

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
            defense=replace(
                base.defense,
                secure_retrieval=defense,
                collapse_topk=defense,
                context_guard_enabled=defense,
            ),
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

        # Per-response composition: of the chunks THIS call actually
        # retrieved, how many are real vs synthetic, read straight off each
        # chunk's own Provenance.data_source (round-tripped byte-for-byte
        # through TenantStore -- see store.py's _chunk_to_metadata). This is
        # the per-response fact; data_source="mixed" below stays the
        # corpus-level, conservative claim ("a synthetic chunk COULD have
        # been retrieved") -- this doesn't replace it, it explains it.
        n_real = sum(1 for sc in answer.retrieval.chunks if sc.chunk.provenance.data_source == "real")
        n_synthetic = sum(1 for sc in answer.retrieval.chunks if sc.chunk.provenance.data_source == "synthetic")

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
            n_chunks_real=n_real,
            n_chunks_synthetic=n_synthetic,
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

    def _target_probe_query(self, target_tenant: str) -> tuple[str, str]:
        """Builds the probe query the same way ``triad.eval.tenant_leak``
        builds its cross-tenant probes: a real EnronQA question whose gold
        answer lives in ``target_tenant``'s own inbox, restricted to an
        email actually indexed in THIS live store (not just anywhere in
        EnronQA -- ``Pipeline.demo`` only ingests a capped subset per
        tenant, and some of that subset may have been quarantined at
        ingest and therefore never written to the store at all).

        Returns ``(question, gold_chunk_id)``. This is deliberately the
        target_tenant's content shaping the query -- the fix for a probe
        that used to ask a fixed generic question regardless of which
        tenant was being probed, which measured "does as_tenant's inbox
        match a generic string" instead of "did isolation fail on a query
        aimed at target_tenant's content."

        Raises ``NoUsableTargetDocument`` if no such question/chunk pair
        exists -- NEVER falls back to ``_PROBE_QUERIES``, which would
        silently reintroduce the exact bug this method exists to fix.
        """
        records = self._qa_cache.get(target_tenant)
        if records is None:
            records = tuple(self._qa_loader(target_tenant))
            self._qa_cache[target_tenant] = records

        store = self._pipeline.store
        for record in records:
            if record.tenant != target_tenant:
                continue
            chunk = store.get(record.email_path)
            if chunk is not None and chunk.tenant == target_tenant:
                return record.question, record.email_path
        raise NoUsableTargetDocument(target_tenant)

    def probe(self, as_tenant: str, target_tenant: str) -> ProbeResult:
        pipeline = self._pipeline
        own_scope = Scope.of(as_tenant)
        query, gold_chunk_id = self._target_probe_query(target_tenant)

        secure_retriever = SecureRetriever(store=pipeline.store, embedder=pipeline.embedder)
        leaky_retriever = LeakyRetriever(store=pipeline.store, embedder=pipeline.embedder)

        secure_result = secure_retriever.retrieve(query, own_scope, k=5)
        leaky_result = leaky_retriever.retrieve(query, own_scope, k=5)

        cross_tenant = as_tenant != target_tenant
        leaky_ids = {sc.chunk.id for sc in leaky_result.chunks}
        # The strong claim: not just "some foreign chunk came back" but
        # "the SPECIFIC chunk this query was built to be answerable from
        # came back for a different requester" -- the same leak_gold
        # definition tenant_leak.py measures (67.6% headline number),
        # reproduced here per-probe instead of aggregated over 500 probes.
        gold_leaked = cross_tenant and gold_chunk_id in leaky_ids

        property_test = None
        if cross_tenant:
            property_test = self._run_property_test(secure_retriever, own_scope)

        return ProbeResult(
            secure=_probe_side(secure_result), leaky=_probe_side(leaky_result), property_test=property_test,
            query=query, target_gold_chunk_id=gold_chunk_id, gold_leaked=gold_leaked,
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

    def chunk_tenant(self, chunk_id: str) -> str | None:
        # Reuses describe_chunk -- the same source trace_for_chunk reads --
        # rather than a second lookup path, so this can never disagree with
        # what trace_for_chunk would show for the same id.
        chunk_trace = self._pipeline.describe_chunk(chunk_id)
        return chunk_trace.tenant if chunk_trace is not None else None

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

        poisonedrag = self._read_allowlisted("poisonedrag_n100_20260910T224217Z.json")
        if poisonedrag is not None:
            rows.append(_row_from_poisonedrag(poisonedrag))

        tenant_leak = self._read_allowlisted("tenant_leak_20260910T120126Z.json")
        if tenant_leak is not None:
            rows.append(_row_from_tenant_leak(tenant_leak))

        # geometry_20260910T142941Z.json and injection_20260910T121109Z.json
        # are on the allowlist (real, vetted results) but produce no
        # ResultRow -- see the comment on _VALID_RESULT_FILES.

        return rows
