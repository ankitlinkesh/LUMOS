"""Convert ``triad.contract`` / ``triad.api.service`` dataclasses into the exact
JSON shapes the UI expects. Kept separate from ``app.py`` so a real
``DemoService`` implementation only has to produce the dataclasses in
``service.py`` (or hand back genuine contract objects) — never hand-roll JSON.
"""

from __future__ import annotations

from triad.api.service import (
    AskResult,
    ChunkTraceStep,
    ProbeResult,
    QuarantineItem,
    ResultRow,
    ServiceMeta,
    TenantInfo,
)
from triad.contract import ScoredChunk, TaintVerdict

__all__ = [
    "preview_of",
    "taint_dict",
    "scored_chunk_dict",
    "tenant_dict",
    "ask_dict",
    "quarantine_dict",
    "probe_dict",
    "trace_step_dict",
    "results_row_dict",
    "meta_dict",
]

_PREVIEW_LEN = 160


def preview_of(text: str, limit: int = _PREVIEW_LEN) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def taint_dict(taint: TaintVerdict) -> dict:
    return {
        "quarantined": taint.quarantined,
        "flags": list(taint.flags),
        "score": taint.score,
        "reasons": list(taint.reasons),
    }


def scored_chunk_dict(sc: ScoredChunk) -> dict:
    c = sc.chunk
    return {
        "id": c.id,
        "tenant": c.tenant,
        "score": sc.score,
        "preview": preview_of(c.text),
        "source_type": c.source_type,
        "data_source": c.provenance.data_source,
        "taint": taint_dict(c.taint),
    }


def tenant_dict(t: TenantInfo) -> dict:
    return {"id": t.id, "label": t.label, "n_docs": t.n_docs}


def ask_dict(r: AskResult) -> dict:
    return {
        "answer": r.answer,
        "declined": r.declined,
        "decline_reason": r.decline_reason,
        "leak_mode": r.leak_mode,
        "latency_ms": r.latency_ms,
        "chunks": [scored_chunk_dict(sc) for sc in r.chunks],
        "trace": [
            {"chunk_id": e.chunk_id, "stage": e.stage, "event": e.event, "detail": e.detail}
            for e in r.trace
        ],
        "cached": r.cached,
        "data_source": r.data_source,
    }


def quarantine_dict(q: QuarantineItem) -> dict:
    return {
        "id": q.id,
        "tenant": q.tenant,
        "preview": q.preview,
        "score": q.score,
        "flags": list(q.flags),
        "reasons": list(q.reasons),
    }


def _probe_side_dict(side) -> dict:
    return {
        "leaked": side.leaked,
        "n_foreign": side.n_foreign,
        "declined": side.declined,
        "chunks": [scored_chunk_dict(sc) for sc in side.chunks],
    }


def probe_dict(r: ProbeResult) -> dict:
    return {
        "secure": _probe_side_dict(r.secure),
        "leaky": _probe_side_dict(r.leaky),
        "property_test": (
            {
                "passed": r.property_test.passed,
                "total": r.property_test.total,
                "fake": r.property_test.fake,
            }
            if r.property_test is not None
            else None
        ),
    }


def trace_step_dict(s: ChunkTraceStep) -> dict:
    return {"stage": s.stage, "status": s.status, "detail": s.detail, "at": s.at}


def results_row_dict(r: ResultRow) -> dict:
    return {
        "attack": r.attack,
        "asr_before": r.asr_before,
        "asr_after": r.asr_after,
        "clean_accuracy": r.clean_accuracy,
        "added_latency_ms": r.added_latency_ms,
        "fpr": r.fpr,
        "n": r.n,
        "data_source": r.data_source,
        "fake": r.fake,
    }


def meta_dict(m: ServiceMeta) -> dict:
    return {"service": m.service, "data_source": m.data_source, "note": m.note}
