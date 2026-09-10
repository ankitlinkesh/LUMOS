"""``QuarantineQueue``: the reviewable holding pen for chunks Stage 1 blocked.

The plan's own framing: "What happens to a false positive?" is the first
question an enterprise judge asks, and deletion is the wrong answer. So a
blocked chunk is never dropped -- it is written here with the ``GuardDecision``
that blocked it (reasons, evidence, score), stays reviewable, and can only
leave the queue through ``release()``, which is itself logged (never a silent
un-block).

Persisted to JSON so the queue survives a process restart (the whole point of
"reviewable" -- a human reviewing an hour later needs it to still be there).
The path is injectable so tests never touch the real ``.cache/`` directory.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from triad.config import CACHE_DIR
from triad.contract import Chunk, GuardDecision, Provenance, TaintVerdict

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_PATH = CACHE_DIR / "quarantine" / "queue.json"
DEFAULT_LOG_PATH = CACHE_DIR / "quarantine" / "release_log.jsonl"


class NotQuarantined(KeyError):
    """Raised by ``release`` when ``chunk_id`` isn't in the queue (never was,
    or was already released)."""


@dataclass(frozen=True)
class QuarantineEntry:
    """One held chunk plus the verdict that held it. ``released``/``released_at``
    make the release itself part of the reviewable record, not just a removal."""

    chunk: Chunk
    decision: GuardDecision
    queued_at: float
    released: bool = False
    released_at: float | None = None


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "text": chunk.text,
        "tenant": chunk.tenant,
        "source_type": chunk.source_type,
        "provenance": {
            "dataset": chunk.provenance.dataset,
            "record_id": chunk.provenance.record_id,
            "data_source": chunk.provenance.data_source,
        },
        "taint": {
            "untrusted": chunk.taint.untrusted,
            "quarantined": chunk.taint.quarantined,
            "flags": list(chunk.taint.flags),
            "score": chunk.taint.score,
            "reasons": list(chunk.taint.reasons),
        },
        "metadata": dict(chunk.metadata),
    }


def _chunk_from_dict(d: dict[str, Any]) -> Chunk:
    prov = Provenance(**d["provenance"])
    t = d["taint"]
    taint = TaintVerdict(
        untrusted=bool(t["untrusted"]),
        quarantined=bool(t["quarantined"]),
        flags=tuple(t["flags"]),
        score=float(t["score"]),
        reasons=tuple(t["reasons"]),
    )
    return Chunk(
        id=d["id"], text=d["text"], tenant=d["tenant"], source_type=d["source_type"],
        provenance=prov, taint=taint, metadata=d["metadata"],
    )


def _decision_to_dict(decision: GuardDecision) -> dict[str, Any]:
    return {
        "stage": decision.stage,
        "allow": decision.allow,
        "escalate": decision.escalate,
        "reasons": list(decision.reasons),
        "evidence": dict(decision.evidence),
        "rewritten": decision.rewritten,
        "latency_ms": decision.latency_ms,
    }


def _decision_from_dict(d: dict[str, Any]) -> GuardDecision:
    return GuardDecision(
        stage=d["stage"], allow=d["allow"], escalate=d.get("escalate", False),
        reasons=tuple(d.get("reasons", ())), evidence=d.get("evidence", {}),
        rewritten=d.get("rewritten"), latency_ms=d.get("latency_ms", 0.0),
    )


def _entry_to_dict(entry: QuarantineEntry) -> dict[str, Any]:
    return {
        "chunk": _chunk_to_dict(entry.chunk),
        "decision": _decision_to_dict(entry.decision),
        "queued_at": entry.queued_at,
        "released": entry.released,
        "released_at": entry.released_at,
    }


def _entry_from_dict(d: dict[str, Any]) -> QuarantineEntry:
    return QuarantineEntry(
        chunk=_chunk_from_dict(d["chunk"]), decision=_decision_from_dict(d["decision"]),
        queued_at=d["queued_at"], released=d.get("released", False), released_at=d.get("released_at"),
    )


@dataclass
class QuarantineQueue:
    """Holds quarantined chunks keyed by chunk id. ``add`` is idempotent
    (re-quarantining the same id overwrites its entry with the newer verdict,
    since a chunk can only be pending in one state at a time). ``release``
    never deletes the record -- it flips ``released`` and appends to a
    separate append-only log, so the queue's own history can never be edited
    away."""

    path: Path = DEFAULT_QUEUE_PATH
    log_path: Path = DEFAULT_LOG_PATH
    clock: Any = time.time  # injectable for deterministic tests

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.log_path = Path(self.log_path)
        self._entries: dict[str, QuarantineEntry] = {}
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # a corrupt queue file must not crash the pipeline
            logger.warning("quarantine queue at %s unreadable (%s); starting empty", self.path, exc)
            return
        for d in raw.get("entries", []):
            entry = _entry_from_dict(d)
            self._entries[entry.chunk.id] = entry

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": [_entry_to_dict(e) for e in self._entries.values()]}
        tmp = self.path.with_suffix(self.path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic, same dir as target (Windows-safe)

    def _append_log(self, event: dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")

    # -- public API ----------------------------------------------------------

    def add(self, chunk: Chunk, decision: GuardDecision) -> QuarantineEntry:
        """Holds ``chunk`` with the ``GuardDecision`` that blocked it. The
        caller is responsible for making sure it never also reaches the
        store -- ``TenantStore.add`` refuses any chunk with
        ``taint.quarantined`` set as a second, independent backstop."""
        entry = QuarantineEntry(chunk=chunk, decision=decision, queued_at=self.clock())
        self._entries[chunk.id] = entry
        self._save()
        logger.info("quarantined chunk %s (tenant=%s): %s", chunk.id, chunk.tenant, decision.reasons)
        return entry

    def list(self, *, released: bool | None = False) -> tuple[QuarantineEntry, ...]:
        """Entries sorted oldest-first. ``released=False`` (default) is the
        review queue proper (still pending); ``released=True`` is history;
        ``None`` is everything."""
        entries = self._entries.values()
        if released is not None:
            entries = (e for e in entries if e.released == released)
        return tuple(sorted(entries, key=lambda e: e.queued_at))

    def get(self, chunk_id: str) -> QuarantineEntry | None:
        return self._entries.get(chunk_id)

    def release(self, chunk_id: str, *, reason: str = "") -> Chunk:
        """Releases ``chunk_id``: marks the entry released (never deleted),
        appends to the release log, and returns the chunk with
        ``taint.quarantined`` cleared so the caller (``Pipeline.release``) can
        actually ingest it. Raises ``NotQuarantined`` if the id was never
        queued or was already released -- a release must act on something
        real, never silently no-op."""
        entry = self._entries.get(chunk_id)
        if entry is None or entry.released:
            raise NotQuarantined(f"{chunk_id!r} is not a pending quarantine entry")

        released_chunk = replace(entry.chunk, taint=replace(entry.chunk.taint, quarantined=False))
        released_at = self.clock()
        self._entries[chunk_id] = replace(entry, released=True, released_at=released_at)
        self._save()

        event = {
            "chunk_id": chunk_id, "tenant": entry.chunk.tenant, "released_at": released_at,
            "original_reasons": list(entry.decision.reasons), "release_reason": reason,
        }
        self._append_log(event)
        logger.info("released quarantined chunk %s (tenant=%s): %s", chunk_id, entry.chunk.tenant, reason)
        return released_chunk

    def __len__(self) -> int:
        return len(self.list(released=False))
