"""``TenantStore``: Chroma-backed storage that makes tenant, provenance and taint
first-class metadata rather than an afterthought bolted onto a generic vector
store.

Two invariants this module exists to enforce:

1. A chunk whose ``taint.quarantined`` is True is never written. Quarantine is
   Stage 1's verdict that a chunk must never be searchable again; if it were
   still addable here, any bug upstream that forgets to check quarantine would
   silently make poisoned content retrievable.
2. ``search`` passes the tenant predicate to Chroma's own ``where`` clause, not
   a Python-side filter after the fact. Measured directly against this chromadb
   version (1.5.9): a query with ``where={"tenant": {"$in": [...]}}`` performs a
   filtered ANN search that returns the true top-k *within* the allowed tenants,
   whereas fetching a global top-k and filtering in Python would silently return
   fewer than k results (or, in the vulnerable baseline, fall back to filling the
   gap from other tenants). See ``tests/test_retrieval_store.py`` for the spike
   that proves this.

Measured gotcha worth recording: ``chromadb.EphemeralClient()`` instances are
NOT isolated from each other within one process in chromadb 1.5.9 -- two
separate ``EphemeralClient()`` calls still see the same named collection. A
fixed default collection name would silently leak state between tests (or
between unrelated ``TenantStore``s in the same process), so the default name
is per-instance-unique; pass an explicit ``collection_name`` to deliberately
share one collection across multiple ``TenantStore`` objects.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from triad.contract import Chunk, Provenance, ScoredChunk, TaintVerdict

_COLLECTION_PREFIX = "triad-chunks"


def _unique_collection_name() -> str:
    return f"{_COLLECTION_PREFIX}-{uuid.uuid4().hex[:12]}"


def _chunk_to_metadata(chunk: Chunk) -> dict[str, Any]:
    """Flattens a ``Chunk`` into the flat scalar metadata Chroma accepts. Tuples
    and the nested ``metadata`` mapping are JSON-encoded so ``_metadata_to_chunk``
    can reconstruct them exactly (a Chunk must round-trip byte-for-byte, since
    Stage 3 re-reads provenance and taint off of what Stage 2 returns)."""
    return {
        "chunk_id": chunk.id,
        "tenant": chunk.tenant,
        "source_type": chunk.source_type,
        "prov_dataset": chunk.provenance.dataset,
        "prov_record_id": chunk.provenance.record_id,
        "prov_data_source": chunk.provenance.data_source,
        "taint_untrusted": chunk.taint.untrusted,
        "taint_quarantined": chunk.taint.quarantined,
        "taint_flags": json.dumps(list(chunk.taint.flags)),
        "taint_score": float(chunk.taint.score),
        "taint_reasons": json.dumps(list(chunk.taint.reasons)),
        "chunk_metadata": json.dumps(dict(chunk.metadata)),
    }


def _metadata_to_chunk(chunk_id: str, text: str, meta: dict[str, Any]) -> Chunk:
    """Inverse of ``_chunk_to_metadata``."""
    provenance = Provenance(
        dataset=meta["prov_dataset"],
        record_id=meta["prov_record_id"],
        data_source=meta["prov_data_source"],
    )
    taint = TaintVerdict(
        untrusted=bool(meta["taint_untrusted"]),
        quarantined=bool(meta["taint_quarantined"]),
        flags=tuple(json.loads(meta["taint_flags"])),
        score=float(meta["taint_score"]),
        reasons=tuple(json.loads(meta["taint_reasons"])),
    )
    return Chunk(
        id=chunk_id,
        text=text,
        tenant=meta["tenant"],
        source_type=meta["source_type"],
        provenance=provenance,
        taint=taint,
        metadata=json.loads(meta["chunk_metadata"]),
    )


@dataclass
class TenantStore:
    """A Chroma collection scoped to hold ``Chunk``s with tenant/provenance/taint
    metadata. ``client`` is injectable (``chromadb.EphemeralClient()`` in tests).

    ``space`` must match the embedder used to produce the vectors passed to
    ``add``/``search`` -- see ``Embedder.space`` in ``triad.embed.base``. Getting
    this wrong doesn't error; it silently ranks results by the wrong metric.
    """

    client: Any
    space: str = "cosine"
    collection_name: str = field(default_factory=_unique_collection_name)

    def __post_init__(self) -> None:
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.space},
        )

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int:
        """Adds ``chunks`` with their pre-computed ``embeddings`` (row i of
        ``embeddings`` corresponds to ``chunks[i]``). Quarantined chunks are
        skipped entirely -- never written, never searchable. Returns the count of
        chunks rejected for that reason.

        Raises ``ValueError`` if ``chunks`` and ``embeddings`` have mismatched
        lengths, since a silent misalignment would attach the wrong vector to the
        wrong chunk's tenant.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have the same length"
            )
        if not chunks:
            return 0

        ids: list[str] = []
        embs: list[list[float]] = []
        metas: list[dict[str, Any]] = []
        docs: list[str] = []
        rejected = 0
        for chunk, vec in zip(chunks, embeddings):
            if chunk.taint.quarantined:
                rejected += 1
                continue
            ids.append(chunk.id)
            embs.append(np.asarray(vec, dtype=np.float32).tolist())
            metas.append(_chunk_to_metadata(chunk))
            docs.append(chunk.text)

        if ids:
            self._collection.add(ids=ids, embeddings=embs, metadatas=metas, documents=docs)
        return rejected

    def search(self, query_vec: np.ndarray, tenants: Iterable[str], k: int) -> list[ScoredChunk]:
        """Searches only within ``tenants``, pushed into Chroma's own ``where``
        so the ANN search itself never surfaces a foreign-tenant chunk in the
        first place (never a post-hoc filter). An empty ``tenants`` searches
        nothing -- fail closed rather than defaulting to "all"."""
        tenant_list = list(tenants)
        if not tenant_list or k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(query_vec, dtype=np.float32).tolist()],
            n_results=k,
            where={"tenant": {"$in": tenant_list}},
            include=["metadatas", "documents", "distances"],
        )
        return self._scored_from_query_result(result)

    def search_unscoped(self, query_vec: np.ndarray, k: int) -> list[ScoredChunk]:
        """Searches WITHOUT a tenant predicate -- no ``where`` clause at all.

        This exists solely so ``LeakyRetriever`` can reproduce the real-world bug
        (global top-k search, tenant filtering applied only afterwards) for the
        leak-rate measurement. It is deliberately the one place in this module
        that does not fail closed. Do not call this from any secure code path;
        ``SecureRetriever`` never touches it.
        """
        if k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(query_vec, dtype=np.float32).tolist()],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )
        return self._scored_from_query_result(result)

    def _scored_from_query_result(self, result: dict[str, Any]) -> list[ScoredChunk]:
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]

        # THE GUARD (see triad/eval/poisonedrag.py's silent-empty-retrieval
        # postmortem): ``zip()`` over four lists silently stops at the
        # shortest one. If the Chroma client ever returns ``ids`` populated
        # but ``documents``/``metadatas``/``distances`` short or empty (a
        # partial/degraded response from a resource-starved or racing query --
        # observed empirically when two chromadb-backed eval processes ran
        # concurrently on this machine), the old code silently produced an
        # EMPTY scored-chunk list instead of an error: exactly the signature
        # that made a defense look like it worked because a condition
        # retrieved nothing. A populated store must never silently degrade to
        # "found nothing" -- it must raise, loudly, right here at the source.
        lengths = {"ids": len(ids), "documents": len(docs), "metadatas": len(metas), "distances": len(distances)}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(
                f"chromadb query() returned mismatched result lengths {lengths} -- "
                "a partial/degraded response would otherwise silently become an "
                "empty retrieval result. Never truncate via zip() over these lists."
            )

        scored: list[ScoredChunk] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            chunk = _metadata_to_chunk(cid, doc, meta)
            # Measured (not assumed) against chromadb 1.5.9: both "cosine" and
            # "ip" hnsw spaces report distance = 1 - similarity, so this
            # conversion is uniform across embedder conventions.
            scored.append(ScoredChunk(chunk=chunk, score=1.0 - float(dist)))
        return scored

    def get(self, chunk_id: str) -> Chunk | None:
        """Fetches one chunk by id, for round-trip verification. Returns None if
        absent (e.g. it was quarantined and therefore never written)."""
        result = self._collection.get(ids=[chunk_id], include=["metadatas", "documents"])
        ids = result["ids"]
        if not ids:
            return None
        return _metadata_to_chunk(ids[0], result["documents"][0], result["metadatas"][0])

    def count(self) -> int:
        return self._collection.count()
