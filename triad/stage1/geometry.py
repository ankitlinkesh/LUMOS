"""Stage 1B: the poison-geometry detector.

PoisonedRAG (USENIX Security '25) writes fluent, factual-sounding text -- the
paper's own ablation shows a perplexity filter does not fire on it, and exact
near-duplicate filtering does nothing because the five injected texts per
target differ (`src/attack.py`: `adv_text_a = question + "."`, then five
DIFFERENT corroborating suffixes are concatenated onto that same prefix). So
this module does not read the text for wording; it looks at the SHAPE the
attack leaves in embedding space, at two points in the pipeline where that
shape is visible:

  ingest_scan   -- at ingestion, before any query exists. Two independent
                   signals, either sufficient alone (see combination below):
      query_echo         the document opens with a question-shaped sentence
                          that dominates the document's own embedding -- the
                          literal fingerprint of `question + "." + text`. This
                          is cheap and precise but a white-box attacker who
                          token-flips the prefix (HotFlip) or drops it defeats
                          it, and it never fires when we replay the raw
                          released texts without the paper's own prepend.
      batch_cluster       the document has at least three other close
                          neighbours in the same ingestion batch. This catches
                          the attack's five-related-chunk shape when wording
                          has been paraphrased or the query prefix is gone.
      manifold_isolation  the document sits off the reference corpus's own
                          manifold. It remains available for measurement but
                          is disabled in the default path after held-out
                          testing found worse-than-chance discrimination.

  collapse_topk -- at query time, on an already-retrieved top-k. PoisonedRAG's
                   attack ONLY works by outnumbering the corpus: 5 mutually
                   near-identical texts occupying 5 of the k retrieval slots.
                   Collapsing each near-duplicate cluster to its single
                   highest-scoring member breaks that arithmetic regardless of
                   whether ingestion caught anything, and does so in embedding
                   space -- not exact-hash dedup, which the paper's own
                   ablation shows does nothing here (the five `sub-text I` are
                   different even though the shared `Q` prefix makes them
                   cluster tightly).

House style matches `triad.stage1.directive`: pure, deterministic, injectable
embedder, every quarantine carries a human-readable reason, and any internal
failure is treated as suspicious -- fail closed, never silently clean.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Callable, Sequence

import numpy as np

from triad.contract import Chunk, GuardDecision, RetrievalResult, TaintVerdict

__all__ = [
    "ingest_scan",
    "collapse_topk",
    "to_taint",
    "query_echo_signal",
    "query_echo_sims_batch",
    "batch_cluster_scores",
    "manifold_isolation_scores",
    "manifold_isolation_baseline",
    "THRESHOLD",
    "QUERY_ECHO_WEIGHT",
    "ISOLATION_WEIGHT",
    "QUERY_ECHO_SIM_THRESHOLD",
    "BATCH_CLUSTER_SIM_THRESHOLD",
    "BATCH_CLUSTER_WEIGHT",
    "BATCH_CLUSTER_MIN_NEIGHBORS",
    "ISOLATION_PERCENTILE",
    "DEFAULT_KNN_K",
    "COLLAPSE_SIM_THRESHOLD",
]

# ---------------------------------------------------------------------------
# Tunable defaults. The eval harness (eval_geometry.py) re-derives these on a
# dev split and passes the tuned values in; these are sane fixed fallbacks for
# callers (and tests) that don't tune.
# ---------------------------------------------------------------------------

# Tuned on the DEV half of PoisonedRAG's NQ targets and 2,500 clean BEIR
# passages: 0.30 is the sensitivity-maximizing point on the 0.30..0.95 sweep
# that stays within the 3% DEV clean-FPR budget (53/2,500 = 2.12%). The
# evaluator reports the full ROC; this value is never selected from holdout.
QUERY_ECHO_SIM_THRESHOLD = 0.30   # cosine(sentence, doc) at/above this = "dominates"
BATCH_CLUSTER_SIM_THRESHOLD = 0.90  # selected on DEV; max in-batch neighbour cosine
BATCH_CLUSTER_MIN_NEIGHBORS = 3     # requires a four-document related group
ISOLATION_PERCENTILE = 0.0        # disabled by default after held-out worse-than-chance result
DEFAULT_KNN_K = 5

QUERY_ECHO_WEIGHT = 0.6
BATCH_CLUSTER_WEIGHT = 0.6
ISOLATION_WEIGHT = 0.6
THRESHOLD = 0.5   # each weight alone clears this: either signal is sufficient to quarantine

COLLAPSE_SIM_THRESHOLD = 0.90   # cosine, on normalized embeddings, for "near-duplicate"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s*(?=[A-Z])")
_QUESTION_WORD_RE = re.compile(
    r"^[\"'“‘(]*\s*(who|what|when|where|which|why|how|is|are|was|were|"
    r"does|do|did|can|could|would|will|should|has|have|had|am)\b",
    re.IGNORECASE,
)


def _normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalize rows so a plain dot product is a true cosine similarity,
    regardless of the embedder's own declared convention (Contriever's raw
    dot-product ranking is a Stage-2 retrieval detail; Stage 1B's internal
    geometry checks always want cosine)."""
    mat = np.atleast_2d(np.asarray(mat, dtype=np.float32))
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _leading_sentences(text: str, max_n: int) -> list[str]:
    """Split off the first ``max_n`` sentences. Splits on sentence punctuation
    followed by a capital letter, with or without a space between them --
    PoisonedRAG's own recipe glues `question + "."` directly onto the next
    sentence with NO space (`adv_text_a + i`, no separator), so a splitter
    that requires whitespace would silently merge the question into the body
    and never see it as its own sentence."""
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts[:max_n] if p.strip()]


def _looks_like_question(sentence: str) -> bool:
    """Interrogative shape: ends with '?', or opens with a wh-word/auxiliary
    verb. Ends-with-'?' alone is not enough on its own here because most raw
    NQ questions carry no question mark at all -- the opening-word check is
    what actually fires on the paper's texts."""
    s = sentence.strip()
    if not s:
        return False
    if s.endswith("?"):
        return True
    return bool(_QUESTION_WORD_RE.match(s))


def _mean_topk(sims: np.ndarray, k: int, self_indices: np.ndarray | None = None) -> np.ndarray:
    """Mean similarity to the k nearest neighbours in ``sims`` (rows = candidates,
    columns = reference pool), optionally excluding each row's own position in
    the reference pool (leave-one-out, for measuring a reference corpus
    against itself)."""
    sims = sims.copy()
    if self_indices is not None:
        rows = np.arange(sims.shape[0])
        sims[rows, self_indices] = -np.inf
    available = sims.shape[1] - (1 if self_indices is not None else 0)
    k = max(1, min(k, available))
    part = np.partition(sims, -k, axis=1)[:, -k:]
    part = np.where(np.isneginf(part), np.nan, part)
    return np.nanmean(part, axis=1).astype(np.float32)


def manifold_isolation_scores(embeddings: np.ndarray, reference_embeddings: np.ndarray,
                               *, knn_k: int = DEFAULT_KNN_K) -> np.ndarray:
    """k-NN density of each candidate embedding against a (fixed, external)
    reference corpus: the mean cosine similarity to its ``knn_k`` nearest
    neighbours in ``reference_embeddings``. Low density = far from everything
    the corpus already trusts."""
    a = _normalize(embeddings)
    r = _normalize(reference_embeddings)
    sims = a @ r.T
    return _mean_topk(sims, knn_k)


def manifold_isolation_baseline(reference_embeddings: np.ndarray, *, knn_k: int = DEFAULT_KNN_K,
                                 sample_n: int = 1000, seed: int = 0) -> np.ndarray:
    """What k-NN density normally looks like among the reference corpus's OWN
    documents (leave-one-out, so a document is never its own neighbour). The
    tail of this distribution is the "unusually low density" cutoff
    ``ingest_scan`` flags against -- computed on a deterministic random
    subsample so it stays cheap on a corpus of thousands of passages."""
    r = _normalize(reference_embeddings)
    n = r.shape[0]
    if n < 2:
        return np.array([], dtype=np.float32)
    rng = np.random.default_rng(seed)
    sample_n = min(sample_n, n)
    idx = rng.choice(n, size=sample_n, replace=False)
    sims = r[idx] @ r.T
    return _mean_topk(sims, knn_k, self_indices=idx)


def query_echo_sims_batch(texts: Sequence[str], doc_embeddings: np.ndarray, embedder,
                           *, max_sentences: int = 2) -> tuple[np.ndarray, list[str | None]]:
    """Batched query-echo similarity: ONE embedder call covering every
    candidate question-shaped sentence across the whole ``texts`` batch,
    rather than one call per document -- what ``ingest_scan`` needs to stay
    cheap over a multi-thousand-document corpus, and what a threshold sweep
    (``eval_geometry.py`` tuning on a dev split) needs to avoid re-embedding
    on every candidate threshold. Returns ``(best_sim per doc, matched
    sentence per doc)``; a document with no question-shaped opening sentence
    gets ``sim=0.0``, ``matched=None``."""
    texts = list(texts)
    doc_embeddings = np.asarray(doc_embeddings, dtype=np.float32)
    n = len(texts)
    candidates: list[tuple[int, str]] = []
    for i, text in enumerate(texts):
        for sentence in _leading_sentences(text, max_sentences):
            if _looks_like_question(sentence):
                candidates.append((i, sentence))

    sim = np.zeros(n, dtype=np.float32)
    matched: list[str | None] = [None] * n
    if candidates:
        sentences = [s for _, s in candidates]
        sent_vecs = _normalize(np.asarray(embedder.embed_documents(sentences), dtype=np.float32))
        doc_vecs_norm = _normalize(doc_embeddings)
        for (idx, sentence), svec in zip(candidates, sent_vecs):
            s = float(np.dot(svec, doc_vecs_norm[idx]))
            if s > sim[idx]:
                sim[idx] = s
                matched[idx] = sentence
    return sim, matched


def query_echo_signal(text: str, doc_embedding: np.ndarray, embedder, *,
                       threshold: float = QUERY_ECHO_SIM_THRESHOLD,
                       max_sentences: int = 2) -> tuple[bool, float, str | None]:
    """Single-document convenience wrapper around ``query_echo_sims_batch``:
    does one of the document's first ``max_sentences`` sentences (a) look
    like a question and (b) have an embedding whose cosine similarity to the
    WHOLE document's embedding is at/above ``threshold``? Returns
    ``(fired, best_similarity, matched_sentence)``."""
    doc_embedding = np.atleast_2d(np.asarray(doc_embedding, dtype=np.float32))
    sims, matched = query_echo_sims_batch([text], doc_embedding, embedder, max_sentences=max_sentences)
    return bool(sims[0] >= threshold), float(sims[0]), matched[0]


def batch_cluster_scores(embeddings: np.ndarray, *, threshold: float = BATCH_CLUSTER_SIM_THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    """Return each document's strongest *other* document cosine in its batch.

    PoisonedRAG emits five related chunks per target, so this catches attacks
    whose wording no longer contains a query-shaped opener. A one-document
    ingestion batch has no neighbour and receives ``-1``; this is deliberately
    a bulk-ingestion signal, not a claim about isolated uploads.
    """
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got {x.shape}")
    if x.shape[0] < 2:
        return (np.full(x.shape[0], -1.0, dtype=np.float32),
                np.zeros(x.shape[0], dtype=np.int32))
    x = _normalize(x)
    sims = x @ x.T
    np.fill_diagonal(sims, -1.0)
    return sims.max(axis=1).astype(np.float32), (sims >= threshold).sum(axis=1)


def ingest_scan(
    chunks: Sequence[Chunk],
    embeddings: np.ndarray,
    *,
    reference_embeddings: np.ndarray,
    embedder,
    query_echo_threshold: float = QUERY_ECHO_SIM_THRESHOLD,
    batch_cluster_threshold: float = BATCH_CLUSTER_SIM_THRESHOLD,
    isolation_percentile: float = ISOLATION_PERCENTILE,
    knn_k: int = DEFAULT_KNN_K,
    max_echo_sentences: int = 2,
    baseline_sample_n: int = 1000,
    baseline_seed: int = 0,
) -> list[GuardDecision]:
    """Score every chunk at ingestion time, before any query exists. Combines
    query_echo and batch_cluster with thresholds: either signal's weight alone
    clears the block threshold. Manifold isolation is retained as an explicit
    diagnostic but disabled by the default zero-percent cutoff.

    ``embeddings`` are the chunks' own already-computed document embeddings
    (parallel to ``chunks``); ``reference_embeddings`` is the clean background
    corpus manifold_isolation measures distance against. Never raises: any
    internal failure blocks every chunk in the batch (fail closed).
    """
    try:
        chunks_list = list(chunks)
    except Exception:
        chunks_list = []

    try:
        n = len(chunks_list)
        if n == 0:
            return []
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != n:
            raise ValueError(f"embeddings shape {embeddings.shape} does not match {n} chunks")

        # --- signal 1: query echo ---
        texts = [c.text for c in chunks_list]
        echo_sim, echo_sentence = query_echo_sims_batch(texts, embeddings, embedder, max_sentences=max_echo_sentences)
        echo_fired = echo_sim >= query_echo_threshold

        # --- signal 2: related-document batch cluster ---
        cluster_sim, cluster_neighbors = batch_cluster_scores(embeddings, threshold=batch_cluster_threshold)
        cluster_fired = ((cluster_sim >= batch_cluster_threshold) &
                         (cluster_neighbors >= BATCH_CLUSTER_MIN_NEIGHBORS))

        # --- signal 3: manifold isolation (disabled by default) ---
        ref = np.asarray(reference_embeddings, dtype=np.float32) if reference_embeddings is not None else np.zeros((0, embeddings.shape[1]), dtype=np.float32)
        if ref.ndim == 2 and ref.shape[0] >= 2:
            density = manifold_isolation_scores(embeddings, ref, knn_k=knn_k)
            baseline = manifold_isolation_baseline(ref, knn_k=knn_k, sample_n=baseline_sample_n, seed=baseline_seed)
            cutoff = float(np.percentile(baseline, isolation_percentile * 100)) if baseline.size else float("-inf")
            iso_fired = density < cutoff
        else:
            # No usable reference corpus yet (e.g. empty index): isolation
            # simply has nothing to say, not an error -- query_echo still runs.
            density = np.zeros(n, dtype=np.float32)
            cutoff = float("nan")
            iso_fired = np.zeros(n, dtype=bool)

        decisions: list[GuardDecision] = []
        for i in range(n):
            score = 0.0
            fired: list[str] = []
            reasons: list[str] = []
            if bool(echo_fired[i]):
                score += QUERY_ECHO_WEIGHT
                fired.append("query_echo")
                reasons.append(
                    f"opens with a question-shaped sentence whose embedding dominates the "
                    f"document (cosine={echo_sim[i]:.3f} >= {query_echo_threshold:.3f}): "
                    f"{echo_sentence[i]!r}"
                )
            if bool(cluster_fired[i]):
                score += BATCH_CLUSTER_WEIGHT
                fired.append("batch_cluster")
                reasons.append(
                    f"document is unusually close to another document in the ingestion batch "
                    f"(cosine={cluster_sim[i]:.3f} >= {batch_cluster_threshold:.3f}, "
                    f"{int(cluster_neighbors[i])} neighbours)"
                )
            if bool(iso_fired[i]):
                score += ISOLATION_WEIGHT
                fired.append("manifold_isolation")
                reasons.append(
                    f"document sits off the reference corpus manifold (k-NN density="
                    f"{density[i]:.3f} < corpus baseline cutoff {cutoff:.3f} at the "
                    f"{isolation_percentile:.0%} tail)"
                )
            score = min(score, 1.0)
            evidence: dict[str, Any] = {
                "score": score,
                "signals": tuple(fired),
                "query_echo_sim": float(echo_sim[i]),
                "batch_cluster_sim": float(cluster_sim[i]),
                "batch_cluster_neighbors": int(cluster_neighbors[i]),
                "batch_cluster_threshold": float(batch_cluster_threshold),
                "manifold_density": float(density[i]),
                "manifold_baseline_cutoff": cutoff,
            }
            if score >= THRESHOLD:
                decisions.append(GuardDecision.block("ingest", *reasons, evidence=evidence))
            else:
                decisions.append(GuardDecision.ok("ingest", evidence=evidence))
        return decisions

    except Exception as exc:  # fail closed: an error scanning the batch quarantines the batch
        n = len(chunks_list)
        evidence = {"score": 1.0, "signals": ("internal_error",), "internal_error": True}
        reason = f"internal error while scanning geometry: {type(exc).__name__}: {exc}"
        return [GuardDecision.block("ingest", reason, evidence=evidence) for _ in range(n)]


def to_taint(decision: GuardDecision) -> TaintVerdict:
    """Project an ``ingest_scan`` decision onto the shared ``TaintVerdict``,
    mirroring ``stage1.directive.to_taint``: retrieved/ingested content is
    always untrusted; ``quarantined`` is the only thing a block changes."""
    blocked = not decision.allow
    score = float(decision.evidence.get("score", 1.0 if blocked else 0.0))
    flags = tuple(decision.evidence.get("signals", ()))
    return TaintVerdict(
        untrusted=True,
        quarantined=blocked,
        flags=flags,
        score=score,
        reasons=decision.reasons,
    )


def collapse_topk(
    result: RetrievalResult,
    embed_fn: Callable[[Sequence[str]], np.ndarray],
    *,
    sim_threshold: float = COLLAPSE_SIM_THRESHOLD,
) -> tuple[RetrievalResult, GuardDecision]:
    """Collapse near-duplicate clusters inside an already-retrieved top-k to
    one vote each: PoisonedRAG's mechanism IS outnumbering the corpus with 5
    mutually near-identical texts, so collapsing each tight cluster to its
    single highest-scoring member breaks that arithmetic before the LLM ever
    sees the retrieved set, regardless of what (if anything) ingestion caught.

    ``embed_fn`` takes a batch of texts and returns their embeddings (pass
    e.g. ``embedder.embed_documents``). Clustering is embedding-space cosine
    similarity via union-find, deliberately NOT exact-hash dedup -- the
    paper's own ablation shows exact duplicate filtering does nothing here,
    because the five injected texts differ; only the shared question prefix
    makes them cluster.

    Preserves every ``RetrievalResult`` invariant (tenant, scope_applied,
    declined/decline_reason, query, latency_ms, leak_mode) untouched -- only
    ``chunks`` changes. Never raises: on internal failure, returns the
    ORIGINAL uncollapsed result plus an ``escalate`` decision (fail closed --
    an error here must not silently claim collapsing happened, and must not
    silently drop chunks either).
    """
    try:
        chunks = result.chunks
        n = len(chunks)
        if n < 2:
            evidence: dict[str, Any] = {
                "num_clusters_collapsed": 0,
                "cluster_sizes": (),
                "collapsed_members": {},
                "original_count": n,
                "collapsed_count": n,
                "sim_threshold": sim_threshold,
            }
            return result, GuardDecision.ok("retrieve", evidence=evidence)

        if not (0.0 < sim_threshold <= 1.0):
            raise ValueError(f"sim_threshold must be in (0, 1], got {sim_threshold}")

        texts = [sc.chunk.text for sc in chunks]
        vecs = _normalize(np.asarray(embed_fn(texts), dtype=np.float32))
        if vecs.shape[0] != n:
            raise ValueError(f"embed_fn returned {vecs.shape[0]} vectors for {n} texts")
        sims = vecs @ vecs.T

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if sims[i, j] >= sim_threshold:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        keep_indices: list[int] = []
        cluster_sizes: list[int] = []
        collapsed_members: dict[str, tuple[str, ...]] = {}
        for members in groups.values():
            if len(members) == 1:
                keep_indices.append(members[0])
                continue
            rep = max(members, key=lambda idx: chunks[idx].score)
            keep_indices.append(rep)
            cluster_sizes.append(len(members))
            collapsed_members[chunks[rep].chunk.id] = tuple(chunks[m].chunk.id for m in members)

        keep_indices.sort()  # subsequence of the original (score-descending) order stays sorted
        new_chunks = tuple(chunks[i] for i in keep_indices)

        evidence = {
            "num_clusters_collapsed": len(cluster_sizes),
            "cluster_sizes": tuple(sorted(cluster_sizes, reverse=True)),
            "collapsed_members": collapsed_members,
            "original_count": n,
            "collapsed_count": len(new_chunks),
            "sim_threshold": sim_threshold,
        }
        new_result = dataclasses.replace(result, chunks=new_chunks)
        return new_result, GuardDecision.ok("retrieve", evidence=evidence)

    except Exception as exc:  # fail closed: don't claim a collapse that may not have happened right
        evidence = {"internal_error": True}
        reason = f"internal error while collapsing near-duplicate clusters: {type(exc).__name__}: {exc}"
        decision = GuardDecision.needs_human("retrieve", reason, evidence=evidence)
        return result, decision
