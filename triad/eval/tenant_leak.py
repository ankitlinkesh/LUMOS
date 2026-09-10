"""Stage 2 evaluation: EnronQA cross-tenant probes. No LLM needed.

    python -m triad.eval.tenant_leak [--n-tenants 20] [--n-probes 500] [--seed 0]

Builds probes the way a real cross-tenant leak would actually be tempting: a
question whose gold email lives in inbox B, asked as tenant A, where A is B's
*nearest topical neighbour* by embedding centroid over the corpus actually
indexed -- not a random pair. An unrelated pair is a coincidence, not a
temptation; this is what makes ``LeakyRetriever``'s overfetch-then-filter
bug (arXiv:2605.05287) worth measuring against.

Leak is measured two ways per probe, matching the task brief's "leak = B's
gold email (OR any B chunk) in A's results":

  - ``leak_gold``: B's specific gold email id is among A's returned chunk ids.
  - ``leak_any_foreign``: ``RetrievalResult.foreign_chunks()`` is non-empty --
    the same invariant ``tests/test_retrieval_property.py`` fuzzes, checked
    here on a real corpus instead of Hypothesis-generated documents.

``leak_rate`` reported below is the OR of the two. Both retrievers are run
on IDENTICAL probes and an identical corpus, so the only variable is the
retrieval bug itself.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import Counter

import chromadb
import numpy as np

from triad.eval import _common

from triad.data.enronqa import load_emails, load_qa
from triad.data.types import DatasetHandle
from triad.embed.sentence_transformer_embedder import SentenceTransformerEmbedder
from triad.retrieval.retriever import LeakyRetriever, SecureRetriever, leak_rate
from triad.retrieval.scope import Scope
from triad.retrieval.store import TenantStore

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LO, DEFAULT_HI = 50, 400  # inbox size band: real, but bounded enough to embed on CPU quickly


def _pick_tenants(n: int, *, lo: int = DEFAULT_LO, hi: int = DEFAULT_HI):
    emails = load_emails()
    counts = Counter(e.tenant for e in emails)
    band = sorted(t for t, c in counts.items() if lo <= c <= hi)
    if len(band) < n:
        raise SystemExit(f"only {len(band)} tenants with {lo}-{hi} emails; lower --n-tenants or widen the band")
    return emails, band[:n]


def _nearest_partner(tenant: str, centroids: dict[str, np.ndarray]) -> str:
    v = centroids[tenant]
    vn = v / (np.linalg.norm(v) + 1e-9)
    best, best_sim = None, -2.0
    for other, ov in centroids.items():
        if other == tenant:
            continue
        on = ov / (np.linalg.norm(ov) + 1e-9)
        sim = float(np.dot(vn, on))
        if sim > best_sim:
            best, best_sim = other, sim
    return best


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
    return s[idx]


def build_corpus_and_probes(n_tenants: int, n_probes: int, seed: int, embedder):
    all_emails, tenants = _pick_tenants(n_tenants)
    tenant_set = set(tenants)
    corpus = [c for c in all_emails if c.tenant in tenant_set]

    store = TenantStore(client=chromadb.EphemeralClient(), space=embedder.space)
    vecs = embedder.embed_documents([c.text for c in corpus])
    store.add(corpus, vecs)

    by_tenant: dict[str, list[np.ndarray]] = {t: [] for t in tenants}
    for c, v in zip(corpus, vecs):
        by_tenant[c.tenant].append(v)
    centroids = {t: np.mean(np.stack(vs), axis=0) for t, vs in by_tenant.items() if vs}
    partner = {t: _nearest_partner(t, centroids) for t in centroids}

    qa = list(load_qa(split="test", users=tenants))
    rng = random.Random(seed)
    rng.shuffle(qa)

    corpus_ids = {c.id for c in corpus}
    probes = []  # (question, asking_tenant, gold_tenant, gold_chunk_id)
    for q in qa:
        if q.tenant not in partner or q.email_path not in corpus_ids:
            continue
        probes.append((q.question, partner[q.tenant], q.tenant, q.email_path))
        if len(probes) >= n_probes:
            break

    return store, tenants, corpus, partner, probes


def _run_and_combine(retriever, probes, k: int) -> dict:
    """Single pass per probe (retrieve once, not twice) computing both leak
    definitions plus latency/decline stats, and cross-checked against the
    shared ``triad.retrieval.retriever.leak_rate`` helper (same invariant,
    independently computed) so the two never silently disagree."""
    leak_gold = leak_any = declined = 0
    latencies: list[float] = []
    for question, asking_tenant, gold_tenant, gold_id in probes:
        scope = Scope.of(asking_tenant)
        result = retriever.retrieve(question, scope, k=k)
        latencies.append(result.latency_ms)
        result_ids = {sc.chunk.id for sc in result.chunks}
        leak_gold += int(gold_id in result_ids)
        leak_any += int(bool(result.foreign_chunks()))
        declined += int(result.declined)

    n = len(probes) or 1
    shared_probes = [(q, Scope.of(a), k) for q, a, g, gid in probes]
    shared_leak_rate = leak_rate(retriever, shared_probes)

    return {
        "n": len(probes),
        "leak_rate_gold_specifically": leak_gold / n,
        "leak_rate_any_foreign_chunk": leak_any / n,
        "leak_rate_shared_helper_cross_check": shared_leak_rate,
        "decline_rate": declined / n,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "latency_ms_mean": statistics.fmean(latencies) if latencies else float("nan"),
    }


def main() -> None:
    _common.require_real()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tenants", type=int, default=20)
    parser.add_argument("--n-probes", type=int, default=500)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embed-model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder(args.embed_model)

    print(f"Building corpus from {args.n_tenants} EnronQA inboxes (band {DEFAULT_LO}-{DEFAULT_HI} emails)...")
    t0 = time.perf_counter()
    store, tenants, corpus, partner, probes = build_corpus_and_probes(
        args.n_tenants, args.n_probes, args.seed, embedder,
    )
    build_s = time.perf_counter() - t0
    print(f"  corpus: {len(corpus)} emails across {len(tenants)} tenants, embedded in {build_s:.1f}s")
    print(f"  probes: {len(probes)} (target {args.n_probes})")
    if len(probes) < args.n_probes:
        print(f"  WARNING: only {len(probes)} probes available, below the {args.n_probes} target", flush=True)

    leaky = LeakyRetriever(store=store, embedder=embedder)
    secure = SecureRetriever(store=store, embedder=embedder)

    print("Running probes against LeakyRetriever (vulnerable baseline)...")
    leaky_stats = _run_and_combine(leaky, probes, args.k)
    print("Running probes against SecureRetriever...")
    secure_stats = _run_and_combine(secure, probes, args.k)

    handle = DatasetHandle(name="enronqa.emails", records=tuple(corpus))

    print()
    print("=" * 72)
    print("STAGE 2 -- cross-tenant probe leak rate (EnronQA)")
    print("=" * 72)
    print(f"corpus: {handle.describe()} across {len(tenants)} tenants")
    print(f"probes: n={len(probes)}, k={args.k}, seed={args.seed}, pairing=nearest-centroid-topic")
    print(f"LeakyRetriever:  leak(gold)={leaky_stats['leak_rate_gold_specifically']:.1%}  "
          f"leak(any foreign)={leaky_stats['leak_rate_any_foreign_chunk']:.1%}  "
          f"decline={leaky_stats['decline_rate']:.1%}  "
          f"p50={leaky_stats['latency_ms_p50']:.2f}ms p95={leaky_stats['latency_ms_p95']:.2f}ms")
    print(f"SecureRetriever: leak(gold)={secure_stats['leak_rate_gold_specifically']:.1%}  "
          f"leak(any foreign)={secure_stats['leak_rate_any_foreign_chunk']:.1%}  "
          f"decline={secure_stats['decline_rate']:.1%}  "
          f"p50={secure_stats['latency_ms_p50']:.2f}ms p95={secure_stats['latency_ms_p95']:.2f}ms")
    added_latency_p50 = secure_stats["latency_ms_p50"] - leaky_stats["latency_ms_p50"]
    print(f"added latency at p50 (secure - leaky): {added_latency_p50:.2f}ms")
    print("=" * 72)

    payload = {
        "data_composition": handle.describe(),
        "data_source": handle.data_source,
        "embedder": {"name": embedder.name, "dim": embedder.dim, "space": embedder.space},
        "n_tenants": len(tenants), "tenants": tenants,
        "n_probes": len(probes), "k": args.k, "seed": args.seed,
        "pairing_method": "nearest topic centroid (cosine over indexed corpus embeddings)",
        "leaky_retriever": leaky_stats,
        "secure_retriever": secure_stats,
        "added_latency_ms_p50_secure_minus_leaky": added_latency_p50,
        "corpus_build_seconds": build_s,
    }
    _common.write_results("tenant_leak", payload)


if __name__ == "__main__":
    main()
