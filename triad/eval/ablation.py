"""Ablation: each defense component toggled individually against the
PoisonedRAG replay, so each stage's contribution is separable rather than
only ever measured as one bundled "on".

    python -m triad.eval.ablation --n 10

Reuses ``triad.eval.poisonedrag``'s exact corpus/attack construction and
generation code -- there is only one way this repo builds a PoisonedRAG
replay -- so an ablation row and the headline off/on numbers are always
directly comparable (same targets, same corpus, same cache).

Rows: baseline (all off), all on, and each of stage1 (1a+1b together --
they act jointly at ingestion, see ``Pipeline.ingest``), collapse_topk alone,
and secure_retrieval alone (expected to be a measured no-op on this
single-tenant corpus -- see ``poisonedrag.py``'s module docstring; included
anyway so that "no-op" is a measured row, not an assumed one).
"""

from __future__ import annotations

import argparse

from triad.eval import _common

from triad.data.beir import gold_ids_for_queries
from triad.data.poisonedrag import load_targets
from triad.data.types import DatasetHandle
from triad.embed.sentence_transformer_embedder import SentenceTransformerEmbedder
from triad.eval.poisonedrag import (
    GENERATOR_MODEL,
    RETRIEVER_MODEL,
    ask_and_score,
    build_stores,
    load_or_build_clean_corpus,
    poison_chunks_for_targets,
)
from triad.llm.cache import DiskCache
from triad.llm.client import AllKeysExhausted, GroqClient
from triad.llm.keys import load_keys
from triad.llm.limiter import RateLimiter
from triad.pipeline import DefenseConfig
from triad.quarantine import QuarantineQueue
from triad.config import CACHE_DIR

ROWS: dict[str, DefenseConfig] = {
    "baseline_all_off": DefenseConfig(stage1a=False, stage1b=False, secure_retrieval=False, collapse_topk=False),
    "stage1_only": DefenseConfig(stage1a=True, stage1b=True, secure_retrieval=False, collapse_topk=False),
    "collapse_topk_only": DefenseConfig(stage1a=False, stage1b=False, secure_retrieval=False, collapse_topk=True),
    "secure_retrieval_only": DefenseConfig(stage1a=False, stage1b=False, secure_retrieval=True, collapse_topk=False),
    "all_on": DefenseConfig(stage1a=True, stage1b=True, secure_retrieval=True, collapse_topk=True),
}


def main() -> None:
    _common.require_real()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--sample-n", type=int, default=10_000)
    parser.add_argument("--corpus", type=str, default="nq", choices=["nq"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    all_targets = sorted(load_targets(corpus=args.corpus), key=lambda t: t.id)
    targets = all_targets[: args.n]
    gold_ids = gold_ids_for_queries([t.id for t in all_targets])

    embedder = SentenceTransformerEmbedder(RETRIEVER_MODEL)
    clean_chunks, clean_embeddings = load_or_build_clean_corpus(embedder, args.sample_n, args.seed, gold_ids)
    poison_chunks = poison_chunks_for_targets(targets)

    keys = load_keys()
    client = GroqClient(keys=keys, limiter=RateLimiter(), cache=DiskCache())
    cache_stats = _common.CacheStats()

    results: dict[str, dict] = {}
    try:
        for row_name, defense in ROWS.items():
            print(f"-- ablation row: {row_name} ({defense}) --")
            q_clean = QuarantineQueue(
                path=CACHE_DIR / "quarantine" / f"ablation_{row_name}_clean_queue.json",
                log_path=CACHE_DIR / "quarantine" / f"ablation_{row_name}_clean_release_log.jsonl",
            )
            q_poisoned = QuarantineQueue(
                path=CACHE_DIR / "quarantine" / f"ablation_{row_name}_poisoned_queue.json",
                log_path=CACHE_DIR / "quarantine" / f"ablation_{row_name}_poisoned_release_log.jsonl",
            )
            # `on_poisoned` always: Pipeline.ingest() correctly no-ops any
            # disabled stage regardless (see DefenseConfig), so it alone is
            # enough to realize every row -- `off_poisoned` (build_stores'
            # hardcoded Stage-1-bypass variant) is not needed for an ablation.
            pipelines = build_stores(embedder, clean_chunks, clean_embeddings, poison_chunks, defense, defense, q_clean, q_poisoned)
            asr_results = ask_and_score(pipelines["on_poisoned"], client, targets, scored_field="asr", k=args.k, cache_stats=cache_stats)
            asr = sum(1 for r in asr_results if r["success"]) / len(asr_results) if asr_results else float("nan")
            results[row_name] = {
                "defense": {
                    "stage1a": defense.stage1a, "stage1b": defense.stage1b,
                    "secure_retrieval": defense.secure_retrieval, "collapse_topk": defense.collapse_topk,
                },
                "asr": asr,
                "poison_quarantined": pipelines["poison_ingest_report_on"].n_quarantined,
                "poison_submitted": pipelines["poison_ingest_report_on"].n_submitted,
            }
            print(f"   ASR={asr:.1%}  poison_quarantined={pipelines['poison_ingest_report_on'].n_quarantined}/{5*len(targets)}")
    except AllKeysExhausted as exc:
        print(f"AllKeysExhausted: {exc}")
        raise SystemExit(1)
    finally:
        client.close()

    print()
    print("=" * 72)
    print(f"ABLATION -- {args.corpus}, n={len(targets)}, k={args.k}")
    print("=" * 72)
    for row_name, r in results.items():
        print(f"{row_name:24s} ASR={r['asr']:.1%}  poison_quarantined={r['poison_quarantined']}/{r['poison_submitted']}")
    print("=" * 72)

    handle = DatasetHandle(name="poisonedrag.targets", records=tuple(targets))
    clean_handle = DatasetHandle(name="beir.nq", records=tuple(clean_chunks))
    payload = {
        "n_targets": len(targets), "corpus": args.corpus, "k": args.k, "seed": args.seed,
        "sample_n": args.sample_n,
        "data_composition": {"targets": handle.describe(), "clean_corpus": clean_handle.describe()},
        "embedder": {"name": embedder.name, "dim": embedder.dim, "space": embedder.space},
        "generator_model": GENERATOR_MODEL,
        "rows": results,
        "cache_stats": cache_stats.as_dict(),
    }
    _common.write_results(f"ablation_n{len(targets)}", payload)


if __name__ == "__main__":
    main()
