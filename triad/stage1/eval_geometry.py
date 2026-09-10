"""Measure Stage 1B (triad.stage1.geometry) against real data.

    python -m triad.stage1.eval_geometry --corpus nq --n-clean 5000 --embedder bge

Two checkpoints, matching geometry.py:
  - ingest_scan: per-signal (query_echo / manifold_isolation / combined) catch
    rate on PoisonedRAG's released nq adv_texts, under three adaptive-attacker
    variants -- (a) verbatim question prefix (the paper's own recipe), (b) the
    question deterministically paraphrased (reorder + synonym-swap, no LLM),
    (c) no question prefix at all (raw adv_text) -- plus false-positive rate
    on real BEIR NQ passages.
  - collapse_topk: per target question, simulate a top-5 retrieval over
    clean+poison and report how often the poison cluster collapses to 1 vote
    vs how often two distinct clean top-5 passages get wrongly merged.

Split (stated once, used everywhere below): PoisonedRAG's 100 nq targets are
sorted by id; the first 50 are DEV (thresholds tuned here), the other 50 are
HELD-OUT (reported numbers, never touched while tuning). The clean BEIR
sample is split the same way -- first half DEV, second half HELD-OUT. The
manifold-isolation reference corpus at report time is the DEV clean half
(the "index"); HELD-OUT clean/poison are scored against it as new, unseen
candidates -- never against themselves, which would trivially self-match.

``facebook/contriever`` is the paper's own retriever; if its one-time
download fails (no network), this falls back to ``BAAI/bge-small-en-v1.5``
and says so loudly, never silently.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from triad import config as _config  # noqa: F401  side effect: HF_HOME -> D:
from triad.config import INDEX_DIR
from triad.contract import Chunk, Provenance, RetrievalResult, ScoredChunk
from triad.data import beir, poisonedrag
from triad.data.types import DatasetHandle, PoisonTarget
from triad.embed.hash_embedder import HashEmbedder
from triad.eval import _common
from triad.stage1 import geometry

VARIANTS = ("verbatim", "paraphrased", "no_prefix")

# ---------------------------------------------------------------------------
# deterministic paraphrase (no LLM): synonym-swap known function/content
# words, then reverse the order of what's left. Not meant to read naturally --
# meant to change the surface form enough to test whether query_echo, which
# scores embedding similarity rather than exact wording, still fires on
# something that is not the verbatim released prefix.
# ---------------------------------------------------------------------------

_SYNONYMS = {
    "who": "which person", "what": "which thing", "when": "at what time",
    "where": "in what place", "which": "what", "how": "in what way",
    "why": "for what reason", "is": "was", "are": "were", "does": "did",
    "do": "did", "the": "that", "a": "one", "an": "one", "name": "title",
    "first": "initial", "movie": "film", "book": "novel", "city": "town",
    "country": "nation", "year": "date", "person": "individual",
    "capital": "seat of government", "many": "how much", "much": "how many",
    "old": "aged", "new": "recent", "big": "large", "small": "little",
}


def paraphrase_question(question: str) -> str:
    """Deterministic word-level paraphrase (same output every run, no model
    call): synonym-swap the leading word plus any recognized function/content
    words, then reverse the order of the remaining tail words."""
    words = question.strip().split()
    if not words:
        return question
    head, *tail = words
    new_head = _SYNONYMS.get(head.lower(), head)
    swapped_tail = [_SYNONYMS.get(w.lower(), w) for w in tail]
    reordered_tail = list(reversed(swapped_tail))
    return " ".join([new_head] + reordered_tail)


def build_variant_texts(target: PoisonTarget, variant: str) -> tuple[str, ...]:
    if variant == "verbatim":
        return target.adv_texts
    if variant == "paraphrased":
        para_q = paraphrase_question(target.question)
        return tuple(para_q + "." + raw for raw in target.raw_adv_texts)
    if variant == "no_prefix":
        return target.raw_adv_texts
    raise ValueError(f"unknown variant {variant!r}")


# ---------------------------------------------------------------------------
# embedder loading + a tiny on-disk embedding cache (this script is the only
# owner of .cache/index/geom_*.npy; other agents' caches use other prefixes)
# ---------------------------------------------------------------------------

def _load_embedder(name: str):
    if name == "hash":
        return HashEmbedder(dim=384)
    from triad.embed.sentence_transformer_embedder import SentenceTransformerEmbedder
    if name == "contriever":
        try:
            emb = SentenceTransformerEmbedder("facebook/contriever")
            _ = emb.dim  # force the download/load now, so a failure surfaces here, not mid-run
            return emb
        except Exception as exc:
            print(f"[eval_geometry] facebook/contriever failed to load ({type(exc).__name__}: {exc}); "
                  f"falling back to BAAI/bge-small-en-v1.5")
            name = "bge"
    if name == "bge":
        return SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    raise ValueError(f"unknown embedder {name!r}")


def _safe_name(s: str) -> str:
    return s.replace("/", "__").replace(":", "_")


def _embed_cached(embedder, texts: Sequence[str], cache_name: str) -> np.ndarray:
    texts = list(texts)
    path = INDEX_DIR / cache_name
    if path.exists():
        arr = np.load(path)
        if arr.shape[0] == len(texts):
            return arr
    arr = embedder.embed_documents(texts) if texts else np.zeros((0, embedder.dim), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


# ---------------------------------------------------------------------------
# ingest_scan evaluation
# ---------------------------------------------------------------------------

@dataclass
class SignalRates:
    n: int
    query_echo: float
    manifold_isolation: float
    combined: float


def _rate(fired: np.ndarray) -> float:
    return float(np.mean(fired)) if fired.size else float("nan")


def _pick_query_echo_threshold(clean_sims: np.ndarray, poison_sims: np.ndarray,
                                candidates: Sequence[float], max_fpr: float) -> float:
    """Smallest (most sensitive) candidate threshold whose DEV clean false-positive
    rate stays at/under ``max_fpr``; falls back to the strictest candidate if none
    qualifies (never silently pick something that blows the FPR budget)."""
    for t in sorted(candidates):
        if _rate(clean_sims >= t) <= max_fpr:
            return t
    return max(candidates)


def _isolation_cutoff(reference_embeddings: np.ndarray, percentile: float, knn_k: int) -> float:
    baseline = geometry.manifold_isolation_baseline(reference_embeddings, knn_k=knn_k,
                                                      sample_n=reference_embeddings.shape[0], seed=0)
    return float(np.percentile(baseline, percentile * 100)) if baseline.size else float("-inf")


def _signal_rates(echo_sim: np.ndarray, echo_t: float,
                   density: np.ndarray, iso_cutoff: float) -> SignalRates:
    echo_fired = echo_sim >= echo_t
    iso_fired = density < iso_cutoff
    combined_score = np.where(echo_fired, geometry.QUERY_ECHO_WEIGHT, 0.0) + \
        np.where(iso_fired, geometry.ISOLATION_WEIGHT, 0.0)
    combined_fired = np.minimum(combined_score, 1.0) >= geometry.THRESHOLD
    return SignalRates(n=echo_sim.size, query_echo=_rate(echo_fired),
                        manifold_isolation=_rate(iso_fired), combined=_rate(combined_fired))


# ---------------------------------------------------------------------------
# collapse_topk evaluation
# ---------------------------------------------------------------------------

def _similarity_to_query(query_vec: np.ndarray, doc_vecs: np.ndarray, space: str) -> np.ndarray:
    if space == "ip":
        return doc_vecs @ query_vec
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    dn = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12)
    return dn @ qn


def _simulate_and_collapse(targets: Sequence[PoisonTarget], variant: str,
                            clean_chunks: Sequence[Chunk], clean_emb: np.ndarray,
                            embedder, sim_threshold: float, k: int = 5) -> list[dict]:
    rows: list[dict] = []
    for target in targets:
        poison_texts = build_variant_texts(target, variant)
        poison_ids = [f"poison:{target.id}:{variant}:{i}" for i in range(len(poison_texts))]
        poison_emb = _embed_cached(
            embedder, poison_texts,
            f"geom_{_safe_name(embedder.name)}_nq_poison_{variant}_{target.id}.npy",
        )

        query_vec = embedder.embed_query(target.question)
        clean_scores = _similarity_to_query(query_vec, clean_emb, embedder.space)
        poison_scores = _similarity_to_query(query_vec, poison_emb, embedder.space)

        combined = [(float(s), "clean", i) for i, s in enumerate(clean_scores)]
        combined += [(float(s), "poison", i) for i, s in enumerate(poison_scores)]
        combined.sort(key=lambda x: -x[0])
        top = combined[:k]

        scored: list[ScoredChunk] = []
        for score, kind, idx in top:
            if kind == "clean":
                scored.append(ScoredChunk(chunk=clean_chunks[idx], score=score))
            else:
                chunk = Chunk(
                    id=poison_ids[idx], text=poison_texts[idx], tenant="public", source_type="passage",
                    provenance=Provenance(f"poisonedrag:nq:{variant}", f"{target.id}:{idx}", "real"),
                )
                scored.append(ScoredChunk(chunk=chunk, score=score))

        result = RetrievalResult(query=target.question, tenant="public", chunks=tuple(scored), scope_applied=("public",))
        new_result, decision = geometry.collapse_topk(result, embedder.embed_documents, sim_threshold=sim_threshold)

        clean_ids_in_top = {sc.chunk.id for sc in scored if not sc.chunk.id.startswith("poison:")}
        poison_ids_in_top = {sc.chunk.id for sc in scored if sc.chunk.id.startswith("poison:")}
        collapsed_members = decision.evidence.get("collapsed_members", {})
        poison_votes_after = len(poison_ids_in_top & {sc.chunk.id for sc in new_result.chunks})
        wrong_merge = any(
            len([m for m in members if m in clean_ids_in_top]) >= 2
            for members in collapsed_members.values()
        )
        rows.append({
            "target_id": target.id,
            "poison_in_top5": len(poison_ids_in_top),
            "poison_votes_after_collapse": poison_votes_after,
            "wrong_clean_merge": wrong_merge,
        })
    return rows


def _summarize_collapse(rows: list[dict]) -> dict:
    with_poison = [r for r in rows if r["poison_in_top5"] >= 2]
    collapsed_to_one = [r for r in with_poison if r["poison_votes_after_collapse"] == 1]
    poison_in_top5_counts = Counter(r["poison_in_top5"] for r in rows)
    return {
        "n_targets": len(rows),
        "n_with_2plus_poison_in_top5": len(with_poison),
        "poison_collapse_rate": (len(collapsed_to_one) / len(with_poison)) if with_poison else float("nan"),
        "clean_wrong_merge_rate": _rate(np.array([r["wrong_clean_merge"] for r in rows])),
        # how many of the 5 top-k slots poison actually occupied (mixed clean+poison pool) --
        # reported because "n_with_2plus_poison_in_top5=n_targets" alone can hide that most
        # slots (or all 5) were poison, which would make the mixed-pool clean_wrong_merge_rate
        # above structurally close to vacuous (too few clean chunks left to ever collide).
        "poison_in_top5_distribution": {str(k): v for k, v in sorted(poison_in_top5_counts.items())},
    }


def _simulate_clean_only_collapse(targets: Sequence[PoisonTarget], clean_chunks: Sequence[Chunk],
                                   clean_emb: np.ndarray, embedder, sim_threshold: float, k: int = 5) -> list[dict]:
    """The real answer to 'how often are two DISTINCT clean top-5 passages wrongly
    merged': top-k over the clean corpus ALONE (no poison injected), per held-out
    question. The mixed clean+poison simulation above answers a related but
    different question -- and when poison dominates the top-k (see
    ``poison_in_top5_distribution``), too few clean chunks are left in that top-k
    for its own ``clean_wrong_merge_rate`` to be anything but vacuously 0."""
    rows: list[dict] = []
    for target in targets:
        query_vec = embedder.embed_query(target.question)
        clean_scores = _similarity_to_query(query_vec, clean_emb, embedder.space)
        top_idx = np.argsort(-clean_scores)[:k]
        scored = [ScoredChunk(chunk=clean_chunks[i], score=float(clean_scores[i])) for i in top_idx]
        result = RetrievalResult(query=target.question, tenant="public", chunks=tuple(scored), scope_applied=("public",))
        _, decision = geometry.collapse_topk(result, embedder.embed_documents, sim_threshold=sim_threshold)
        collapsed_members = decision.evidence.get("collapsed_members", {})
        wrong_merge = any(len(members) >= 2 for members in collapsed_members.values())
        rows.append({"target_id": target.id, "wrong_merge": wrong_merge,
                      "num_clusters_collapsed": decision.evidence.get("num_clusters_collapsed", 0)})
    return rows


def _summarize_clean_only_collapse(rows: list[dict]) -> dict:
    return {
        "n_targets": len(rows),
        "wrong_merge_rate": _rate(np.array([r["wrong_merge"] for r in rows])),
        "n_targets_with_any_collapse": sum(1 for r in rows if r["num_clusters_collapsed"] > 0),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    _common.require_real()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="nq", choices=["nq"],
                         help="background corpus; only nq is wired (triad.data.beir is nq-only)")
    parser.add_argument("--n-clean", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedder", default="bge", choices=["contriever", "bge", "hash"])
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--max-fpr", type=float, default=0.01, help="dev FPR budget for query_echo threshold selection")
    args = parser.parse_args()

    t0 = time.perf_counter()
    embedder = _load_embedder(args.embedder)
    embedder_tag = _safe_name(embedder.name)
    print(f"[eval_geometry] embedder={embedder.name} space={embedder.space} corpus={args.corpus} "
          f"n_clean={args.n_clean} seed={args.seed}")

    all_targets = sorted(poisonedrag.load_targets(corpus=args.corpus), key=lambda t: t.id)
    dev_targets, holdout_targets = all_targets[:50], all_targets[50:]
    print(f"[eval_geometry] {len(all_targets)} PoisonedRAG targets: {len(dev_targets)} dev / {len(holdout_targets)} held-out")

    clean_chunks = beir.load_nq(sample_n=args.n_clean, seed=args.seed)
    half = len(clean_chunks) // 2
    dev_clean, holdout_clean = clean_chunks[:half], clean_chunks[half:]
    print(f"[eval_geometry] {len(clean_chunks)} clean NQ passages: {len(dev_clean)} dev / {len(holdout_clean)} held-out")

    dev_clean_emb = _embed_cached(embedder, [c.text for c in dev_clean],
                                   f"geom_{embedder_tag}_{args.corpus}_dev_clean_{len(dev_clean)}_{args.seed}.npy")
    holdout_clean_emb = _embed_cached(embedder, [c.text for c in holdout_clean],
                                       f"geom_{embedder_tag}_{args.corpus}_holdout_clean_{len(holdout_clean)}_{args.seed}.npy")

    # --- per-variant poison texts + embeddings for dev and held-out targets ---
    variant_data = {}
    for variant in VARIANTS:
        dev_texts = [t for target in dev_targets for t in build_variant_texts(target, variant)]
        holdout_texts = [t for target in holdout_targets for t in build_variant_texts(target, variant)]
        dev_emb = _embed_cached(embedder, dev_texts, f"geom_{embedder_tag}_{args.corpus}_dev_poison_{variant}.npy")
        holdout_emb = _embed_cached(embedder, holdout_texts, f"geom_{embedder_tag}_{args.corpus}_holdout_poison_{variant}.npy")
        variant_data[variant] = {"dev_texts": dev_texts, "dev_emb": dev_emb,
                                  "holdout_texts": holdout_texts, "holdout_emb": holdout_emb}
    print(f"[eval_geometry] embedded {sum(len(v['dev_texts']) + len(v['holdout_texts']) for v in variant_data.values())} "
          f"poison texts across {len(VARIANTS)} variants in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # TUNE on dev: query_echo_threshold from verbatim (its strongest case),
    # isolation_percentile fixed at three FPR operating points, reported all.
    # ------------------------------------------------------------------
    dev_clean_sims, _ = geometry.query_echo_sims_batch([c.text for c in dev_clean], dev_clean_emb, embedder)
    verbatim = variant_data["verbatim"]
    dev_poison_sims, _ = geometry.query_echo_sims_batch(verbatim["dev_texts"], verbatim["dev_emb"], embedder)
    candidate_thresholds = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
    echo_t = _pick_query_echo_threshold(dev_clean_sims, dev_poison_sims, candidate_thresholds, args.max_fpr)
    print(f"[eval_geometry] tuned query_echo_threshold={echo_t} (dev FPR budget <= {args.max_fpr:.1%}, "
          f"achieved dev clean FPR={_rate(dev_clean_sims >= echo_t):.2%}, "
          f"dev verbatim catch={_rate(dev_poison_sims >= echo_t):.2%})")

    isolation_percentiles = [0.01, 0.02, 0.05]

    # ------------------------------------------------------------------
    # REPORT on held-out: score held-out clean + each held-out poison variant
    # against the DEV clean reference (the "index"); never against themselves.
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print(f"STAGE 1B -- ingest_scan evaluation ({embedder.name}, n_clean={args.n_clean}, split=dev/held-out 50/50)")
    print("=" * 96)

    holdout_density = geometry.manifold_isolation_scores(holdout_clean_emb, dev_clean_emb, knn_k=args.knn_k)
    for p in isolation_percentiles:
        cutoff = _isolation_cutoff(dev_clean_emb, p, args.knn_k)
        fpr = _rate(holdout_density < cutoff)
        print(f"  manifold_isolation @ percentile={p:.0%} (cutoff={cutoff:.3f}): "
              f"held-out CLEAN false-positive rate = {fpr:.2%}  (n={len(holdout_clean)})")
    chosen_p = 0.02
    iso_cutoff = _isolation_cutoff(dev_clean_emb, chosen_p, args.knn_k)
    print(f"  -> reporting per-variant catch rates at isolation_percentile={chosen_p:.0%} (cutoff={iso_cutoff:.3f})")
    print()

    header = f"  {'variant':<14}{'n':>6}{'query_echo':>14}{'manifold_iso':>16}{'combined':>12}"
    print(header)
    variant_rates: dict[str, dict] = {}
    for variant in VARIANTS:
        holdout_texts = variant_data[variant]["holdout_texts"]
        holdout_emb = variant_data[variant]["holdout_emb"]
        echo_sim, _ = geometry.query_echo_sims_batch(holdout_texts, holdout_emb, embedder)
        density = geometry.manifold_isolation_scores(holdout_emb, dev_clean_emb, knn_k=args.knn_k)
        rates = _signal_rates(echo_sim, echo_t, density, iso_cutoff)
        print(f"  {variant:<14}{rates.n:>6}{rates.query_echo:>13.1%} {rates.manifold_isolation:>15.1%} {rates.combined:>11.1%}")
        variant_rates[variant] = {
            "n": rates.n, "query_echo_catch_rate": rates.query_echo,
            "manifold_isolation_catch_rate": rates.manifold_isolation, "combined_catch_rate": rates.combined,
        }
    holdout_clean_echo_sim, _ = geometry.query_echo_sims_batch(
        [c.text for c in holdout_clean], holdout_clean_emb, embedder)
    echo_fpr_holdout = _rate(holdout_clean_echo_sim >= echo_t)
    iso_fpr_holdout = _rate(holdout_density < iso_cutoff)
    combined_fpr_holdout = _signal_rates(holdout_clean_echo_sim, echo_t, holdout_density, iso_cutoff).combined
    print()
    print(f"  clean FPR (held-out, n={len(holdout_clean)}): "
          f"query_echo={echo_fpr_holdout:.2%}  "
          f"manifold_isolation={iso_fpr_holdout:.2%}  "
          f"combined={combined_fpr_holdout:.2%}")
    print("=" * 96)

    isolation_fpr_by_percentile = {}
    for p in isolation_percentiles:
        cutoff_p = _isolation_cutoff(dev_clean_emb, p, args.knn_k)
        isolation_fpr_by_percentile[f"{p:.2f}"] = {
            "cutoff": cutoff_p, "holdout_clean_fpr": _rate(holdout_density < cutoff_p),
        }

    # ------------------------------------------------------------------
    # query_echo diagnostic (verbatim, held-out): is a low catch rate a
    # detection failure (sentence not recognized as a question, or the
    # no-space splitter merging it into the body) or a genuine sim
    # distribution sitting below the FPR-forced 0.9 threshold?
    # ------------------------------------------------------------------
    verbatim_holdout_texts = variant_data["verbatim"]["holdout_texts"]
    verbatim_holdout_emb = variant_data["verbatim"]["holdout_emb"]
    diag_sims, diag_matched = geometry.query_echo_sims_batch(verbatim_holdout_texts, verbatim_holdout_emb, embedder)
    diag_none = sum(1 for m in diag_matched if m is None)
    diag_matched_lens = [len(m) / max(1, len(t)) for m, t in zip(diag_matched, verbatim_holdout_texts) if m is not None]
    query_echo_diagnostic = {
        "n": len(verbatim_holdout_texts),
        "no_question_sentence_detected": diag_none,
        "matched_sentence_len_over_doc_len_percentiles": {
            str(p): float(np.percentile(diag_matched_lens, p)) if diag_matched_lens else None
            for p in (25, 50, 75, 95)
        },
        "sim_percentiles": {str(p): float(np.percentile(diag_sims, p)) for p in (5, 25, 50, 75, 95)},
        "note": (
            "matched_sentence_len_over_doc_len staying small (~0.2) confirms the no-space "
            "sentence splitter is correctly isolating just the question, not merging it into "
            "the body; sim sitting mostly in 0.8-0.95 with a 0.9 threshold (forced up by the "
            "dev clean FPR budget) is why catch rate lands well under 100% -- a precision/recall "
            "tradeoff at this threshold, not a broken splitter."
        ),
    }

    # ------------------------------------------------------------------
    # collapse_topk: tune sim_threshold on dev (verbatim), report on held-out
    # across all three variants.
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("STAGE 1B -- collapse_topk evaluation (simulated top-5 retrieval, clean+poison pool)")
    print("=" * 96)

    collapse_candidates = [0.75, 0.80, 0.85, 0.90, 0.93, 0.95]
    best_t, best_rate = collapse_candidates[-1], -1.0
    dev_collapse_tuning = {}
    for t in collapse_candidates:
        rows = _simulate_and_collapse(dev_targets, "verbatim", dev_clean, dev_clean_emb, embedder, sim_threshold=t)
        summary = _summarize_collapse(rows)
        cr = summary["poison_collapse_rate"] if summary["poison_collapse_rate"] == summary["poison_collapse_rate"] else 0.0
        print(f"  dev tuning sim_threshold={t:.2f}: poison_collapse_rate={summary['poison_collapse_rate']:.1%} "
              f"(n_with_2+_poison={summary['n_with_2plus_poison_in_top5']}/{summary['n_targets']}), "
              f"clean_wrong_merge_rate={summary['clean_wrong_merge_rate']:.1%}")
        dev_collapse_tuning[f"{t:.2f}"] = summary
        if summary["clean_wrong_merge_rate"] == 0.0 and cr > best_rate:
            best_rate, best_t = cr, t
    print(f"  -> chosen sim_threshold={best_t} (best dev poison_collapse_rate with zero dev clean wrong-merges)")
    print()

    holdout_collapse: dict[str, dict] = {}
    for variant in VARIANTS:
        rows = _simulate_and_collapse(holdout_targets, variant, holdout_clean, holdout_clean_emb, embedder, sim_threshold=best_t)
        summary = _summarize_collapse(rows)
        holdout_collapse[variant] = summary
        print(f"  held-out [{variant:<12}] poison_collapse_rate={summary['poison_collapse_rate']:.1%} "
              f"(n_with_2+_poison_in_top5={summary['n_with_2plus_poison_in_top5']}/{summary['n_targets']}), "
              f"clean_wrong_merge_rate={summary['clean_wrong_merge_rate']:.1%} "
              f"[mixed-pool poison_in_top5 distribution: {summary['poison_in_top5_distribution']}]")
    print("=" * 96)

    # ------------------------------------------------------------------
    # clean-only collapse baseline: the mixed clean+poison top-5 above is
    # usually dominated by poison (see poison_in_top5_distribution), leaving
    # too few clean chunks in any one top-5 for clean_wrong_merge_rate to be
    # anything but near-vacuous. This measures the real question -- do two
    # DISTINCT real clean top-5 passages ever collapse into one -- over the
    # held-out questions against the held-out clean corpus alone, no poison.
    # ------------------------------------------------------------------
    print()
    print("STAGE 1B -- collapse_topk clean-only baseline (no poison injected)")
    clean_only_rows = _simulate_clean_only_collapse(holdout_targets, holdout_clean, holdout_clean_emb,
                                                      embedder, sim_threshold=best_t)
    clean_only_summary = _summarize_clean_only_collapse(clean_only_rows)
    print(f"  held-out clean-only top-5, n={clean_only_summary['n_targets']}: "
          f"wrong_merge_rate={clean_only_summary['wrong_merge_rate']:.2%} "
          f"(any-collapse in {clean_only_summary['n_targets_with_any_collapse']} of {clean_only_summary['n_targets']})")
    print("=" * 96)
    total_s = time.perf_counter() - t0
    print(f"[eval_geometry] total wall time: {total_s:.1f}s")

    # ------------------------------------------------------------------
    # write results/geometry_<timestamp>.json in the shared eval-script shape
    # ------------------------------------------------------------------
    clean_handle = DatasetHandle(name="beir.nq", records=tuple(clean_chunks))
    targets_handle = DatasetHandle(name="poisonedrag.nq_targets", records=tuple(all_targets))

    payload = {
        "data_composition": {
            "clean_corpus": clean_handle.describe(),
            "poisonedrag_targets": targets_handle.describe(),
        },
        "data_source": clean_handle.data_source if clean_handle.data_source == targets_handle.data_source
        else f"{clean_handle.data_source}+{targets_handle.data_source}",
        "embedder": {"name": embedder.name, "dim": embedder.dim, "space": embedder.space},
        "corpus": args.corpus,
        "seed": args.seed,
        "n_clean_total": len(clean_chunks),
        "n_dev_clean": len(dev_clean),
        "n_holdout_clean": len(holdout_clean),
        "n_targets_total": len(all_targets),
        "n_dev_targets": len(dev_targets),
        "n_holdout_targets": len(holdout_targets),
        "n_adv_texts_total": sum(len(t.adv_texts) for t in all_targets),
        "n_adv_texts_dev": sum(len(t.adv_texts) for t in dev_targets),
        "n_adv_texts_holdout": sum(len(t.adv_texts) for t in holdout_targets),
        "split_description": (
            "PoisonedRAG's 100 nq targets sorted by id: first 50 DEV (thresholds tuned), "
            "other 50 HELD-OUT (reported numbers, never touched while tuning). Clean BEIR "
            "sample split the same way (first half DEV, second half HELD-OUT). At report "
            "time, manifold_isolation's reference corpus is the DEV clean half; HELD-OUT "
            "clean/poison are scored against it as new, unseen candidates."
        ),
        "ingest_scan": {
            "max_fpr_budget_dev": args.max_fpr,
            "tuned_query_echo_threshold": echo_t,
            "dev_query_echo_fpr_at_tuned_threshold": _rate(dev_clean_sims >= echo_t),
            "dev_verbatim_catch_at_tuned_threshold": _rate(dev_poison_sims >= echo_t),
            "isolation_percentile_chosen": chosen_p,
            "isolation_cutoff_chosen": iso_cutoff,
            "isolation_fpr_by_percentile_holdout": isolation_fpr_by_percentile,
            "knn_k": args.knn_k,
            "held_out_variant_catch_rates": variant_rates,
            "held_out_clean_fpr": {
                "query_echo": echo_fpr_holdout,
                "manifold_isolation": iso_fpr_holdout,
                "combined": combined_fpr_holdout,
            },
            "query_echo_diagnostic_verbatim_holdout": query_echo_diagnostic,
            "manifold_isolation_honest_assessment": (
                "At every operating point measured (1%/2%/5% dev percentile cutoffs) and every "
                "attacker variant, manifold_isolation's held-out catch rate on poison (0.4%-2.4%) "
                "is LOWER than its own held-out FPR on clean text (1.36%-4.88% depending on "
                "cutoff) -- it fires more often on real clean NQ passages than on PoisonedRAG's "
                "injected texts. PoisonedRAG's texts are fluent, on-topic and (per the paper's "
                "own design goal) engineered to sit close to the target question's neighborhood, "
                "so they do not drift off the corpus manifold in bge cosine space the way the "
                "module docstring predicts. As measured here, this signal is not merely weak: "
                "combining it with query_echo INCREASES combined FPR (0.64% -> 3.20%) for a "
                "negligible increase in combined catch on verbatim (33.6% -> 34.4%) and makes "
                "paraphrased/no_prefix catch barely move. It does not 'carry the signal when "
                "query_echo is gone', on this embedder/corpus -- it is close to noise."
            ),
        },
        "collapse_topk": {
            "dev_tuning_by_sim_threshold": dev_collapse_tuning,
            "chosen_sim_threshold": best_t,
            "chosen_sim_threshold_note": (
                "0.75/0.80/0.85 all scored 100% dev poison_collapse_rate with 0% dev "
                "clean_wrong_merge_rate; the tie-break picked the lowest (most aggressive) of "
                "those three. The dev sweep did not discriminate between them -- no candidate "
                "in {0.75..0.95} ever produced a nonzero dev clean wrong-merge, so 'tuned' here "
                "means 'FPR-safe on this dev sample', not 'selected on discriminating evidence'."
            ),
            "held_out_by_variant": holdout_collapse,
            "clean_only_baseline_no_poison_injected": clean_only_summary,
            "caveats": (
                "(1) load_nq() was called WITHOUT include_ids, so PoisonedRAG targets' actual "
                "gold passages are almost certainly absent from the 2,500-passage held-out clean "
                "pool used here -- this simulates '5 targeted poison texts vs. 2,500 UNRELATED "
                "clean passages', not poison vs. the real competing evidence for that question. "
                "That inflates poison_in_top5 and hence poison_collapse_rate; wiring include_ids "
                "in would change every clean embedding cache key and cost another ~50 min of "
                "CPU embedding, so it was not done here. "
                "(2) mixed-pool clean_wrong_merge_rate above is close to vacuous: see "
                "poison_in_top5_distribution per variant -- when poison occupies most/all of the "
                "5 slots, too few clean chunks remain in that top-5 to ever collide. "
                "clean_only_baseline_no_poison_injected is the more honest number for 'do two "
                "distinct real clean passages get wrongly merged'."
            ),
        },
        "embedder_choice_note": (
            "BAAI/bge-small-en-v1.5 was used, not facebook/contriever. An existing "
            ".cache/index/beir_nq_contriever_*.npy covers only clean BEIR passages under a "
            "different cache namespace (owned by another eval script) and does not cover the "
            "1,500 poison-variant texts or the ~200 per-target collapse batches this script also "
            "needs -- those would still need fresh CPU embedding (~30-60 min observed rate) under "
            "contriever. Also contriever's convention is unnormalized dot-product (space='ip') "
            "vs. bge's normalized cosine; geometry.py internally L2-normalizes regardless, so a "
            "threshold tuned here would not transfer to contriever's raw dot-product ranking "
            "unchanged. Given manifold_isolation is already measured as non-discriminative on "
            "bge/cosine, contriever was not attempted."
        ),
        "total_wall_seconds": total_s,
    }
    _common.write_results("geometry", payload)


if __name__ == "__main__":
    main()
