"""Prints the four-number table from the latest results JSONs.

    python -m triad.eval.report

Reads whatever ``triad.eval.tenant_leak``, ``triad.eval.injection``,
``triad.eval.poisonedrag`` and ``triad.eval.ablation`` most recently wrote to
``results/`` -- never recomputes anything itself, so this is always exactly
what those scripts measured, with ``n`` printed next to every number so a
smoke run (``--n 10``) can never be mistaken for the headline run.
"""

from __future__ import annotations

import re
import sys

from triad.eval import _common


def _fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and x != x):  # NaN
        return "n/a"
    return f"{x:.1%}"


def _fmt_ms(x) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x:.2f}ms"


def _latest_poisonedrag(results_dir):
    """Prefers the largest-n poisonedrag run available (the headline), falling
    back to whatever smaller smoke run exists."""
    if not results_dir.exists():
        return None
    candidates = []
    for p in results_dir.glob("poisonedrag_n*_*.json"):
        m = re.match(r"poisonedrag_n(\d+)_", p.name)
        if m:
            candidates.append((int(m.group(1)), p.name, p))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))  # biggest n, then latest timestamp within that n
    best_n = candidates[-1][0]
    same_n = [c for c in candidates if c[0] == best_n]
    same_n.sort(key=lambda t: t[1])
    import json
    return json.loads(same_n[-1][2].read_text(encoding="utf-8"))


def _latest_ablation(results_dir):
    if not results_dir.exists():
        return None
    candidates = sorted(results_dir.glob("ablation_n*_*.json"))
    if not candidates:
        return None
    import json
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def main() -> None:
    results_dir = _common.RESULTS_DIR
    tenant_leak = _common.latest_result("tenant_leak", results_dir=results_dir)
    injection = _common.latest_result("injection", results_dir=results_dir)
    poisonedrag = _latest_poisonedrag(results_dir)
    ablation = _latest_ablation(results_dir)

    print("=" * 78)
    print("TRIAD-RAG -- measured results (latest run of each script)")
    print("=" * 78)

    print()
    print("-- ASR before -> after (PoisonedRAG replay on BEIR NQ) --")
    if poisonedrag:
        asr = poisonedrag["asr"]
        n = poisonedrag["n_targets"]
        print(f"  n={n}  ASR off={_fmt_pct(asr['off'])}  ->  on={_fmt_pct(asr['on'])}  (delta {asr['delta']:+.1%})")
        print(f"  poison caught at ingestion: {poisonedrag['ingestion']['poison_quarantined']}/{poisonedrag['ingestion']['poison_submitted']}")
        print(f"  data: {poisonedrag['data_composition']['targets']}")
        print(f"  corpus: {poisonedrag['data_composition']['clean_corpus']} (poison ratio {poisonedrag['poison_ratio']:.4%})")
    else:
        print("  no poisonedrag results yet -- run `python -m triad.eval.poisonedrag --n 10` (smoke) then --n 100")

    print()
    print("-- Clean accuracy retained --")
    if poisonedrag:
        ca = poisonedrag["clean_accuracy"]
        print(f"  n={poisonedrag['n_targets']}  clean accuracy off={_fmt_pct(ca['off'])}  on={_fmt_pct(ca['on'])}")
    else:
        print("  (see poisonedrag results above)")

    print()
    print("-- Added latency per stage --")
    if poisonedrag:
        lat = poisonedrag["latency_ms"]
        print(f"  Stage 2 retrieval (single-tenant BEIR corpus): off mean={_fmt_ms(lat['stage2_retrieval_off_mean'])}  "
              f"on mean={_fmt_ms(lat['stage2_retrieval_on_mean'])}")
        print(f"  Stage 1 ingest, clean corpus (once, n={poisonedrag['clean_corpus_size']}): {_fmt_ms(lat['stage1_ingest_clean_corpus_once'])}")
        print(f"  Stage 1 ingest, poison batch: {_fmt_ms(lat['stage1_ingest_poison_batch'])} total, "
              f"{_fmt_ms(lat['stage1_ingest_poison_batch_per_chunk_mean'])}/chunk")
    if tenant_leak:
        print(f"  Stage 2 retrieval (real multi-tenant EnronQA, n_probes={tenant_leak['n_probes']}): "
              f"leaky p50={_fmt_ms(tenant_leak['leaky_retriever']['latency_ms_p50'])}  "
              f"secure p50={_fmt_ms(tenant_leak['secure_retriever']['latency_ms_p50'])}  "
              f"added={_fmt_ms(tenant_leak['added_latency_ms_p50_secure_minus_leaky'])}")
    if not poisonedrag and not tenant_leak:
        print("  no results yet")

    print()
    print("-- FPR (false positive rate) --")
    if injection:
        d = injection["directive"]
        print(f"  Stage 1A on LLMail-Inject PHASE 2 (n={injection['phase2_attacks_unique_after_dedupe']} unique, "
              f"{injection['phase2_attacks_raw_all_objectives']} raw): detection={_fmt_pct(d['detection_rate_phase2'])}")
        print(f"  FPR on LLMail-Inject benign set: {_fmt_pct(d['fpr_llmail_benign'])}")
        print(f"  FPR on held-out Enron sample (n={injection['n_enron']}, seed={injection['seed']}): {_fmt_pct(d['fpr_enron'])}")
        bc = injection.get("baseline_classifier", {})
        if bc.get("probed"):
            print(f"  baseline ({bc['model']}) on the same sets: detection={_fmt_pct(bc['detection_rate_phase2'])}  "
                  f"FPR(llmail)={_fmt_pct(bc['fpr_llmail_benign'])}  FPR(enron)={_fmt_pct(bc['fpr_enron'])}")
    if poisonedrag:
        ing = poisonedrag["ingestion"]
        if ing["clean_submitted"]:
            fpr_clean_corpus = ing["clean_quarantined"] / ing["clean_submitted"]
            print(f"  Stage 1 (1a+1b) FPR on BEIR NQ clean background corpus (n={ing['clean_submitted']}): {_fmt_pct(fpr_clean_corpus)}")
    if tenant_leak:
        print(f"  Stage 2 cross-tenant leak rate: leaky={_fmt_pct(tenant_leak['leaky_retriever']['leak_rate_any_foreign_chunk'])}  "
              f"secure={_fmt_pct(tenant_leak['secure_retriever']['leak_rate_any_foreign_chunk'])}  "
              f"(n_probes={tenant_leak['n_probes']})")
    if not injection and not poisonedrag and not tenant_leak:
        print("  no results yet")

    print()
    print("-- Ablation (each defense component toggled individually) --")
    if ablation:
        for row_name, r in ablation["rows"].items():
            print(f"  {row_name:24s} ASR={_fmt_pct(r['asr'])}  poison_quarantined={r['poison_quarantined']}/{r['poison_submitted']}")
    else:
        print("  no ablation results yet -- run `python -m triad.eval.ablation --n 10`")

    print()
    print("=" * 78)
    if not any([tenant_leak, injection, poisonedrag, ablation]):
        print("No results found in results/. Run the eval scripts first.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
