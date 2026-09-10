"""Stage 1A evaluation on BIPIA (Microsoft's indirect-prompt-injection
benchmark) -- the FIRST measured result against a benchmark ``directive.py``
was never built or thresholded against. Stage 1A's structural signals and
THRESHOLD were tuned only on LLMail-Inject phase 1; this script does not
touch the detector, it only measures what falls out on a different research
group's attack style.

    python -m triad.eval.bipia [--seed 7] [--n-enron 300] [--probe-baseline]

HELD OUT BY DESIGN -- READ THIS BEFORE TOUCHING A NUMBER BELOW: nothing here
adjusts THRESHOLD, signal weights, or adds a new signal. A low number is the
result, not a bug. Tuning against this script's own output would defeat the
reason it exists (see the task brief this script was written to satisfy).

=== How an "attack" is built ================================================
BIPIA does not ship pre-poisoned documents. Each of the three usable tasks
(email n=50, table n=100, code n=50 -- see ``triad/data/bipia.py``, which
also documents the two tasks that are NOT usable here) ships CLEAN `context`
records plus a SEPARATE pool of attack-instruction strings
(``bipia.load_attack_pool``). BIPIA's own protocol is to insert one attack
string into one clean context at one of three positions ("start", "end",
"middle") and treat the result as one attack trial. We follow that same
protocol -- every (context record, attack text, position) triple becomes one
attacked document, scanned once -- but we reimplement the three insertion
helpers ourselves instead of importing BIPIA's own ``bipia.data.utils``
(vendored under ``data/raw/BIPIA``), which requires ``nltk`` + ``transformers``
purely to pick a random *sentence* boundary for "middle"; neither package is
in this repo's venv, and pulling them in (nltk additionally needs a
``punkt`` tokenizer data download) would add a heavy, network-dependent
detector-adjacent dependency to answer a data-construction question. Our
``_insert_middle`` instead splits at the nearest word boundary to the
character midpoint -- deterministic, no new dependency, same intent (the
attack text lands inside the body, not just bracketing it). This is
faithful-in-spirit to BIPIA's construction, not byte-identical to it.

=== What "attack" means per task, and what our number does NOT claim =======
- email / table: the attack pool (``load_attack_pool(kind="text")``) is
  generic natural-language injected instructions (task automation, scams,
  cipher/encoding tricks, ...) -- these map directly onto what
  ``directive.scan()`` looks for: an instruction embedded in retrieved
  content. Directly comparable in kind to LLMail-Inject's attack texts.
- code: the attack pool (``load_attack_pool(kind="code")``) is malicious
  CODE SNIPPETS an attacker wants inserted into the user's own buggy code --
  framed as an instruction ("add/ensure/merge the following snippet"), so it
  is still in-scope for a scanner built to catch "do X" -- but the payload
  register (Python/shell source, not business prose) is genuinely different
  from anything LLMail-Inject or the detector's calibration corpus contains.
  Reported separately from email/table for exactly this reason; a low code
  number would say something different (payload-register mismatch) than a
  low email number would (a genuine detection miss).
- IMPORTANT SCOPE NOTE: BIPIA's own published metric is Attack Success Rate
  (ASR) -- whether a downstream LLM, having been SHOWN the poisoned context,
  actually complies with the injected instruction in its answer. Stage 1A is
  an upstream, pre-LLM scanner: it never sees a downstream answer, it only
  decides allow/block on the retrieved text itself. The "detection rate"
  this script reports is therefore NOT a BIPIA ASR number and is not
  directly comparable to BIPIA paper results -- it is the same kind of
  number ``triad.eval.injection`` already reports for LLMail-Inject: the
  fraction of attacked documents Stage 1A blocks before an LLM ever sees
  them. Comparable across BIPIA/LLMail within this repo; not comparable to
  BIPIA's own leaderboard.

=== False positives ==========================================================
FPR is measured the same way ``triad.eval.injection`` measures it -- on the
LLMail-Inject benign set and a held-out Enron sample -- reusing exactly that
FPR methodology so a false positive rate on real business content means the
same thing here as it does there. ``--seed`` must differ from
``triad.eval.injection``'s default seed=2 to keep this a genuinely separate
held-out draw (default here is 7; passing --seed 2 prints a warning but does
not refuse, in case a future run has a deliberate reason to reuse it).
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter

from triad.eval import _common

from triad.data import bipia
from triad.data.registry import get_dataset
from triad.stage1.directive import THRESHOLD, scan

BASELINE_MODEL = "meta-llama/llama-prompt-guard-2-86m"
BASELINE_THRESHOLD = 0.5
BASELINE_MAX_CHARS = 1500  # same limit triad.eval.injection measured (86M model's window is small)

# The full (context x attack x position) cross-product is what the Stage 1A
# DETECTOR is scored against below (tens of thousands of pure-regex scans,
# no LLM involved, cheap). The optional Groq baseline probe is NOT run
# against that full grid -- that would be tens of thousands of live API
# calls -- it runs against a bounded random subsample per task instead. See
# `_subsample` and the --baseline-sample-per-task flag.
DEFAULT_BASELINE_SAMPLE_PER_TASK = 25

_ATTACK_POOL_KIND = {"email": "text", "table": "text", "code": "code"}


# --------------------------------------------------------------------------- construction

def _insert_end(context: str, attack: str) -> str:
    return f"{context}\n\n{attack}"


def _insert_start(context: str, attack: str) -> str:
    return f"{attack}\n\n{context}"


def _insert_middle(context: str, attack: str) -> str:
    """Deterministic stand-in for BIPIA's random-sentence-boundary insertion
    (see module docstring): split at the nearest word boundary to the
    character midpoint."""
    mid = len(context) // 2
    idx = context.find(" ", mid)
    if idx == -1:
        idx = mid
    return f"{context[:idx]}\n\n{attack}\n\n{context[idx:]}"


_POSITIONS = (("start", _insert_start), ("end", _insert_end), ("middle", _insert_middle))


def _flatten_attack_pool(pool: dict[str, tuple[str, ...]]) -> list[tuple[str, str]]:
    """``[(category, attack_text), ...]`` in a stable (sorted-by-category)
    order, so repeated runs build the identical grid."""
    out: list[tuple[str, str]] = []
    for category in sorted(pool):
        for text in pool[category]:
            out.append((category, text))
    return out


def _build_attacked_samples(records, attack_pool_flat: list[tuple[str, str]]):
    """Yields ``(combined_text, category, position)`` for every (context
    record, attack, position) triple -- one attack trial each, matching
    BIPIA's own construction protocol (see module docstring)."""
    for rec in records:
        for category, attack_text in attack_pool_flat:
            for position, insert_fn in _POSITIONS:
                yield insert_fn(rec.context, attack_text), category, position


# --------------------------------------------------------------------------- scanning

def _run_scan_batch(texts: list[str]) -> tuple[int, Counter, list[float]]:
    """Same shape as ``triad.eval.injection._run_scan_batch`` -- kept as a
    local copy rather than an import so this module has no dependency on
    another eval script's private helpers (other agents are working in
    ``triad/eval/``)."""
    blocked = 0
    signals: Counter = Counter()
    scores: list[float] = []
    for text in texts:
        decision = scan(text)
        if not decision.allow:
            blocked += 1
        for sig in decision.evidence.get("signals", ()):
            signals[sig] += 1
        scores.append(float(decision.evidence.get("score", 0.0)))
    return blocked, signals, scores


def _rate(numerator: int, denominator: int, label: str) -> float:
    """A rate with a guard: an empty/degenerate denominator raises loudly
    (this repo was burned this week by a rate whose denominator was silently
    empty) instead of writing a misleading number to the results JSON."""
    if denominator <= 0:
        raise RuntimeError(f"{label}: denominator is {denominator} -- refusing to report a vacuous rate")
    return numerator / denominator


def _subsample(items: list, n: int, seed: int) -> list:
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(items)), min(n, len(items))))
    return [items[i] for i in idx]


# --------------------------------------------------------------------------- baseline probe (optional)

def _probe_baseline(texts: list[str], principal_prefix: str):
    """One Groq call per text against the baseline classifier -- same
    contract as ``triad.eval.injection._probe_baseline``: stops (does not
    retry in a loop) on AllKeysExhausted, so a swallowed rate-limit failure
    here can never silently understate the baseline."""
    from triad.llm.cache import DiskCache
    from triad.llm.client import AllKeysExhausted, EmptyResponse, GroqClient
    from triad.llm.keys import load_keys
    from triad.llm.limiter import RateLimiter

    keys = load_keys()
    client = GroqClient(keys=keys, limiter=RateLimiter(), cache=DiskCache())
    cache_stats = _common.CacheStats()
    scores: list[float | None] = []
    errors = 0
    exhausted = False
    try:
        for i, text in enumerate(texts):
            try:
                resp = client.chat(
                    [{"role": "user", "content": text[:BASELINE_MAX_CHARS]}],
                    principal=f"{principal_prefix}:{i}", model=BASELINE_MODEL, max_tokens=16,
                )
            except AllKeysExhausted as exc:
                print(f"AllKeysExhausted probing baseline classifier at {i}/{len(texts)}: {exc}", file=sys.stderr)
                exhausted = True
                break
            except EmptyResponse:
                errors += 1
                scores.append(None)
                continue
            except RuntimeError as exc:
                errors += 1
                scores.append(None)
                print(f"  baseline probe error at {i}/{len(texts)} (recorded as unscored): {exc}", file=sys.stderr)
                continue
            cache_stats.record(resp.cached)
            try:
                scores.append(float(resp.text.strip()))
            except ValueError:
                errors += 1
                scores.append(None)
    finally:
        client.close()
    return scores, errors, exhausted, cache_stats


def _baseline_rate(scores: list[float | None]) -> float:
    usable = [s for s in scores if s is not None]
    if not usable:
        return float("nan")
    return sum(1 for s in usable if s >= BASELINE_THRESHOLD) / len(usable)


# --------------------------------------------------------------------------- main

def main() -> None:
    _common.require_real()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=7,
                         help="held-out Enron sample seed / baseline subsample seed (default 7; "
                              "must differ from triad.eval.injection's seed=2 to stay a separate held-out draw)")
    parser.add_argument("--n-enron", type=int, default=300)
    parser.add_argument("--probe-baseline", action="store_true",
                         help="also run the Groq Prompt Guard 2 baseline on a BOUNDED subsample (live calls)")
    parser.add_argument("--baseline-sample-per-task", type=int, default=DEFAULT_BASELINE_SAMPLE_PER_TASK,
                         help="attacked documents per task sent to the baseline probe (NOT the full grid -- "
                              "see module docstring for why)")
    args = parser.parse_args()

    if args.seed == 2:
        print("WARNING: --seed 2 collides with triad.eval.injection's Enron sample seed; "
              "this run's Enron FPR sample will no longer be a genuinely independent held-out draw.",
              file=sys.stderr)

    per_task_payload: dict[str, dict] = {}
    baseline_texts_by_task: dict[str, list[str]] = {}

    for task in bipia.USABLE_TASKS:
        print(f"Loading BIPIA {task!r} contexts + attack pool (first use of this task in this repo)...")
        ctx_handle = get_dataset(f"bipia.{task}")
        pool_kind = _ATTACK_POOL_KIND[task]
        pool = bipia.load_attack_pool(kind=pool_kind, split="test")
        flat_pool = _flatten_attack_pool(pool)

        combined = list(_build_attacked_samples(ctx_handle.records, flat_pool))
        n = len(combined)
        if n == 0:
            raise RuntimeError(f"bipia.{task}: zero attacked samples constructed -- refusing to report a vacuous rate")

        blocked = 0
        signals: Counter = Counter()
        position_n: Counter = Counter()
        position_blocked: Counter = Counter()
        for text, category, position in combined:
            decision = scan(text)
            position_n[position] += 1
            if not decision.allow:
                blocked += 1
                position_blocked[position] += 1
            for sig in decision.evidence.get("signals", ()):
                signals[sig] += 1

        detection_rate = _rate(blocked, n, f"bipia.{task} detection_rate")
        by_position = {
            pos: {"rate": _rate(position_blocked[pos], position_n[pos], f"bipia.{task}/{pos}"),
                  "blocked": position_blocked[pos], "n": position_n[pos]}
            for pos in position_n
        }

        print(f"BIPIA {task}: detection rate {detection_rate:.1%}  ({blocked}/{n})")
        print(f"  by position: " + ", ".join(f"{p}={v['rate']:.1%} ({v['blocked']}/{v['n']})" for p, v in by_position.items()))
        print(f"  signals fired: {signals.most_common()}")

        per_task_payload[task] = {
            "n_context_records": len(ctx_handle.records),
            "context_data_composition": ctx_handle.describe(),
            "attack_pool_kind": pool_kind,
            "attack_pool_categories": sorted(pool),
            "n_attack_pool_texts": len(flat_pool),
            "positions": [p for p, _ in _POSITIONS],
            "n_attacked_documents": n,
            "blocked": blocked,
            "detection_rate": detection_rate,
            "detection_rate_by_position": by_position,
            "signals_on_attacks": dict(signals),
        }

        if args.probe_baseline:
            baseline_texts_by_task[task] = [t for t, _cat, _pos in _subsample(combined, args.baseline_sample_per_task, args.seed)]
        del combined  # bound memory before moving to the next (potentially 22.5k-row) task

    print()
    print("Loading LLMail-Inject benign set + held-out Enron sample for false positives...")
    benign_handle = get_dataset("llmail.benign")
    benign_texts = [c.text for c in benign_handle.records]

    enron_handle = get_dataset("enronqa.emails")
    enron_sample = _subsample(list(enron_handle.records), args.n_enron, args.seed)
    enron_texts = [c.text for c in enron_sample]

    n_benign, n_enron = len(benign_texts), len(enron_texts)
    if n_benign == 0 or n_enron == 0:
        raise RuntimeError(f"FPR denominator empty (benign={n_benign}, enron={n_enron}) -- refusing to report a vacuous rate")

    benign_blocked, benign_signals, _ = _run_scan_batch(benign_texts)
    enron_blocked, enron_signals, _ = _run_scan_batch(enron_texts)
    fpr_llmail = _rate(benign_blocked, n_benign, "fpr_llmail_benign")
    fpr_enron = _rate(enron_blocked, n_enron, "fpr_enron")

    print(f"FPR on LLMail-Inject benign, n={n_benign}: {fpr_llmail:.1%}  ({benign_blocked}/{n_benign})")
    print(f"  signals fired: {benign_signals.most_common()}")
    print(f"FPR on held-out Enron sample (seed={args.seed}), n={n_enron}: {fpr_enron:.1%}  ({enron_blocked}/{n_enron})")
    print(f"  signals fired: {enron_signals.most_common()}")

    baseline_payload: dict = {"probed": False}
    if args.probe_baseline:
        print()
        print(f"Probing baseline classifier {BASELINE_MODEL} on a BOUNDED per-task subsample "
              f"(n={args.baseline_sample_per_task}/task, NOT the full grid) plus the FPR sets...")
        per_task_baseline: dict[str, dict] = {}
        cache_totals = _common.CacheStats()
        any_exhausted = False
        for task, texts in baseline_texts_by_task.items():
            scores, errors, exhausted, cs = _probe_baseline(texts, f"bipia-eval:baseline:{task}")
            cache_totals.hits += cs.hits
            cache_totals.live += cs.live
            any_exhausted = any_exhausted or exhausted
            rate = _baseline_rate(scores)
            n_scored = sum(1 for s in scores if s is not None)
            per_task_baseline[task] = {
                "detection_rate": rate, "n_sampled": len(texts), "n_scored": n_scored, "errors": errors,
            }
            print(f"  {task}: baseline detection rate {rate:.1%} (n scored={n_scored}/{len(texts)})")
            if exhausted:
                break

        if not any_exhausted:
            b_benign_scores, b_benign_err, exh_b, cs_b = _probe_baseline(benign_texts, "bipia-eval:baseline:benign")
            cache_totals.hits += cs_b.hits
            cache_totals.live += cs_b.live
            any_exhausted = any_exhausted or exh_b
        else:
            b_benign_scores, b_benign_err = [], 0
        if not any_exhausted:
            b_enron_scores, b_enron_err, exh_e, cs_e = _probe_baseline(enron_texts, "bipia-eval:baseline:enron")
            cache_totals.hits += cs_e.hits
            cache_totals.live += cs_e.live
            any_exhausted = any_exhausted or exh_e
        else:
            b_enron_scores, b_enron_err = [], 0

        b_fpr_llmail = _baseline_rate(b_benign_scores)
        b_fpr_enron = _baseline_rate(b_enron_scores)
        print(f"  baseline FPR on LLMail benign: {b_fpr_llmail:.1%}")
        print(f"  baseline FPR on Enron sample: {b_fpr_enron:.1%}")
        if any_exhausted:
            print("  WARNING: stopped early on AllKeysExhausted -- baseline numbers above are PARTIAL", file=sys.stderr)

        baseline_payload = {
            "probed": True, "model": BASELINE_MODEL, "threshold": BASELINE_THRESHOLD,
            "sampling_note": "per-task subsample of the full (context x attack x position) grid, NOT the full grid -- see module docstring",
            "baseline_sample_per_task": args.baseline_sample_per_task,
            "per_task": per_task_baseline,
            "fpr_llmail_benign": b_fpr_llmail, "fpr_enron": b_fpr_enron,
            "errors": {"benign": b_benign_err, "enron": b_enron_err},
            "stopped_on_all_keys_exhausted": bool(any_exhausted),
            "cache_stats": cache_totals.as_dict(),
        }

    payload = {
        "threshold": THRESHOLD,
        "seed": args.seed,
        "n_enron": n_enron,
        "tasks": per_task_payload,
        "false_positive_rate": {
            "llmail_benign": {"rate": fpr_llmail, "blocked": benign_blocked, "n": n_benign,
                               "data_composition": benign_handle.describe(),
                               "signals": dict(benign_signals)},
            "enron_holdout": {"rate": fpr_enron, "blocked": enron_blocked, "n": n_enron,
                               "data_composition": f"enronqa.emails: {n_enron:,} real (held-out sample, seed={args.seed})",
                               "signals": dict(enron_signals)},
        },
        "baseline_classifier": baseline_payload,
        "scope_note": (
            "detection_rate here is 'Stage 1A blocked the attacked document before an LLM ever saw it', "
            "NOT BIPIA's own published Attack Success Rate (which measures downstream LLM compliance). "
            "See module docstring 'What attack means per task' section."
        ),
    }
    _common.write_results("bipia", payload)


if __name__ == "__main__":
    main()
