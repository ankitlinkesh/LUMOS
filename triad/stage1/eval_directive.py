"""Measure Stage 1A against the real corpora.

    python -m triad.stage1.eval_directive --n 500 --seed 0

Reports:
  - detection rate on a sample of LLMail-Inject PHASE 1 submissions that met
    ALL FIVE objectives (these are the attacks that beat Microsoft's deployed
    defenses -- the hardest real examples available).
  - false-positive rate on the 203 curated benign emails shipped with
    LLMail-Inject for exactly this purpose.
  - false-positive rate on a random sample of real Enron emails.
  - per-signal hit counts, so it's clear which structural signals are doing
    the work.

PHASE 1 ONLY. Phase 2 of LLMail-Inject is reserved for final reported numbers
and is never opened by this script or during tuning -- see the module
docstring in ``directive.py`` and the task brief for why.

This file writes its own tiny streaming readers rather than importing a
shared loader: other agents are building shared data-loading code in this
repo concurrently, and depending on it here would create a race.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from triad.config import DATA_RAW
from triad.stage1.directive import scan

PHASE1_PATH = DATA_RAW / "llmail_inject" / "data" / "raw_submissions_phase1.jsonl"
FP_EMAILS_PATH = DATA_RAW / "llmail_inject" / "data" / "emails_for_fp_tests.json"
ENRON_PARQUET_PATH = DATA_RAW / "enronqa" / "data" / "test-00000-of-00001.parquet"

# Deliberately never touched:
PHASE2_PATH = DATA_RAW / "llmail_inject" / "data" / "raw_submissions_phase2.jsonl"


def _met_all_objectives(objectives_field: str) -> bool:
    try:
        obj = json.loads(objectives_field)
    except (TypeError, ValueError):
        return False
    if not obj:
        return False
    return all(bool(v) for v in obj.values())


def iter_phase1_all_objectives_attacks(path: Path = PHASE1_PATH):
    """Stream raw_submissions_phase1.jsonl line by line (it's ~370k lines,
    too large to load whole) and yield the attack text (subject + body) for
    every submission that met all five LLMail-Inject objectives."""

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _met_all_objectives(rec.get("objectives", "")):
                continue
            subject = rec.get("subject") or ""
            body = rec.get("body") or ""
            yield f"{subject}\n\n{body}"


def load_fp_benign_emails(path: Path = FP_EMAILS_PATH) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data)


def sample_enron_emails(path: Path = ENRON_PARQUET_PATH, n: int = 500, seed: int = 0) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["email"])
    col = table.column("email")
    total = len(col)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(total), min(n, total)))
    return [col[i].as_py() for i in idx]


def _reservoir_sample(iterable, n: int, seed: int) -> list[str]:
    """Reservoir-sample n items from a streaming iterable without materializing it."""
    rng = random.Random(seed)
    sample: list[str] = []
    for i, item in enumerate(iterable):
        if i < n:
            sample.append(item)
        else:
            j = rng.randint(0, i)
            if j < n:
                sample[j] = item
    return sample


def _run_scan_batch(texts: list[str]) -> tuple[int, Counter]:
    """Returns (blocked_count, signal_hit_counter) over texts."""
    blocked = 0
    signal_counts: Counter = Counter()
    for text in texts:
        decision = scan(text)
        if not decision.allow:
            blocked += 1
        for sig in decision.evidence.get("signals", ()):
            signal_counts[sig] += 1
    return blocked, signal_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500, help="sample size for phase-1 attacks and Enron benign mail")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading LLMail-Inject PHASE 1 (all-objectives-met attacks), sampling up to {args.n} (seed={args.seed})...")
    attack_sample = _reservoir_sample(iter_phase1_all_objectives_attacks(), args.n, args.seed)
    attack_blocked, attack_signals = _run_scan_batch(attack_sample)
    n_attacks = len(attack_sample)
    detection_rate = attack_blocked / n_attacks if n_attacks else float("nan")

    print(f"Loading the {FP_EMAILS_PATH.name} benign set...")
    fp_emails = load_fp_benign_emails()
    fp_blocked, fp_signals = _run_scan_batch(fp_emails)
    n_fp = len(fp_emails)
    fp_rate_llmail = fp_blocked / n_fp if n_fp else float("nan")

    print(f"Sampling up to {args.n} real Enron emails (seed={args.seed})...")
    enron_sample = sample_enron_emails(n=args.n, seed=args.seed)
    enron_blocked, enron_signals = _run_scan_batch(enron_sample)
    n_enron = len(enron_sample)
    fp_rate_enron = enron_blocked / n_enron if n_enron else float("nan")

    print()
    print("=" * 72)
    print("STAGE 1A -- directive.scan() evaluation (PHASE 1 ONLY; phase 2 untouched)")
    print("=" * 72)
    print(f"LLMail-Inject phase-1 attacks (ALL 5 objectives met), n={n_attacks}")
    print(f"  detection rate (blocked): {detection_rate:.1%}  ({attack_blocked}/{n_attacks})")
    print(f"  top signals fired: {attack_signals.most_common()}")
    print()
    print(f"LLMail-Inject benign FP-test emails, n={n_fp}")
    print(f"  false positive rate: {fp_rate_llmail:.1%}  ({fp_blocked}/{n_fp})")
    print(f"  signals fired on false positives: {fp_signals.most_common()}")
    print()
    print(f"Enron real email sample, n={n_enron}")
    print(f"  false positive rate: {fp_rate_enron:.1%}  ({enron_blocked}/{n_enron})")
    print(f"  signals fired on false positives: {enron_signals.most_common()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
