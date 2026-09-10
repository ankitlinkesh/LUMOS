"""Open every downloaded dataset and check the fields the TRIAD-RAG plan relies on.

    .venv/Scripts/python scripts/inspect_data.py

Prints a short report; exits non-zero if an expected file or field is missing, so a
teammate with a broken download finds out now rather than mid-sweep.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
problems: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        problems.append(msg)
        print(f"  !! {msg}")


def enronqa() -> None:
    print("\n[EnronQA]")
    files = sorted((RAW / "enronqa" / "data").glob("*.parquet"))
    check(len(files) == 4, f"expected 4 parquet files, found {len(files)}")
    if not files:
        return
    cols = pq.read_schema(files[0]).names
    print(f"  columns: {cols}")
    for need in ("email", "user", "questions", "gold_answers", "incorrect_answers"):
        check(need in cols, f"EnronQA missing column {need!r}")
    users: Counter[str] = Counter()
    rows = questions = 0
    for f in files:
        t = pq.read_table(f, columns=["user", "questions"])
        rows += t.num_rows
        users.update(t.column("user").to_pylist())
        questions += sum(len(q or []) for q in t.column("questions").to_pylist())
        print(f"  {f.name}: {t.num_rows:,} rows")
    paths = set()
    for f in files:
        paths.update(pq.read_table(f, columns=["path"]).column("path").to_pylist())
    print(f"  total: {rows:,} rows, {questions:,} questions, {len(users)} distinct users (tenants)")
    print(f"  UNIQUE emails by path: {len(paths):,}. Splits share the same emails, so dedupe by `path`")
    print(f"  before indexing; duplicates would look like a poison cluster to the Stage 1B detector.")
    check(len(paths) < rows, "expected EnronQA splits to share emails; dedupe logic may be wrong")
    print(f"  largest inboxes: {users.most_common(5)}")
    sample = pq.read_table(files[0]).slice(0, 1).to_pylist()[0]
    print(f"  sample user={sample['user']!r}  q={sample['questions'][0]!r}")
    print(f"         gold={sample['gold_answers'][0]!r}  incorrect={sample['incorrect_answers'][0]!r}")


def enron_corpus() -> None:
    print("\n[Enron corpus]")
    p = RAW / "enron_corpus" / "emails_adj_dedup.csv"
    check(p.exists(), "enron corpus CSV missing")
    if p.exists():
        with open(p, encoding="utf-8", errors="replace") as fh:
            header = fh.readline().strip()
        print(f"  {p.stat().st_size / 1e6:.0f} MB, header: {header[:200]}")


def llmail() -> None:
    print("\n[LLMail-Inject]")
    d = RAW / "llmail_inject" / "data"
    for phase in (1, 2):
        lab = d / f"labelled_unique_submissions_phase{phase}.json"
        check(lab.exists(), f"missing {lab.name}")
        if lab.exists():
            labelled = json.loads(lab.read_text(encoding="utf-8"))  # {attack text: {attack_attempt, reason, ...}}
            reasons = Counter(v.get("reason") for v in labelled.values())
            print(f"  phase{phase} labelled: {len(labelled):,} unique attacks, reason={dict(reasons)}")
        raw = d / f"raw_submissions_phase{phase}.jsonl"
        check(raw.exists(), f"missing {raw.name}: the only source of defense.undetected / exfil.* labels")
        if not raw.exists():
            continue
        n = retrieved = undetected = exfil_sent = full = 0
        with open(raw, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line).get("objectives")
                obj = json.loads(obj) if isinstance(obj, str) else (obj or {})
                n += 1
                retrieved += bool(obj.get("email.retrieved"))
                undetected += bool(obj.get("defense.undetected"))
                exfil_sent += bool(obj.get("exfil.sent"))
                full += all(obj.get(k) for k in ("email.retrieved", "defense.undetected", "exfil.sent", "exfil.destination", "exfil.content"))
        print(f"  phase{phase} raw: {n:,} attempts | retrieved {retrieved:,} | defense.undetected {undetected:,} | exfil.sent {exfil_sent:,} | ALL objectives met {full:,}")
        check(undetected > 0, f"phase{phase}: zero defense.undetected labels parsed; the headline subset would be empty")
    fp = d / "emails_for_fp_tests.json"
    check(fp.exists(), "missing emails_for_fp_tests.json (benign set)")
    if fp.exists():
        benign = json.loads(fp.read_text(encoding="utf-8"))
        print(f"  benign emails for false-positive tests: {len(benign):,}")


def poisonedrag() -> None:
    print("\n[PoisonedRAG released attack texts]")
    d = RAW / "PoisonedRAG" / "results" / "adv_targeted_results"
    for name in ("nq", "hotpotqa", "msmarco"):
        p = d / f"{name}.json"
        check(p.exists(), f"missing PoisonedRAG {name}.json")
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        entry = next(iter(data.values()))
        print(f"  {name}: {len(data)} target questions, {len(entry['adv_texts'])} adv_texts each, fields={sorted(entry)}")
        starts_with_q = entry["adv_texts"][0].lower().startswith(entry["question"].lower()[:30])
        print(f"    adv_text already begins with the question: {starts_with_q}  (False = prepend 'question.' at injection, as the paper's code does)")


def beir() -> None:
    print("\n[BEIR NQ]")
    d = RAW / "beir" / "nq"
    for f in ("corpus.jsonl", "queries.jsonl"):
        check((d / f).exists(), f"missing beir/nq/{f}")
    if (d / "corpus.jsonl").exists():
        with open(d / "corpus.jsonl", "rb") as fh:
            n = sum(1 for _ in fh)
        print(f"  corpus: {n:,} passages  ({(d / 'corpus.jsonl').stat().st_size / 1e9:.2f} GB)")


def bipia() -> None:
    print("\n[BIPIA]")
    d = RAW / "BIPIA"
    check(d.exists(), "BIPIA not cloned")
    if d.exists():
        bench = d / "benchmark"
        print(f"  benchmark dirs: {sorted(p.name for p in bench.iterdir() if p.is_dir()) if bench.exists() else 'none'}")
        lic = next((p.name for p in d.iterdir() if p.name.upper().startswith("LICENSE")), None)
        print(f"  license file: {lic}")


for fn in (enronqa, enron_corpus, llmail, poisonedrag, beir, bipia):
    try:
        fn()
    except Exception as e:  # report and keep going: one broken source shouldn't hide the rest
        check(False, f"{fn.__name__} failed: {type(e).__name__}: {e}")

print(f"\n{'OK: all expected files and fields present' if not problems else f'{len(problems)} PROBLEM(S)'}")
sys.exit(1 if problems else 0)
