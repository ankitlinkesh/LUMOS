# TRIAD-RAG

**A three-stage integrity and access-control pipeline for RAG systems** — hackathon PS 3.

Retrieval-augmented generation trusts whatever the retriever hands it. That trust is the
vulnerability: a document can carry the attacker's answer (PoisonedRAG), instructions aimed at the
model rather than the reader (EchoLeak / CVE-2025-32711), or simply belong to another customer and
be returned anyway (the post-filter-then-widen cross-tenant leak). TRIAD-RAG enforces one rule at
three points in the data flow.

## The invariant

> **A retrieved chunk may supply facts. It may never (a) widen its own retrieval scope,
> (b) authorize a tool call, or (c) cause an egress.**

Each stage is one tooth of that single rule, not an unrelated filter:

| Stage | Tooth | Attack it is aimed at |
|---|---|---|
| **1 — Ingest** | untrusted content cannot enter the corpus unlabeled or unreviewed | PoisonedRAG, hidden-payload email |
| **2 — Retrieve** | untrusted content cannot widen its own scope | cross-tenant leak / fallback widening |
| **3 — Egress** | untrusted content cannot cause an outbound effect | EchoLeak exfiltration |

Everything is a pure, deterministic function over frozen contract types with injectable
dependencies (embedder, store, LLM). No LLM sits in the detection hot path.

## Measured results

Every number below was produced by a script in this repo against **real published data**. Each
results JSON records its git hash, seed, embedder, and the exact real/synthetic composition of its
inputs. Read the caveats — several of these numbers are weaker than they first look, and saying so
is the point.

### Stage 2 — cross-tenant retrieval

`results/tenant_leak_20260910T120126Z.json` · 500 probes, 20 real EnronQA inboxes, 3,277 emails,
bge-small, k=5. A probe asks tenant A a question whose gold email lives in tenant B, so leakage is
exact rather than judged.

| Retriever | Any foreign chunk | The specific gold email | p50 latency |
|---|---|---|---|
| Leaky (search, then filter) | **67.6%** | 52.2% | 288 ms |
| Secure (predicate inside the ANN search) | **0.0%** | **0.0%** | **207 ms** |

The secure retriever is **80 ms faster**: pre-filtering shrinks the candidate set the KNN has to
search. Security is not paying a latency tax here.

### Stage 1A — structural injection detection

`results/injection_20260910T121109Z.json` · LLMail-Inject **phase 2**, the held-out split, 222
unique attacks that met *every* objective against Microsoft's deployed defenses (Prompt Shield,
TaskTracker, Spotlighting, an LLM judge).

| Detector | Detection rate | FPR, 203 benign | FPR, 500 real Enron |
|---|---|---|---|
| **Ours** (structural) | **51.4%** | 0.0% | 0.0% |
| Llama Prompt Guard 2 86M (baseline, 922 live Groq calls) | 23.4% | 0.0% | 0.0% |

2.2× the baseline on attacks that already beat a shipped defense stack, at zero false positives.

**Generalization gap, stated plainly:** the same detector scores 72–75% on phase 1, which its
thresholds were tuned on. The drop to 51.4% on phase 2 is what a held-out split is for. Quote 51.4%.

### Stage 1B — poison geometry

`results/geometry_20260910T142941Z.json` · 5,000 BEIR NQ passages + PoisonedRAG's 100 NQ targets
(500 adv_texts). Targets sorted by id: the first 50 tune thresholds, **the other 50 are never
touched until report time**; the clean sample is split the same way, and manifold isolation's
reference corpus is the dev half only.

Against an adaptive attacker — (a) the paper's verbatim question prefix, (b) a deterministically
paraphrased prefix, (c) no prefix at all:

**Ingest-time scan** (held-out, 250 poison texts per variant, echo threshold 0.90 tuned to a ≤1% dev FPR):

| Attack variant | Query-echo | Manifold isolation | Combined |
|---|---|---|---|
| verbatim (the paper's attack) | 33.6% | 0.8% | **34.4%** |
| paraphrased prefix | 6.4% | 0.4% | **6.8%** |
| no prefix | 0.0% | 2.4% | **2.4%** |

False positives on the 2,500 held-out clean passages: query-echo 0.64%, manifold isolation 2.56%,
combined 3.2%.

Two findings here, both negative, both measured:

- **Ingest-time detection collapses under adaptation**: 34.4% → 6.8% → 2.4%. Query-echo is a cheap
  trap for the *published* attack, not a defense. An attacker who paraphrases the prefix costs us
  five sixths of the catch rate; one who drops it costs us all of it.
- **Manifold isolation does not work at all.** Its poison catch (0.4–2.4%) is *lower than its own
  clean false-positive rate* (1.36–4.88% across candidate cutoffs) — it is worse than chance on this
  data, and adding it to the combined score raises FPR from 0.64% to 3.2% for almost no catch gain.
  PoisonedRAG's texts are fluent and on-topic, so they simply do not sit off the corpus manifold in
  this embedding space. This contradicts the design premise that isolation would carry the detector
  when query-echo went dark. It does not.

**Query-time cluster collapse** (held-out, k=5, similarity threshold 0.75):

| Attack variant | Poison cluster collapsed to one vote | Poison held 5 of 5 top-k slots |
|---|---|---|
| verbatim | 100% | 50/50 targets |
| paraphrased | 100% | 50/50 targets |
| no prefix | 96% | 48/50 targets |

The five injected texts share an embedding neighbourhood that survives paraphrasing and prefix
removal, which is why this holds where query-echo collapses. It also answers the paper's own finding
that *duplicate-text filtering has zero effect*: this clusters in embedding space, not by hash.

> **Three things this number does not prove — read before quoting it.**
>
> 1. **It is a mechanism, not an outcome.** `poison_collapse_rate` counts targets where 2+ poison
>    documents were merged into a single vote. There is no LLM call anywhere in this eval. It does
>    not show the attack fails; that is ASR before→after, still pending below.
> 2. **The 0% "clean wrong merge" figure in the mixed pool is vacuous, and is not reported here.**
>    Poison occupied *all five* top-k slots in every verbatim and paraphrased target, so no clean
>    chunk remained that *could* be wrongly merged. The honest measurement is a clean-only baseline
>    with no poison injected: **14.0% of held-out questions (7/50) had two genuinely distinct clean
>    passages merged.** Collapse does damage real retrieval, and that cost is real.
> 3. **The clean pool does not contain the right answers.** `load_nq` was called without
>    `include_ids`, so each target's actual gold passage is almost certainly absent from the 2,500
>    held-out clean passages. This measures 5 targeted poison texts against 2,500 *unrelated* clean
>    passages, not against the real competing evidence for that question — which inflates how far
>    poison dominates the top-k, and therefore the collapse rate. Fixing it invalidates every clean
>    embedding cache key and costs ~50 minutes of CPU embedding; it has not been done.
>
> Also: the similarity threshold was not selected on discriminating evidence. Every candidate from
> 0.75 to 0.95 scored 0% dev wrong-merge, so 0.75 won on a tie-break, not on measurement.

Embedder is bge-small, not Contriever (the paper's retriever). Contriever's cache covers only clean
passages under another script's namespace, and its unnormalized dot-product convention would not
carry these thresholds across unchanged.

### Stage 3 — output and egress

**Paused** by decision, to finish stages 1 and 2 first. The code exists and is tested
(`triad/stage3/`: URL-taint egress check, rendering-boundary sanitizer, tool-call authorization,
`[UNTRUSTED DATA]` fencing; the egress property test is mutation-verified — it fails when the taint
check is disabled). It has **no end-to-end measured numbers yet**, and is not part of any headline
claim.

### Attack success rate, before → after

**PENDING.** Only an n=2 smoke run exists (`results/poisonedrag_n2_20260910T113806Z.json`: ASR
1.0 → 0.5 on two targets), which is far too small to report. A reportable-scale replay — 100
targets against a ≥10k-passage BEIR NQ corpus — is running now. This row is deliberately left
visible rather than omitted.

## Reproducibility caveats

- **Every results file currently records `"dirty": true`** — the working tree had uncommitted
  changes when the run happened, so the recorded git hash does not exactly reproduce it. Disclosed
  rather than hidden; a corpus build alone is ~1,772 s, so these are not re-run casually. Commit
  before the next run so future results are clean.
- There are two `injection_*` and two `tenant_leak_*` files. **The later timestamp in each pair is
  the live one**; the earlier is a smaller preliminary run kept for history.
- `signals_on_enron_fp` shows `action_verb_near_address` firing on **126 of 500 real Enron emails**.
  Overall FPR is 0.0% only because that signal alone sits below the 0.5 quarantine threshold.
  Raising signal weights will cost false positives on real business mail first — that is the
  binding constraint on Stage 1A, not the attack set.
- Thresholds are tuned on a dev split and reported on a held-out split, per eval. LLMail phase 2 was
  opened once, for the reported number, and never used for tuning.

## Data

Real published datasets only. Synthetic records are never a silent fallback: a real loader that
fails raises `DataUnavailable` loudly. Synthetic data can be **mixed into** real data on explicit
opt-in (`get_dataset(..., synthetic_n=N, mix_reason="...")`), every record keeps its own
`real`/`synthetic` label, the handle reports its exact composition, and `TRIAD_REQUIRE_REAL=1`
forbids mixing entirely for any run whose numbers get reported.

| Dataset | Measured facts | Used for |
|---|---|---|
| **EnronQA** (`MichaelR207/enron_qa_0922`) | **73,772 unique emails** — not the paper's 103,638; all four parquet splits contain the *same* emails, so dedupe by `path` or three copies of every email look exactly like a poison cluster. 150 inboxes | Stage 2 corpus, clean accuracy, FPR |
| **PoisonedRAG** release | 100 targets × 5 `adv_texts` for nq/hotpotqa/msmarco. The released texts do **not** contain the question; the attack code prepends `question + "."` at injection, and so do we — otherwise the "before" ASR is wrong | Stage 1B, ASR |
| **BEIR NQ** | 2,681,468 passages | clean background corpus |
| **LLMail-Inject** (`microsoft/llmail-inject-challenge`) | Success labels live **only** in `raw_submissions_phase{1,2}.jsonl` — the labelled_unique files carry just `{attack_attempt, reason}`. All objectives met: 3,018 phase 1, 306 phase 2 → **222 unique** after dedupe | Stage 1A, Stage 3 |
| **BIPIA** | email/table/code = 50/100/50 | held-out second injection source |

Data lives in `data/raw` (5.10 GB, sha256 in `MANIFEST.json`) — on `D:` because `C:` is nearly
full. `scripts/download_data.py` re-fetches; `scripts/inspect_data.py` verifies and **exits
non-zero** if a file or label is missing.

*Licenses:* PoisonedRAG ships no license file and BIPIA is NOASSERTION — fine for evaluation, do
not redistribute their files.

## Quickstart

```
# Python 3.11+; CPU-only torch keeps the install small
python -m venv .venv
.venv\Scripts\activate                 # Windows; use .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m pytest                       # 294 tests; network and slow excluded by default
python -m pytest -m slow               # also loads real datasets and models
```

Data (once, ~5 GB):

```
python scripts/download_data.py
python scripts/inspect_data.py         # non-zero exit if anything is missing
```

## Running the evaluations

Each writes a timestamped JSON to `results/` carrying git hash, seed, embedder, and data
composition.

```
python -m triad.eval.tenant_leak     --n-tenants 20 --n-probes 500 --k 5
python -m triad.eval.injection       --n-enron 500 --probe-baseline   # --probe-baseline makes live Groq calls
python -m triad.stage1.eval_geometry --n-clean 5000 --embedder bge
python -m triad.eval.poisonedrag     --n 100 --sample-n 10000         # the ASR run
python -m triad.eval.ablation        --n 10                           # one row per stage on/off
python -m triad.eval.report                                           # collate results/ into a table
```

## Demo UI

```
python -m triad.api            # fake demo service, for UI development
python -m triad.api --real     # the real pipeline
```

The UI is a React + Vite build served offline as static files by the same FastAPI app — **no second
code path**. When the backend does not positively confirm it is serving real data, the page shows a
sticky amber **"DEMO MODE — FAKE DATA, NOT MEASURED RESULTS"** banner, so a screenshot can never be
mistaken for a measurement.

`triad/api/real_adapter.py` is **not finished**: only `meta()` is implemented, the rest raise
`NotImplementedError`. Wiring it to `Pipeline.demo()` is the remaining task.

## Layout

```
triad/
  contract.py        FROZEN interface types: Chunk, TaintVerdict, RetrievalResult, GuardDecision
  config.py
  data/              real dataset loaders + registry (real-first) + synthetic generators
  embed/             embedder interface; bge-small (real) and a hash embedder (fast tests)
  llm/               Groq key pool, official free-tier limiter, principal-keyed cache, doctor
  retrieval/         scope algebra, Chroma store, SecureRetriever + LeakyRetriever (the bug)
  stage1/            1A hidden_text + directive; 1B geometry; their eval drivers
  stage3/            egress, tool-call authorization, fencing  (PAUSED)
  eval/              the measurement harness, one module per experiment
  api/               FastAPI app, demo service, real adapter (incomplete)
  pipeline.py        end-to-end wiring
  quarantine.py      reviewable queue with a reason string and a release path — never deletion
tests/               294 tests, incl. Hypothesis property tests for the Stage 2 invariant
scripts/             dataset download + verification
ui/                  React + Vite demo front end
results/             timestamped measurement JSONs
```

### Design notes worth knowing before reading the code

- **`contract.py` is frozen.** Every stage returns a `GuardDecision`, which cannot simultaneously
  allow and escalate, and cannot block without a reason. `Chunk` raises if constructed without a
  tenant. `RetrievalResult.foreign_chunks()` is the property test's oracle.
- **`SecureRetriever` never calls `store.search_unscoped`.** An empty scope searches nothing, any
  exception becomes a decline, and results are re-filtered by scope after Chroma as defense in
  depth. `LeakyRetriever` exists to reproduce the CVE-class bug live on stage.
- **Scope nesting intersects, never replaces.** Replacement is a real containment escape.
- **The LLM cache key includes the principal.** Without it the cache is itself a cross-tenant
  channel. The same trap bit the key-pool health check: a fixed ping prompt was served from cache,
  so keys 2–6 "passed" without ever reaching Groq.
- **Quarantine, never delete.** "What happens to a false positive?" is the first question an
  enterprise reviewer asks.

## Secrets

`secrets/` is gitignored. The Groq key pool reads `secrets/groq_keys.txt`, one key per line. Groq's
free-tier limits are hardcoded in `triad/llm/limits.py` from the official docs and are enforced
**per organization**, which is also the scope Groq's AUP applies to multi-account pooling. Tests
never touch the real file — they build temp files with fake keys.

## Scope and limits

- Close-ended factoid QA, matching the PoisonedRAG paper's own stated limitation.
- Corpus scale is thousands of passages, not millions. The poison ratio is stated per run rather
  than implying million-document scale.
- Threat model: the attacker can write documents or send email, and is black-box to the retriever
  and the LLM. **The tenant identity comes from the authenticated session, never from the query.**
  Out of scope: a compromised embedder, a malicious administrator.
- Stage 3 is paused and unmeasured. Stage 1B's ingest-time signals are weak under adaptation, by
  measurement. The ASR row is pending.

## Sources

- PoisonedRAG, USENIX Security 2025 · [arXiv:2402.07867](https://arxiv.org/html/2402.07867v3)
- EchoLeak / CVE-2025-32711 · [arXiv:2509.10540](https://arxiv.org/abs/2509.10540)
- Relevance–authorization gap · [arXiv:2605.05287](https://arxiv.org/html/2605.05287v1) (ACM CAIS 2026)
- RobustRAG · [arXiv:2405.15556](https://arxiv.org/html/2405.15556v2)
- LLMail-Inject, IEEE SaTML 2025 · [arXiv:2506.09956](https://arxiv.org/html/2506.09956v1)
- EnronQA · [arXiv:2505.00263](https://arxiv.org/html/2505.00263) — BIPIA · [arXiv:2312.14197](https://arxiv.org/abs/2312.14197) — BEIR · [arXiv:2104.08663](https://arxiv.org/pdf/2104.08663)
- OWASP [LLM01:2025 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
