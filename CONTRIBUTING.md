# Contributing to TRIAD-RAG

Thanks for looking at this. It's early, research-grade software (see the README's
"Project status" section) and contributions are welcome — code, new attack/eval
coverage, documentation, or just filing an issue about a number that doesn't add up.

## Dev setup

```
git clone https://github.com/ankitlinkesh/LUMOS.git triad-rag
cd triad-rag
python -m venv .venv
.venv\Scripts\activate          # Windows; `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"
```

`.[dev]` pulls in every optional piece (real embeddings, a real vector store, a
real LLM client, dataset loaders, the HTTP service) plus `pytest`/`hypothesis`,
so the full test suite can run. If you only want to work on Stage 1/Stage 3
detection logic, `pip install -e .` (no extras) is enough — see the README's
"Install" section for what each extra actually buys you.

## Running the tests

```
python -m pytest                # fast: no network, no real models/datasets
python -m pytest -m slow        # also loads real datasets and models
python -m pytest -m network     # needs a real Groq key in secrets/groq_keys.txt
```

The fast suite is what CI (and you, before opening a PR) should run on every
change; it should stay well under a minute. `slow`/`network` tests are for
verifying against real data/models and real API calls respectively — never
required for a documentation or refactoring PR.

## House rules

These aren't style preferences — they're the invariants that make the numbers
in the README trustworthy, and PRs that violate them will be asked to change.

- **Real data only, and never a silent synthetic fallback.** A real dataset
  loader that fails raises `DataUnavailable` loudly; it never quietly
  substitutes synthetic data. Synthetic data can be *mixed into* a real
  dataset, but only on explicit opt-in (`get_dataset(..., synthetic_n=N,
  mix_reason="...")`), with every record keeping its own `real`/`synthetic`
  label and the handle reporting its exact composition. `TRIAD_REQUIRE_REAL=1`
  forbids mixing entirely, and any run whose numbers get reported anywhere
  (a README table, a paper, a slide) must be run with it set.
- **Tune on a dev split, report on a held-out split.** If you add or retune a
  detector, split your evaluation data before you look at it, tune thresholds
  only on the dev half, and report the number the held-out half gives you —
  even when it's worse than the dev number. Several sections of the README
  exist specifically to show when this discipline caught an inflated result.
- **Every rate prints its denominator.** "51.4%" without "(114/222)" next to
  it is not an acceptable result to report, log, or put in a docstring. A
  structurally empty denominator (n=0) is reported as n=0, never as "0%" or
  omitted.
- **No LLM in the detection path.** Stage 1 and Stage 3 are pure, deterministic
  functions over the frozen contract types in `triad/contract.py`, with
  injectable dependencies (embedder, store, LLM) for testing. An LLM call is
  fine in an *eval harness* (measuring against a model's behavior) but must
  never become part of what decides whether a chunk is quarantined or a tool
  call is authorized.
- **`triad/contract.py` is frozen.** Every stage codes against these types;
  changing them ripples through Stage 1/2/3, the eval harness, and the API.
  Changes here need explicit agreement first, not just a passing test suite —
  open an issue before sending a PR that touches this file.

## Where to look first

- `triad/contract.py` — the frozen types every stage codes against.
- `triad/stage1/`, `triad/stage3/` — the two stages you can exercise with pure
  unit tests, no corpus or model required.
- `triad/pipeline.py` — how the stages wire together end to end.
- `triad/eval/` — one module per experiment; `python -m triad.eval.report`
  collates `results/` into a table.
- `tests/` — read a few before writing one; the fixtures (`HashEmbedder`,
  `chromadb.EphemeralClient()`, a fake LLM implementing the `chat()` protocol)
  are what keep the fast suite fast and network-free.
