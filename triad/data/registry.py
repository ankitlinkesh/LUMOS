"""`get_dataset(name, **kw)` -- the one entry point every eval script should call.

Real data first, always. The real loader runs and, if it fails, the failure is
raised LOUDLY: nothing is ever substituted behind the caller's back. Synthetic data
exists to be MIXED INTO a real dataset when the real one doesn't cover a scenario
well enough (for example planting synthetic hidden-payload emails in real Enron
inboxes). Mixing is an explicit opt-in (`synthetic_n=`, with a stated `mix_reason`),
every record keeps its own real/synthetic label, and the handle reports its exact
composition. `TRIAD_REQUIRE_REAL=1` forbids mixing entirely: the eval harness sets it
for any run whose numbers get reported as real.

Read from the environment INSIDE `get_dataset`, never cached at import time: a module-
level constant would make `monkeypatch.setenv` in a test a no-op.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any, Callable, Sequence

from triad.data import beir, bipia, enronqa, llmail, poisonedrag, synthetic
from triad.data.errors import DataUnavailable
from triad.data.types import DatasetHandle

REQUIRE_REAL_ENV = "TRIAD_REQUIRE_REAL"

REAL_LOADERS: dict[str, Callable[..., list[Any]]] = {
    "enronqa.emails": enronqa.load_emails,
    "enronqa.qa": enronqa.load_qa,
    "poisonedrag.targets": poisonedrag.load_targets,
    "beir.nq": beir.load_nq,
    "llmail.attacks": llmail.load_attacks,
    "llmail.benign": llmail.load_benign,
    "bipia.email": lambda **kw: bipia.load(task="email", **kw),
    "bipia.table": lambda **kw: bipia.load(task="table", **kw),
    "bipia.code": lambda **kw: bipia.load(task="code", **kw),
}

SYNTHETIC_GENERATORS: dict[str, Callable[..., list[Any]]] = {
    "enronqa.emails": synthetic.generate_emails,
    "enronqa.qa": synthetic.generate_qa,
    "poisonedrag.targets": synthetic.generate_poison_targets,
    "beir.nq": synthetic.generate_beir,
    "llmail.attacks": synthetic.generate_llmail,
    "llmail.benign": synthetic.generate_benign,
    "bipia.email": lambda **kw: synthetic.generate_bipia(task="email", **{k: v for k, v in kw.items() if k != "task"}),
    "bipia.table": lambda **kw: synthetic.generate_bipia(task="table", **{k: v for k, v in kw.items() if k != "task"}),
    "bipia.code": lambda **kw: synthetic.generate_bipia(task="code", **{k: v for k, v in kw.items() if k != "task"}),
}

# Every real dataset needs a synthetic generator it can be mixed with.
assert set(REAL_LOADERS) <= set(SYNTHETIC_GENERATORS), (
    f"missing synthetic generator for: {set(REAL_LOADERS) - set(SYNTHETIC_GENERATORS)}"
)


def _require_real() -> bool:
    return os.environ.get(REQUIRE_REAL_ENV, "") == "1"


def _warn_mixed(handle: DatasetHandle, requested: int) -> None:
    short = f" (requested {requested:,}; the generator produced fewer)" if handle.n_synthetic < requested else ""
    rule = "=" * 72
    lines = [
        "",
        rule,
        f"MIXED DATASET  {handle.describe()}{short}",
        "  synthetic records are tagged data_source='synthetic'; report this",
        "  composition with any number computed from it",
        rule,
        "",
    ]
    print("\n".join(lines), file=sys.stderr)


def _assign_tenants(records: Sequence[Any], tenants: Sequence[str]) -> list[Any]:
    """Plant synthetic records into existing (real) tenants, round-robin, so e.g. a
    synthetic hidden-payload email lands in a real Enron inbox. IDs get a
    `synthetic:` prefix so they can never collide with a real record's id."""
    out = []
    for i, r in enumerate(records):
        changes: dict[str, Any] = {}
        if hasattr(r, "tenant"):
            changes["tenant"] = tenants[i % len(tenants)]
        if hasattr(r, "id") and not str(r.id).startswith("synthetic:"):
            changes["id"] = f"synthetic:{r.id}"
        out.append(dataclasses.replace(r, **changes) if changes else r)
    return out


def get_dataset(
    name: str,
    *,
    synthetic_n: int = 0,
    mix_reason: str | None = None,
    into_tenants: Sequence[str] | None = None,
    seed: int = synthetic.SEED,
    **kw: Any,
) -> DatasetHandle:
    """Load the real dataset `name`; optionally mix in up to `synthetic_n` synthetic
    records of the same type (`mix_reason` required, `into_tenants` to plant them in
    existing tenants). Real-loader failures always raise `DataUnavailable`."""
    if name not in REAL_LOADERS:
        raise ValueError(f"unknown dataset {name!r}; known datasets: {sorted(REAL_LOADERS)}")
    if synthetic_n < 0:
        raise ValueError("synthetic_n must be >= 0")

    try:
        real = list(REAL_LOADERS[name](**kw))
    except DataUnavailable:
        raise
    except (OSError, ValueError, KeyError) as e:  # present-but-broken files: still loud, never substituted
        raise DataUnavailable(f"{name}: real loader failed: {type(e).__name__}: {e}") from e
    if not real:
        raise DataUnavailable(f"{name}: real loader returned zero records")

    if synthetic_n == 0:
        return DatasetHandle(name=name, records=tuple(real))

    if _require_real():
        raise DataUnavailable(f"{name}: {REQUIRE_REAL_ENV}=1 forbids mixing synthetic records into reported data")
    if not mix_reason or not mix_reason.strip():
        raise ValueError(f"{name}: say why synthetic records are being mixed in (mix_reason=...)")

    synth = list(SYNTHETIC_GENERATORS[name](seed=seed, **kw))[:synthetic_n]
    if into_tenants:
        synth = _assign_tenants(synth, list(into_tenants))
    handle = DatasetHandle(name=name, records=tuple(real + synth), mix_reason=mix_reason.strip())
    _warn_mixed(handle, synthetic_n)
    return handle
