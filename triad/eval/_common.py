"""Shared plumbing every eval script uses: git provenance, cache-hit
accounting, and one uniform results-JSON writer, so ``python -m triad.eval.report``
can read every script's output the same way.

Call ``require_real()`` at the START of an eval script's ``main()`` -- never
at import time. It used to be a module-level import-time side effect; that
leaked ``TRIAD_REQUIRE_REAL=1`` into every OTHER test in the same pytest
session (env vars are process-global, not scoped to one test file) and broke
``test_data_registry.py``'s mixing tests, which rely on synthetic mixing
being allowed by default. Importing ``_common`` (e.g. from a unit test that
only exercises a script's pure helper functions) must never have that side
effect; only actually *running* an eval script should.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from triad.config import ROOT

RESULTS_DIR = ROOT / "results"


def require_real() -> None:
    """Sets ``TRIAD_REQUIRE_REAL=1`` for the current process so
    ``triad.data.registry.get_dataset`` refuses to silently mix in synthetic
    records. The registry reads this env var INSIDE ``get_dataset`` on every
    call (never cached at import time -- see the registry's own docstring),
    so calling this once at the top of ``main()`` is enough for every
    ``get_dataset`` call the rest of that script makes."""
    os.environ["TRIAD_REQUIRE_REAL"] = "1"


def git_commit() -> dict[str, Any]:
    """``{"hash": <sha or "unknown">, "dirty": bool | None}``. A hash alone is
    misleading if the tree had uncommitted changes when the run happened, so
    ``dirty`` travels with it. Never raises -- a missing git binary degrades
    to "unknown", never a crashed eval run."""
    try:
        hash_ = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return {"hash": "unknown", "dirty": None}
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True,
        )
        dirty: bool | None = bool(status.strip())
    except Exception:
        dirty = None
    return {"hash": hash_, "dirty": dirty}


@dataclass
class CacheStats:
    """Cache hits vs. live LLM calls -- the two counts the task brief requires
    in every results JSON that touches the LLM. Call ``record(response.cached)``
    after every ``GroqClient.chat()``."""

    hits: int = 0
    live: int = 0

    def record(self, cached: bool) -> None:
        if cached:
            self.hits += 1
        else:
            self.live += 1

    def as_dict(self) -> dict[str, int]:
        return {"cache_hits": self.hits, "live_calls": self.live, "total_calls": self.hits + self.live}


def write_results(name: str, payload: dict[str, Any], *, results_dir: Path = RESULTS_DIR) -> Path:
    """Writes ``results/<name>_<timestamp>.json``. ``payload`` should already
    carry the fields the brief requires (data composition via
    ``DatasetHandle.describe()``, embedder, model id, n, k, cache stats); this
    adds the timestamp and git provenance and writes atomically (Windows-safe
    temp-file-then-replace, matching ``triad.llm.cache``'s own pattern)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = results_dir / f"{name}_{ts}.json"
    full = {"script": name, "written_at_utc": ts, "git": git_commit(), **payload}
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(full, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    print(f"wrote {path}", file=sys.stderr)
    return path


def latest_result(name_prefix: str, *, results_dir: Path = RESULTS_DIR) -> dict[str, Any] | None:
    """The most recently written results file whose name starts with
    ``name_prefix`` (matched on the filename, e.g. "poisonedrag_off"), or
    None if there isn't one yet. Sorted by filename, which embeds the
    timestamp, so this is always the latest run, not just the latest mtime."""
    if not results_dir.exists():
        return None
    candidates = sorted(results_dir.glob(f"{name_prefix}*.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text(encoding="utf-8"))
