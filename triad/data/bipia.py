"""BIPIA (Microsoft, indirect prompt injection benchmark).

Usable as downloaded, no extra steps: `email`, `table`, `code` tasks each ship a
`test.jsonl`/`train.jsonl` context file plus a separate attack-text pool
(`text_attack_{test,train}.json` for email/table, `code_attack_{test,train}.json`
for code).

NOT usable without extra work we were told not to guess at: `qa` (WebQA) and
`abstract` (Summarization) ship no context files -- `benchmark/qa/process.py` and
`benchmark/abstract/process.py` require building `newsqa` from its own repo (via a
docker image) and the XSum dataset respectively, both external downloads outside
`data/raw`. `load(task="qa")` / `load(task="abstract")` raise `DataUnavailable`
naming exactly what's missing rather than fabricating a loader for data we don't have.
"""

from __future__ import annotations

import json
from pathlib import Path

from triad.config import DATA_RAW
from triad.contract import Provenance
from triad.data.errors import DataUnavailable
from triad.data.types import BipiaRecord

DEFAULT_ROOT = DATA_RAW / "BIPIA" / "benchmark"

USABLE_TASKS = ("email", "table", "code")
UNAVAILABLE_TASKS = {
    "qa": "needs newsqa (build via the newsqa repo's docker image; see benchmark/qa/process.py)",
    "abstract": "needs the XSum dataset; see benchmark/abstract/process.py",
}

_ATTACK_POOL_FILES = {
    "text": "text_attack_{split}.json",   # attacks for email/table (and any free-text task)
    "code": "code_attack_{split}.json",
}


def load(task: str, split: str = "test", root: Path = DEFAULT_ROOT) -> list[BipiaRecord]:
    if task in UNAVAILABLE_TASKS:
        raise DataUnavailable(f"BIPIA task {task!r} unavailable: {UNAVAILABLE_TASKS[task]}")
    if task not in USABLE_TASKS:
        raise DataUnavailable(f"unknown BIPIA task {task!r}; expected one of {USABLE_TASKS}")
    path = root / task / f"{split}.jsonl"
    if not path.exists():
        raise DataUnavailable(f"missing BIPIA context file: {path}")

    out: list[BipiaRecord] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            context = row.get("context")
            context = context if isinstance(context, str) else json.dumps(context)
            record_id = f"{task}-{split}-{i}"
            out.append(BipiaRecord(
                task=task,
                context=context,
                question=row.get("question", ""),
                ideal=row.get("ideal", ""),
                provenance=Provenance(f"bipia:{task}", record_id, "real"),
            ))
    if not out:
        raise DataUnavailable(f"{path}: zero records")
    return out


def load_attack_pool(kind: str = "text", split: str = "test", root: Path = DEFAULT_ROOT) -> dict[str, tuple[str, ...]]:
    """The pool of injected-instruction strings BIPIA ships separately from its
    contexts, grouped by category (e.g. "Task Automation"). `kind="text"` covers
    email/table; `kind="code"` is the code-task pool."""

    if kind not in _ATTACK_POOL_FILES:
        raise DataUnavailable(f"unknown BIPIA attack pool kind {kind!r}; expected one of {tuple(_ATTACK_POOL_FILES)}")
    path = root / _ATTACK_POOL_FILES[kind].format(split=split)
    if not path.exists():
        raise DataUnavailable(f"missing BIPIA attack pool file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        raise DataUnavailable(f"{path}: zero attack categories")
    return {category: tuple(texts) for category, texts in data.items()}
