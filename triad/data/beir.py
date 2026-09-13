"""BEIR NQ: a background corpus for the PoisonedRAG replay.

`corpus.jsonl` is 2.68M lines / 1.5GB -- never `json.load` the whole file. Streams
line by line, seeded reservoir sampling for the random slice, so `load_nq(seed=0)`
is reproducible across runs (NOT process-salted `hash()`, which would pass a
same-process determinism test and silently fail across runs/machines).

`sample_n` is the size of the RANDOM slice; `include_ids` (typically a PoisonedRAG
target's gold passages, resolved via `gold_ids_for_queries`) are added ON TOP,
guaranteed present even if the random draw misses them -- so the returned corpus can
be larger than `sample_n` by up to `len(include_ids)`. Document the ratio you used
(sample_n vs. corpus size) wherever you report ASR on this slice; PoisonedRAG's
own million-doc framing does not apply to a corpus this small.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable

from triad.config import DATA_RAW
from triad.contract import Chunk, Provenance
from triad.data.errors import DataUnavailable

DEFAULT_ROOT = DATA_RAW / "beir" / "nq"


def _corpus_path(root: Path) -> Path:
    p = root / "corpus.jsonl"
    if not p.exists():
        raise DataUnavailable(f"missing BEIR corpus file: {p}")
    return p


def gold_ids_for_queries(query_ids: Iterable[str], root: Path = DEFAULT_ROOT,
                          split: str = "test") -> tuple[str, ...]:
    """Corpus doc-ids qrels marks relevant for the given query ids -- the join that
    lets `load_nq(include_ids=...)` guarantee a PoisonedRAG target's gold passage is
    actually indexed. PoisonedRAG's target id (e.g. "test1") IS the BEIR query `_id`
    for the `nq` corpus -- confirmed by inspection, not assumed."""

    path = root / "qrels" / f"{split}.tsv"
    if not path.exists():
        raise DataUnavailable(f"missing BEIR qrels file: {path}")
    wanted = set(query_ids)
    if not wanted:
        return ()
    found: list[str] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row["query-id"] in wanted:
                found.append(row["corpus-id"])
    return tuple(found)


def load_nq(root: Path = DEFAULT_ROOT, sample_n: int = 1000, seed: int = 0,
            include_ids: tuple[str, ...] = ()) -> list[Chunk]:
    corpus_path = _corpus_path(root)
    rng = random.Random(seed)
    forced = set(include_ids)
    forced_found: dict[str, dict] = {}
    reservoir: list[dict] = []

    with open(corpus_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj["_id"] in forced:
                forced_found[obj["_id"]] = obj
            if len(reservoir) < sample_n:
                reservoir.append(obj)
            else:
                j = rng.randint(0, i)
                if j < sample_n:
                    reservoir[j] = obj

    if forced and len(forced_found) < len(forced):
        missing = forced - forced_found.keys()
        raise DataUnavailable(f"{len(missing)} include_ids not found in BEIR NQ corpus: {sorted(missing)[:5]}...")

    seen_ids: set[str] = set()
    out: list[Chunk] = []
    # Forced ids first, so a caller relying on include_ids never has to search for them.
    for obj in list(forced_found.values()) + reservoir:
        if obj["_id"] in seen_ids:
            continue
        seen_ids.add(obj["_id"])
        meta = {"title": obj.get("title", "")} if obj.get("title") else {}
        out.append(Chunk(
            id=obj["_id"],
            text=obj["text"],
            tenant="public",
            source_type="passage",
            provenance=Provenance("beir:nq", obj["_id"], "real"),
            metadata=meta,
        ))
    return out
