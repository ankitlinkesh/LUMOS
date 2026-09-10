"""PoisonedRAG's own released attack texts, reproduced with its own prefix recipe.

MEASURED TRAP: the released JSON's `adv_texts` do NOT start with the target
question -- the paper's black-box attack (`LM_targeted`) prepends it at injection
time, not at generation time. Cited verbatim from the paper's own source, so the
"before" ASR we measure matches what the paper measured, not a weaker version of it:

    data/raw/PoisonedRAG/src/attack.py:92-93 (LM_targeted)
        adv_text_a = question + "."
        adv_texts = [adv_text_a + i for i in adv_texts_b]

Note there is NO space in `adv_text_a + i` -- it is bare string concatenation of
"<question>." directly onto the released text. A "helpful" added space silently
breaks the reproduction and produces a doc that is measurably easier to defend
against than the paper's real attack.
"""

from __future__ import annotations

import json
from pathlib import Path

from triad.config import DATA_RAW
from triad.contract import Provenance
from triad.data.errors import DataUnavailable
from triad.data.types import PoisonTarget

DEFAULT_ROOT = DATA_RAW / "PoisonedRAG" / "results" / "adv_targeted_results"
CORPORA = ("nq", "hotpotqa", "msmarco")


def load_targets(root: Path = DEFAULT_ROOT, corpus: str = "nq") -> list[PoisonTarget]:
    if corpus not in CORPORA:
        raise DataUnavailable(f"unknown PoisonedRAG corpus {corpus!r}; expected one of {CORPORA}")
    path = root / f"{corpus}.json"
    if not path.exists():
        raise DataUnavailable(f"missing PoisonedRAG results file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise DataUnavailable(f"{path}: not valid JSON ({e})") from e
    if not data:
        raise DataUnavailable(f"{path}: zero target questions")

    out: list[PoisonTarget] = []
    for tid, entry in data.items():
        try:
            question = entry["question"]
            correct = entry["correct answer"]
            incorrect = entry["incorrect answer"]
            raw_adv = tuple(entry["adv_texts"])
        except KeyError as e:
            raise DataUnavailable(f"{path}: target {tid!r} missing field {e}") from e
        prefix = question + "."  # attack.py:92, LM_targeted -- bare concatenation, no space
        prefixed = tuple(prefix + text for text in raw_adv)  # attack.py:93
        out.append(PoisonTarget(
            id=tid,
            corpus=corpus,
            question=question,
            correct_answer=correct,
            incorrect_answer=incorrect,
            raw_adv_texts=raw_adv,
            adv_texts=prefixed,
            provenance=Provenance(f"poisonedrag:{corpus}", tid, "real"),
        ))
    return out
