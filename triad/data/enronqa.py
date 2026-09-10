"""EnronQA loaders.

MEASURED TRAP (2026-09-10, `scripts/inspect_data.py`): all four parquet files
(train x2, dev, test) contain the SAME 73,772 emails -- the splits are over
*questions*, not emails. Loading all four files without deduplicating by `path`
indexes every email three times over, which looks exactly like a poison cluster to
the Stage 1B near-duplicate detector. `load_emails` always dedupes by `path`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

from triad.config import DATA_RAW
from triad.contract import Chunk, Provenance
from triad.data.errors import DataUnavailable
from triad.data.types import QARecord

DEFAULT_ROOT = DATA_RAW / "enronqa" / "data"

# The four parquet files are named train-00000-of-00002 / train-00001-of-00002 /
# dev-00000-of-00001 / test-00000-of-00001 -- "train" needs a glob of TWO files, not
# one. Verified by listing the directory before writing this map.
_SPLIT_GLOBS = {
    "train": "train-*.parquet",
    "dev": "dev-*.parquet",
    "test": "test-*.parquet",
}

_SUBJECT_RE = re.compile(r"^Subject:\s*(.*)$", re.MULTILINE)
_SENDER_RE = re.compile(r"^Sender:\s*(.*)$", re.MULTILINE)


def _files(root: Path, pattern: str = "*.parquet") -> list[Path]:
    files = sorted(root.glob(pattern))
    if not files:
        raise DataUnavailable(f"no parquet files matching {pattern!r} under {root}")
    return files


def _parse_email_header(email_text: str) -> dict[str, str]:
    """Best-effort subject/sender extraction. Never raises: a header EnronQA didn't
    format the way we expect should degrade to an empty metadata dict, not kill the
    whole load."""

    meta: dict[str, str] = {}
    if (m := _SUBJECT_RE.search(email_text)):
        meta["subject"] = m.group(1).strip()
    if (m := _SENDER_RE.search(email_text)):
        meta["sender"] = m.group(1).strip()
    return meta


def load_emails(root: Path = DEFAULT_ROOT, users: Iterable[str] | None = None,
                 limit: int | None = None) -> list[Chunk]:
    """All EnronQA emails, deduplicated by `path`. `tenant` = the `user` column
    (150 distinct values). Raises `DataUnavailable` if no parquet files are found."""

    files = _files(root)
    users_set = set(users) if users is not None else None
    seen_paths: set[str] = set()
    out: list[Chunk] = []
    for f in files:
        table = pq.read_table(f, columns=["email", "user", "path"])
        for email_text, user, path in zip(
            table.column("email").to_pylist(),
            table.column("user").to_pylist(),
            table.column("path").to_pylist(),
            strict=True,
        ):
            if path in seen_paths:
                continue
            if users_set is not None and user not in users_set:
                continue
            seen_paths.add(path)
            out.append(Chunk(
                id=path,
                text=email_text,
                tenant=user,
                source_type="email",
                provenance=Provenance("enronqa", path, "real"),
                metadata=_parse_email_header(email_text),
            ))
            if limit is not None and len(out) >= limit:
                return out
    return out


def load_qa(root: Path = DEFAULT_ROOT, split: str = "test",
            users: Iterable[str] | None = None) -> list[QARecord]:
    """One QARecord per question. `incorrect_answers` can be a nested list in the
    parquet (one row of alternates per question) -- flattened to a flat tuple for
    THAT question only; never pooled across the email's other questions, which
    would misattribute an incorrect answer to the wrong question."""

    if split not in _SPLIT_GLOBS:
        raise DataUnavailable(f"unknown EnronQA split {split!r}; expected one of {sorted(_SPLIT_GLOBS)}")
    files = _files(root, _SPLIT_GLOBS[split])
    users_set = set(users) if users is not None else None
    out: list[QARecord] = []
    for f in files:
        table = pq.read_table(f, columns=["questions", "gold_answers", "incorrect_answers", "path", "user"])
        rows = zip(
            table.column("questions").to_pylist(),
            table.column("gold_answers").to_pylist(),
            table.column("incorrect_answers").to_pylist(),
            table.column("path").to_pylist(),
            table.column("user").to_pylist(),
            strict=True,
        )
        for questions, golds, incorrects, path, user in rows:
            if users_set is not None and user not in users_set:
                continue
            questions = questions or []
            golds = golds or []
            incorrects = incorrects or []
            if not (len(questions) == len(golds) == len(incorrects)):
                raise DataUnavailable(
                    f"{path}: questions/gold_answers/incorrect_answers length mismatch "
                    f"({len(questions)}/{len(golds)}/{len(incorrects)}) -- EnronQA's per-question "
                    f"alignment is broken for this row"
                )
            for i, (question, gold, incorrect_list) in enumerate(zip(questions, golds, incorrects, strict=True)):
                flat_incorrect = tuple(x for x in (incorrect_list or []) if x)
                out.append(QARecord(
                    question=question,
                    gold_answers=(gold,),
                    incorrect_answers=flat_incorrect,
                    email_path=path,
                    tenant=user,
                    provenance=Provenance("enronqa", f"{path}#{i}", "real"),
                ))
    return out
