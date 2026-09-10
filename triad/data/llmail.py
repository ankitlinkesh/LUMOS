"""LLMail-Inject: real attack emails that already beat Microsoft's deployed defenses.

`objectives` is a JSON-encoded STRING inside each JSONL row, not a nested object --
matches `scripts/inspect_data.py`'s handling exactly (parse when it's a str, treat
missing/None as met-nothing) so the "all objectives met" count this module reports
reproduces the measured baseline: phase 1 = 3,018, phase 2 = 306, BEFORE dedupe by
body text. Report the number AFTER dedupe too -- that's the real unique attack count
and it is what `load_attacks` returns.
"""

from __future__ import annotations

import json
from pathlib import Path

from triad.config import DATA_RAW
from triad.contract import Provenance
from triad.data.errors import DataUnavailable
from triad.data.types import InjectionAttack
from triad.contract import Chunk

DEFAULT_ROOT = DATA_RAW / "llmail_inject" / "data"

ALL_OBJECTIVE_KEYS = ("email.retrieved", "defense.undetected", "exfil.sent", "exfil.destination", "exfil.content")


def _parse_objectives(raw: object) -> dict[str, bool]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def load_attacks(root: Path = DEFAULT_ROOT, phase: int = 1, all_objectives_only: bool = True,
                  dedupe: bool = True) -> list[InjectionAttack]:
    if phase not in (1, 2):
        raise DataUnavailable(f"unknown LLMail-Inject phase {phase!r}; expected 1 or 2")
    path = root / f"raw_submissions_phase{phase}.jsonl"
    if not path.exists():
        raise DataUnavailable(f"missing LLMail-Inject file: {path}")

    seen_bodies: set[str] = set()
    out: list[InjectionAttack] = []
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n += 1
            row = json.loads(line)
            objectives = _parse_objectives(row.get("objectives"))
            body = row.get("body", "")
            if all_objectives_only and not all(objectives.get(k) for k in ALL_OBJECTIVE_KEYS):
                continue
            if dedupe:
                if body in seen_bodies:
                    continue
                seen_bodies.add(body)
            record_id = row.get("RowKey") or row.get("job_id") or f"{phase}:{n}"
            out.append(InjectionAttack(
                id=str(record_id),
                subject=row.get("subject", ""),
                body=body,
                scenario=row.get("scenario", ""),
                objectives=objectives,
                provenance=Provenance(f"llmail:phase{phase}", str(record_id), "real"),
            ))
    if n == 0:
        raise DataUnavailable(f"{path}: zero lines")
    return out


def load_benign(root: Path = DEFAULT_ROOT) -> list[Chunk]:
    """The 203 benign emails LLMail-Inject ships for false-positive testing. Each
    entry is a single string `"Subject of the email: X.   Body: Y"`; split on the
    literal "Body:" marker (best-effort -- a record missing the marker keeps its
    full text rather than being dropped)."""

    path = root / "emails_for_fp_tests.json"
    if not path.exists():
        raise DataUnavailable(f"missing LLMail-Inject benign fixture: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        raise DataUnavailable(f"{path}: zero benign emails")

    out: list[Chunk] = []
    for i, entry in enumerate(data):
        text = entry if isinstance(entry, str) else json.dumps(entry)
        meta: dict[str, str] = {}
        if "Body:" in text:
            subject_part, _, body_part = text.partition("Body:")
            meta["subject"] = subject_part.replace("Subject of the email:", "").strip().rstrip(".")
        out.append(Chunk(
            id=f"llmail-benign-{i}",
            text=text,
            tenant="public",
            source_type="email",
            provenance=Provenance("llmail:benign", f"benign-{i}", "real"),
            metadata=meta,
        ))
    return out
