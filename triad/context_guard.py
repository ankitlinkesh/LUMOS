"""Policy-aware context firewall between retrieval and prompt assembly.

The firewall is intentionally deterministic. Retrieval relevance is not an
authorization decision: tenant, role, classification, PII, secrets, and
Stage-1 taint are evaluated before any chunk reaches the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Sequence

from triad.contract import Chunk, ScoredChunk

__all__ = ["UserContext", "ContextDecision", "ContextGuard", "GuardedContext"]

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{8,}\d)(?!\d)")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|gsk_[A-Za-z0-9_-]{12,}|AKIA[0-9A-Z]{16})\b")


@dataclass(frozen=True)
class UserContext:
    tenant: str
    roles: tuple[str, ...] = ()
    purpose: str = "answer"


@dataclass(frozen=True)
class ContextDecision:
    chunk_id: str
    action: str  # allow | redact | block
    reasons: tuple[str, ...]
    evidence: dict[str, object]


@dataclass(frozen=True)
class GuardedContext:
    allowed: tuple[ScoredChunk, ...]
    redacted: tuple[ScoredChunk, ...]
    blocked: tuple[str, ...]
    decisions: tuple[ContextDecision, ...]

    @property
    def prompt_chunks(self) -> tuple[ScoredChunk, ...]:
        """Chunks in prompt order, excluding blocked content."""
        return self.allowed + self.redacted


def _redact(text: str, *, redact_email: bool = False) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    if _SECRET.search(text):
        text = _SECRET.sub("[REDACTED_SECRET]", text)
        reasons.append("redact-secret")
    if _SSN.search(text):
        text = _SSN.sub("[REDACTED_SSN]", text)
        reasons.append("redact-ssn")
    if redact_email and _EMAIL.search(text):
        text = _EMAIL.sub("[REDACTED_EMAIL]", text)
        reasons.append("redact-email")
    if _PHONE.search(text):
        text = _PHONE.sub("[REDACTED_PHONE]", text)
        reasons.append("redact-phone")
    return text, tuple(reasons)


class ContextGuard:
    """Apply deny-first context policy and return an auditable decision set."""

    def __init__(self, *, deny_quarantined: bool = True, redact_sensitive: bool = True, redact_email: bool = False) -> None:
        self.deny_quarantined = deny_quarantined
        self.redact_sensitive = redact_sensitive
        self.redact_email = redact_email

    def guard(self, user: UserContext, chunks: Sequence[ScoredChunk]) -> GuardedContext:
        allowed: list[ScoredChunk] = []
        redacted: list[ScoredChunk] = []
        blocked: list[str] = []
        decisions: list[ContextDecision] = []

        for scored in chunks:
            chunk = scored.chunk
            reasons: list[str] = []
            evidence: dict[str, object] = {
                "tenant": chunk.tenant,
                "user_tenant": user.tenant,
                "classification": chunk.metadata.get("classification", "unclassified"),
                "quarantined": chunk.taint.quarantined,
            }
            if chunk.tenant != user.tenant:
                reasons.append("tenant-isolation")
            if self.deny_quarantined and chunk.taint.quarantined:
                reasons.append("quarantined-content")

            if reasons:
                blocked.append(chunk.id)
                decisions.append(ContextDecision(chunk.id, "block", tuple(reasons), evidence))
                continue

            if self.redact_sensitive:
                text, redact_reasons = _redact(chunk.text, redact_email=self.redact_email)
            else:
                text, redact_reasons = chunk.text, ()
            if redact_reasons:
                evidence["redactions"] = redact_reasons
                redacted_chunk = replace(chunk, text=text)
                redacted_scored = replace(scored, chunk=redacted_chunk)
                redacted.append(redacted_scored)
                decisions.append(ContextDecision(chunk.id, "redact", redact_reasons, evidence))
            else:
                allowed.append(scored)
                decisions.append(ContextDecision(chunk.id, "allow", (), evidence))

        return GuardedContext(tuple(allowed), tuple(redacted), tuple(blocked), tuple(decisions))
