"""Deterministic document-hygiene checks run before embedding.

These checks are deliberately narrow and high-confidence. They catch content
that can be invisible to a human reviewer or that is clearly carrying an
encoded instruction. They are evidence for Stage 1A, not a semantic truth
classifier: ordinary words such as ``ignore`` are not enough to block a file.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass

from triad.contract import GuardDecision

__all__ = ["scan", "HygieneReport"]

_BIDI = re.compile(r"[\u202a-\u202e\u2066-\u2069\u200e\u200f]")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_HIDDEN_MARKUP = re.compile(
    r"<!--.*?-->|<[^>]+(?:display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0)[^>]*>.*?</[^>]+>",
    re.IGNORECASE | re.DOTALL,
)
_BASE64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{48,}={0,2}(?![A-Za-z0-9+/])")
_INSTRUCTION = re.compile(
    r"\b(?:ignore|disregard|override)\s+(?:all\s+|the\s+)?(?:previous|prior|above)\s+instructions\b"
    r"|\b(?:system|developer)\s*:\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HygieneReport:
    signals: tuple[str, ...]
    reasons: tuple[str, ...]
    evidence: dict[str, object]

    @property
    def suspicious(self) -> bool:
        return bool(self.signals)


def _decoded_instruction(token: str) -> bool:
    try:
        raw = base64.b64decode(token + "=" * (-len(token) % 4), validate=True)
        text = raw.decode("utf-8", errors="strict")
    except (ValueError, UnicodeError, binascii.Error):
        return False
    return bool(_INSTRUCTION.search(text))


def scan(text_or_chunk) -> GuardDecision:
    """Return a Stage 1 decision for high-confidence hygiene violations.

    The function never raises. A malformed input is blocked rather than
    silently treated as clean, matching the rest of Stage 1's fail-closed rule.
    """
    try:
        text = getattr(text_or_chunk, "text", text_or_chunk)
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        text = "" if text is None else str(text)

        signals: list[str] = []
        reasons: list[str] = []
        evidence: dict[str, object] = {
            "zero_width_count": len(_ZERO_WIDTH.findall(text)),
            "bidi_count": len(_BIDI.findall(text)),
            "hidden_markup_count": len(_HIDDEN_MARKUP.findall(text)),
        }

        if evidence["bidi_count"]:
            signals.append("bidi_override")
            reasons.append("bidirectional override characters can make displayed text differ from parsed text")
        if evidence["zero_width_count"] >= 2:
            signals.append("zero_width_text")
            reasons.append("multiple zero-width characters can hide content from a human reviewer")
        if evidence["hidden_markup_count"]:
            signals.append("hidden_markup")
            reasons.append("HTML or markup hides content from a normal document reader")

        encoded_hits = [token for token in _BASE64_TOKEN.findall(text) if _decoded_instruction(token)]
        evidence["encoded_instruction_count"] = len(encoded_hits)
        if encoded_hits:
            signals.append("encoded_instruction")
            reasons.append("a Base64 payload decodes to instruction-override text")

        evidence["signals"] = tuple(signals)
        evidence["score"] = 1.0 if signals else 0.0
        if signals:
            return GuardDecision.block("ingest", *reasons, evidence=evidence)
        return GuardDecision.ok("ingest", evidence=evidence)
    except Exception as exc:
        return GuardDecision.block(
            "ingest",
            f"internal error while scanning hygiene: {type(exc).__name__}: {exc}",
            evidence={"score": 1.0, "signals": ("internal_error",), "internal_error": True},
        )
