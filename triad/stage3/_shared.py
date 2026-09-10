"""Helpers shared by Stage 3's egress and action guards.

Not part of the public stage3 API (leading underscore): fence.py, egress.py and
actions.py all need the same primitives -- normalize/shingle text, decode a URL
component through a few common encodings, and measure how "random" a string
looks -- so they live here once instead of drifting apart in two modules.

House rule this module exists to serve: taint detection is DATA FLOW, not
keyword matching. We never grep for words like "secret" or "confidential" --
we decode what the model tried to put in a URL/argument and compare the
decoded bytes against what was actually retrieved.
"""

from __future__ import annotations

import base64
import math
import re
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from urllib.parse import unquote

import tldextract

from triad.contract import Chunk

# Offline PSL snapshot only: tldextract fetches over the network by default,
# which this sandbox cannot reach and which would make a security check flaky
# on data flow it happens to run in an airgapped room.
_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

SHINGLE_SIZE = 4
LONG_SUBSTRING_MIN = 20
DECODE_DEPTH = 2                # "nested once": one decode, then one more
ENTROPY_MIN_LEN = 20
ENTROPY_THRESHOLD = 4.0         # bits/char; ~random base64/hex clears this comfortably

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+", re.IGNORECASE)
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def normalize(text: str) -> str:
    """Case/whitespace-fold text so 'A  secret\\n' and 'a secret' compare equal."""
    return re.sub(r"\s+", " ", text.strip().lower())


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def shingles(text: str, n: int = SHINGLE_SIZE) -> frozenset[tuple[str, ...]]:
    """Word n-grams of ``text``. Used to detect a phrase lifted verbatim from
    retrieved context, without caring about exact punctuation/casing."""
    w = words(text)
    if len(w) < n:
        return frozenset()
    return frozenset(tuple(w[i : i + n]) for i in range(len(w) - n + 1))


def longest_common_substring_len(a: str, b: str) -> int:
    """Length of the longest run ``a`` and ``b`` share, on normalized text."""
    if not a or not b:
        return 0
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    return match.size


def shannon_entropy(s: str) -> float:
    """Bits/char. Near 0 for repeated/simple strings, ~4.5-6 for base64/hash-like
    noise. Used only as a supporting signal (long + high-entropy param), never
    alone, since it also fires on legitimate opaque tokens (session ids etc.)."""
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_emails(text: str) -> list[str]:
    return _EMAIL_RE.findall(text or "")


def extract_urls(text: str) -> list[str]:
    return _URL_RE.findall(text or "")


def looks_textual(s: str, min_len: int = 3) -> bool:
    """Reject decode results that are just decoding noise (binary, mostly
    unprintable) so we don't chase every byte string as a "decoded payload"."""
    if not s or len(s) < min_len:
        return False
    printable = sum(1 for ch in s if ch.isprintable())
    return printable / len(s) >= 0.9


def _decode_percent(s: str) -> str | None:
    if "%" not in s:
        return None
    try:
        d = unquote(s, errors="strict")
    except Exception:
        return None
    return d if d != s and looks_textual(d) else None


def _decode_base64(s: str) -> str | None:
    candidate = s.strip()
    if len(candidate) < 8 or not re.fullmatch(r"[A-Za-z0-9+/\-_=]+", candidate):
        return None
    padded = candidate + "=" * (-len(candidate) % 4)
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            raw = decoder(padded, validate=False)
        except Exception:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if looks_textual(text):
            return text
    return None


def _decode_hex(s: str) -> str | None:
    candidate = s.strip()
    if len(candidate) < 8 or len(candidate) % 2 != 0 or not _HEX_RE.match(candidate):
        return None
    try:
        text = bytes.fromhex(candidate).decode("utf-8")
    except Exception:
        return None
    return text if looks_textual(text) else None


_DECODERS: tuple[tuple[str, "callable[[str], str | None]"], ...] = (
    ("percent", _decode_percent),
    ("base64", _decode_base64),
    ("hex", _decode_hex),
)


def decode_layers(s: str, depth: int = DECODE_DEPTH) -> list[tuple[str, str]]:
    """All the strings we're willing to compare against context: ``s`` itself,
    plus every percent/base64/hex decoding of it, decoded again once more
    ("nested once") so a base64-of-percent-encoded payload is still caught.
    Depth-limited and dedup'd so this can't blow up on adversarial input."""
    out: list[tuple[str, str]] = [("raw", s)]
    seen = {s}
    frontier = [s]
    for _ in range(depth):
        next_frontier: list[str] = []
        for val in frontier:
            for name, fn in _DECODERS:
                try:
                    decoded = fn(val)
                except Exception:
                    decoded = None
                if decoded and decoded not in seen:
                    seen.add(decoded)
                    out.append((name, decoded))
                    next_frontier.append(decoded)
        frontier = next_frontier
        if not frontier:
            break
    return out


def hostname_of(url: str) -> str | None:
    from urllib.parse import urlparse

    try:
        host = urlparse(url).hostname
    except Exception:
        return None
    return host.lower() if host else None


def host_allowed(hostname: str | None, allow_hosts: Iterable[str]) -> bool:
    if not hostname:
        return False
    allow = {h.lower() for h in allow_hosts}
    return any(hostname == h or hostname.endswith("." + h) for h in allow)


def registrable_domain(hostname: str | None) -> str | None:
    """Best-effort eTLD+1, offline. Evidence-only (e.g. "this url's domain is
    exfil.example.co.uk") -- never used as the allow/deny check itself, so a
    PSL edge case can't accidentally widen trust."""
    if not hostname:
        return None
    try:
        ext = _EXTRACTOR(hostname)
    except Exception:
        return None
    if not ext.suffix:
        return ext.domain or None
    return f"{ext.domain}.{ext.suffix}" if ext.domain else None


def chunk_shingle_index(context: Sequence[Chunk]) -> dict[str, frozenset[tuple[str, ...]]]:
    """chunk id -> its shingle set, for the whole context in one pass."""
    return {c.id: shingles(c.text) for c in context}


def matching_chunk_ids(
    candidate_shingles: frozenset[tuple[str, ...]],
    index: dict[str, frozenset[tuple[str, ...]]],
) -> list[str]:
    return [cid for cid, sset in index.items() if candidate_shingles & sset]
