"""Stage 1A, step 1: find content a human reading a rendered document would never
see, and separate it from what they would.

EchoLeak-style attacks (CVE-2025-32711) and the LLMail-Inject corpus both rely on
the same trick: a human skimming an email sees nothing unusual, but the text handed
to the LLM contains an instruction. The instruction can be hidden by several
independent *mechanisms* (CSS, HTML comments, zero-width Unicode, base64, markdown
that never renders inline, fake conversation-turn syntax). This module's only job
is mechanical extraction of each mechanism -- no judgement about intent. Judgement
(is the hidden or visible text actually an attack?) belongs to ``directive.py``.

Kept deliberately dumb and structural (no keyword lists here) so it cannot be
evaded by rewording -- it targets the *rendering gap* itself, not phrasing.
"""

from __future__ import annotations

import base64
import re
import string
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment

__all__ = ["HiddenSegment", "HiddenTextReport", "extract"]


@dataclass(frozen=True)
class HiddenSegment:
    """One piece of content a human would not see when reading the rendered
    document normally. ``kind`` names the mechanism that hid it."""

    kind: str
    text: str


@dataclass(frozen=True)
class HiddenTextReport:
    visible_text: str
    hidden_segments: tuple[HiddenSegment, ...] = ()


# ---------------------------------------------------------------------------
# Zero-width / invisible Unicode
# ---------------------------------------------------------------------------

# U+200B ZERO WIDTH SPACE, U+200C ZWNJ, U+200D ZWJ, U+2060 WORD JOINER,
# U+FEFF ZERO WIDTH NO-BREAK SPACE (a.k.a. BOM used mid-text).
_ZERO_WIDTH_RE = re.compile("[​-‍⁠﻿]+")

# Unicode "tag characters" U+E0000-U+E007F. These are fully invisible in every
# renderer but map 1:1 onto ASCII (U+E0020..U+E007E -> 0x20..0x7E), so an
# attacker can smuggle an entire hidden ASCII instruction through them
# ("ASCII smuggling"). We decode, we don't just flag.
_TAG_CHAR_RE = re.compile("[\U000e0000-\U000e007f]+")


def _decode_tag_chars(run: str) -> str:
    out = []
    for ch in run:
        cp = ord(ch) - 0xE0000
        if 0x20 <= cp <= 0x7E:
            out.append(chr(cp))
    return "".join(out)


def _strip_invisible_unicode(text: str) -> tuple[str, list[HiddenSegment]]:
    segments: list[HiddenSegment] = []

    def _tag_sub(m: re.Match) -> str:
        decoded = _decode_tag_chars(m.group(0)).strip()
        if decoded:
            segments.append(HiddenSegment("invisible_tag_chars", decoded))
        return ""

    text = _TAG_CHAR_RE.sub(_tag_sub, text)

    def _zw_sub(m: re.Match) -> str:
        run = m.group(0)
        codepoints = " ".join(f"U+{ord(c):04X}" for c in run)
        segments.append(HiddenSegment("zero_width", codepoints))
        return ""

    text = _ZERO_WIDTH_RE.sub(_zw_sub, text)
    return text, segments


# ---------------------------------------------------------------------------
# Base64 blobs
# ---------------------------------------------------------------------------

# Long enough to not catch incidental short tokens (hashes, ids); no nested
# quantifiers, so no catastrophic-backtracking risk on huge input.
_BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/=])(?:[A-Za-z0-9+/]{4}){10,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?(?![A-Za-z0-9+/=])")

_PRINTABLE = set(string.printable) - set("\x0b\x0c")


def _looks_printable(s: str) -> bool:
    if not s.strip():
        return False
    printable_count = sum(1 for c in s if c in _PRINTABLE)
    return printable_count / len(s) >= 0.95


def _strip_base64(text: str) -> tuple[str, list[HiddenSegment]]:
    segments: list[HiddenSegment] = []

    def _sub(m: re.Match) -> str:
        blob = m.group(0)
        try:
            decoded = base64.b64decode(blob, validate=True)
        except Exception:
            return blob  # not actually valid base64 -- leave it as visible text
        try:
            decoded_text = decoded.decode("utf-8")
        except Exception:
            return blob  # decodes to bytes, not text -- nothing readable to report
        if not _looks_printable(decoded_text):
            return blob
        segments.append(HiddenSegment("base64", decoded_text))
        return ""

    text = _BASE64_RE.sub(_sub, text)
    return text, segments


# ---------------------------------------------------------------------------
# Markdown reference-style link definitions
# ---------------------------------------------------------------------------

# [label]: https://evil.example/x "title"  -- never rendered inline, only its
# *use* ([label]) is; the URL/title live entirely outside what a reader sees.
_MD_REF_LINK_RE = re.compile(
    r"^[ \t]{0,3}\[([^\]\n]+)\]:[ \t]*(\S+)(?:[ \t]+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^)\n]*\)))?[ \t]*$",
    re.MULTILINE,
)


def _strip_markdown_ref_links(text: str) -> tuple[str, list[HiddenSegment]]:
    segments: list[HiddenSegment] = []

    def _sub(m: re.Match) -> str:
        segments.append(HiddenSegment("markdown_ref_link", f"{m.group(1)} -> {m.group(2)}"))
        return ""

    text = _MD_REF_LINK_RE.sub(_sub, text)
    return text, segments


# ---------------------------------------------------------------------------
# Fake conversation / turn markers
# ---------------------------------------------------------------------------

# These stay VISIBLE as raw characters if you read the plain-text body -- they
# are not hidden by any rendering mechanism. They are extracted here anyway
# (as evidence for directive.py) because that is where every other structural
# signal for "this document is trying to look like a conversation" lives, and
# because LLMail-Inject's winning submissions lean on exactly this family:
# fabricated turn boundaries meant to make the model believe a new,
# instruction-bearing "user turn" has begun.
_TURN_MARKER_RE = re.compile(
    r"""
    <\|/?\s*(?:im_start|im_end|user|assistant|system)\s*\|>          # <|im_start|>, <|user|>
    | <{1,2}-{0,2}\s*(?:begin|end|start)\s+(?:user|assistant|system)\s*-{0,2}\s*>{1,2}   # <-- BEGIN USER -->
    | \[\s*(?:begin|end|start)\s+(?:user|assistant|system)\s*\]      # [END USER]
    | ^[ \t]*\#{2,4}[ \t]*(?:user|assistant|system)[ \t]*:            # ### Assistant:
    | <<\s*/?(?:end|start|begin)?[_ ]?(?:context|prompt|user_query|user_session|email_context|processing)\s*>>
    | \b(?:NEW_USER_SESSION|END_USER_SESSION|EXECUTE_USERQUERY|END_QUERY|END_CONTEXT|END_PROCESSING|START_PROMPT|END_PROMPT|NEXT_INSTRUCTION)\b
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)


# Turn markers (e.g. "<<end_context>>") are literal characters a human reading
# the raw body would see -- but an HTML parser sees "<<...>>" as tag-like
# gibberish and silently eats the middle of it (BeautifulSoup turns
# "<<end_context>>" into "<>"). So before any HTML parsing we swap each match
# for a placeholder built from Private Use Area code points (never produced by
# real text, never touched by the zero-width/tag-char/base64/markdown steps),
# run the rest of the pipeline, then restore the original text verbatim.
_PLACEHOLDER_OPEN = ""
_PLACEHOLDER_CLOSE = ""


def _protect_turn_markers(text: str) -> tuple[str, dict[str, str], list[HiddenSegment]]:
    segments: list[HiddenSegment] = []
    restore: dict[str, str] = {}
    out: list[str] = []
    last = 0
    for i, m in enumerate(_TURN_MARKER_RE.finditer(text)):
        out.append(text[last:m.start()])
        placeholder = f"{_PLACEHOLDER_OPEN}TM{i}{_PLACEHOLDER_CLOSE}"
        out.append(placeholder)
        restore[placeholder] = m.group(0)
        segments.append(HiddenSegment("fake_turn_marker", m.group(0)))
        last = m.end()
    out.append(text[last:])
    return "".join(out), restore, segments


# ---------------------------------------------------------------------------
# HTML: comments, CSS-hidden elements, alt/title attributes
# ---------------------------------------------------------------------------

_HIDDEN_DISPLAY = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_HIDDEN_VISIBILITY = re.compile(r"visibility\s*:\s*hidden", re.IGNORECASE)
_HIDDEN_OPACITY = re.compile(r"opacity\s*:\s*0(?:\.0+)?\b|opacity\s*:\s*0%", re.IGNORECASE)
_HIDDEN_FONT_SIZE = re.compile(r"font-size\s*:\s*0(?:\.0*)?(?:px|pt|em|rem|%)?\b", re.IGNORECASE)
_HIDDEN_OFFSCREEN = re.compile(
    r"(?:position\s*:\s*(?:absolute|fixed)).{0,80}?(?:left|top)\s*:\s*-\d{3,}px"
    r"|(?:left|top)\s*:\s*-\d{3,}px.{0,80}?position\s*:\s*(?:absolute|fixed)"
    r"|text-indent\s*:\s*-\d{3,}px",
    re.IGNORECASE | re.DOTALL,
)

_WHITE_NAMES = {"white", "#fff", "#ffffff", "rgb(255,255,255)", "rgb(255, 255, 255)"}


def _normalize_color(value: str) -> str:
    return re.sub(r"\s+", "", value.strip().lower())


def _style_props(style: str) -> dict[str, str]:
    props: dict[str, str] = {}
    for decl in style.split(";"):
        if ":" not in decl:
            continue
        k, _, v = decl.partition(":")
        props[k.strip().lower()] = v.strip()
    return props


def _is_hidden_style(style: str) -> bool:
    if not style:
        return False
    if _HIDDEN_DISPLAY.search(style) or _HIDDEN_VISIBILITY.search(style):
        return True
    if _HIDDEN_OPACITY.search(style) or _HIDDEN_FONT_SIZE.search(style):
        return True
    if _HIDDEN_OFFSCREEN.search(style):
        return True
    props = _style_props(style)
    color = props.get("color")
    bg = props.get("background-color") or props.get("background")
    if color:
        c_norm = _normalize_color(color)
        if bg and _normalize_color(bg) == c_norm:
            return True  # text colour == its own background: invisible regardless of hue
        if c_norm in _WHITE_NAMES and not bg:
            return True  # white text, no background override -> invisible on a normal white page
    return False


def _extract_html(raw: str) -> tuple[str, list[HiddenSegment]]:
    segments: list[HiddenSegment] = []
    soup = BeautifulSoup(raw, "html.parser")

    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        text = str(comment).strip()
        if text:
            segments.append(HiddenSegment("html_comment", text))
        comment.extract()

    for tag in soup.find_all(True):
        style = tag.get("style") if hasattr(tag, "get") else None
        is_hidden_attr = tag.has_attr("hidden") if hasattr(tag, "has_attr") else False
        if is_hidden_attr or (style and _is_hidden_style(style)):
            text = tag.get_text(" ", strip=True)
            if text:
                segments.append(HiddenSegment("css_hidden", text))
            tag.decompose()

    for tag in soup.find_all(True):
        for attr in ("alt", "title"):
            if hasattr(tag, "get"):
                value = tag.get(attr)
                if value and value.strip():
                    kind = "alt_text" if attr == "alt" else "title_attr"
                    segments.append(HiddenSegment(kind, value.strip()))

    visible = soup.get_text(" ")
    return visible, segments


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _coerce_text(raw) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if raw is None:
        return ""
    return str(raw)


def extract(raw: str) -> HiddenTextReport:
    """Split ``raw`` into what a human reader would see (``visible_text``) and
    every mechanically-hidden segment found along the way.

    Never raises: any single step that fails is skipped (its content simply
    stays in ``visible_text`` unextracted) rather than aborting the whole scan,
    because callers (``directive.py``) must fail *closed* -- a scan that raises
    would otherwise let unscanned content through.
    """

    text = _coerce_text(raw)
    segments: list[HiddenSegment] = []

    turn_segments: list[HiddenSegment] = []
    restore_map: dict[str, str] = {}
    try:
        text, restore_map, turn_segments = _protect_turn_markers(text)
    except Exception:
        pass

    try:
        text, html_segments = _extract_html(text)
        segments.extend(html_segments)
    except Exception:
        pass

    try:
        text, unicode_segments = _strip_invisible_unicode(text)
        segments.extend(unicode_segments)
    except Exception:
        pass

    try:
        text, b64_segments = _strip_base64(text)
        segments.extend(b64_segments)
    except Exception:
        pass

    try:
        text, md_segments = _strip_markdown_ref_links(text)
        segments.extend(md_segments)
    except Exception:
        pass

    try:
        for placeholder, original in restore_map.items():
            text = text.replace(placeholder, original)
    except Exception:
        pass
    segments.extend(turn_segments)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return HiddenTextReport(visible_text=text, hidden_segments=tuple(segments))
