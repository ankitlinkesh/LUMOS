"""Stage 1A, step 2: decide whether a document is trying to give the model an
instruction it shouldn't take -- an EchoLeak/LLMail-Inject-style exfiltration
directive hidden (or hiding in plain sight) inside retrieved content.

Deliberately NOT a keyword-substring detector (see the author's earlier
``eva-agent/backend/eva/threat_defense/injection_detector.py`` for what that
looks like, and why LLMail-Inject beat it: its winning submissions never say
"ignore previous instructions"). Instead this scores *structural* signals and
combines them:

  - hidden_text.extract() runs first, so hidden content and visible content
    are scored separately -- a hidden segment that itself reads as an
    instruction is far more suspicious than either hidden content alone or
    an instruction-shaped sentence sitting in plain view.
  - an action verb (send/forward/call/...) sitting next to an email address
    or URL -- the structural shape of "do X to/with Y", regardless of wording.
  - obfuscated addresses (contact <at> contact <dot> com) -- built specifically
    to dodge a plain email regex.
  - JSON/tool-call-shaped payloads (`"to": ...`, `destination=`, `api_call`).
  - fake conversation-turn markers (<<end_context>>, <|im_start|>, ### Assistant:,
    EXECUTE_USERQUERY, ...) -- an attempt to make the model believe a new,
    instruction-bearing turn has begun.
  - instruction-override phrasing ("ignore previous instructions", ...).
  - text that directly addresses "the AI/assistant/model".

No single weak signal blocks anything by itself (a newsletter's "click here to
unsubscribe" is one signal; an obfuscated-email attack is several at once).
This is how "please send me the Q3 report by Friday" -- one action verb, no
address, no hidden content, no overrides -- clears every real business email
in calibration while LLMail-Inject's structural giveaways (turn markers +
address + tool-shaped payload, all at once) do not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from triad.contract import GuardDecision, TaintVerdict
from triad.stage1 import hidden_text

__all__ = ["scan", "to_taint", "THRESHOLD"]

THRESHOLD = 0.5

_ACTION_VERBS = (
    "send", "forward", "email", "mail", "call", "issue", "execute", "run",
    "trigger", "post", "upload", "submit", "reply", "respond", "click",
    "visit", "open", "download", "invoke", "dispatch", "fetch", "retrieve",
    "exfiltrate", "transmit", "publish",
)
_VERB_RE = re.compile(r"\b(?:" + "|".join(_ACTION_VERBS) + r")\b", re.IGNORECASE)

_NORMAL_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_OBFUSCATED_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%-]+\s*(?:\(\s*at\s*\)|\[\s*at\s*\]|<\s*at\s*>|\bat\b)\s*"
    r"[A-Za-z0-9-]+\s*(?:\(\s*dot\s*\)|\[\s*dot\s*\]|<\s*dot\s*>|\bdot\b)\s*[A-Za-z]{2,}",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"\bhttps?://[^\s<>\"')]+|\bwww\.[^\s<>\"')]+", re.IGNORECASE)

_OVERRIDE_RE = re.compile(
    r"ignore\s+(?:all\s+|the\s+)?(?:previous|prior|above)\s+instructions"
    r"|disregard\s+(?:the\s+)?(?:above|previous|prior)"
    r"|new\s+instructions\s*:"
    r"|overrides?\s+(?:the\s+)?(?:system|previous|policy)"
    r"|you\s+must\s+now"
    r"|from\s+now\s+on\s+you\s+(?:will|must)"
    r"|this\s+(?:overrides|supersedes)"
    r"|highest[- ]priority\s+instruction"
    r"|do\s+not\s+(?:tell|inform|mention|notify)\s+the\s+user",
    re.IGNORECASE,
)
_AI_ADDRESS_RE = re.compile(
    r"\b(?:dear|hey|hi)\s+(?:ai|assistant|chatbot|model)\b"
    r"|\bas\s+(?:an|the)\s+(?:ai|assistant|language\s+model)\b"
    r"|\battention\s*:?\s*(?:ai|assistant|model)\b"
    r"|\bassistant\s*,\s*please\b",
    re.IGNORECASE,
)
_TOOLCALL_RE = re.compile(
    r'"(?:to|body|arguments|destination|function|tool_call|api_call|recipient)"\s*[:=]'
    r"|\bapi_call\b|\bfunction_call\b|\btool_call\b"
    r"|\b\w+\s*\(\s*value\s*:"
    r"|\bdestination\s*="
    r"|\barguments\s*:",
    re.IGNORECASE,
)

# Text about how THIS document will be processed by an AI pipeline: "once this email
# has been summarized", "as part of your ... task", "the tool you have access to".
# Deliberately excludes human phrasing like "when you read/receive this", which is
# ordinary business email. Alone it never blocks; it tips an action+address over.
_PIPELINE_META_RE = re.compile(
    r"\b(?:once|after|when|while|if)\b[^.\n]{0,80}\b(?:summari[sz](?:e|ed|es|ing|ation)|processed|processing|parsed|parsing)\b"
    r"|\bthis\s+(?:email|message|e-mail)\s+(?:has\s+been|is\s+being|is|was|gets)\s+(?:summari[sz]ed|processed|parsed|analy[sz]ed)"
    r"|\bas\s+part\s+of\s+your\b"
    r"|\byour\s+(?:\w+\s+){0,3}task\b"
    r"|\b(?:tool|api|function)s?\s+(?:call|you\s+have\s+access\s+to)\b|\byou\s+have\s+access\s+to\b|\bappropriate\s+api\b"
    r"|\byou\s+are\s+(?:expected|required|instructed)\s+to\b"
    r"|\bfor\s+summari[sz]ation\b|\bemails?\s+for\s+summari[sz]",
    re.IGNORECASE,
)

_PROXIMITY_WINDOW = 80


def _normalize(text: str) -> str:
    """Undo cheap word-gluing obfuscation seen in LLMail-Inject (`Also,`in`addition`)
    so the regexes see ordinary spacing. Hidden-text extraction has already run."""
    return text.replace("`", " ")


@dataclass(frozen=True)
class _Signals:
    verb_address_hit: bool = False
    obfuscated_address: bool = False
    json_toolcall: bool = False
    override: bool = False
    ai_address: bool = False
    pipeline_meta: bool = False

    def any_instructional(self) -> bool:
        return (self.verb_address_hit or self.obfuscated_address or self.json_toolcall
                or self.override or self.ai_address or self.pipeline_meta)


def _has_verb_near(text: str, start: int, end: int) -> bool:
    lo = max(0, start - _PROXIMITY_WINDOW)
    hi = min(len(text), end + _PROXIMITY_WINDOW)
    return bool(_VERB_RE.search(text[lo:hi]))


def _action_signals(text: str) -> _Signals:
    if not text:
        return _Signals()
    text = _normalize(text)

    verb_address_hit = False
    for pattern in (_NORMAL_EMAIL_RE, _URL_RE):
        for m in pattern.finditer(text):
            if _has_verb_near(text, m.start(), m.end()):
                verb_address_hit = True
                break
        if verb_address_hit:
            break

    obfuscated = bool(_OBFUSCATED_EMAIL_RE.search(text))
    if obfuscated and not verb_address_hit:
        for m in _OBFUSCATED_EMAIL_RE.finditer(text):
            if _has_verb_near(text, m.start(), m.end()):
                verb_address_hit = True
                break

    return _Signals(
        verb_address_hit=verb_address_hit,
        obfuscated_address=obfuscated,
        json_toolcall=bool(_TOOLCALL_RE.search(text)),
        override=bool(_OVERRIDE_RE.search(text)),
        ai_address=bool(_AI_ADDRESS_RE.search(text)),
        pipeline_meta=bool(_PIPELINE_META_RE.search(text)),
    )


def _extract_raw(text_or_chunk) -> str:
    raw = getattr(text_or_chunk, "text", text_or_chunk)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if raw is None:
        return ""
    return str(raw)


def scan(text_or_chunk) -> GuardDecision:
    """Score a document (raw text, or anything with a ``.text`` attribute,
    e.g. a ``Chunk``) at ingestion. Never raises: any internal failure is
    treated as suspicious and blocked (fail closed), never as clean content
    let through unscanned.
    """

    try:
        raw = _extract_raw(text_or_chunk)
        report = hidden_text.extract(raw)

        visible_sig = _action_signals(report.visible_text)

        hidden_other = tuple(s for s in report.hidden_segments if s.kind != "fake_turn_marker")
        turn_markers = tuple(s for s in report.hidden_segments if s.kind == "fake_turn_marker")
        hidden_blob = "\n".join(s.text for s in hidden_other)
        hidden_sig = _action_signals(hidden_blob)

        score = 0.0
        fired: set[str] = set()
        reasons: list[str] = []

        if turn_markers:
            # Specific tokens (<|im_start|>, <-- BEGIN USER -->, [END USER], ...): 0 hits in
            # 703 real benign emails, so one is enough to quarantine on its own.
            score += 0.5
            fired.add("fake_turn_marker")
            reasons.append(
                f"fake conversation/turn markers found ({len(turn_markers)}), "
                f"e.g. {turn_markers[0].text!r}"
            )

        if hidden_sig.any_instructional():
            score += 0.4
            fired.add("hidden_content_with_instruction")
            reasons.append("content hidden from a human reader contains an actionable instruction")
        elif hidden_other:
            bonus = min(0.2, 0.07 * len(hidden_other))
            score += bonus
            fired.add("hidden_content_present")
            kinds = sorted({s.kind for s in hidden_other})
            reasons.append(f"document contains {len(hidden_other)} segment(s) hidden from a human reader: {kinds}")

        if visible_sig.verb_address_hit or hidden_sig.verb_address_hit:
            score += 0.3
            fired.add("action_verb_near_address")
            reasons.append("an action verb (send/forward/call/...) appears next to an email address or URL")

        if visible_sig.obfuscated_address or hidden_sig.obfuscated_address:
            score += 0.2
            fired.add("obfuscated_address")
            reasons.append("an obfuscated email address (e.g. 'contact <at> contact <dot> com') was found")

        if visible_sig.json_toolcall or hidden_sig.json_toolcall:
            score += 0.2
            fired.add("tool_call_shaped_payload")
            reasons.append("a JSON- or tool-call-shaped payload (to/body/arguments/api_call/destination=) was found")

        if visible_sig.override or hidden_sig.override:
            score += 0.25
            fired.add("instruction_override_phrasing")
            reasons.append("instruction-override phrasing ('ignore previous instructions', ...) was found")

        if visible_sig.pipeline_meta or hidden_sig.pipeline_meta:
            score += 0.25
            fired.add("pipeline_meta_reference")
            reasons.append("text refers to how an AI pipeline will process this document ('once this email is summarized', 'your task', 'the tool you have access to')")

        if visible_sig.ai_address or hidden_sig.ai_address:
            score += 0.15
            fired.add("ai_addressed_imperative")
            reasons.append("text directly addresses the assistant/AI/model")

        score = min(score, 1.0)

        evidence = {
            "score": score,
            "signals": tuple(sorted(fired)),
            "hidden_segment_kinds": tuple(sorted({s.kind for s in report.hidden_segments})),
            "hidden_segment_count": len(report.hidden_segments),
        }

        if score >= THRESHOLD:
            return GuardDecision.block("ingest", *reasons, evidence=evidence)
        return GuardDecision.ok("ingest", evidence=evidence)

    except Exception as exc:  # fail closed: an error scanning is treated as suspicious
        return GuardDecision.block(
            "ingest",
            f"internal error while scanning: {type(exc).__name__}: {exc}",
            evidence={"score": 1.0, "signals": ("internal_error",), "internal_error": True},
        )


def to_taint(decision: GuardDecision) -> TaintVerdict:
    """Project a Stage 1A ``GuardDecision`` onto the shared ``TaintVerdict``
    other stages read. Retrieved content is always untrusted; ``quarantined``
    is the only thing that changes."""

    blocked = not decision.allow
    score = float(decision.evidence.get("score", 1.0 if blocked else 0.0))
    flags = tuple(decision.evidence.get("signals", ()))
    return TaintVerdict(
        untrusted=True,
        quarantined=blocked,
        flags=flags,
        score=score,
        reasons=decision.reasons,
    )
