"""Groq free-tier rate limits, hardcoded from Groq's official page.

Source:  https://console.groq.com/docs/rate-limits   (fetched 2026-09-10)
Scope:   "Rate limits apply at the organization level, not individual users."
         Several keys from the SAME organization share ONE quota.
Exceed:  HTTP 429 Too Many Requests, with a ``retry-after`` header.
Headers: x-ratelimit-{limit,remaining,reset}-{requests,tokens}

These are client-side ceilings so we stop before Groq has to stop us. The server's
x-ratelimit-* headers are the live authority and override these when stricter.
If Groq changes its limits, update this table and the fetch date together.

Policy note: Groq's Acceptable Use Policy prohibits exceeding published limits
"by registering multiple accounts or orchestrating usage between multiple
organizations" (https://console.groq.com/docs/legal/ai-policy).
"""

from __future__ import annotations

from dataclasses import dataclass

SOURCE_URL = "https://console.groq.com/docs/rate-limits"
FETCHED_ON = "2026-09-10"
LIMIT_SCOPE = "organization"


@dataclass(frozen=True)
class ModelLimits:
    rpm: int | None   # requests per minute
    rpd: int | None   # requests per day
    tpm: int | None   # tokens per minute (prompt + completion, incl. reasoning tokens)
    tpd: int | None   # tokens per day; None = no limit listed


# Text models only; audio/TTS models from the same table are omitted (unused).
# "K" on Groq's page = 1,000.
FREE_TIER: dict[str, ModelLimits] = {
    "openai/gpt-oss-120b":                  ModelLimits(rpm=30, rpd=1_000,  tpm=8_000,  tpd=200_000),
    "openai/gpt-oss-20b":                   ModelLimits(rpm=30, rpd=1_000,  tpm=8_000,  tpd=200_000),
    "openai/gpt-oss-safeguard-20b":         ModelLimits(rpm=30, rpd=1_000,  tpm=8_000,  tpd=200_000),
    "qwen/qwen3.6-27b":                     ModelLimits(rpm=30, rpd=1_000,  tpm=8_000,  tpd=200_000),
    "qwen/qwen3.8-27b":                     ModelLimits(rpm=30, rpd=1_000,  tpm=8_000,  tpd=200_000),
    "groq/compound":                        ModelLimits(rpm=30, rpd=250,    tpm=70_000, tpd=None),
    "groq/compound-mini":                   ModelLimits(rpm=30, rpd=250,    tpm=70_000, tpd=None),
    "meta-llama/llama-prompt-guard-2-22m":  ModelLimits(rpm=30, rpd=14_400, tpm=15_000, tpd=500_000),
    "meta-llama/llama-prompt-guard-2-86m":  ModelLimits(rpm=30, rpd=14_400, tpm=15_000, tpd=500_000),
}

# The answer generator for the RAG pipeline, and the Stage 1A baseline classifier
# we measure against (Prompt Guard 2 is served by Groq on the free tier).
DEFAULT_GENERATOR = "openai/gpt-oss-20b"
BASELINE_INJECTION_CLASSIFIER = "meta-llama/llama-prompt-guard-2-86m"


def limits_for(model: str) -> ModelLimits:
    try:
        return FREE_TIER[model]
    except KeyError:
        raise KeyError(f"no free-tier limits recorded for {model!r}; add it from {SOURCE_URL}") from None
