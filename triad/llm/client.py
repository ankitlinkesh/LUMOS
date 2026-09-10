"""Groq chat-completion client: cache-first, rate-limited, multi-key, loud on failure.

Design principle (see module docstring in cache.py and limiter.py): an empty
answer would be silently scored as a "successful defense" by the eval harness,
so this module never returns one. Every failure mode either raises or falls
through to another key -- it never manufactures a blank ``LLMResponse``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import httpx

from triad import config
from triad.llm import limits
from triad.llm.cache import CacheEntry, DiskCache, cache_key
from triad.llm.keys import ApiKey
from triad.llm.limiter import Clock, RateLimiter, RealClock, parse_retry_after

logger = logging.getLogger(__name__)

_MAX_5XX_RETRIES = 2
_5XX_BACKOFF_BASE_S = 1.0


class AllKeysExhausted(RuntimeError):
    """No key/bucket had capacity within ``max_wait_s``."""


class EmptyResponse(RuntimeError):
    """Groq returned a completion with no usable text. Never cached."""


class _NoCapacityNow(Exception):
    """Internal signal: this key just lost capacity (429, marked dead, or a
    losing reservation race). Caught by chat()'s outer loop, which re-checks
    capacity across ALL keys and either picks another one, waits (bounded by
    max_wait_s), or raises AllKeysExhausted -- the single place that decision
    is made, so a 429 with a short retry-after is retried within the budget
    instead of failing immediately."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: Mapping[str, Any]
    cached: bool
    key_label: str | None  # masked; None only for a cache hit whose original call predates this field


def _estimate_tokens(messages: Sequence[Mapping[str, str]], max_tokens: int) -> int:
    """Cheap pre-flight estimate: prompt chars/4 (a common English-text rule of
    thumb) plus the requested completion budget. Reconciled against real
    ``usage`` after the call, so a rough estimate here is fine -- it only needs
    to keep us from ever knowingly sending a call that would already blow TPM."""
    chars = sum(len(m.get("content", "")) for m in messages)
    return (chars // 4) + max_tokens


def _extract_text(body: Mapping[str, Any]) -> str | None:
    choices = body.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        return None
    if not content.strip():
        return None
    return content


class GroqClient:
    def __init__(
        self,
        keys: Sequence[ApiKey],
        limiter: RateLimiter,
        cache: DiskCache,
        transport: httpx.BaseTransport | None = None,
        clock: Clock | None = None,
        base_url: str = config.GROQ_BASE_URL,
        max_wait_s: float = 60.0,
    ):
        if not keys:
            raise ValueError("GroqClient needs at least one key")
        self._keys = list(keys)
        self._rr_index = 0
        self.limiter = limiter
        self.cache = cache
        self.clock = clock or RealClock()
        self.base_url = base_url.rstrip("/")
        self.max_wait_s = max_wait_s
        self._dead: set[str] = set()  # masked values of keys marked dead by 401/403
        # httpx.Client is created lazily so importing/constructing this class never
        # opens a socket (doctor.py and tests must be able to build one for free).
        self._transport = transport
        self._http: httpx.Client | None = None

    def _http_client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(transport=self._transport, timeout=30.0)
        return self._http

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    # -- key selection -------------------------------------------------

    def _candidate_keys(self) -> list[ApiKey]:
        return [k for k in self._keys if str(k) not in self._dead]

    def _next_key_with_capacity(self, model: str, est_tokens: int) -> ApiKey | None:
        candidates = self._candidate_keys()
        if not candidates:
            return None
        n = len(candidates)
        for i in range(n):
            idx = (self._rr_index + i) % n
            key = candidates[idx]
            if self.limiter.has_capacity(key.bucket, model, est_tokens):
                self._rr_index = (idx + 1) % n
                return key
        return None

    def _earliest_capacity_wait(self, model: str, est_tokens: int) -> float | None:
        """Min wait across all live buckets, or None if every bucket is done for
        the day (so waiting longer inside this process cannot help)."""
        waits: list[float] = []
        any_today = False
        for key in self._candidate_keys():
            w = self.limiter.time_until_capacity(key.bucket, model, est_tokens)
            if w is not None:
                any_today = True
                waits.append(w)
        if not any_today:
            return None
        return min(waits) if waits else 0.0

    # -- public API ------------------------------------------------------

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        principal: str,
        scope: Sequence[str] = (),
        model: str = limits.DEFAULT_GENERATOR,
        max_tokens: int = 300,
        temperature: float = 0,
        **params: Any,
    ) -> LLMResponse:
        if not principal:
            raise ValueError("chat() requires a principal (tenant) for cache isolation")

        resolved_params: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature, **params}
        if model.startswith("openai/gpt-oss-") and "reasoning_effort" not in resolved_params:
            resolved_params["reasoning_effort"] = "low"  # reasoning tokens count against TPM

        key = cache_key(principal=principal, scope=scope, model=model, messages=messages, params=resolved_params)
        hit = self.cache.get(key)
        if hit is not None:
            return LLMResponse(text=hit.text, model=hit.model, usage=hit.usage, cached=True, key_label=hit.key_label)

        est_tokens = _estimate_tokens(messages, max_tokens)
        deadline = self.clock.now() + self.max_wait_s

        while True:
            if not self._candidate_keys():
                raise AllKeysExhausted(
                    f"all keys exhausted for {model}: no valid keys remain (all rejected as invalid by Groq, 401/403)"
                )

            api_key = self._next_key_with_capacity(model, est_tokens)
            if api_key is None:
                wait = self._earliest_capacity_wait(model, est_tokens)
                now = self.clock.now()
                if wait is None:
                    raise AllKeysExhausted(
                        f"all keys exhausted for {model}: today's ceiling is spent on every bucket; try again tomorrow (UTC)"
                    )
                if now + wait > deadline:
                    raise AllKeysExhausted(
                        f"all keys exhausted for {model}: no capacity within {self.max_wait_s}s "
                        f"(next capacity frees up in {wait:.1f}s)"
                    )
                self.clock.sleep(min(wait, deadline - now))
                continue

            try:
                return self._attempt_call(api_key, model, messages, resolved_params, est_tokens, key)
            except _NoCapacityNow:
                continue  # this key just lost capacity; loop re-evaluates ALL keys, honoring max_wait_s

    # -- HTTP -------------------------------------------------------------

    def _attempt_call(
        self,
        api_key: ApiKey,
        model: str,
        messages: Sequence[Mapping[str, str]],
        params: Mapping[str, Any],
        est_tokens: int,
        cache_key_: str,
    ) -> LLMResponse:
        """One key, one logical attempt. 5xx/network errors get a short bounded
        retry ON THIS SAME KEY (they're not a capacity problem); 429/401/403
        raise _NoCapacityNow so the caller's outer loop re-picks across all keys."""
        attempts_5xx = 0

        while True:
            reservation = self.limiter.reserve(api_key.bucket, model, est_tokens)
            if reservation is None:
                # Capacity vanished between selection and reservation (e.g. a
                # concurrent caller in this process raced us for it).
                raise _NoCapacityNow()

            try:
                http_response = self._http_client().post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key.value}", "Content-Type": "application/json"},
                    json={"model": model, "messages": list(messages), **params},
                )
            except httpx.HTTPError as exc:
                self.limiter.reconcile(reservation, actual_tokens=None)
                attempts_5xx += 1
                logger.warning("network error calling Groq with key %s: %s", api_key, exc)
                if attempts_5xx > _MAX_5XX_RETRIES:
                    raise _NoCapacityNow() from exc  # give another key a turn
                self.clock.sleep(_5XX_BACKOFF_BASE_S * (2 ** (attempts_5xx - 1)))
                continue

            self.limiter.tighten_from_headers(api_key.bucket, model, http_response.headers)

            if http_response.status_code == 429:
                self.limiter.reconcile(reservation, actual_tokens=None)
                retry_after = parse_retry_after(http_response.headers.get("retry-after")) or 5.0
                self.limiter.cooldown(api_key.bucket, model, retry_after)
                logger.warning("429 from Groq for key %s; cooling down %.1fs", api_key, retry_after)
                raise _NoCapacityNow()

            if http_response.status_code in (401, 403):
                self.limiter.reconcile(reservation, actual_tokens=None)
                self._dead.add(str(api_key))
                logger.warning("key %s rejected with %s; marking dead for this process", api_key, http_response.status_code)
                raise _NoCapacityNow()

            if http_response.status_code >= 500:
                self.limiter.reconcile(reservation, actual_tokens=None)
                attempts_5xx += 1
                if attempts_5xx > _MAX_5XX_RETRIES:
                    raise RuntimeError(
                        f"Groq returned {http_response.status_code} after {_MAX_5XX_RETRIES} retries for key {api_key}"
                    )
                backoff = _5XX_BACKOFF_BASE_S * (2 ** (attempts_5xx - 1))
                logger.warning("5xx (%s) from Groq for key %s; retrying in %.1fs", http_response.status_code, api_key, backoff)
                self.clock.sleep(backoff)
                continue

            if http_response.status_code != 200:
                self.limiter.reconcile(reservation, actual_tokens=None)
                raise RuntimeError(f"unexpected Groq status {http_response.status_code} for key {api_key}: {http_response.text[:200]}")

            try:
                body = http_response.json()
            except json.JSONDecodeError as exc:
                self.limiter.reconcile(reservation, actual_tokens=None)
                raise RuntimeError(f"Groq returned non-JSON body for key {api_key}") from exc

            usage = body.get("usage") or {}
            actual_tokens = usage.get("total_tokens")
            self.limiter.reconcile(reservation, actual_tokens=actual_tokens if isinstance(actual_tokens, int) else None)

            text = _extract_text(body)
            if text is None:
                # Do NOT cache: an empty completion is not a valid answer, and
                # caching it would make an outage look like a stable "answer".
                raise EmptyResponse(f"Groq returned an empty completion for key {api_key} / model {model}")

            entry = CacheEntry(text=text, model=model, usage=usage, created_at=self.clock.now(), key_label=str(api_key))
            self.cache.put(cache_key_, entry)
            return LLMResponse(text=text, model=model, usage=usage, cached=False, key_label=str(api_key))
