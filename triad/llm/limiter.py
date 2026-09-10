"""Per-(bucket, model) rate limiter enforcing Groq's free-tier RPM/RPD/TPM/TPD.

A "bucket" is an organization (or a lone key with no declared org) -- see
``keys.ApiKey.bucket``. Token counts are ESTIMATED before a call (so we never
send a request we already know will blow the ceiling) and RECONCILED against
the real ``usage`` the API reports afterwards, so the running total stays
accurate without ever double-charging the estimate.

Daily counters (RPD/TPD) are persisted to disk so a process restart doesn't
forget what was already spent today; per-minute counters are not persisted
(a restart losing a few seconds of RPM/TPM headroom is harmless and simpler
than reasoning about partial-minute windows across a restart).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Protocol

from triad import config
from triad.llm import limits

logger = logging.getLogger(__name__)


class Clock(Protocol):
    """Injectable time source. Tests use a fake one so no test ever really sleeps."""

    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class RealClock:
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


@dataclass(frozen=True)
class Reservation:
    """An accepted claim against a bucket's capacity, made before the HTTP call
    with an ESTIMATED token count. Pass it to ``reconcile`` afterwards so the
    real usage (or the fact that the call failed and used zero tokens) replaces
    the estimate rather than stacking on top of it."""

    bucket: str
    model: str
    est_tokens: int
    minute_key: int


def _utc_date(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()


def _minute_key(ts: float) -> int:
    return int(ts // 60)


_DURATION_RE = re.compile(
    r"(?:(?P<h>\d+(?:\.\d+)?)h)?(?:(?P<m>\d+(?:\.\d+)?)m)?(?:(?P<s>\d+(?:\.\d+)?)s)?$"
)


def parse_duration_seconds(text: str | None) -> float | None:
    """Parse a Groq-style duration header ('7.66s', '2m59.56s', '1h2m3s') into
    seconds. Never raises: unparseable input returns None so a malformed header
    degrades to a default cooldown instead of crashing the request path."""
    if not text:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)  # plain seconds, e.g. "12" or "12.5"
    except ValueError:
        pass
    m = _DURATION_RE.fullmatch(text)
    if not m or not any(m.groups()):
        return None
    h = float(m.group("h") or 0)
    mi = float(m.group("m") or 0)
    s = float(m.group("s") or 0)
    total = h * 3600 + mi * 60 + s
    return total if total > 0 else None


def parse_retry_after(text: str | None) -> float | None:
    """Parse a Retry-After header: either integer/float seconds or an HTTP-date.
    Never raises; returns None if neither form parses."""
    if not text:
        return None
    text = text.strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(text)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


@dataclass
class _WindowState:
    minute_key: int = -1
    minute_requests: int = 0
    minute_tokens: int = 0
    cooldown_until: float = 0.0
    # Server-reported tightening (only shrinks capacity, never widens it).
    server_remaining_requests: int | None = None
    server_requests_reset_at: float = 0.0
    server_remaining_tokens: int | None = None
    server_tokens_reset_at: float = 0.0


class RateLimiter:
    """One limiter instance covers all buckets/models; state is per-(bucket, model)."""

    def __init__(self, clock: Clock | None = None, usage_path: Path = config.LLM_CACHE_DIR / "usage.json"):
        self.clock = clock or RealClock()
        self.usage_path = Path(usage_path)
        self._windows: dict[tuple[str, str], _WindowState] = {}
        self._day: str = _utc_date(self.clock.now())
        self._daily: dict[str, dict[str, int]] = {}  # "bucket|model" -> {"requests": N, "tokens": N}
        self._load_daily()

    # -- persistence -----------------------------------------------------

    def _load_daily(self) -> None:
        if not self.usage_path.exists():
            return
        try:
            raw = json.loads(self.usage_path.read_text(encoding="utf-8"))
            date = raw.get("date")
            buckets = raw.get("buckets", {})
            if date == self._day and isinstance(buckets, dict):
                self._daily = {k: {"requests": int(v.get("requests", 0)), "tokens": int(v.get("tokens", 0))}
                                for k, v in buckets.items()}
            # a stale date is a rollover: leave self._daily empty (today starts fresh)
        except Exception as exc:  # never brick the demo over a corrupt cache file
            logger.warning("usage.json unreadable (%s); starting today's counters at zero", exc)
            self._daily = {}

    def _save_daily(self) -> None:
        try:
            self.usage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"date": self._day, "buckets": self._daily}
            tmp = self.usage_path.with_suffix(self.usage_path.suffix + f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, self.usage_path)  # atomic even on Windows; same dir as target
        except Exception as exc:
            logger.warning("could not persist usage.json: %s", exc)

    def _roll_day_if_needed(self) -> None:
        today = _utc_date(self.clock.now())
        if today != self._day:
            self._day = today
            self._daily = {}
            self._save_daily()

    def _daily_key(self, bucket: str, model: str) -> str:
        return f"{bucket}|{model}"

    # -- state -------------------------------------------------------------

    def _window(self, bucket: str, model: str) -> _WindowState:
        key = (bucket, model)
        w = self._windows.get(key)
        if w is None:
            w = _WindowState()
            self._windows[key] = w
        now_minute = _minute_key(self.clock.now())
        if w.minute_key != now_minute:
            w.minute_key = now_minute
            w.minute_requests = 0
            w.minute_tokens = 0
        return w

    def _effective_remaining_requests(self, w: _WindowState, rpm: int | None) -> int | None:
        remaining = None if rpm is None else rpm - w.minute_requests
        now = self.clock.now()
        if w.server_remaining_requests is not None and now < w.server_requests_reset_at:
            srv = w.server_remaining_requests
            remaining = srv if remaining is None else min(remaining, srv)
        return remaining

    def _effective_remaining_tokens(self, w: _WindowState, tpm: int | None) -> int | None:
        remaining = None if tpm is None else tpm - w.minute_tokens
        now = self.clock.now()
        if w.server_remaining_tokens is not None and now < w.server_tokens_reset_at:
            srv = w.server_remaining_tokens
            remaining = srv if remaining is None else min(remaining, srv)
        return remaining

    # -- public API ----------------------------------------------------

    def has_capacity(self, bucket: str, model: str, est_tokens: int) -> bool:
        return self._check(bucket, model, est_tokens)[0]

    def time_until_capacity(self, bucket: str, model: str, est_tokens: int) -> float | None:
        """Seconds until this bucket/model could accept a call of this size, or
        None if it cannot happen today at all (daily ceiling already spent)."""
        ok, wait, daily_blocked = self._check(bucket, model, est_tokens)
        if ok:
            return 0.0
        if daily_blocked:
            return None
        return wait

    def _check(self, bucket: str, model: str, est_tokens: int) -> tuple[bool, float, bool]:
        """Returns (has_capacity, wait_seconds_if_not, blocked_for_the_rest_of_today)."""
        self._roll_day_if_needed()
        lim = limits.limits_for(model)
        w = self._window(bucket, model)
        now = self.clock.now()

        if now < w.cooldown_until:
            return False, w.cooldown_until - now, False

        day_key = self._daily_key(bucket, model)
        day = self._daily.get(day_key, {"requests": 0, "tokens": 0})

        if lim.rpd is not None and day["requests"] + 1 > lim.rpd:
            return False, 0.0, True
        if lim.tpd is not None and day["tokens"] + est_tokens > lim.tpd:
            return False, 0.0, True

        remaining_req = self._effective_remaining_requests(w, lim.rpm)
        remaining_tok = self._effective_remaining_tokens(w, lim.tpm)

        blocked = (remaining_req is not None and remaining_req < 1) or (
            remaining_tok is not None and remaining_tok < est_tokens
        )
        if blocked:
            seconds_left_in_minute = 60 - (now % 60)
            return False, seconds_left_in_minute, False
        return True, 0.0, False

    def reserve(self, bucket: str, model: str, est_tokens: int) -> Reservation | None:
        """Charge ``est_tokens`` against this bucket/model now, optimistically.
        Returns None (no mutation) if there isn't currently capacity."""
        ok, _, _ = self._check(bucket, model, est_tokens)
        if not ok:
            return None
        self._roll_day_if_needed()
        w = self._window(bucket, model)
        w.minute_requests += 1
        w.minute_tokens += est_tokens
        day_key = self._daily_key(bucket, model)
        day = self._daily.setdefault(day_key, {"requests": 0, "tokens": 0})
        day["requests"] += 1
        day["tokens"] += est_tokens
        self._save_daily()
        return Reservation(bucket=bucket, model=model, est_tokens=est_tokens, minute_key=w.minute_key)

    def reconcile(self, reservation: Reservation, actual_tokens: int | None) -> None:
        """Replace the estimate with reality. ``actual_tokens=None`` means the
        call failed (429/5xx/exception): the request still counts (an HTTP call
        was made, so it still counts against RPD/RPM), but its token charge is
        reconciled down to zero since no tokens were actually billed."""
        billed = 0 if actual_tokens is None else actual_tokens
        delta = billed - reservation.est_tokens

        w = self._windows.get((reservation.bucket, reservation.model))
        if w is not None and w.minute_key == reservation.minute_key:
            w.minute_tokens = max(0, w.minute_tokens + delta)

        day_key = self._daily_key(reservation.bucket, reservation.model)
        day = self._daily.get(day_key)
        if day is not None:
            day["tokens"] = max(0, day["tokens"] + delta)
            self._save_daily()

    def cooldown(self, bucket: str, model: str, seconds: float) -> None:
        """Called on a 429: this bucket/model gets no more capacity for ``seconds``."""
        w = self._window(bucket, model)
        w.cooldown_until = max(w.cooldown_until, self.clock.now() + max(0.0, seconds))

    def tighten_from_headers(self, bucket: str, model: str, headers: Mapping[str, str]) -> None:
        """Adopt the server's x-ratelimit-remaining-* / reset-* if they're stricter
        than what we've computed ourselves. Never raises on a malformed header --
        worst case we just don't tighten from it."""
        w = self._window(bucket, model)
        now = self.clock.now()

        def _get(name: str) -> str | None:
            for k, v in headers.items():
                if k.lower() == name:
                    return v
            return None

        try:
            rr = _get("x-ratelimit-remaining-requests")
            if rr is not None:
                w.server_remaining_requests = int(float(rr))
                reset_s = parse_duration_seconds(_get("x-ratelimit-reset-requests")) or 60.0
                w.server_requests_reset_at = now + reset_s
        except Exception as exc:
            logger.warning("could not parse rate-limit request headers: %s", exc)

        try:
            rt = _get("x-ratelimit-remaining-tokens")
            if rt is not None:
                w.server_remaining_tokens = int(float(rt))
                reset_s = parse_duration_seconds(_get("x-ratelimit-reset-tokens")) or 60.0
                w.server_tokens_reset_at = now + reset_s
        except Exception as exc:
            logger.warning("could not parse rate-limit token headers: %s", exc)

    def snapshot(self, bucket: str, model: str) -> dict:
        """Read-only view for doctor.py: today's usage vs limits."""
        self._roll_day_if_needed()
        lim = limits.limits_for(model)
        day = self._daily.get(self._daily_key(bucket, model), {"requests": 0, "tokens": 0})
        return {
            "bucket": bucket,
            "model": model,
            "requests_today": day["requests"],
            "rpd": lim.rpd,
            "tokens_today": day["tokens"],
            "tpd": lim.tpd,
        }
