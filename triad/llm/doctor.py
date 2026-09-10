"""Diagnostic CLI: ``python -m triad.llm.doctor [--ping]``.

Reports how many keys are loaded (masked), what bucket each is in, and
today's usage vs. limits per model. Does nothing at import time -- all file
and network I/O happens inside ``main()`` -- so importing this module (e.g.
from a test) never touches the real secrets file or the network.
"""

from __future__ import annotations

import time
import sys

import httpx

from triad import config
from triad.llm import limits
from triad.llm.client import GroqClient
from triad.llm.keys import ApiKey, KeysMissing, load_keys
from triad.llm.limiter import RateLimiter


def _report_keys(keys: tuple[ApiKey, ...]) -> None:
    print(f"Loaded {len(keys)} key(s):")
    for k in keys:
        org = k.org or "(no org — its own bucket)"
        label = f" [{k.label}]" if k.label else ""
        print(f"  {k}{label}  bucket={k.bucket}  org={org}")


def _report_usage(keys: tuple[ApiKey, ...], limiter: RateLimiter) -> None:
    buckets = sorted({k.bucket for k in keys})
    print("\nToday's usage vs. free-tier limits:")
    for bucket in buckets:
        for model, lim in limits.FREE_TIER.items():
            snap = limiter.snapshot(bucket, model)
            if snap["requests_today"] == 0 and snap["tokens_today"] == 0:
                continue  # keep the report readable: skip untouched models
            rpd = "∞" if lim.rpd is None else lim.rpd
            tpd = "∞" if lim.tpd is None else lim.tpd
            print(
                f"  {bucket}  {model}: "
                f"{snap['requests_today']}/{rpd} req/day, "
                f"{snap['tokens_today']}/{tpd} tok/day"
            )


def _ping(keys: tuple[ApiKey, ...]) -> None:
    print("\n--ping: sending one tiny live request per key...")
    limiter = RateLimiter()
    cache_dir = config.LLM_CACHE_DIR / "responses"
    from triad.llm.cache import DiskCache

    for k in keys:
        client = GroqClient(keys=[k], limiter=limiter, cache=DiskCache(directory=cache_dir))
        try:
            # A unique principal per key and per run: the cache must never answer a ping,
            # or keys 2..N would "pass" on key 1's cached reply without touching Groq.
            # 64 tokens because gpt-oss spends its budget on reasoning before any content.
            resp = client.chat(
                [{"role": "user", "content": "reply with the single word: ok"}],
                principal=f"doctor-ping:{k.bucket}:{time.time_ns()}",
                model=limits.DEFAULT_GENERATOR,
                max_tokens=64,
            )
            live = "live" if not resp.cached else "FROM CACHE - not a live test"
            print(f"  {k}: OK ({live}) -> {resp.text!r}")
        except Exception as exc:
            print(f"  {k}: FAILED -> {exc}")
        finally:
            client.close()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    do_ping = "--ping" in argv

    try:
        keys = load_keys()
    except KeysMissing as exc:
        print(str(exc))
        print(f"\nPaste one or more Groq API keys (one per line, starting with 'gsk_') into:\n  {config.KEYS_FILE}")
        return 1

    _report_keys(keys)
    limiter = RateLimiter()
    _report_usage(keys, limiter)

    if do_ping:
        _ping(keys)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
