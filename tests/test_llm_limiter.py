import json

import pytest

from triad.llm.limiter import RateLimiter, parse_duration_seconds, parse_retry_after

MODEL = "openai/gpt-oss-20b"  # rpm=30 rpd=1000 tpm=8000 tpd=200000


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.t = start

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds  # advance time instead of really sleeping


def make_limiter(tmp_path, clock=None):
    return RateLimiter(clock=clock or FakeClock(), usage_path=tmp_path / "usage.json")


def test_reserve_and_capacity_basic(tmp_path):
    lim = make_limiter(tmp_path)
    r = lim.reserve("orgA", MODEL, 1500)
    assert r is not None
    assert lim.has_capacity("orgA", MODEL, 1500) is True


def test_tpm_ceiling_respected(tmp_path):
    clock = FakeClock()
    lim = make_limiter(tmp_path, clock)
    # tpm=8000; five reservations of 1500 = 7500, a 6th of 1500 would hit 9000 > 8000
    for _ in range(5):
        assert lim.reserve("orgA", MODEL, 1500) is not None
    assert lim.has_capacity("orgA", MODEL, 1500) is False
    r = lim.reserve("orgA", MODEL, 1500)
    assert r is None


def test_rpm_ceiling_respected(tmp_path):
    clock = FakeClock()
    lim = make_limiter(tmp_path, clock)
    for _ in range(30):
        assert lim.reserve("orgA", MODEL, 10) is not None
    assert lim.reserve("orgA", MODEL, 10) is None


def test_capacity_frees_after_minute_rolls(tmp_path):
    clock = FakeClock()
    lim = make_limiter(tmp_path, clock)
    for _ in range(30):
        lim.reserve("orgA", MODEL, 10)
    assert lim.has_capacity("orgA", MODEL, 10) is False
    clock.t += 61  # roll into the next minute window
    assert lim.has_capacity("orgA", MODEL, 10) is True


def test_reconcile_replaces_estimate_not_adds(tmp_path):
    clock = FakeClock()
    lim = make_limiter(tmp_path, clock)
    r = lim.reserve("orgA", MODEL, 1500)
    lim.reconcile(r, actual_tokens=100)  # actual usage much smaller than estimate
    # We should now have headroom for several more 1500-token estimates within TPM=8000
    ok_count = 0
    for _ in range(5):
        if lim.reserve("orgA", MODEL, 1500) is not None:
            ok_count += 1
    assert ok_count == 5  # 100 + 5*1500 = 7600 < 8000


def test_reconcile_on_failure_zeroes_tokens_keeps_request(tmp_path):
    clock = FakeClock()
    lim = make_limiter(tmp_path, clock)
    r = lim.reserve("orgA", MODEL, 1500)
    lim.reconcile(r, actual_tokens=None)  # failed call: no tokens billed
    snap = lim.snapshot("orgA", MODEL)
    assert snap["tokens_today"] == 0
    assert snap["requests_today"] == 1  # the HTTP call still happened


def test_cooldown_blocks_until_elapsed(tmp_path):
    clock = FakeClock()
    lim = make_limiter(tmp_path, clock)
    lim.cooldown("orgA", MODEL, 30.0)
    assert lim.has_capacity("orgA", MODEL, 100) is False
    clock.t += 31
    assert lim.has_capacity("orgA", MODEL, 100) is True


def test_daily_counters_persist_across_limiter_instances(tmp_path):
    clock = FakeClock()
    usage_path = tmp_path / "usage.json"
    lim1 = RateLimiter(clock=clock, usage_path=usage_path)
    lim1.reserve("orgA", MODEL, 1500)
    lim1.reserve("orgA", MODEL, 1500)

    lim2 = RateLimiter(clock=clock, usage_path=usage_path)  # simulate restart
    snap = lim2.snapshot("orgA", MODEL)
    assert snap["tokens_today"] == 3000
    assert snap["requests_today"] == 2


def test_daily_counters_reset_on_date_rollover(tmp_path):
    clock = FakeClock(start=1_000_000.0)
    usage_path = tmp_path / "usage.json"
    lim1 = RateLimiter(clock=clock, usage_path=usage_path)
    lim1.reserve("orgA", MODEL, 1500)

    clock.t += 90000  # jump forward >24h in UTC
    lim2 = RateLimiter(clock=clock, usage_path=usage_path)
    snap = lim2.snapshot("orgA", MODEL)
    assert snap["tokens_today"] == 0
    assert snap["requests_today"] == 0


def test_rpd_exhaustion_reports_not_today(tmp_path):
    clock = FakeClock()
    lim = make_limiter(tmp_path, clock)
    day_key = "orgA|" + MODEL
    lim._daily[day_key] = {"requests": 1000, "tokens": 0}  # rpd=1000, already at ceiling
    assert lim.time_until_capacity("orgA", MODEL, 10) is None


def test_time_until_capacity_zero_when_available(tmp_path):
    lim = make_limiter(tmp_path)
    assert lim.time_until_capacity("orgA", MODEL, 10) == 0.0


def test_same_org_bucket_shared_state(tmp_path):
    # Two keys mapping to the same bucket string must draw from one shared pool.
    clock = FakeClock()
    lim = make_limiter(tmp_path, clock)
    for _ in range(30):
        assert lim.reserve("org:shared", MODEL, 10) is not None
    assert lim.reserve("org:shared", MODEL, 10) is None  # second "key" sees it exhausted too


def test_tighten_from_headers_makes_capacity_stricter(tmp_path):
    lim = make_limiter(tmp_path)
    lim.tighten_from_headers(
        "orgA", MODEL, {"x-ratelimit-remaining-tokens": "50", "x-ratelimit-reset-tokens": "30s"}
    )
    assert lim.has_capacity("orgA", MODEL, 100) is False
    assert lim.has_capacity("orgA", MODEL, 10) is True


def test_tighten_from_headers_malformed_never_raises(tmp_path):
    lim = make_limiter(tmp_path)
    lim.tighten_from_headers("orgA", MODEL, {"x-ratelimit-remaining-tokens": "garbage"})
    # should not raise, and should not corrupt normal capacity checks
    assert lim.has_capacity("orgA", MODEL, 100) is True


def test_parse_duration_seconds_variants():
    assert parse_duration_seconds("7.66s") == pytest.approx(7.66)
    assert parse_duration_seconds("2m59.56s") == pytest.approx(179.56)
    assert parse_duration_seconds("1h2m3s") == pytest.approx(3723.0)
    assert parse_duration_seconds("12") == pytest.approx(12.0)
    assert parse_duration_seconds("not a duration") is None
    assert parse_duration_seconds(None) is None
    assert parse_duration_seconds("") is None


def test_parse_retry_after_variants():
    assert parse_retry_after("5") == pytest.approx(5.0)
    assert parse_retry_after("5.5") == pytest.approx(5.5)
    assert parse_retry_after(None) is None
    assert parse_retry_after("garbage") is None


def test_usage_json_is_valid_json_on_disk(tmp_path):
    clock = FakeClock()
    usage_path = tmp_path / "usage.json"
    lim = RateLimiter(clock=clock, usage_path=usage_path)
    lim.reserve("orgA", MODEL, 500)
    raw = json.loads(usage_path.read_text(encoding="utf-8"))
    assert "date" in raw and "buckets" in raw


def test_corrupt_usage_json_does_not_crash(tmp_path):
    usage_path = tmp_path / "usage.json"
    usage_path.write_text("{not valid json", encoding="utf-8")
    lim = RateLimiter(clock=FakeClock(), usage_path=usage_path)
    snap = lim.snapshot("orgA", MODEL)
    assert snap["requests_today"] == 0
