import json
import logging

import httpx
import pytest

from triad.llm.cache import DiskCache, cache_key
from triad.llm.client import AllKeysExhausted, EmptyResponse, GroqClient
from triad.llm.keys import ApiKey
from triad.llm.limiter import RateLimiter

MODEL = "openai/gpt-oss-20b"


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.t = start
        self.slept = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept += seconds
        self.t += seconds  # never a real sleep


def make_client(tmp_path, handler, keys=None, max_wait_s=60.0, clock=None):
    clock = clock or FakeClock()
    keys = keys or [ApiKey("gsk_keyoneaaaaaaaaaaaa1111"), ApiKey("gsk_keytwobbbbbbbbbbbb2222")]
    limiter = RateLimiter(clock=clock, usage_path=tmp_path / "usage.json")
    cache = DiskCache(directory=tmp_path / "responses")
    transport = httpx.MockTransport(handler)
    client = GroqClient(keys=keys, limiter=limiter, cache=cache, transport=transport,
                         clock=clock, max_wait_s=max_wait_s)
    return client, clock, cache


def ok_response(text="the answer", total_tokens=50):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": total_tokens},
    })


def test_cache_hit_makes_zero_http_calls(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return ok_response()

    client, clock, cache = make_client(tmp_path, handler)
    # Pre-populate the cache with EXACTLY what chat() will hash: resolved params
    # include the reasoning_effort default the client injects for gpt-oss models.
    messages = [{"role": "user", "content": "hello"}]
    params = {"max_tokens": 300, "temperature": 0, "reasoning_effort": "low"}
    key = cache_key(principal="alice", scope=(), model=MODEL, messages=messages, params=params)
    from triad.llm.cache import CacheEntry
    cache.put(key, CacheEntry(text="cached answer", model=MODEL, usage={"total_tokens": 1},
                               created_at=clock.now(), key_label="gsk_…zzzz"))

    resp = client.chat(messages, principal="alice")
    assert resp.cached is True
    assert resp.text == "cached answer"
    assert calls == []


def test_round_robin_across_keys(tmp_path):
    seen_keys = []

    def handler(request):
        seen_keys.append(request.headers["authorization"])
        return ok_response()

    client, clock, cache = make_client(tmp_path, handler)
    r1 = client.chat([{"role": "user", "content": "q1"}], principal="alice")
    r2 = client.chat([{"role": "user", "content": "q2"}], principal="alice")
    assert seen_keys[0] != seen_keys[1]  # alternated keys
    assert r1.key_label != r2.key_label


def test_429_cools_down_and_tries_next_key(tmp_path):
    key1 = ApiKey("gsk_keyoneaaaaaaaaaaaa1111")
    key2 = ApiKey("gsk_keytwobbbbbbbbbbbb2222")

    def handler(request):
        auth = request.headers["authorization"]
        if auth == f"Bearer {key1.value}":
            return httpx.Response(429, headers={"retry-after": "30"}, json={"error": "rate limited"})
        return ok_response(text="from key2")

    client, clock, cache = make_client(tmp_path, handler, keys=[key1, key2])
    resp = client.chat([{"role": "user", "content": "q"}], principal="alice")
    assert resp.text == "from key2"
    assert resp.key_label == str(key2)


def test_all_keys_exhausted_raises_never_empty_answer(tmp_path):
    def handler(request):
        return httpx.Response(429, headers={"retry-after": "5"}, json={"error": "rate limited"})

    key1 = ApiKey("gsk_keyoneaaaaaaaaaaaa1111", org="shared")
    key2 = ApiKey("gsk_keytwobbbbbbbbbbbb2222", org="shared")
    client, clock, cache = make_client(tmp_path, handler, keys=[key1, key2], max_wait_s=1.0)
    with pytest.raises(AllKeysExhausted):
        client.chat([{"role": "user", "content": "q"}], principal="alice")


def test_same_org_keys_both_cooled_by_one_429(tmp_path):
    # The sharp version: a 429 on key1 must ALSO block key2 immediately, since
    # they share one org bucket -- not just "map to the same bucket string".
    calls = []

    def handler(request):
        calls.append(request.headers["authorization"])
        return httpx.Response(429, headers={"retry-after": "30"}, json={"error": "rate limited"})

    key1 = ApiKey("gsk_keyoneaaaaaaaaaaaa1111", org="shared")
    key2 = ApiKey("gsk_keytwobbbbbbbbbbbb2222", org="shared")
    client, clock, cache = make_client(tmp_path, handler, keys=[key1, key2], max_wait_s=1.0)
    with pytest.raises(AllKeysExhausted):
        client.chat([{"role": "user", "content": "q"}], principal="alice")
    # Only ONE http call should have happened: after key1's 429 cools the shared
    # bucket, key2 must see it as unavailable too, not get its own live attempt.
    assert len(calls) == 1


def test_different_tenants_get_different_cache_entries(tmp_path):
    def handler(request):
        return ok_response(text="dynamic-" + request.headers["authorization"][-4:])

    client, clock, cache = make_client(tmp_path, handler)
    messages = [{"role": "user", "content": "what is my balance?"}]
    resp_a = client.chat(messages, principal="tenantA")
    resp_b = client.chat(messages, principal="tenantB")
    # Both hit the network (different cache keys) rather than B getting A's cached answer.
    assert resp_a.cached is False
    assert resp_b.cached is False


def test_raw_api_key_never_appears_in_logs(tmp_path, caplog):
    key1 = ApiKey("gsk_keyoneaaaaaaaaaaaa1111")
    key2 = ApiKey("gsk_keytwobbbbbbbbbbbb2222")

    def handler(request):
        return httpx.Response(401, json={"error": "invalid api key"})

    caplog.set_level(logging.WARNING)
    client, clock, cache = make_client(tmp_path, handler, keys=[key1, key2], max_wait_s=1.0)
    with pytest.raises(AllKeysExhausted):
        client.chat([{"role": "user", "content": "q"}], principal="alice")
    assert "keyoneaaaaaaaaaaaa1111" not in caplog.text
    assert "keytwobbbbbbbbbbbb2222" not in caplog.text


def test_empty_completion_raises_and_is_not_cached(tmp_path):
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": ""}}],
            "usage": {"total_tokens": 5},
        })

    client, clock, cache = make_client(tmp_path, handler)
    messages = [{"role": "user", "content": "q"}]
    with pytest.raises(EmptyResponse):
        client.chat(messages, principal="alice")
    params = {"max_tokens": 300, "temperature": 0, "reasoning_effort": "low"}
    key = cache_key(principal="alice", scope=(), model=MODEL, messages=messages, params=params)
    assert cache.get(key) is None


def test_empty_choices_list_raises(tmp_path):
    def handler(request):
        return httpx.Response(200, json={"choices": [], "usage": {"total_tokens": 0}})

    client, clock, cache = make_client(tmp_path, handler)
    with pytest.raises(EmptyResponse):
        client.chat([{"role": "user", "content": "q"}], principal="alice")


def test_whitespace_only_completion_raises(tmp_path):
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "   \n  "}}],
            "usage": {"total_tokens": 5},
        })

    client, clock, cache = make_client(tmp_path, handler)
    with pytest.raises(EmptyResponse):
        client.chat([{"role": "user", "content": "q"}], principal="alice")


def test_reasoning_effort_default_and_override(tmp_path):
    seen_bodies = []

    def handler(request):
        seen_bodies.append(json.loads(request.content))
        return ok_response()

    client, clock, cache = make_client(tmp_path, handler)
    client.chat([{"role": "user", "content": "a"}], principal="alice")
    assert seen_bodies[0]["reasoning_effort"] == "low"

    client.chat([{"role": "user", "content": "b"}], principal="alice", reasoning_effort="high")
    assert seen_bodies[1]["reasoning_effort"] == "high"


def test_successful_call_is_cached_for_next_time(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return ok_response(text="fresh answer")

    client, clock, cache = make_client(tmp_path, handler)
    messages = [{"role": "user", "content": "q"}]
    r1 = client.chat(messages, principal="alice")
    r2 = client.chat(messages, principal="alice")
    assert r1.cached is False
    assert r2.cached is True
    assert r2.text == "fresh answer"
    assert len(calls) == 1


def test_5xx_retries_then_succeeds(tmp_path):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] < 2:
            return httpx.Response(503, json={"error": "server error"})
        return ok_response(text="recovered")

    client, clock, cache = make_client(tmp_path, handler)
    resp = client.chat([{"role": "user", "content": "q"}], principal="alice")
    assert resp.text == "recovered"


def test_5xx_exhausts_retries_and_raises(tmp_path):
    def handler(request):
        return httpx.Response(500, json={"error": "server error"})

    client, clock, cache = make_client(tmp_path, handler)
    with pytest.raises(RuntimeError):
        client.chat([{"role": "user", "content": "q"}], principal="alice")


def test_dead_key_marked_after_403_and_not_retried(tmp_path):
    key1 = ApiKey("gsk_keyoneaaaaaaaaaaaa1111")
    key2 = ApiKey("gsk_keytwobbbbbbbbbbbb2222")
    calls_by_key = {"key1": 0, "key2": 0}

    def handler(request):
        auth = request.headers["authorization"]
        if auth == f"Bearer {key1.value}":
            calls_by_key["key1"] += 1
            return httpx.Response(403, json={"error": "forbidden"})
        calls_by_key["key2"] += 1
        return ok_response(text="from key2")

    client, clock, cache = make_client(tmp_path, handler, keys=[key1, key2])
    client.chat([{"role": "user", "content": "q1"}], principal="alice")
    client.chat([{"role": "user", "content": "q2"}], principal="alice")
    assert calls_by_key["key1"] == 1  # marked dead after the first 403, never retried
    assert calls_by_key["key2"] == 2
