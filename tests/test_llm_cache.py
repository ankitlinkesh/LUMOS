import json

from triad.llm.cache import CacheEntry, DiskCache, cache_key


def make_entry(text="hello world"):
    return CacheEntry(text=text, model="openai/gpt-oss-20b", usage={"total_tokens": 42},
                       created_at=1234.0, key_label="gsk_...abcd")


def test_put_then_get_roundtrips(tmp_path):
    cache = DiskCache(directory=tmp_path)
    key = cache_key(principal="alice", scope=(), model="openai/gpt-oss-20b",
                     messages=[{"role": "user", "content": "hi"}], params={"temperature": 0})
    cache.put(key, make_entry())
    got = cache.get(key)
    assert got is not None
    assert got.text == "hello world"
    assert got.usage["total_tokens"] == 42


def test_miss_returns_none(tmp_path):
    cache = DiskCache(directory=tmp_path)
    assert cache.get("nonexistent" * 4) is None


def test_different_principal_gives_different_key():
    common = dict(scope=(), model="openai/gpt-oss-20b",
                  messages=[{"role": "user", "content": "what is my balance?"}], params={"temperature": 0})
    key_a = cache_key(principal="tenantA", **common)
    key_b = cache_key(principal="tenantB", **common)
    assert key_a != key_b


def test_tenant_b_cannot_read_tenant_a_cache_entry(tmp_path):
    cache = DiskCache(directory=tmp_path)
    common = dict(scope=(), model="openai/gpt-oss-20b",
                  messages=[{"role": "user", "content": "what is my balance?"}], params={"temperature": 0})
    key_a = cache_key(principal="tenantA", **common)
    key_b = cache_key(principal="tenantB", **common)
    cache.put(key_a, make_entry("tenant A's private answer"))
    assert cache.get(key_b) is None  # not just "different file" -- an actual miss for B


def test_different_scope_gives_different_key():
    common = dict(principal="alice", model="openai/gpt-oss-20b",
                  messages=[{"role": "user", "content": "hi"}], params={"temperature": 0})
    assert cache_key(scope=(), **common) != cache_key(scope=("team-x",), **common)


def test_different_params_gives_different_key():
    common = dict(principal="alice", scope=(), model="openai/gpt-oss-20b",
                  messages=[{"role": "user", "content": "hi"}])
    key_low = cache_key(params={"temperature": 0, "reasoning_effort": "low"}, **common)
    key_high = cache_key(params={"temperature": 0, "reasoning_effort": "high"}, **common)
    assert key_low != key_high


def test_key_is_order_independent_for_scope():
    common = dict(principal="alice", model="openai/gpt-oss-20b",
                  messages=[{"role": "user", "content": "hi"}], params={"temperature": 0})
    assert cache_key(scope=("b", "a"), **common) == cache_key(scope=("a", "b"), **common)


def test_raw_key_never_appears_in_cache_file(tmp_path):
    cache = DiskCache(directory=tmp_path)
    key = cache_key(principal="alice", scope=(), model="openai/gpt-oss-20b",
                     messages=[{"role": "user", "content": "hi"}], params={})
    entry = CacheEntry(text="answer", model="openai/gpt-oss-20b", usage={},
                        created_at=1.0, key_label="gsk_...abcd")
    cache.put(key, entry)
    on_disk = (tmp_path / f"{key}.json").read_text(encoding="utf-8")
    raw = json.loads(on_disk)
    assert raw["key_label"] == "gsk_...abcd"  # masked label is fine to store
    # but nothing resembling a full raw key (16+ chars after gsk_) is present
    import re
    assert not re.search(r"gsk_[A-Za-z0-9]{16,}", on_disk)


def test_put_is_atomic_no_leftover_tmp_files(tmp_path):
    cache = DiskCache(directory=tmp_path)
    key = cache_key(principal="alice", scope=(), model="openai/gpt-oss-20b",
                     messages=[{"role": "user", "content": "hi"}], params={})
    cache.put(key, make_entry())
    leftovers = list(tmp_path.glob("*.tmp*"))
    assert leftovers == []


def test_corrupt_cache_file_is_a_miss_not_a_crash(tmp_path):
    cache = DiskCache(directory=tmp_path)
    key = "deadbeef" * 8
    (tmp_path / f"{key}.json").write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None
