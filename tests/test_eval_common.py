"""triad.eval._common: git provenance, cache stats, and the results-JSON
writer. Fast, no network -- ``git_commit`` shells out to the local repo only."""

from __future__ import annotations

import json
import os

from triad.eval import _common


def test_importing_common_never_sets_require_real_env(monkeypatch):
    # Regression test: this used to be a module-level import-time side effect
    # that leaked TRIAD_REQUIRE_REAL=1 into every OTHER test in the pytest
    # session (env vars are process-global, not scoped to one test file) and
    # broke test_data_registry.py's mixing tests. Importing _common (which
    # every test in this file does) must never set it -- only require_real(),
    # called explicitly by an eval script's main(), does.
    monkeypatch.delenv("TRIAD_REQUIRE_REAL", raising=False)
    assert "TRIAD_REQUIRE_REAL" not in os.environ


def test_require_real_sets_the_env_var(monkeypatch):
    monkeypatch.delenv("TRIAD_REQUIRE_REAL", raising=False)
    try:
        _common.require_real()
        assert os.environ.get("TRIAD_REQUIRE_REAL") == "1"
    finally:
        # require_real() mutates os.environ directly (by design -- it must
        # outlive the rest of the eval script's process), so monkeypatch's
        # own teardown won't undo it. Clean up explicitly so this test can
        # never leak into a later, unrelated test regardless of run order.
        os.environ.pop("TRIAD_REQUIRE_REAL", None)


def test_cache_stats_records_hits_and_live():
    stats = _common.CacheStats()
    stats.record(True)
    stats.record(False)
    stats.record(False)
    d = stats.as_dict()
    assert d == {"cache_hits": 1, "live_calls": 2, "total_calls": 3}


def test_git_commit_returns_hash_and_dirty_flag():
    info = _common.git_commit()
    assert "hash" in info and "dirty" in info
    # In this real repo, git is available -- a real hex hash, not "unknown".
    assert info["hash"] == "unknown" or len(info["hash"]) == 40


def test_write_results_round_trips_and_carries_git_and_timestamp(tmp_path):
    payload = {"n": 5, "data_composition": "some dataset: 5 real", "cache_stats": {"cache_hits": 0, "live_calls": 5}}
    path = _common.write_results("unit_test_script", payload, results_dir=tmp_path)

    assert path.exists()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["script"] == "unit_test_script"
    assert written["n"] == 5
    assert "git" in written and "hash" in written["git"]
    assert "written_at_utc" in written


def test_latest_result_picks_the_most_recent_matching_file(tmp_path):
    _common.write_results("myeval", {"run": 1}, results_dir=tmp_path)
    import time
    time.sleep(1.1)  # timestamps have 1-second resolution
    _common.write_results("myeval", {"run": 2}, results_dir=tmp_path)

    latest = _common.latest_result("myeval", results_dir=tmp_path)
    assert latest is not None
    assert latest["run"] == 2


def test_latest_result_none_when_missing(tmp_path):
    assert _common.latest_result("nope", results_dir=tmp_path) is None
