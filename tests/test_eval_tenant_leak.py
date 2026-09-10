"""Pure, fast pieces of triad.eval.tenant_leak: percentile helper and the
nearest-topic-centroid pairing. No network, no real EnronQA loading."""

from __future__ import annotations

import numpy as np

from triad.eval.tenant_leak import _nearest_partner, _percentile


def test_percentile_basic():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile(values, 0.0) == 10.0
    assert _percentile(values, 1.0) == 50.0
    assert _percentile([], 0.5) != _percentile([], 0.5)  # NaN != NaN


def test_nearest_partner_picks_highest_cosine_similarity():
    centroids = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.99, 0.01]),   # nearly identical to a -- a's real partner
        "c": np.array([0.0, 1.0]),     # orthogonal -- not a's partner
    }
    assert _nearest_partner("a", centroids) == "b"
    assert _nearest_partner("c", centroids) in ("a", "b")  # c is equidistant-ish from both; just must not be itself


def test_nearest_partner_never_returns_self():
    centroids = {"x": np.array([1.0, 1.0]), "y": np.array([1.0, 1.0])}
    assert _nearest_partner("x", centroids) == "y"
