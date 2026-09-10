"""HashEmbedder: deterministic, model-free, used as the fake embedder everywhere
else in this test suite. These tests are about the embedder itself, not Chroma."""

from __future__ import annotations

import numpy as np
import pytest

from triad.embed import HashEmbedder


def test_deterministic_across_instances():
    """Same text -> same vector, in a fresh instance, every time (no hidden
    randomness like a per-process salted hash)."""
    a = HashEmbedder(dim=32).embed_query("quarterly earnings report")
    b = HashEmbedder(dim=32).embed_query("quarterly earnings report")
    assert np.array_equal(a, b)


def test_shape_and_dtype():
    emb = HashEmbedder(dim=64)
    docs = emb.embed_documents(["alpha beta", "gamma delta", "epsilon"])
    assert docs.shape == (3, 64)
    assert docs.dtype == np.float32
    q = emb.embed_query("alpha")
    assert q.shape == (64,)
    assert q.dtype == np.float32


def test_empty_document_list_returns_empty_array_with_right_dim():
    emb = HashEmbedder(dim=16)
    docs = emb.embed_documents([])
    assert docs.shape == (0, 16)


def test_shared_words_are_closer_than_unrelated_text():
    """The property the whole test suite leans on: texts sharing words must be
    closer (higher cosine similarity) than texts sharing none."""
    emb = HashEmbedder(dim=256)
    a = emb.embed_query("quarterly earnings report finance numbers")
    b = emb.embed_query("quarterly earnings report finance summary")
    c = emb.embed_query("purple giraffe skateboard umbrella")

    sim_ab = float(np.dot(a, b))
    sim_ac = float(np.dot(a, c))
    assert sim_ab > sim_ac


def test_vectors_are_l2_normalized():
    emb = HashEmbedder(dim=64)
    vec = emb.embed_query("some words here for norm check")
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)


def test_empty_text_is_the_zero_vector_not_an_error():
    emb = HashEmbedder(dim=32)
    vec = emb.embed_query("   ")
    assert np.array_equal(vec, np.zeros(32, dtype=np.float32))


def test_space_is_cosine():
    assert HashEmbedder().space == "cosine"


def test_rejects_nonpositive_dim():
    with pytest.raises(ValueError):
        HashEmbedder(dim=0)
