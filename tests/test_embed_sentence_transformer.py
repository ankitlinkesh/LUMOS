"""Real-model smoke test. Marked slow (excluded by default via addopts = -m 'not
network'... actually via the `slow` marker, run explicitly) because it downloads
and loads an actual sentence-transformers model. HF_HOME is redirected onto D:
via ``triad.config`` (imported by the module under test) before this ever
touches the network, since C: is nearly full.
"""

from __future__ import annotations

import numpy as np
import pytest

from triad.embed import SentenceTransformerEmbedder


@pytest.mark.slow
def test_bge_small_embeds_and_normalizes():
    emb = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    assert emb.space == "cosine"

    texts = [
        "quarterly earnings report finance numbers",
        "the cat sat on the mat",
        "authorization gating stops cross-tenant leakage",
    ]
    vecs = emb.embed_documents(texts)
    assert vecs.shape == (3, emb.dim)
    assert vecs.dtype == np.float32
    # bge convention: normalized -> unit-norm rows.
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-3)

    q = emb.embed_query("earnings report")
    assert q.shape == (emb.dim,)
    assert float(np.linalg.norm(q)) == pytest.approx(1.0, abs=1e-3)
