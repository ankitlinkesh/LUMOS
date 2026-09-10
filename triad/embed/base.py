"""The Embedder protocol every Stage 2 embedding backend implements.

Why a protocol instead of an ABC: ``HashEmbedder`` (tests/CI) and
``SentenceTransformerEmbedder`` (real models) share no useful base behaviour --
only a shape contract. Structural typing lets ``TenantStore``/``SecureRetriever``
depend on the shape without caring which concrete class they got.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

# Chroma's ``hnsw:space`` values this pipeline actually configures.
SPACES = frozenset({"cosine", "ip"})


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors, plus the similarity convention that makes those
    vectors comparable.

    ``space`` matters because it is NOT a detail: PoisonedRAG's Contriever setup
    uses dot product (``ip``) on unnormalized mean-pooled vectors, while bge-style
    models use normalized cosine (``cosine``). Comparing across conventions -- or
    configuring Chroma with the wrong one -- silently returns a plausible-looking
    but wrong ranking. A caller building a ``TenantStore`` must read this field and
    set the collection's ``hnsw:space`` to match.
    """

    name: str
    dim: int
    space: str  # one of SPACES

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Returns float32 array of shape (len(texts), dim)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Returns float32 array of shape (dim,)."""
        ...
