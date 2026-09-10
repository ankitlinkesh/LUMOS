"""A deterministic, model-free embedder for tests and CI.

No download, no torch forward pass, no randomness (Python's ``hash()`` is salted
per-process; this uses ``hashlib.sha256`` so the same text always lands in the
same bucket, in this process or any other). It is a plain bag-of-words hashing
trick: each word increments the count in one of ``dim`` buckets, then the vector
is L2-normalized. Two texts that share words end up with overlapping nonzero
buckets and therefore high cosine similarity -- good enough to exercise tenant
filtering, k, and score thresholds without needing real semantics.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")


def _bucket(word: str, dim: int) -> int:
    digest = hashlib.sha256(word.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dim


def _hash_vector(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for word in _WORD_RE.findall(text.lower()):
        vec[_bucket(word, dim)] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


class HashEmbedder:
    """Deterministic bag-of-words hash embedder. ``space="cosine"`` because
    vectors are always L2-normalized."""

    space = "cosine"

    def __init__(self, dim: int = 256, name: str = "hash") -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.name = name

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([_hash_vector(t, self.dim) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return _hash_vector(text, self.dim)
