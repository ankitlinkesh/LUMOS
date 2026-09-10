"""Real embedding models, loaded lazily so importing this module -- or even
constructing this class -- never triggers a download. The model only loads on
the first ``embed_documents``/``embed_query``/``dim`` access, which keeps unit
tests (which must never touch the network) safe even if this module is imported.

Importing ``triad.config`` first is required in real use: it sets ``HF_HOME``
onto D: (via ``os.environ.setdefault``) before any Hugging Face library reads it,
keeping model weights off the nearly-full C: drive. This module imports it for
that side effect before ``sentence_transformers`` ever gets imported.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from triad import config as _config  # noqa: F401  side effect: HF_HOME -> D:

# PoisonedRAG's reference setup (arXiv:2605.05287) runs Contriever with dot
# product on *unnormalized* mean-pooled embeddings -- normalizing would change
# the ranking it reports. bge-style models are trained for normalized cosine and
# use an asymmetric instruction prefix on the query side only. Matched by
# substring against the model name; unknown models fall back to the safer
# default (normalized cosine, no prefix) since an unnormalized *unknown*
# convention risks producing near-meaningless dot products.
_KNOWN_CONVENTIONS = {
    "contriever": {"space": "ip", "normalize": False, "query_prefix": ""},
    "bge": {
        "space": "cosine",
        "normalize": True,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
}
_DEFAULT_CONVENTION = {"space": "cosine", "normalize": True, "query_prefix": ""}


def _convention_for(model_name: str) -> dict:
    lowered = model_name.lower()
    for key, convention in _KNOWN_CONVENTIONS.items():
        if key in lowered:
            return convention
    return _DEFAULT_CONVENTION


class SentenceTransformerEmbedder:
    """Lazily-loading wrapper around ``sentence_transformers.SentenceTransformer``.

    ``space``, ``normalize`` and the query instruction prefix are picked from the
    model name via ``_convention_for`` so callers get the right Chroma
    configuration (``hnsw:space``) without having to know each model's paper.
    """

    def __init__(self, model_name: str) -> None:
        self.name = model_name
        convention = _convention_for(model_name)
        self.space = convention["space"]
        self._normalize = convention["normalize"]
        self._query_prefix = convention["query_prefix"]
        self._model = None  # loaded lazily
        self._dim: int | None = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, deferred

            self._model = SentenceTransformer(self.name)
            # sentence-transformers renamed this method; support both so this
            # doesn't break on whichever side of the rename is installed.
            dim_fn = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
            self._dim = int(dim_fn())
        return self._model

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_loaded()
        vecs = model.encode(
            list(texts),
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        model = self._ensure_loaded()
        vec = model.encode(
            [self._query_prefix + text],
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )[0]
        return np.asarray(vec, dtype=np.float32)
