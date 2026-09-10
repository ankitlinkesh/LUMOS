"""Stage 2 embedding backends.

Every embedder declares its similarity convention (``space``: ``"cosine"`` or
``"ip"``) so callers can configure Chroma's HNSW index (``hnsw:space``) to match.
Mixing conventions silently (e.g. treating Contriever's unnormalized dot-product
embeddings as cosine) produces a working-looking but wrong ranking, which is why
the convention lives on the embedder itself rather than being assumed by callers.
"""

from triad.embed.base import Embedder
from triad.embed.hash_embedder import HashEmbedder
from triad.embed.sentence_transformer_embedder import SentenceTransformerEmbedder

__all__ = ["Embedder", "HashEmbedder", "SentenceTransformerEmbedder"]
