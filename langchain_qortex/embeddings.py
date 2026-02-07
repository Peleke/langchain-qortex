"""Embedding wrappers for qortex <-> LangChain interop.

qortex uses `model.embed(texts) -> list[list[float]]`.
LangChain uses `model.embed_documents(texts)` and `model.embed_query(text)`.

These wrappers bridge the two interfaces in both directions so consumers
don't need to care which convention their embedding model follows.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings


class QortexEmbeddings(Embeddings):
    """Wrap a qortex embedding model in LangChain's Embeddings interface.

    Any object with `.embed(texts) -> list[list[float]]` works.
    """

    def __init__(self, qortex_model: Any) -> None:
        """Initialize with a qortex-style embedding model.

        Args:
            qortex_model: Any object with `.embed(texts: list[str]) -> list[list[float]]`.
        """
        self._model = qortex_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return self._model.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._model.embed([text])[0]


class LangChainEmbeddingWrapper:
    """Wrap a LangChain Embeddings instance in qortex's embed() interface.

    This lets you pass a LangChain embedding model to QortexVectorStore.from_texts()
    and have it work with qortex internals.
    """

    def __init__(self, lc_embedding: Embeddings) -> None:
        """Initialize with a LangChain Embeddings instance.

        Probes the model to determine embedding dimensions.

        Args:
            lc_embedding: A LangChain Embeddings instance.
        """
        self._lc = lc_embedding
        probe = lc_embedding.embed_query("test")
        self._dimensions = len(probe)

    @property
    def dimensions(self) -> int:
        """The dimensionality of the embedding vectors."""
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the LangChain embedding model."""
        return self._lc.embed_documents(texts)
