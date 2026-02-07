"""Tests for embedding wrappers."""

from __future__ import annotations

import hashlib

from langchain_core.embeddings import Embeddings

from langchain_qortex import LangChainEmbeddingWrapper, QortexEmbeddings

DIMS = 32


class FakeQortexModel:
    """qortex-style embedding: .embed(texts) -> list[list[float]]."""

    @property
    def dimensions(self) -> int:
        return DIMS

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = [float(b) / 255.0 for b in h[:DIMS]]
            norm = sum(v * v for v in vec) ** 0.5
            result.append([v / norm for v in vec])
        return result


class FakeLCEmbedding(Embeddings):
    """LangChain-style embedding."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = [float(b) / 255.0 for b in h[:DIMS]]
            norm = sum(v * v for v in vec) ** 0.5
            result.append([v / norm for v in vec])
        return result

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class TestQortexEmbeddings:
    """Wrap qortex model in LangChain Embeddings interface."""

    def test_is_embeddings(self):
        wrapper = QortexEmbeddings(FakeQortexModel())
        assert isinstance(wrapper, Embeddings)

    def test_embed_documents(self):
        wrapper = QortexEmbeddings(FakeQortexModel())
        vecs = wrapper.embed_documents(["hello", "world"])
        assert len(vecs) == 2
        assert len(vecs[0]) == DIMS

    def test_embed_query(self):
        wrapper = QortexEmbeddings(FakeQortexModel())
        vec = wrapper.embed_query("test")
        assert len(vec) == DIMS


class TestLangChainEmbeddingWrapper:
    """Wrap LangChain Embeddings in qortex embed() interface."""

    def test_dimensions(self):
        wrapper = LangChainEmbeddingWrapper(FakeLCEmbedding())
        assert wrapper.dimensions == DIMS

    def test_embed(self):
        wrapper = LangChainEmbeddingWrapper(FakeLCEmbedding())
        vecs = wrapper.embed(["hello", "world"])
        assert len(vecs) == 2
        assert len(vecs[0]) == DIMS

    def test_roundtrip_consistency(self):
        """qortex -> langchain -> qortex wrapping produces identical vectors."""
        qortex_model = FakeQortexModel()
        lc_wrapper = QortexEmbeddings(qortex_model)
        qortex_wrapper = LangChainEmbeddingWrapper(lc_wrapper)

        original = qortex_model.embed(["test"])[0]
        roundtripped = qortex_wrapper.embed(["test"])[0]
        assert original == roundtripped
