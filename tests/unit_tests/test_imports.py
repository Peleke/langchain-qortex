"""Verify all public exports are importable."""

from langchain_qortex import LangChainEmbeddingWrapper, QortexEmbeddings, QortexVectorStore


def test_qortex_vectorstore_importable():
    assert QortexVectorStore is not None


def test_qortex_embeddings_importable():
    assert QortexEmbeddings is not None


def test_langchain_embedding_wrapper_importable():
    assert LangChainEmbeddingWrapper is not None


def test_vectorstore_is_langchain_subclass():
    from langchain_core.vectorstores import VectorStore

    assert issubclass(QortexVectorStore, VectorStore)


def test_embeddings_is_langchain_subclass():
    from langchain_core.embeddings import Embeddings

    assert issubclass(QortexEmbeddings, Embeddings)
