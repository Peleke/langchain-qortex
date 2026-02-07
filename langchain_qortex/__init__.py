"""LangChain integration for qortex graph-enhanced retrieval.

This package provides a LangChain VectorStore backed by qortex's knowledge graph.
Same API as Chroma, FAISS, or Pinecone, with graph structure, rules,
and feedback-driven learning layered on top.

Usage:
    from langchain_qortex import QortexVectorStore

    # Zero-config (like Chroma.from_texts):
    vs = QortexVectorStore.from_texts(texts, embedding, domain="security")
    docs = vs.similarity_search("query")

    # From existing qortex client:
    vs = QortexVectorStore(client=client, domain="security")
    docs = vs.similarity_search("query", k=5)
    retriever = vs.as_retriever()

    # qortex extras (graph + rules + feedback):
    explore = vs.explore(docs[0].metadata["node_id"])
    rules = vs.rules(concept_ids=[d.metadata["node_id"] for d in docs])
    vs.feedback({docs[0].id: "accepted"})
"""

from langchain_qortex.embeddings import LangChainEmbeddingWrapper, QortexEmbeddings
from langchain_qortex.vectorstores import QortexVectorStore

__all__ = [
    "LangChainEmbeddingWrapper",
    "QortexEmbeddings",
    "QortexVectorStore",
]
