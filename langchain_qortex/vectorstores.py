"""QortexVectorStore: qortex as a LangChain VectorStore.

Drop-in replacement for Chroma, FAISS, Pinecone, or any LangChain VectorStore.
Same API. Same chains. Same retriever. Plus graph structure, rules, and
feedback-driven learning.

Usage:
    from langchain_qortex import QortexVectorStore

    # From texts (like Chroma.from_texts):
    vs = QortexVectorStore.from_texts(texts, embedding, domain="security")
    docs = vs.similarity_search("authentication", k=5)

    # From an existing qortex client:
    vs = QortexVectorStore(client=client, domain="security")
    retriever = vs.as_retriever(search_kwargs={"k": 10})
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_qortex.embeddings import LangChainEmbeddingWrapper, QortexEmbeddings

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from qortex.client import ExploreResult, RulesResult


class QortexVectorStore(VectorStore):
    """qortex as a LangChain VectorStore.

    Wraps QortexClient and exposes the full VectorStore interface:
    similarity_search, add_texts, from_texts, as_retriever, etc.

    Unlike plain vector stores, qortex results carry graph structure:
    - Documents include node_id in metadata for graph navigation
    - explore() traverses typed edges from any retrieved concept
    - rules() returns projected rules linked to retrieved concepts
    - feedback() closes the learning loop via PPR teleportation updates
    """

    def __init__(
        self,
        client: Any,
        domain: str = "default",
        feedback_source: str = "langchain",
    ) -> None:
        """Initialize QortexVectorStore.

        Args:
            client: A QortexClient (LocalQortexClient or future McpQortexClient).
            domain: Default domain for searches and add operations.
            feedback_source: Source identifier for feedback events.
        """
        self._client = client
        self._domain = domain
        self._feedback_source = feedback_source
        self._last_query_id: str | None = None

    @property
    def embeddings(self) -> Embeddings | None:
        """Access the embedding model if available."""
        emb = getattr(self._client, "embedding_model", None)
        if emb is None:
            return None
        if isinstance(emb, Embeddings):
            return emb
        return QortexEmbeddings(emb)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vector store.

        Creates ConceptNodes in the backend and indexes their embeddings.
        Each text becomes a concept in the configured domain.

        Args:
            texts: Texts to add.
            metadatas: Optional metadata dicts per text.
            ids: Optional IDs. Auto-generated if not provided.
            **kwargs: domain (str) to override the default domain.

        Returns:
            List of IDs of added concepts.
        """
        domain = kwargs.get("domain", self._domain)
        return self._client.add_concepts(
            texts=list(texts),
            domain=domain,
            metadatas=metadatas,
            ids=ids,
        )

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Input text.
            k: Number of documents to return.
            **kwargs: Additional args (domains, min_confidence).

        Returns:
            List of langchain Document objects.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Return docs and similarity scores.

        Args:
            query: Input text.
            k: Number of documents to return.
            **kwargs: Additional args.

        Returns:
            List of (Document, score) tuples. Score is 0-1 similarity.
        """
        domains = kwargs.get("domains", [self._domain])
        min_confidence = kwargs.get("min_confidence", 0.0)

        result = self._client.query(
            context=query, domains=domains, top_k=k, min_confidence=min_confidence,
        )
        self._last_query_id = result.query_id

        docs_and_scores = []
        for item in result.items:
            meta = {
                "score": item.score,
                "domain": item.domain,
                "node_id": item.node_id,
            }
            meta.update(item.metadata)

            if result.rules:
                meta["rules"] = [
                    {"id": r.id, "text": r.text, "relevance": r.relevance}
                    for r in result.rules
                    if item.node_id in r.source_concepts
                ]

            doc = Document(page_content=item.content, metadata=meta, id=item.id)
            docs_and_scores.append((doc, item.score))

        return docs_and_scores

    def _select_relevance_score_fn(self):
        """qortex scores are already cosine similarity in [0, 1]."""
        return lambda score: score

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> QortexVectorStore:
        """Create a QortexVectorStore from texts.

        Sets up a full qortex backend (InMemoryBackend + NumpyVectorIndex),
        adds the texts, and returns a ready-to-use VectorStore.

        Args:
            texts: Texts to add.
            embedding: LangChain Embeddings instance.
            metadatas: Optional metadata per text.
            ids: Optional IDs.
            **kwargs: domain (str), backend, vector_index for advanced use.

        Returns:
            Initialized QortexVectorStore.
        """
        domain = kwargs.pop("domain", "default")
        backend = kwargs.pop("backend", None)
        vector_index = kwargs.pop("vector_index", None)

        qortex_embedding = LangChainEmbeddingWrapper(embedding)

        if backend is None:
            from qortex.core.memory import InMemoryBackend
            from qortex.vec.index import NumpyVectorIndex

            if vector_index is None:
                vector_index = NumpyVectorIndex(dimensions=qortex_embedding.dimensions)
            backend = InMemoryBackend(vector_index=vector_index)
            backend.connect()

        from qortex.client import LocalQortexClient

        client = LocalQortexClient(
            vector_index=vector_index,
            backend=backend,
            embedding_model=qortex_embedding,
        )

        store = cls(client=client, domain=domain)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their node IDs."""
        nodes = self._client.get_nodes(list(ids))
        docs = []
        for node in nodes:
            meta = {
                "domain": node.domain,
                "node_id": node.id,
            }
            meta.update(node.properties)
            docs.append(Document(
                page_content=f"{node.name}: {node.description}",
                metadata=meta,
                id=node.id,
            ))
        return docs

    # -- qortex extras: graph exploration + rules + feedback --

    def explore(self, node_id: str, depth: int = 1) -> ExploreResult | None:
        """Explore a concept's graph neighborhood.

        Call this on a node_id from search results to see typed edges,
        neighbors, and linked rules.

        Args:
            node_id: The graph node ID (from doc.metadata["node_id"]).
            depth: Hops to traverse (1-3). Default 1 = immediate neighbors.

        Returns:
            ExploreResult with node, edges, neighbors, and rules. None if not found.
        """
        return self._client.explore(node_id, depth)

    def rules(self, **kwargs: Any) -> RulesResult:
        """Get projected rules from the knowledge graph.

        Args:
            **kwargs: Passed to client.rules(). Accepts domains, concept_ids,
                categories, include_derived, min_confidence.

        Returns:
            RulesResult with rules list and metadata.
        """
        return self._client.rules(**kwargs)

    def feedback(self, outcomes: dict[str, str]) -> None:
        """Report feedback for the last search. Closes the learning loop.

        Accepted concepts get higher PPR teleportation probability on future
        queries. Rejected concepts get lower. Over time, results improve.

        Args:
            outcomes: Mapping of item ID to "accepted" or "rejected".
        """
        if self._last_query_id is None:
            return
        self._client.feedback(
            query_id=self._last_query_id,
            outcomes=outcomes,
            source=self._feedback_source,
        )

    @property
    def last_query_id(self) -> str | None:
        """The query_id from the most recent search. Used internally by feedback()."""
        return self._last_query_id
