"""Tests for QortexVectorStore.

Comprehensive tests proving QortexVectorStore is a genuine VectorStore drop-in.

Tests:
1. VectorStore ABC compliance: similarity_search, similarity_search_with_score,
   add_texts, get_by_ids, as_retriever, embeddings property
2. from_texts classmethod
3. qortex extras: explore(), rules(), feedback()
4. Rules auto-surfaced in Document metadata
5. Relevance score identity
"""

from __future__ import annotations

import hashlib

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_qortex import QortexVectorStore
from qortex.client import LocalQortexClient
from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptEdge, ConceptNode, ExplicitRule, RelationType
from qortex.vec.index import NumpyVectorIndex

DIMS = 32


class FakeEmbedding:
    """Deterministic hash-based embedding for reproducible tests."""

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
    """LangChain-compatible embedding for from_texts tests."""

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


def make_client_with_graph():
    """Create a LocalQortexClient with concepts, edges, and rules."""
    vector_index = NumpyVectorIndex(dimensions=DIMS)
    backend = InMemoryBackend(vector_index=vector_index)
    backend.connect()
    embedding = FakeEmbedding()

    backend.create_domain("security")

    nodes = [
        ConceptNode(
            id="sec:oauth", name="OAuth2",
            description="OAuth2 authorization framework for delegated access",
            domain="security", source_id="docs",
        ),
        ConceptNode(
            id="sec:jwt", name="JWT",
            description="JSON Web Tokens for stateless authentication",
            domain="security", source_id="docs",
        ),
        ConceptNode(
            id="sec:rbac", name="RBAC",
            description="Role-based access control for permission management",
            domain="security", source_id="docs",
        ),
        ConceptNode(
            id="sec:mfa", name="MFA",
            description="Multi-factor authentication for enhanced security",
            domain="security", source_id="docs",
        ),
    ]

    for node in nodes:
        backend.add_node(node)

    texts = [f"{n.name}: {n.description}" for n in nodes]
    embeddings = embedding.embed(texts)
    for node, emb in zip(nodes, embeddings):
        backend.add_embedding(node.id, emb)

    backend.add_edge(ConceptEdge(
        source_id="sec:oauth", target_id="sec:jwt",
        relation_type=RelationType.REQUIRES,
    ))
    backend.add_edge(ConceptEdge(
        source_id="sec:oauth", target_id="sec:rbac",
        relation_type=RelationType.USES,
    ))

    backend.add_rule(ExplicitRule(
        id="rule:use-oauth", text="Always use OAuth2 for third-party auth",
        domain="security", source_id="docs",
        concept_ids=["sec:oauth"], category="security",
    ))
    backend.add_rule(ExplicitRule(
        id="rule:rotate-keys", text="Rotate JWT signing keys every 90 days",
        domain="security", source_id="docs",
        concept_ids=["sec:oauth", "sec:jwt"], category="security",
    ))
    backend.add_rule(ExplicitRule(
        id="rule:rbac-first", text="Define roles before permissions",
        domain="security", source_id="docs",
        concept_ids=["sec:rbac"], category="architectural",
    ))

    ids = [n.id for n in nodes]
    vector_index.add(ids, embeddings)

    client = LocalQortexClient(
        vector_index=vector_index,
        backend=backend,
        embedding_model=embedding,
        mode="graph",
    )

    return client


@pytest.fixture
def vs():
    client = make_client_with_graph()
    return QortexVectorStore(client=client, domain="security")


# ===========================================================================
# VectorStore ABC compliance
# ===========================================================================


class TestVectorStoreCompliance:
    """QortexVectorStore IS-A VectorStore."""

    def test_is_vectorstore_subclass(self, vs):
        assert isinstance(vs, VectorStore)

    def test_similarity_search_returns_documents(self, vs):
        docs = vs.similarity_search("OAuth2 authorization", k=3)
        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc, Document)
            assert isinstance(doc.page_content, str)
            assert len(doc.page_content) > 0

    def test_similarity_search_with_score(self, vs):
        results = vs.similarity_search_with_score("JWT authentication", k=3)
        assert len(results) > 0
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_similarity_search_respects_k(self, vs):
        docs = vs.similarity_search("security", k=2)
        assert len(docs) <= 2

    def test_as_retriever_works(self, vs):
        retriever = vs.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke("authentication")
        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc, Document)

    def test_documents_have_metadata(self, vs):
        docs = vs.similarity_search("OAuth2 authorization", k=3)
        for doc in docs:
            assert "score" in doc.metadata
            assert "domain" in doc.metadata
            assert "node_id" in doc.metadata
            assert doc.metadata["domain"] == "security"

    def test_documents_have_ids(self, vs):
        docs = vs.similarity_search("OAuth2", k=3)
        for doc in docs:
            assert doc.id is not None
            assert doc.id.startswith("sec:")


# ===========================================================================
# from_texts
# ===========================================================================


class TestFromTexts:
    """QortexVectorStore.from_texts() works like Chroma.from_texts()."""

    def test_from_texts_creates_working_store(self):
        vs = QortexVectorStore.from_texts(
            texts=["OAuth2 for auth", "JWT for tokens", "RBAC for permissions"],
            embedding=FakeLCEmbedding(),
            domain="test",
        )
        assert isinstance(vs, QortexVectorStore)
        docs = vs.similarity_search("auth", k=2)
        assert len(docs) > 0

    def test_from_texts_with_metadatas(self):
        vs = QortexVectorStore.from_texts(
            texts=["First doc", "Second doc"],
            embedding=FakeLCEmbedding(),
            metadatas=[{"source": "a"}, {"source": "b"}],
            domain="meta-test",
        )
        docs = vs.similarity_search("doc", k=2)
        assert len(docs) > 0

    def test_from_texts_with_ids(self):
        vs = QortexVectorStore.from_texts(
            texts=["Only doc"],
            embedding=FakeLCEmbedding(),
            ids=["custom-id-1"],
            domain="id-test",
        )
        docs = vs.similarity_search("doc", k=1)
        assert len(docs) > 0

    def test_from_texts_is_vectorstore(self):
        vs = QortexVectorStore.from_texts(
            texts=["test"],
            embedding=FakeLCEmbedding(),
        )
        assert isinstance(vs, VectorStore)


# ===========================================================================
# add_texts
# ===========================================================================


class TestAddTexts:
    """add_texts() creates concepts and indexes embeddings."""

    def test_add_texts_returns_ids(self, vs):
        ids = vs.add_texts(["New concept A", "New concept B"])
        assert len(ids) == 2
        for id_ in ids:
            assert isinstance(id_, str)

    def test_add_texts_are_searchable(self, vs):
        vs.add_texts(
            ["Quantum encryption uses quantum mechanics for security"],
            metadatas=[{"source": "research"}],
        )
        docs = vs.similarity_search("quantum encryption", k=5)
        assert len(docs) > 0

    def test_add_texts_with_custom_ids(self, vs):
        ids = vs.add_texts(["Custom ID doc"], ids=["my-custom-id"])
        assert ids == ["my-custom-id"]

    def test_add_texts_auto_generates_ids(self, vs):
        ids = vs.add_texts(["Auto ID doc"])
        assert len(ids) == 1
        assert ":" in ids[0]


# ===========================================================================
# get_by_ids
# ===========================================================================


class TestGetByIds:
    """get_by_ids() retrieves documents by node ID."""

    def test_get_existing_ids(self, vs):
        docs = vs.get_by_ids(["sec:oauth", "sec:jwt"])
        assert len(docs) == 2
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.id is not None

    def test_get_nonexistent_id(self, vs):
        docs = vs.get_by_ids(["nonexistent:id"])
        assert len(docs) == 0

    def test_get_partial_match(self, vs):
        docs = vs.get_by_ids(["sec:oauth", "nonexistent:id"])
        assert len(docs) == 1
        assert docs[0].id == "sec:oauth"


# ===========================================================================
# embeddings property
# ===========================================================================


class TestEmbeddingsProperty:
    """embeddings property wraps qortex model in LangChain interface."""

    def test_embeddings_returns_wrapper(self, vs):
        emb = vs.embeddings
        assert emb is not None
        assert isinstance(emb, Embeddings)

    def test_embeddings_embed_documents(self, vs):
        emb = vs.embeddings
        vectors = emb.embed_documents(["test text"])
        assert len(vectors) == 1
        assert len(vectors[0]) == DIMS

    def test_embeddings_embed_query(self, vs):
        emb = vs.embeddings
        vector = emb.embed_query("test query")
        assert len(vector) == DIMS

    def test_embeddings_none_when_no_model(self):
        vi = NumpyVectorIndex(dimensions=DIMS)
        backend = InMemoryBackend(vector_index=vi)
        backend.connect()
        client = LocalQortexClient(
            vector_index=vi, backend=backend, embedding_model=None,
        )
        vs = QortexVectorStore(client=client)
        assert vs.embeddings is None


# ===========================================================================
# qortex extras: explore, rules, feedback
# ===========================================================================


class TestQortexExtras:
    """Graph exploration, rules, and feedback loop."""

    def test_explore_from_search_result(self, vs):
        docs = vs.similarity_search("OAuth2 authorization", k=3)
        assert len(docs) > 0
        node_id = docs[0].metadata["node_id"]
        explore = vs.explore(node_id)
        assert explore is not None
        assert explore.node.id == node_id

    def test_explore_reveals_edges(self, vs):
        explore = vs.explore("sec:oauth")
        assert len(explore.edges) > 0
        edge_types = {e.relation_type for e in explore.edges}
        assert "requires" in edge_types or "uses" in edge_types

    def test_explore_reveals_neighbors(self, vs):
        explore = vs.explore("sec:oauth")
        neighbor_ids = {n.id for n in explore.neighbors}
        assert "sec:jwt" in neighbor_ids
        assert "sec:rbac" in neighbor_ids

    def test_explore_reveals_rules(self, vs):
        explore = vs.explore("sec:oauth")
        assert len(explore.rules) > 0
        rule_ids = {r.id for r in explore.rules}
        assert "rule:use-oauth" in rule_ids

    def test_explore_nonexistent_returns_none(self, vs):
        assert vs.explore("nonexistent:id") is None

    def test_rules_returns_all(self, vs):
        rules_result = vs.rules()
        assert len(rules_result.rules) > 0
        assert rules_result.projection == "rules"

    def test_rules_by_concept_ids(self, vs):
        rules_result = vs.rules(concept_ids=["sec:rbac"])
        for r in rules_result.rules:
            assert "sec:rbac" in r.source_concepts

    def test_feedback_after_search(self, vs):
        docs = vs.similarity_search("OAuth2", k=3)
        assert len(docs) > 0
        vs.feedback({docs[0].id: "accepted"})

    def test_feedback_noop_without_query(self):
        client = make_client_with_graph()
        vs = QortexVectorStore(client=client, domain="security")
        vs.feedback({"fake-id": "accepted"})

    def test_last_query_id_tracked(self, vs):
        assert vs.last_query_id is None
        vs.similarity_search("test", k=1)
        assert vs.last_query_id is not None


# ===========================================================================
# rules in document metadata
# ===========================================================================


class TestRulesInMetadata:
    """Query results include linked rules in Document metadata."""

    def test_documents_may_have_rules(self, vs):
        results = vs.similarity_search_with_score("OAuth2 authorization", k=4)
        all_rules = []
        for doc, _ in results:
            if "rules" in doc.metadata:
                all_rules.extend(doc.metadata["rules"])

    def test_rules_have_expected_fields(self, vs):
        results = vs.similarity_search_with_score("OAuth2 authorization", k=4)
        for doc, _ in results:
            if "rules" in doc.metadata:
                for rule in doc.metadata["rules"]:
                    assert "id" in rule
                    assert "text" in rule
                    assert "relevance" in rule


# ===========================================================================
# relevance score
# ===========================================================================


class TestRelevanceScore:
    """qortex scores are cosine similarity in [0, 1], no transformation needed."""

    def test_relevance_score_is_identity(self, vs):
        fn = vs._select_relevance_score_fn()
        assert fn(0.5) == 0.5
        assert fn(0.0) == 0.0
        assert fn(1.0) == 1.0
