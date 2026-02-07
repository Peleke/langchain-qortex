"""E2E integration: QortexVectorStore as a LangChain VectorStore drop-in.

Proves the full integration: from_texts, search, explore, rules, feedback.
If these tests pass, the package is shippable.
"""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_qortex import QortexVectorStore
from qortex.client import LocalQortexClient
from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptEdge, ConceptNode, ExplicitRule, RelationType
from qortex.vec.index import NumpyVectorIndex

DIMS = 32


class FakeLCEmbedding(Embeddings):
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


class FakeQortexEmbedding:
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


def make_security_graph():
    """Build a security knowledge graph with concepts, edges, and rules."""
    vector_index = NumpyVectorIndex(dimensions=DIMS)
    backend = InMemoryBackend(vector_index=vector_index)
    backend.connect()
    embedding = FakeQortexEmbedding()

    backend.create_domain("security")

    nodes = [
        ConceptNode(
            id="sec:oauth", name="OAuth2",
            description="OAuth2 authorization framework for delegated access to APIs",
            domain="security", source_id="security-handbook",
        ),
        ConceptNode(
            id="sec:jwt", name="JWT",
            description="JSON Web Tokens for stateless authentication and claims transfer",
            domain="security", source_id="security-handbook",
        ),
        ConceptNode(
            id="sec:rbac", name="RBAC",
            description="Role-based access control restricts system access by user roles",
            domain="security", source_id="security-handbook",
        ),
        ConceptNode(
            id="sec:mfa", name="MFA",
            description="Multi-factor authentication requires multiple verification factors",
            domain="security", source_id="security-handbook",
        ),
    ]

    for node in nodes:
        backend.add_node(node)

    texts = [f"{n.name}: {n.description}" for n in nodes]
    embeddings = embedding.embed(texts)
    for node, emb in zip(nodes, embeddings):
        backend.add_embedding(node.id, emb)

    ids = [n.id for n in nodes]
    vector_index.add(ids, embeddings)

    backend.add_edge(ConceptEdge(
        source_id="sec:oauth", target_id="sec:jwt",
        relation_type=RelationType.REQUIRES,
    ))
    backend.add_edge(ConceptEdge(
        source_id="sec:oauth", target_id="sec:rbac",
        relation_type=RelationType.USES,
    ))
    backend.add_edge(ConceptEdge(
        source_id="sec:mfa", target_id="sec:oauth",
        relation_type=RelationType.SUPPORTS,
    ))

    backend.add_rule(ExplicitRule(
        id="rule:oauth-required", text="Always use OAuth2 for third-party API access",
        domain="security", source_id="security-handbook",
        concept_ids=["sec:oauth"], category="security",
    ))
    backend.add_rule(ExplicitRule(
        id="rule:rotate-jwt", text="Rotate JWT signing keys every 90 days",
        domain="security", source_id="security-handbook",
        concept_ids=["sec:oauth", "sec:jwt"], category="operations",
    ))
    backend.add_rule(ExplicitRule(
        id="rule:rbac-before-code", text="Define RBAC roles before writing authorization code",
        domain="security", source_id="security-handbook",
        concept_ids=["sec:rbac"], category="architectural",
    ))

    client = LocalQortexClient(
        vector_index=vector_index,
        backend=backend,
        embedding_model=embedding,
        mode="graph",
    )
    return client


class TestFromTextsOneliner:
    """from_texts: one line to a working VectorStore."""

    def test_from_texts_and_search(self):
        vs = QortexVectorStore.from_texts(
            texts=[
                "OAuth2 is an authorization framework for API access",
                "JWT tokens carry signed claims between parties",
                "RBAC assigns permissions based on user roles",
            ],
            embedding=FakeLCEmbedding(),
            metadatas=[
                {"source": "handbook", "chapter": "auth"},
                {"source": "handbook", "chapter": "tokens"},
                {"source": "handbook", "chapter": "access"},
            ],
            domain="security",
        )

        assert isinstance(vs, VectorStore)
        docs = vs.similarity_search("authentication tokens", k=2)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_from_texts_as_retriever(self):
        vs = QortexVectorStore.from_texts(
            texts=["Python is great", "Rust is fast", "Go is concurrent"],
            embedding=FakeLCEmbedding(),
            domain="languages",
        )

        retriever = vs.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke("fast programming language")
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)


class TestSimilaritySearchE2E:
    """Standard VectorStore search."""

    def test_similarity_search(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        docs = vs.similarity_search("OAuth2 authorization", k=3)
        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.metadata["domain"] == "security"
            assert "node_id" in doc.metadata

    def test_similarity_search_with_score(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        results = vs.similarity_search_with_score("JWT tokens", k=3)
        assert len(results) > 0
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_add_then_search(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        ids = vs.add_texts(
            ["Zero-trust architecture assumes no implicit trust"],
            metadatas=[{"source": "new-research"}],
        )
        assert len(ids) == 1

        docs = vs.similarity_search("zero trust", k=5)
        assert len(docs) > 0


class TestGraphExplorationE2E:
    """Search, then explore the graph from results."""

    def test_search_then_explore(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        docs = vs.similarity_search("OAuth2 authorization", k=3)
        assert len(docs) > 0

        node_id = docs[0].metadata["node_id"]
        explore = vs.explore(node_id)
        assert explore is not None
        assert explore.node.id == node_id
        assert len(explore.edges) > 0
        assert len(explore.neighbors) > 0

    def test_explore_reveals_rules(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        explore = vs.explore("sec:oauth")
        assert len(explore.rules) > 0
        rule_ids = {r.id for r in explore.rules}
        assert "rule:oauth-required" in rule_ids


class TestRulesProjectionE2E:
    """rules() returns projected rules from the knowledge graph."""

    def test_rules_for_retrieved_concepts(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        docs = vs.similarity_search("OAuth2 authorization", k=3)
        activated_ids = [doc.metadata["node_id"] for doc in docs]

        rules_result = vs.rules(concept_ids=activated_ids)
        assert len(rules_result.rules) > 0
        assert rules_result.projection == "rules"

    def test_rules_by_category(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        arch_rules = vs.rules(categories=["architectural"])
        assert all(r.category == "architectural" for r in arch_rules.rules)


class TestFeedbackLoopE2E:
    """feedback() closes the learning loop."""

    def test_search_feedback_re_search(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        docs1 = vs.similarity_search("authentication protocol", k=4)
        assert len(docs1) > 0

        vs.feedback({docs1[0].id: "accepted"})

        docs2 = vs.similarity_search("authentication protocol", k=4)
        assert len(docs2) > 0

    def test_feedback_tracked_by_query_id(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        vs.similarity_search("OAuth2", k=2)
        qid1 = vs.last_query_id
        assert qid1 is not None

        vs.similarity_search("JWT", k=2)
        qid2 = vs.last_query_id
        assert qid2 is not None
        assert qid1 != qid2


class TestFullIntegrationLoop:
    """The full workflow: create, search, explore, rules, feedback, repeat."""

    def test_complete_vectorstore_workflow(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        docs = vs.similarity_search("how to authenticate API requests", k=4)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

        top_node = docs[0].metadata["node_id"]
        explore = vs.explore(top_node)
        assert explore is not None

        activated = [d.metadata["node_id"] for d in docs]
        rules_result = vs.rules(concept_ids=activated)
        assert isinstance(rules_result.rules, list)

        outcomes = {docs[0].id: "accepted"}
        if len(docs) > 1:
            outcomes[docs[-1].id] = "rejected"
        vs.feedback(outcomes)

        docs2 = vs.similarity_search("how to authenticate API requests", k=4)
        assert len(docs2) > 0
