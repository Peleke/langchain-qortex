# QortexVectorStore Guide

`QortexVectorStore` implements LangChain's `VectorStore` ABC. It works anywhere LangChain expects a VectorStore: chains, agents, `as_retriever()`, `similarity_search()`, `from_texts()`, etc.

qortex augments standard vector search with graph structure, rules, and feedback-driven learning. This package gives LangChain users access to those capabilities through the API they already know.

## Setup

### From texts (zero-config)

```python
from langchain_qortex import QortexVectorStore

vs = QortexVectorStore.from_texts(
    texts=[
        "OAuth2 is an authorization framework for API access",
        "JWT tokens carry signed claims between parties",
        "RBAC assigns permissions based on user roles",
    ],
    embedding=my_embedding,
    metadatas=[
        {"source": "handbook", "chapter": "auth"},
        {"source": "handbook", "chapter": "tokens"},
        {"source": "handbook", "chapter": "access"},
    ],
    domain="security",
)
```

This creates an in-memory backend with a numpy vector index, adds the texts as concepts, and returns a ready-to-use VectorStore. Comparable to `Chroma.from_texts()` or `FAISS.from_texts()`.

### From an existing qortex client

```python
from qortex.client import LocalQortexClient
from qortex.core.memory import InMemoryBackend
from qortex.vec.index import NumpyVectorIndex

vector_index = NumpyVectorIndex(dimensions=384)
backend = InMemoryBackend(vector_index=vector_index)
backend.connect()

client = LocalQortexClient(
    vector_index=vector_index,
    backend=backend,
    embedding_model=my_embedding,
    mode="graph",
)

vs = QortexVectorStore(client=client, domain="security")
```

## Standard VectorStore API

### similarity_search

```python
docs = vs.similarity_search("authentication protocols", k=5)

for doc in docs:
    print(doc.page_content)       # Concept content
    print(doc.metadata["score"])   # 0-1 relevance
    print(doc.metadata["domain"])  # Domain name
    print(doc.metadata["node_id"]) # Graph node ID (use for explore())
    print(doc.id)                  # Unique document ID
```

### similarity_search_with_score

```python
results = vs.similarity_search_with_score("OAuth2 auth", k=3)
for doc, score in results:
    print(f"{score:.3f}: {doc.page_content}")
```

### as_retriever

```python
retriever = vs.as_retriever(search_kwargs={"k": 10})
docs = retriever.invoke("how to authenticate API requests")
```

Works in any LangChain chain or agent that accepts a retriever.

### add_texts

```python
ids = vs.add_texts(
    ["Zero-trust architecture assumes no implicit trust"],
    metadatas=[{"source": "new-research"}],
)
```

### get_by_ids

```python
docs = vs.get_by_ids(["sec:oauth", "sec:jwt"])
```

## Graph exploration

After a search, use `node_id` from any result to explore the knowledge graph:

```python
docs = vs.similarity_search("OAuth2 authorization", k=3)
node_id = docs[0].metadata["node_id"]

explore = vs.explore(node_id)

# The node itself
print(explore.node.name)         # "OAuth2"
print(explore.node.description)  # Full description

# Typed edges (structurally related, not just textually similar)
for edge in explore.edges:
    print(f"{edge.source_id} --{edge.relation_type}--> {edge.target_id}")
    # "sec:oauth --requires--> sec:jwt"
    # "sec:oauth --uses--> sec:rbac"

# Neighbor nodes
for neighbor in explore.neighbors:
    print(f"{neighbor.name}: {neighbor.description}")

# Rules linked to this concept
for rule in explore.rules:
    print(f"[{rule.category}] {rule.text}")
```

`explore()` supports depth 1-3 (default 1 = immediate neighbors). Returns `None` if the node doesn't exist.

## Rules in query results

When concepts returned by `similarity_search` have linked rules, those rules appear in the Document metadata automatically:

```python
results = vs.similarity_search_with_score("OAuth2 authorization", k=4)
for doc, score in results:
    if "rules" in doc.metadata:
        for rule in doc.metadata["rules"]:
            print(f"  Rule: {rule['text']} (relevance: {rule['relevance']:.2f})")
```

You can also query rules directly:

```python
# Rules for specific concepts
activated_ids = [doc.metadata["node_id"] for doc in docs]
rules_result = vs.rules(concept_ids=activated_ids)
for rule in rules_result.rules:
    print(f"[{rule.category}] {rule.text}")

# Rules by category
arch_rules = vs.rules(categories=["architectural"])

# Rules by domain
security_rules = vs.rules(domains=["security"])
```

## Feedback loop

Tell qortex which results were useful. Accepted concepts get higher PPR teleportation probability on future queries. Rejected concepts get lower. Over time, results improve.

```python
# Search
docs = vs.similarity_search("authentication", k=5)

# Use the results in your application...

# Then report what worked
vs.feedback({
    docs[0].id: "accepted",   # This was useful
    docs[-1].id: "rejected",  # This was not
})

# Future searches benefit from this signal
docs2 = vs.similarity_search("authentication", k=5)
```

## What's proven

| Claim | Evidence |
|-------|----------|
| VectorStore ABC compliance | `isinstance(vs, VectorStore)` verified in tests |
| similarity_search returns Documents | All results are `langchain_core.documents.Document` |
| from_texts zero-config | Creates backend, indexes, adds texts in one call |
| as_retriever works | `retriever.invoke()` returns Documents |
| Graph exploration | `explore()` returns typed edges, neighbors, linked rules |
| Rules auto-surfaced | Query results include rules in Document metadata |
| Feedback recorded | `feedback()` adjusts PPR teleportation factors |
| 99% code coverage | 58 tests, unit + integration, all passing |

## Embedding interop

langchain-qortex handles embedding format differences automatically:

- **LangChain -> qortex**: `from_texts()` wraps your LangChain `Embeddings` in qortex's `embed()` interface
- **qortex -> LangChain**: The `embeddings` property wraps qortex's model in LangChain's `Embeddings` interface

```python
# Both directions work transparently
from langchain_qortex import QortexEmbeddings, LangChainEmbeddingWrapper

# Wrap qortex model for LangChain use
lc_emb = QortexEmbeddings(my_qortex_model)

# Wrap LangChain model for qortex use
qortex_emb = LangChainEmbeddingWrapper(my_lc_embedding)
```
