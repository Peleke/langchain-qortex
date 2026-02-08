<div align="center">

# langchain-qortex

### LangChain VectorStore Backed by Graph-Enhanced Retrieval

[![PyPI](https://img.shields.io/pypi/v/langchain-qortex?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/langchain-qortex/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-qortex?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/langchain-qortex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

LangChain integration for [qortex](https://github.com/Peleke/qortex) graph-enhanced retrieval.

[Install](#install) · [Quick Start](#quick-start) · [VectorStore Guide](docs/vectorstore.md) · [qortex](https://github.com/Peleke/qortex)

</div>

## Install

```bash
pip install langchain-qortex
```

## What is this?

This package provides `QortexVectorStore`, a LangChain `VectorStore` backed by qortex's knowledge graph. It works anywhere LangChain expects a VectorStore: chains, agents, `as_retriever()`, `similarity_search()`, `add_documents()`, etc.

qortex augments standard vector search with graph structure, rules, and a feedback-driven learning loop. This package gives LangChain users access to those capabilities through the API they already know.

## Quick start

```python
from langchain_qortex import QortexVectorStore

# Zero-config (like Chroma.from_texts)
vs = QortexVectorStore.from_texts(
    texts=["OAuth2 is an authorization framework", "JWT carries signed claims"],
    embedding=my_embedding,
    domain="security",
)

# Standard VectorStore API
docs = vs.similarity_search("authentication", k=5)
retriever = vs.as_retriever()
```

## What qortex adds

Beyond standard vector search, `QortexVectorStore` exposes three capabilities that flat vector stores cannot provide:

```python
# Explore: traverse typed edges from any search result
explore = vs.explore(docs[0].metadata["node_id"])
for edge in explore.edges:
    print(f"{edge.source_id} --{edge.relation_type}--> {edge.target_id}")

# Rules: get projected rules linked to retrieved concepts
rules = vs.rules(concept_ids=[d.metadata["node_id"] for d in docs])
for rule in rules.rules:
    print(f"[{rule.category}] {rule.text}")

# Feedback: close the learning loop
vs.feedback({docs[0].id: "accepted", docs[-1].id: "rejected"})
# Future queries adjust PPR teleportation weights based on this signal
```

## Documentation

- [Full VectorStore guide](docs/vectorstore.md) (graph exploration, rules, feedback loop)
- [qortex docs](https://peleke.github.io/qortex/)
- [Querying guide](https://peleke.github.io/qortex/guides/querying/)

## License

MIT
