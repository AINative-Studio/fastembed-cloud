# fastembed-cloud

Cloud text embeddings via AINative API. Drop-in replacement for [fastembed](https://github.com/qdrant/fastembed) — same interface, no model downloads, no ONNX runtime.

## Why?

fastembed is great, but requires downloading 100MB+ ONNX models locally. `fastembed-cloud` gives you the same API backed by a free cloud service — zero setup, zero downloads.

| | fastembed | fastembed-cloud |
|---|---|---|
| First-run latency | 30-120s (model download) | 0s |
| Disk usage | 100MB-1GB per model | 0 |
| ONNX runtime | Required | Not needed |
| Offline support | Yes | No (cloud API) |
| Cost | Free (local compute) | Free (AINative API) |

## Install

```bash
pip install fastembed-cloud
```

## Quick Start

```python
from fastembed_cloud import CloudTextEmbedding

# Auto-provisions a free API key on first run
model = CloudTextEmbedding()

# Single query
embedding = model.query_embed("What is semantic search?")
print(f"Dimensions: {len(embedding)}")  # 384

# Batch embedding
docs = ["First document", "Second document", "Third document"]
embeddings = model.embed(docs)
print(f"Embedded {len(embeddings)} documents")
```

## Models

| Model | Dimensions | ID |
|---|---|---|
| BGE-small-en-v1.5 | 384 | `BAAI/bge-small-en-v1.5` (default) |
| BGE-base-en-v1.5 | 768 | `BAAI/bge-base-en-v1.5` |
| BGE-large-en-v1.5 | 1024 | `BAAI/bge-large-en-v1.5` |
| BGE-M3 (multilingual) | 1024 | `bge-m3` |

```python
model = CloudTextEmbedding(model_name="bge-m3")
embedding = model.query_embed("Multilingual embedding")
print(len(embedding))  # 1024
```

## Smart Hybrid: Local + Cloud

`TextEmbedding` automatically uses local fastembed if installed, cloud otherwise:

```python
from fastembed_cloud import TextEmbedding

model = TextEmbedding()
print(model.is_cloud)  # True if fastembed not installed

embeddings = model.embed(["works either way"])
```

Install fastembed alongside for local-first with cloud fallback:

```bash
pip install fastembed-cloud[local]
```

## Authentication

Credentials are resolved in this order:

1. `api_key` parameter: `CloudTextEmbedding(api_key="your-key")`
2. `AINATIVE_API_KEY` environment variable
3. `ZERODB_API_KEY` environment variable (shared with ZeroDB ecosystem)
4. `~/.zerodb/credentials.json` (auto-saved from any ZeroDB tool)
5. Auto-provisioning (free 72-hour account, claim to keep permanently)

### Auto-Provisioning

On first use with no credentials, fastembed-cloud automatically provisions a free account:

```
$ python -c "from fastembed_cloud import CloudTextEmbedding; CloudTextEmbedding().query_embed('test')"

  No API key found — provisioning a free AINative account for embeddings...

  Auto-provisioned! Free embeddings API ready.

  API Key:   zdb_abc12345...
  Expires:   72 hours
  Saved to:  ~/.zerodb/credentials.json

  To keep access permanently, claim your account:
  https://ainative.studio/signup
```

## Batch Embedding

Handles large datasets efficiently with automatic batching:

```python
model = CloudTextEmbedding(batch_size=100)

# Automatically batches into chunks of 100
large_dataset = ["document " + str(i) for i in range(10000)]
embeddings = model.embed(large_dataset)
```

## Use with Vector Databases

### Qdrant

```python
from qdrant_client import QdrantClient
from fastembed_cloud import CloudTextEmbedding

client = QdrantClient(":memory:")
model = CloudTextEmbedding()

docs = ["AI is transforming healthcare", "Machine learning for finance"]
embeddings = model.embed(docs)

client.add(
    collection_name="my_docs",
    documents=docs,
    embeddings=embeddings,
)
```

### ChromaDB

```python
import chromadb
from fastembed_cloud import CloudTextEmbedding

client = chromadb.Client()
collection = client.create_collection("my_docs")
model = CloudTextEmbedding()

docs = ["First doc", "Second doc"]
embeddings = model.embed(docs)

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=["id1", "id2"],
)
```

## API Reference

### CloudTextEmbedding

Always uses the cloud API.

```python
CloudTextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",  # Model to use
    api_key=None,                           # API key (auto-resolved)
    base_url=None,                          # API URL (default: api.ainative.studio)
    batch_size=64,                          # Max texts per API call
    normalize=True,                         # Normalize to unit vectors
)
```

**Methods:**
- `embed(documents, batch_size=None)` — Embed a list of texts. Returns `list[list[float]]`.
- `query_embed(query)` — Embed a single query. Returns `list[float]`.
- `passage_embed(texts)` — Alias for `embed()` (fastembed compatibility).

**Properties:**
- `dim` — Embedding dimensions (e.g., 384 for bge-small).
- `model_name` — Resolved model name.

### TextEmbedding

Smart hybrid: local fastembed if available, cloud otherwise.

```python
TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    api_key=None,
    **kwargs,
)
```

**Properties:**
- `is_cloud` — `True` if using cloud API, `False` if using local fastembed.

## License

MIT
