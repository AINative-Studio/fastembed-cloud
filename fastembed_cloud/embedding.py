"""
fastembed-cloud — Embedding Classes

CloudTextEmbedding: Always uses the AINative API (no local models).
TextEmbedding: Smart hybrid — uses local fastembed if installed, cloud otherwise.

API endpoint: POST /api/v1/public/embeddings/generate
Request body: {"texts": [...], "model": "bge-m3", "normalize": true}
Response: {"embeddings": [[...], ...], "model": ..., "dimensions": ..., "count": ...}

Refs #3943
"""

import os
from typing import Iterator, List, Optional, Union

import requests

from fastembed_cloud.provision import resolve_api_key

# Model aliases — map fastembed model names to AINative model names
MODEL_MAP = {
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "BAAI/bge-m3": "bge-m3",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "bge-m3": "bge-m3",
}

# Model dimensions for reference
MODEL_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "bge-m3": 1024,
}

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BASE_URL = "https://api.ainative.studio"
GENERATE_PATH = "/api/v1/public/embeddings/generate"


def _chunked(iterable, size):
    """Split an iterable into chunks of the given size."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class CloudTextEmbedding:
    """
    Generate text embeddings via AINative's free Embeddings API.

    Drop-in replacement for fastembed.TextEmbedding — same interface,
    no local model downloads, no ONNX runtime needed.

    Args:
        model_name: Model to use. Defaults to BAAI/bge-small-en-v1.5 (384d).
            Supported: bge-small (384d), bge-base (768d), bge-large (1024d), bge-m3 (1024d).
        api_key: AINative API key. Auto-resolved from env/config/provisioning if not set.
        base_url: API base URL. Defaults to https://api.ainative.studio.
        batch_size: Max texts per API call. Defaults to 64.
        normalize: Normalize embeddings to unit length. Defaults to True.

    Example:
        >>> from fastembed_cloud import CloudTextEmbedding
        >>> model = CloudTextEmbedding()
        >>> embeddings = list(model.embed(["hello world", "semantic search"]))
        >>> len(embeddings[0])  # 384 dimensions
        384
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 64,
        normalize: bool = True,
        **kwargs,  # Accept extra kwargs for fastembed compat
    ):
        self.model_name = MODEL_MAP.get(model_name, model_name)
        self._api_key = resolve_api_key(api_key)
        self._base_url = (
            base_url
            or os.environ.get("AINATIVE_API_URL")
            or os.environ.get("ZERODB_API_URL")
            or DEFAULT_BASE_URL
        )
        self._batch_size = batch_size
        self._normalize = normalize
        self._session = requests.Session()
        self._session.headers.update({
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
        })

    @property
    def dim(self) -> int:
        """Return the embedding dimension for the current model."""
        return MODEL_DIMENSIONS.get(self.model_name, 384)

    def embed(
        self,
        documents: Union[List[str], Iterator[str]],
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            documents: List of text strings to embed (max 100 per batch).
            batch_size: Override default batch size.

        Returns:
            List of embedding vectors (each a list of floats).

        Raises:
            RuntimeError: If the API returns an error.
        """
        docs = list(documents)
        if not docs:
            return []

        bs = batch_size or self._batch_size
        results = []

        for batch in _chunked(docs, min(bs, 100)):
            embeddings = self._call_api(batch)
            results.extend(embeddings)

        return results

    def query_embed(self, query: str, **kwargs) -> List[float]:
        """
        Embed a single query string.

        Args:
            query: Text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        result = self.embed([query])
        if not result:
            raise RuntimeError("Empty response from embedding API")
        return result[0]

    def passage_embed(self, texts: Union[List[str], Iterator[str]], **kwargs) -> List[List[float]]:
        """
        Embed passages (alias for embed, provided for fastembed compatibility).

        Args:
            texts: List of passage strings.

        Returns:
            List of embedding vectors.
        """
        return self.embed(texts, **kwargs)

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """
        Call the AINative embeddings API.

        Args:
            texts: Batch of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: On API error.
        """
        url = f"{self._base_url}{GENERATE_PATH}"
        payload = {
            "texts": texts,
            "model": self.model_name,
            "normalize": self._normalize,
        }

        try:
            resp = self._session.post(url, json=payload, timeout=30)
        except requests.RequestException as exc:
            raise RuntimeError(f"Embedding API request failed: {exc}") from exc

        if resp.status_code == 401:
            raise RuntimeError(
                "Invalid API key. Set AINATIVE_API_KEY or pass api_key= parameter."
            )
        if resp.status_code == 429:
            raise RuntimeError(
                "Rate limited. Wait a moment and try again, or upgrade at https://ainative.studio"
            )
        if resp.status_code != 200:
            detail = resp.text[:200] if resp.text else "Unknown error"
            raise RuntimeError(
                f"Embedding API error (HTTP {resp.status_code}): {detail}"
            )

        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise RuntimeError(f"Unexpected API response format: {list(data.keys())}")

        return embeddings

    def __repr__(self) -> str:
        return f"CloudTextEmbedding(model={self.model_name!r}, dim={self.dim})"


class TextEmbedding:
    """
    Smart embedding: uses local fastembed if installed, falls back to cloud.

    This is a hybrid class that checks for the `fastembed` package at init time.
    If fastembed is available, it uses local ONNX inference (faster, offline).
    If not, it transparently falls back to AINative's cloud API (no downloads).

    Args:
        model_name: Model to use. Defaults to BAAI/bge-small-en-v1.5.
        api_key: AINative API key (only used if falling back to cloud).
        **kwargs: Additional kwargs passed to the underlying embedding class.

    Example:
        >>> from fastembed_cloud import TextEmbedding
        >>> model = TextEmbedding()  # Uses local if fastembed installed, cloud otherwise
        >>> embeddings = list(model.embed(["hello world"]))
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self._local = None
        self._cloud = None

        try:
            from fastembed import TextEmbedding as LocalTextEmbedding
            self._local = LocalTextEmbedding(model_name=model_name, **kwargs)
        except ImportError:
            self._cloud = CloudTextEmbedding(
                model_name=model_name, api_key=api_key, **kwargs
            )

    @property
    def is_cloud(self) -> bool:
        """True if using cloud API, False if using local fastembed."""
        return self._cloud is not None

    @property
    def dim(self) -> int:
        """Return the embedding dimension for the current model."""
        if self._cloud:
            return self._cloud.dim
        # Local fastembed doesn't always expose dim, use our map
        model_name = getattr(self._local, "model_name", DEFAULT_MODEL)
        return MODEL_DIMENSIONS.get(model_name, 384)

    def embed(self, documents: Union[List[str], Iterator[str]], **kwargs) -> List[List[float]]:
        """
        Embed documents using the best available backend.

        Args:
            documents: Texts to embed.

        Returns:
            List of embedding vectors.
        """
        if self._cloud:
            return self._cloud.embed(documents, **kwargs)
        # Local fastembed returns a generator of numpy arrays
        return [emb.tolist() for emb in self._local.embed(list(documents), **kwargs)]

    def query_embed(self, query: str, **kwargs) -> List[float]:
        """Embed a single query."""
        if self._cloud:
            return self._cloud.query_embed(query, **kwargs)
        result = list(self._local.query_embed(query))
        if hasattr(result[0], "tolist"):
            return result[0].tolist()
        return result[0]

    def passage_embed(self, texts: Union[List[str], Iterator[str]], **kwargs) -> List[List[float]]:
        """Embed passages."""
        return self.embed(texts, **kwargs)

    def __repr__(self) -> str:
        backend = "local" if self._local else "cloud"
        return f"TextEmbedding(backend={backend!r})"
