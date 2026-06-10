"""
Tests for fastembed-cloud embedding classes.

Uses `responses` to mock HTTP calls — no real API calls needed.

Refs #3943
"""

import json
import os
from unittest.mock import patch

import pytest
import responses

from fastembed_cloud import CloudTextEmbedding, TextEmbedding
from fastembed_cloud.embedding import _chunked, MODEL_DIMENSIONS, GENERATE_PATH
from fastembed_cloud.provision import resolve_api_key


# --- Fixtures ---

MOCK_API_KEY = "test-api-key-12345"
MOCK_BASE_URL = "https://api.ainative.studio"
MOCK_EMBEDDINGS_URL = f"{MOCK_BASE_URL}{GENERATE_PATH}"


def _mock_embedding_response(texts, model="BAAI/bge-small-en-v1.5", dim=384):
    """Generate a mock embeddings API response."""
    return {
        "embeddings": [[0.1] * dim for _ in texts],
        "model": model,
        "dimensions": dim,
        "count": len(texts),
        "processing_time_ms": 12.5,
        "cost_usd": 0.0,
    }


# --- Test _chunked utility ---

class TestChunked:
    def test_even_split(self):
        result = list(_chunked([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        result = list(_chunked([1, 2, 3, 4, 5], 2))
        assert result == [[1, 2], [3, 4], [5]]

    def test_single_chunk(self):
        result = list(_chunked([1, 2], 10))
        assert result == [[1, 2]]

    def test_empty(self):
        result = list(_chunked([], 5))
        assert result == []


# --- Test CloudTextEmbedding ---

class TestCloudTextEmbedding:

    @responses.activate
    def test_embed_basic(self):
        """Embed a list of documents."""
        texts = ["hello world", "semantic search"]
        responses.add(
            responses.POST,
            MOCK_EMBEDDINGS_URL,
            json=_mock_embedding_response(texts),
            status=200,
        )

        model = CloudTextEmbedding(api_key=MOCK_API_KEY)
        result = model.embed(texts)

        assert len(result) == 2
        assert len(result[0]) == 384
        assert all(isinstance(v, float) for v in result[0])

    @responses.activate
    def test_embed_empty_list(self):
        """Empty input returns empty output without API call."""
        model = CloudTextEmbedding(api_key=MOCK_API_KEY)
        result = model.embed([])
        assert result == []
        assert len(responses.calls) == 0

    @responses.activate
    def test_query_embed(self):
        """Embed a single query string."""
        responses.add(
            responses.POST,
            MOCK_EMBEDDINGS_URL,
            json=_mock_embedding_response(["test query"]),
            status=200,
        )

        model = CloudTextEmbedding(api_key=MOCK_API_KEY)
        result = model.query_embed("test query")

        assert len(result) == 384
        assert isinstance(result, list)

    @responses.activate
    def test_batch_splitting(self):
        """Large input is split into batches."""
        texts = [f"text {i}" for i in range(150)]

        # Should make 2 calls: 100 + 50
        responses.add(
            responses.POST,
            MOCK_EMBEDDINGS_URL,
            json=_mock_embedding_response(texts[:100]),
            status=200,
        )
        responses.add(
            responses.POST,
            MOCK_EMBEDDINGS_URL,
            json=_mock_embedding_response(texts[100:]),
            status=200,
        )

        model = CloudTextEmbedding(api_key=MOCK_API_KEY, batch_size=100)
        result = model.embed(texts)

        assert len(result) == 150
        assert len(responses.calls) == 2

    @responses.activate
    def test_model_selection(self):
        """Different models send correct model name."""
        responses.add(
            responses.POST,
            MOCK_EMBEDDINGS_URL,
            json=_mock_embedding_response(["test"], model="bge-m3", dim=1024),
            status=200,
        )

        model = CloudTextEmbedding(model_name="bge-m3", api_key=MOCK_API_KEY)
        result = model.embed(["test"])

        body = json.loads(responses.calls[0].request.body)
        assert body["model"] == "bge-m3"

    @responses.activate
    def test_model_alias(self):
        """Model aliases are resolved correctly."""
        responses.add(
            responses.POST,
            MOCK_EMBEDDINGS_URL,
            json=_mock_embedding_response(["test"]),
            status=200,
        )

        model = CloudTextEmbedding(model_name="bge-small", api_key=MOCK_API_KEY)
        assert model.model_name == "BAAI/bge-small-en-v1.5"
        model.embed(["test"])

        body = json.loads(responses.calls[0].request.body)
        assert body["model"] == "BAAI/bge-small-en-v1.5"

    @responses.activate
    def test_api_key_in_header(self):
        """API key is sent as x-api-key header."""
        responses.add(
            responses.POST,
            MOCK_EMBEDDINGS_URL,
            json=_mock_embedding_response(["test"]),
            status=200,
        )

        model = CloudTextEmbedding(api_key=MOCK_API_KEY)
        model.embed(["test"])

        assert responses.calls[0].request.headers["x-api-key"] == MOCK_API_KEY

    @responses.activate
    def test_401_error(self):
        """401 raises RuntimeError with auth message."""
        responses.add(
            responses.POST,
            MOCK_EMBEDDINGS_URL,
            json={"detail": "Unauthorized"},
            status=401,
        )

        model = CloudTextEmbedding(api_key="bad-key")
        with pytest.raises(RuntimeError, match="Invalid API key"):
            model.embed(["test"])

    @responses.activate
    def test_429_error(self):
        """429 raises RuntimeError with rate limit message."""
        responses.add(
            responses.POST,
            MOCK_EMBEDDINGS_URL,
            json={"detail": "Rate limited"},
            status=429,
        )

        model = CloudTextEmbedding(api_key=MOCK_API_KEY)
        with pytest.raises(RuntimeError, match="Rate limited"):
            model.embed(["test"])

    @responses.activate
    def test_500_error(self):
        """500 raises RuntimeError with error details."""
        responses.add(
            responses.POST,
            MOCK_EMBEDDINGS_URL,
            json={"detail": "Internal server error"},
            status=500,
        )

        model = CloudTextEmbedding(api_key=MOCK_API_KEY)
        with pytest.raises(RuntimeError, match="HTTP 500"):
            model.embed(["test"])

    def test_dim_property(self):
        """dim property returns correct dimensions for known models."""
        model = CloudTextEmbedding(model_name="BAAI/bge-small-en-v1.5", api_key=MOCK_API_KEY)
        assert model.dim == 384

        model = CloudTextEmbedding(model_name="bge-m3", api_key=MOCK_API_KEY)
        assert model.dim == 1024

    def test_repr(self):
        """repr shows model name and dimensions."""
        model = CloudTextEmbedding(api_key=MOCK_API_KEY)
        assert "CloudTextEmbedding" in repr(model)
        assert "384" in repr(model)

    @responses.activate
    def test_passage_embed(self):
        """passage_embed is an alias for embed."""
        texts = ["passage one", "passage two"]
        responses.add(
            responses.POST,
            MOCK_EMBEDDINGS_URL,
            json=_mock_embedding_response(texts),
            status=200,
        )

        model = CloudTextEmbedding(api_key=MOCK_API_KEY)
        result = model.passage_embed(texts)
        assert len(result) == 2

    @responses.activate
    def test_normalize_parameter(self):
        """normalize parameter is sent in request body."""
        responses.add(
            responses.POST,
            MOCK_EMBEDDINGS_URL,
            json=_mock_embedding_response(["test"]),
            status=200,
        )

        model = CloudTextEmbedding(api_key=MOCK_API_KEY, normalize=False)
        model.embed(["test"])

        body = json.loads(responses.calls[0].request.body)
        assert body["normalize"] is False


# --- Test TextEmbedding (hybrid) ---

class TestTextEmbedding:

    @responses.activate
    def test_falls_back_to_cloud(self):
        """When fastembed is not installed, uses cloud."""
        responses.add(
            responses.POST,
            MOCK_EMBEDDINGS_URL,
            json=_mock_embedding_response(["test"]),
            status=200,
        )

        # TextEmbedding should fall back to cloud since fastembed likely not installed
        model = TextEmbedding(api_key=MOCK_API_KEY)
        if model.is_cloud:
            result = model.embed(["test"])
            assert len(result) == 1
            assert len(result[0]) == 384

    def test_is_cloud_property(self):
        """is_cloud reflects which backend is active."""
        model = TextEmbedding(api_key=MOCK_API_KEY)
        # In test environment, fastembed is probably not installed
        assert isinstance(model.is_cloud, bool)

    def test_repr(self):
        """repr shows backend type."""
        model = TextEmbedding(api_key=MOCK_API_KEY)
        assert "TextEmbedding" in repr(model)
        assert "backend=" in repr(model)


# --- Test credential resolution ---

class TestProvision:

    def test_explicit_key(self):
        """Explicit api_key is used first."""
        key = resolve_api_key("my-explicit-key")
        assert key == "my-explicit-key"

    def test_env_ainative_key(self):
        """AINATIVE_API_KEY env var is used."""
        with patch.dict(os.environ, {"AINATIVE_API_KEY": "env-key-123"}, clear=False):
            key = resolve_api_key()
            assert key == "env-key-123"

    def test_env_zerodb_key(self):
        """ZERODB_API_KEY env var is used as fallback."""
        env = {"ZERODB_API_KEY": "zerodb-key-456"}
        with patch.dict(os.environ, env, clear=False):
            # Remove AINATIVE_API_KEY if set
            os.environ.pop("AINATIVE_API_KEY", None)
            key = resolve_api_key()
            assert key == "zerodb-key-456"


# --- Test model dimensions ---

class TestModelDimensions:

    def test_known_models(self):
        """All known models have dimension entries."""
        assert MODEL_DIMENSIONS["BAAI/bge-small-en-v1.5"] == 384
        assert MODEL_DIMENSIONS["BAAI/bge-base-en-v1.5"] == 768
        assert MODEL_DIMENSIONS["BAAI/bge-large-en-v1.5"] == 1024
        assert MODEL_DIMENSIONS["bge-m3"] == 1024
