"""
fastembed-cloud — Cloud embeddings via AINative API.

Drop-in replacement for fastembed that generates embeddings via API
instead of downloading and running ONNX models locally.

Usage:
    from fastembed_cloud import CloudTextEmbedding

    model = CloudTextEmbedding()
    embeddings = model.embed(["hello world", "semantic search"])

Or use the smart TextEmbedding that prefers local fastembed when installed:

    from fastembed_cloud import TextEmbedding

    model = TextEmbedding()
    embeddings = model.embed(["hello world"])
"""

from fastembed_cloud.embedding import CloudTextEmbedding, TextEmbedding

__version__ = "0.1.0"
__all__ = ["CloudTextEmbedding", "TextEmbedding"]
