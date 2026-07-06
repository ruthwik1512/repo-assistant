"""
embedder sub-package

Exposes three public components:
  - EmbeddingBackend          : abstract interface for embedding providers
  - SentenceTransformerBackend: local embedding backend (sentence-transformers)
  - FileEmbedder              : chunks + embeds IndexedFile objects
  - EmbeddedFile              : result object grouping all chunks for one file
  - EmbeddedChunk             : one chunk of a file and its embedding vector
"""

from .backends import EmbeddingBackend, SentenceTransformerBackend
from .embedder import FileEmbedder
from .models import EmbeddedChunk, EmbeddedFile

__all__ = [
    "EmbeddingBackend",
    "SentenceTransformerBackend",
    "FileEmbedder",
    "EmbeddedChunk",
    "EmbeddedFile",
]
