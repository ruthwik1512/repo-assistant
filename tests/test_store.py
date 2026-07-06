"""
tests/test_store.py

Unit tests for the store sub-package (VectorStore + SearchResult).

Testing strategy:
  - Use real ChromaDB (EphemeralClient) for fast, in-memory testing.
  - Use FakeBackend to produce controllable, deterministic vectors.
  - Test build and search functionality.
  - Test persistence using temporary directories.
"""

import os
import tempfile
from typing import List

import pytest

from repo_assistant.embedder.backends import EmbeddingBackend
from repo_assistant.embedder.models import EmbeddedChunk, EmbeddedFile
from repo_assistant.indexer.models import IndexedFile
from repo_assistant.store import VectorStore, SearchResult


# =============================================================================
# Test doubles
# =============================================================================

DIMS = 4  # small dimension — makes test assertions easy to reason about

class FakeBackend(EmbeddingBackend):
    """
    Returns controllable deterministic vectors for query embedding.
    """
    def __init__(self, query_vector: List[float] = None):
        self._query_vector = query_vector or [1.0] * DIMS

    @property
    def model_name(self) -> str:
        return "fake-model"

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [list(self._query_vector) for _ in texts]


# =============================================================================
# Helpers
# =============================================================================

def make_indexed_file(relative_path: str = "src/main.py") -> IndexedFile:
    return IndexedFile(
        path=f"/repo/{relative_path}",
        relative_path=relative_path,
        extension=".py",
        content="def hello(): pass",
        line_count=1,
        size_bytes=18,
    )


def make_embedded_file(
    relative_path: str,
    vectors: List[List[float]],
) -> EmbeddedFile:
    """
    Builds an EmbeddedFile with the given pre-computed chunk vectors.
    """
    indexed_file = make_indexed_file(relative_path)
    chunks = [
        EmbeddedChunk(
            content=f"chunk {i} of {relative_path}",
            embedding=vec,
            chunk_index=i,
            start_char=i * 100,
        )
        for i, vec in enumerate(vectors)
    ]
    return EmbeddedFile(
        indexed_file=indexed_file,
        chunks=chunks,
        model_name="fake-model",
    )


def build_store(embedded_files: List[EmbeddedFile], query_vector=None) -> VectorStore:
    """Convenience: build an ephemeral VectorStore and return it."""
    store = VectorStore(FakeBackend(query_vector))
    store.build(embedded_files)
    return store


# =============================================================================
# VectorStore — build()
# =============================================================================

class TestVectorStoreBuild:
    def test_raises_when_no_chunks(self):
        """Building from files with no chunks should raise ValueError."""
        empty_file = EmbeddedFile(
            indexed_file=make_indexed_file(),
            chunks=[],
            model_name="m",
        )
        store = VectorStore(FakeBackend())
        with pytest.raises(ValueError, match="No chunks found"):
            store.build([empty_file])

    def test_size_equals_total_chunks(self):
        """store.size should equal the total number of chunks across all files."""
        ef1 = make_embedded_file("a.py", [[1.0, 0.0, 0.0, 0.0]] * 3)
        ef2 = make_embedded_file("b.py", [[0.0, 1.0, 0.0, 0.0]] * 2)

        store = build_store([ef1, ef2])
        assert store.size == 5

    def test_size_is_zero_before_build(self):
        store = VectorStore(FakeBackend())
        assert store.size == 0


# =============================================================================
# VectorStore — search()
# =============================================================================

class TestVectorStoreSearch:
    def test_raises_when_not_built(self):
        store = VectorStore(FakeBackend())
        with pytest.raises(RuntimeError, match="VectorStore is empty"):
            store.search("some query")

    def test_returns_correct_number_of_results(self):
        ef = make_embedded_file("a.py", [[1.0, 0.0, 0.0, 0.0]] * 5)
        store = build_store([ef])

        results = store.search("query", top_k=3)
        assert len(results) == 3

    def test_top_k_clamped_to_index_size(self):
        """If top_k > number of indexed chunks, return all chunks (no error)."""
        ef = make_embedded_file("a.py", [[1.0, 0.0, 0.0, 0.0]] * 2)
        store = build_store([ef])

        results = store.search("query", top_k=100)
        assert len(results) == 2

    def test_best_match_has_highest_cosine_score(self):
        """
        The chunk whose vector is most aligned with the query vector
        should be the top result.
        """
        ef = make_embedded_file("a.py", [
            [1.0, 0.0, 0.0, 0.0],   # chunk 0
            [0.0, 0.0, 1.0, 0.0],   # chunk 1
            [0.0, 1.0, 0.0, 0.0],   # chunk 2 ← should win
        ])

        # Query aligns with chunk 2's direction
        store = build_store([ef], query_vector=[0.0, 1.0, 0.0, 0.0])
        results = store.search("query", top_k=3)

        best = results[0]
        assert "chunk 2" in best.chunk.content

    def test_result_contains_correct_indexed_file(self):
        ef = make_embedded_file("utils.py", [[1.0, 0.0, 0.0, 0.0]])
        store = build_store([ef])

        results = store.search("query", top_k=1)
        assert results[0].indexed_file.relative_path == "utils.py"

    def test_search_across_multiple_files(self):
        """Results can come from different source files."""
        ef1 = make_embedded_file("alpha.py", [[1.0, 0.0, 0.0, 0.0]])
        ef2 = make_embedded_file("beta.py",  [[0.9, 0.1, 0.0, 0.0]])
        store = build_store([ef1, ef2])

        results = store.search("query", top_k=2)
        paths = {r.indexed_file.relative_path for r in results}
        assert "alpha.py" in paths
        assert "beta.py" in paths


# =============================================================================
# VectorStore — Persistence
# =============================================================================

class TestVectorStorePersistence:
    def test_persistent_client_creates_directory(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store_path = os.path.join(tmp, "mydb")
            VectorStore(FakeBackend(), persist_directory=store_path)
            assert os.path.exists(store_path)

    def test_persistent_client_saves_and_loads(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store_path = os.path.join(tmp, "mydb")
            
            # Create and build
            store1 = VectorStore(FakeBackend(), persist_directory=store_path)
            ef = make_embedded_file("app.py", [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            store1.build([ef])
            assert store1.size == 2

            # Load into a new instance
            store2 = VectorStore(FakeBackend(), persist_directory=store_path)
            assert store2.size == 2
            
            # Ensure it is queryable and metadata survived
            results = store2.search("query", top_k=1)
            assert len(results) == 1
            assert results[0].indexed_file.relative_path == "app.py"
