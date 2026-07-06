"""
tests/test_embedder.py

Unit tests for the embedder sub-package.

Testing strategy:
  - Never load a real model in tests (slow, ~90MB download, GPU dependency).
  - Use a FakeBackend that returns deterministic dummy vectors.
  - Test _chunk_text directly — it is pure logic with no dependencies.
  - Test FileEmbedder.embed by checking structure, ordering, and chunk assembly.
  - Test EmbeddedFile and EmbeddedChunk field assignment and repr.
"""

from typing import List

import pytest

from repo_assistant.embedder import (
    EmbeddingBackend,
    FileEmbedder,
    EmbeddedChunk,
    EmbeddedFile,
)
from repo_assistant.indexer.models import IndexedFile


# =============================================================================
# Test doubles (fakes)
# =============================================================================

class FakeBackend(EmbeddingBackend):
    """
    A minimal EmbeddingBackend that returns predictable dummy vectors.

    Why a fake instead of a mock (unittest.mock.MagicMock)?
      A fake is a real implementation of the interface — it just does something
      simpler. Here it returns a fixed-length vector of 1.0 values.
      Fakes are preferred over mocks when you want to test the interaction
      between two real objects (FileEmbedder + a backend) without using
      a heavy model. Mocks would work too, but fakes make assertions clearer.

    DIMS controls the output vector length. Kept small (4) to make
    test assertions readable.
    """

    DIMS = 4

    def __init__(self, model: str = "fake-model"):
        self._model_name = model
        self.call_count = 0       # lets tests assert how many times embed_batch ran
        self.last_input: List[str] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        self.last_input = texts
        # Return one [1.0, 1.0, 1.0, 1.0] vector per text
        return [[1.0] * self.DIMS for _ in texts]


# =============================================================================
# Helpers
# =============================================================================

def make_indexed_file(
    relative_path: str = "src/main.py",
    content: str = "def hello(): pass",
    extension: str = ".py",
) -> IndexedFile:
    """Build an IndexedFile with sensible defaults."""
    return IndexedFile(
        path=f"/repo/{relative_path}",
        relative_path=relative_path,
        extension=extension,
        content=content,
        line_count=content.count("\n") + 1,
        size_bytes=len(content.encode()),
    )


# =============================================================================
# EmbeddedChunk and EmbeddedFile tests
# =============================================================================

class TestEmbeddedChunk:
    def test_fields_are_set_correctly(self):
        chunk = EmbeddedChunk(
            content="def foo(): pass",
            embedding=[0.1, 0.2, 0.3],
            chunk_index=0,
            start_char=0,
        )
        assert chunk.content == "def foo(): pass"
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.chunk_index == 0
        assert chunk.start_char == 0

    def test_repr_shows_preview_not_full_content(self):
        chunk = EmbeddedChunk(
            content="x" * 200,
            embedding=[0.0] * 384,
            chunk_index=2,
            start_char=500,
        )
        r = repr(chunk)
        # Key metadata present
        assert "index=2" in r
        assert "start_char=500" in r
        assert "dims=384" in r
        # Full content not dumped
        assert "x" * 100 not in r


class TestEmbeddedFile:
    def _make(self, chunk_count: int = 2) -> EmbeddedFile:
        chunks = [
            EmbeddedChunk(content="x", embedding=[1.0], chunk_index=i, start_char=i * 10)
            for i in range(chunk_count)
        ]
        return EmbeddedFile(
            indexed_file=make_indexed_file(),
            chunks=chunks,
            model_name="test-model",
        )

    def test_chunk_count_property(self):
        ef = self._make(chunk_count=3)
        assert ef.chunk_count == 3

    def test_chunk_count_zero_when_empty(self):
        ef = EmbeddedFile(indexed_file=make_indexed_file(), chunks=[], model_name="m")
        assert ef.chunk_count == 0

    def test_repr_contains_key_info(self):
        ef = self._make(chunk_count=5)
        r = repr(ef)
        assert "src/main.py" in r
        assert "chunks=5" in r
        assert "test-model" in r


# =============================================================================
# FakeBackend tests — validates our test double is correct
# =============================================================================

class TestFakeBackend:
    def test_returns_correct_number_of_vectors(self):
        backend = FakeBackend()
        result = backend.embed_batch(["hello", "world", "foo"])
        assert len(result) == 3

    def test_each_vector_has_correct_dims(self):
        backend = FakeBackend()
        result = backend.embed_batch(["a", "b"])
        assert all(len(v) == FakeBackend.DIMS for v in result)

    def test_model_name_property(self):
        backend = FakeBackend(model="my-model")
        assert backend.model_name == "my-model"


# =============================================================================
# FileEmbedder — constructor validation
# =============================================================================

class TestFileEmbedderInit:
    def test_raises_if_overlap_exceeds_chunk_size(self):
        """overlap must be strictly less than chunk_size."""
        with pytest.raises(ValueError, match="overlap"):
            FileEmbedder(FakeBackend(), chunk_size=100, overlap=100)

    def test_raises_if_overlap_greater_than_chunk_size(self):
        with pytest.raises(ValueError, match="overlap"):
            FileEmbedder(FakeBackend(), chunk_size=100, overlap=200)

    def test_valid_params_do_not_raise(self):
        # Should construct without error
        embedder = FileEmbedder(FakeBackend(), chunk_size=500, overlap=50)
        assert embedder.chunk_size == 500
        assert embedder.overlap == 50


# =============================================================================
# FileEmbedder — _chunk_text
# =============================================================================

class TestChunkText:
    """
    Tests for the private _chunk_text method.

    We test it directly because it contains non-trivial logic (overlap
    arithmetic, boundary conditions) that is easier to verify in isolation.
    """

    def setup_method(self):
        self.embedder = FileEmbedder(FakeBackend(), chunk_size=10, overlap=3)

    def test_empty_string_returns_no_chunks(self):
        assert self.embedder._chunk_text("") == []

    def test_short_text_produces_single_chunk(self):
        """Text shorter than chunk_size should produce exactly one chunk."""
        result = self.embedder._chunk_text("hello")
        assert len(result) == 1
        assert result[0] == ("hello", 0)

    def test_exact_chunk_size_produces_single_chunk(self):
        text = "A" * 10  # exactly chunk_size
        result = self.embedder._chunk_text(text)
        assert len(result) == 1
        assert result[0] == (text, 0)

    def test_long_text_produces_multiple_chunks(self):
        # chunk_size=10, overlap=3 → step=7
        # "AAAAAAAAAA BBBBBBBBB..." → chunk at 0, 7, 14, ...
        text = "X" * 25
        result = self.embedder._chunk_text(text)
        assert len(result) > 1

    def test_first_chunk_starts_at_zero(self):
        result = self.embedder._chunk_text("hello world foo bar")
        assert result[0][1] == 0   # start_char of first chunk

    def test_chunks_advance_by_chunk_size_minus_overlap(self):
        """Each chunk's start_char should be chunk_size - overlap ahead of the previous."""
        text = "A" * 50
        result = self.embedder._chunk_text(text)
        expected_step = self.embedder.chunk_size - self.embedder.overlap  # 10 - 3 = 7
        for i in range(1, len(result)):
            assert result[i][1] - result[i - 1][1] == expected_step

    def test_overlap_region_appears_in_consecutive_chunks(self):
        """The last `overlap` chars of chunk N should appear at the start of chunk N+1."""
        # chunk_size=10, overlap=3 → step=7
        # chunk 0: text[0:10], chunk 1: text[7:17]
        # overlap region: text[7:10] (3 chars)
        text = "ABCDEFGHIJKLMNOPQRST"
        result = self.embedder._chunk_text(text)

        if len(result) >= 2:
            end_of_first = result[0][0][-self.embedder.overlap:]   # last 3 chars of chunk 0
            start_of_second = result[1][0][:self.embedder.overlap]  # first 3 chars of chunk 1
            assert end_of_first == start_of_second

    def test_no_chunk_exceeds_chunk_size(self):
        text = "Z" * 100
        result = self.embedder._chunk_text(text)
        assert all(len(chunk) <= self.embedder.chunk_size for chunk, _ in result)


# =============================================================================
# FileEmbedder — embed()
# =============================================================================

class TestFileEmbedderEmbed:
    def setup_method(self):
        self.backend = FakeBackend()
        self.embedder = FileEmbedder(self.backend, chunk_size=50, overlap=10)

    def test_empty_input_returns_empty_list(self):
        result = self.embedder.embed([])
        assert result == []

    def test_returns_one_embedded_file_per_input(self):
        files = [make_indexed_file("a.py"), make_indexed_file("b.py")]
        result = self.embedder.embed(files)
        assert len(result) == 2

    def test_output_order_matches_input_order(self):
        """EmbeddedFile at index i must correspond to indexed_files[i]."""
        files = [
            make_indexed_file("first.py"),
            make_indexed_file("second.py"),
            make_indexed_file("third.py"),
        ]
        result = self.embedder.embed(files)
        for i, ef in enumerate(result):
            assert ef.indexed_file is files[i]

    def test_model_name_stored_on_each_file(self):
        backend = FakeBackend(model="custom-model")
        embedder = FileEmbedder(backend, chunk_size=50, overlap=5)
        result = embedder.embed([make_indexed_file()])
        assert result[0].model_name == "custom-model"

    def test_embed_batch_called_exactly_once(self):
        """
        All chunks from all files must be sent in ONE batch call.
        Multiple calls would mean we are not batching properly.
        """
        # Use large content to force multiple chunks per file
        long_content = "x " * 200
        files = [
            make_indexed_file("a.py", content=long_content),
            make_indexed_file("b.py", content=long_content),
        ]
        self.embedder.embed(files)
        assert self.backend.call_count == 1

    def test_each_chunk_has_correct_vector_dims(self):
        result = self.embedder.embed([make_indexed_file(content="hello world")])
        for chunk in result[0].chunks:
            assert len(chunk.embedding) == FakeBackend.DIMS

    def test_empty_file_content_produces_no_chunks(self):
        result = self.embedder.embed([make_indexed_file(content="")])
        assert result[0].chunk_count == 0

    def test_short_file_produces_exactly_one_chunk(self):
        result = self.embedder.embed([make_indexed_file(content="tiny")])
        assert result[0].chunk_count == 1

    def test_chunk_index_is_sequential(self):
        long_content = "word " * 100
        result = self.embedder.embed([make_indexed_file(content=long_content)])
        chunks = result[0].chunks
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_start_char_of_first_chunk_is_zero(self):
        result = self.embedder.embed([make_indexed_file(content="hello world")])
        assert result[0].chunks[0].start_char == 0
