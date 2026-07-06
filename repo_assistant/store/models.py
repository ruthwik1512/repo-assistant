"""
models.py

Data structure returned by VectorStore.search().
"""

from dataclasses import dataclass

from repo_assistant.indexer.models import IndexedFile
from repo_assistant.embedder.models import EmbeddedChunk


@dataclass
class SearchResult:
    """
    One result from a vector similarity search.

    Attributes:
        chunk:        The matching EmbeddedChunk (contains the raw text + embedding).
        indexed_file: The source file this chunk came from.
        score:        Cosine similarity score in range [-1.0, 1.0].
                      Higher is more similar. 1.0 means identical.
        rank:         1-based position in the result list (rank 1 = best match).
    """

    chunk: EmbeddedChunk
    indexed_file: IndexedFile
    score: float
    rank: int

    def __repr__(self) -> str:
        preview = self.chunk.content[:60].replace("\n", " ")
        return (
            f"SearchResult("
            f"rank={self.rank}, "
            f"score={self.score:.4f}, "
            f"file={self.indexed_file.relative_path!r}, "
            f"preview={preview!r})"
        )
