"""
models.py

Data structures produced by the embedder.

Two classes work together:
  - EmbeddedChunk  : one slice of a file's content + its embedding vector
  - EmbeddedFile   : groups all chunks for a single source file

Why two separate classes instead of one flat structure?
  A file can produce N chunks. If we flattened everything into one object
  we'd have to repeat file metadata (path, language, etc.) for every chunk,
  making downstream code more complex and storage less efficient.
  Nesting EmbeddedChunk inside EmbeddedFile keeps the relationship clear.
"""

from dataclasses import dataclass, field
from typing import List

from repo_assistant.indexer.models import IndexedFile


@dataclass
class EmbeddedChunk:
    """
    One chunk of a source file and its embedding vector.

    Attributes:
        content:     The raw text of this chunk (a slice of the file).
        embedding:   The vector representation as a list of floats.
                     Length depends on the model (e.g. 384 for all-MiniLM-L6-v2).
        chunk_index: Zero-based position of this chunk within its file.
                     Chunk 0 is the start of the file.
        start_char:  Character offset in the original file where this chunk begins.
                     Useful for reconstructing which part of the file a result came from.
    """

    content: str
    embedding: List[float]
    chunk_index: int
    start_char: int

    def __repr__(self) -> str:
        preview = self.content[:40].replace("\n", " ")
        return (
            f"EmbeddedChunk("
            f"index={self.chunk_index}, "
            f"start_char={self.start_char}, "
            f"dims={len(self.embedding)}, "
            f"preview={preview!r})"
        )


@dataclass
class EmbeddedFile:
    """
    All embedding chunks produced from a single source file.

    Attributes:
        indexed_file: The original IndexedFile (path, content, metadata).
                      Stored by reference — no data is duplicated.
        chunks:       The list of EmbeddedChunk objects for this file.
                      Most files produce 1 chunk; large files produce many.
        model_name:   Name of the embedding model used.
                      Stored here so we know how to interpret the vector dimensions.
    """

    indexed_file: IndexedFile
    chunks: List[EmbeddedChunk] = field(default_factory=list)
    model_name: str = ""

    @property
    def chunk_count(self) -> int:
        """Convenience property — avoids len(f.chunks) at call sites."""
        return len(self.chunks)

    def __repr__(self) -> str:
        return (
            f"EmbeddedFile("
            f"relative_path={self.indexed_file.relative_path!r}, "
            f"chunks={self.chunk_count}, "
            f"model={self.model_name!r})"
        )
