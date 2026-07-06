"""
embedder.py

Responsibility: Orchestrate chunking + embedding for a list of IndexedFile objects.

FileEmbedder does three things in sequence:
  1. Chunk  — split each file's content into overlapping text windows
  2. Batch  — collect all chunks across all files into one flat list
  3. Embed  — send the flat list to the backend in one call, then
              reassemble results back into per-file EmbeddedFile objects

Why flatten into one batch before embedding?
  Embedding models process all texts in a batch together (one forward pass).
  Sending 500 chunks in one call is roughly as fast as sending 1.
  Sending them one by one would be ~500× slower.
"""

from collections import defaultdict
from typing import List, Tuple

from repo_assistant.indexer.models import IndexedFile
from .backends import EmbeddingBackend
from .models import EmbeddedChunk, EmbeddedFile


class FileEmbedder:
    """
    Chunks and embeds a list of IndexedFile objects.

    Parameters are set at construction so they can be reused across
    multiple .embed() calls without repetition.

    Example:
        backend = SentenceTransformerBackend()
        embedder = FileEmbedder(backend)
        embedded_files = embedder.embed(indexed_files)
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        chunk_size: int = 1_500,
        overlap: int = 200,
    ) -> None:
        """
        Args:
            backend:    The embedding provider (SentenceTransformerBackend, etc.)
            chunk_size: Maximum number of characters per chunk.
                        all-MiniLM-L6-v2 handles ~512 tokens; 1500 chars ≈ 300-400
                        tokens for typical code, leaving headroom.
            overlap:    Number of characters shared between consecutive chunks.
                        Prevents a logical unit (e.g. a function) from being
                        split in half with no context on either side.
        """
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})."
            )
        self.backend = backend
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, indexed_files: List[IndexedFile]) -> List[EmbeddedFile]:
        """
        Chunks and embeds all files, returning one EmbeddedFile per input file.

        Args:
            indexed_files: Output of FileIndexer.index().

        Returns:
            List of EmbeddedFile objects in the same order as indexed_files.
            Files with empty content produce an EmbeddedFile with zero chunks.
        """
        if not indexed_files:
            return []

        # ------------------------------------------------------------------
        # Stage 1: Chunk every file and record which file each chunk belongs to
        # ------------------------------------------------------------------
        # flat_chunks[i] = (file_index, chunk_index, chunk_text, start_char)
        flat_chunks: List[Tuple[int, int, str, int]] = []

        for file_idx, indexed_file in enumerate(indexed_files):
            raw_chunks = self._chunk_text(indexed_file.content)
            for chunk_idx, (text, start_char) in enumerate(raw_chunks):
                flat_chunks.append((file_idx, chunk_idx, text, start_char))

        if not flat_chunks:
            # All files were empty — return EmbeddedFile shells with no chunks
            return [
                EmbeddedFile(
                    indexed_file=f,
                    chunks=[],
                    model_name=self.backend.model_name,
                )
                for f in indexed_files
            ]

        # ------------------------------------------------------------------
        # Stage 2: One batch call to the embedding model
        # ------------------------------------------------------------------
        print(
            f"[embedder] Embedding {len(flat_chunks)} chunk(s) "
            f"from {len(indexed_files)} file(s) ..."
        )

        texts = [chunk[2] for chunk in flat_chunks]
        vectors = self.backend.embed_batch(texts)

        print(f"[embedder] Done. Vector dims: {len(vectors[0])}.")

        # ------------------------------------------------------------------
        # Stage 3: Reassemble chunks back into per-file EmbeddedFile objects
        # ------------------------------------------------------------------
        # Group EmbeddedChunk objects by their file index
        file_chunk_map: dict = defaultdict(list)

        for i, (file_idx, chunk_idx, text, start_char) in enumerate(flat_chunks):
            file_chunk_map[file_idx].append(
                EmbeddedChunk(
                    content=text,
                    embedding=vectors[i],
                    chunk_index=chunk_idx,
                    start_char=start_char,
                )
            )

        return [
            EmbeddedFile(
                indexed_file=indexed_files[file_idx],
                chunks=file_chunk_map[file_idx],
                model_name=self.backend.model_name,
            )
            for file_idx in range(len(indexed_files))
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> List[Tuple[str, int]]:
        """
        Splits text into overlapping windows of chunk_size characters.

        Returns:
            A list of (chunk_text, start_char) pairs.
            start_char is the character offset of the chunk in the original text.

        Example (chunk_size=10, overlap=3):
            text = "ABCDEFGHIJKLMNOP"
            → [("ABCDEFGHIJ", 0), ("HIJKLMNOP", 7)]
                                    ^^^  overlapping region

        Edge cases:
            - Empty string → returns empty list (no chunks)
            - Text shorter than chunk_size → one chunk containing the full text
        """
        if not text:
            return []

        chunks: List[Tuple[str, int]] = []
        start = 0
        step = self.chunk_size - self.overlap  # how far to advance each iteration

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append((chunk_text, start))

            # If this chunk reached the end of the text, we're done
            if end >= len(text):
                break

            start += step

        return chunks
