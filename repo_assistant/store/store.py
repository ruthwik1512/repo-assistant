"""
store.py

Responsibility: Build a searchable vector index from EmbeddedFile objects
and answer semantic similarity queries against it using ChromaDB.

Why ChromaDB instead of FAISS?
  - ChromaDB is a full Vector Database. It stores vectors, text, and metadata
    all together in one place.
  - With FAISS, we had to maintain a separate Pickle file for metadata.
  - ChromaDB handles persistence natively.
"""

import hashlib
import os
from typing import List, Optional
import uuid

import chromadb

from repo_assistant.embedder.backends import EmbeddingBackend
from repo_assistant.embedder.models import EmbeddedChunk, EmbeddedFile
from repo_assistant.indexer.models import IndexedFile
from .models import SearchResult


class VectorStore:
    """
    Stores EmbeddedChunk vectors in ChromaDB and supports semantic search.

    Lifecycle (Ephemeral - In Memory):
        store = VectorStore(backend)
        store.build(embedded_files)
        results = store.search("query")

    Lifecycle (Persistent - On Disk):
        store = VectorStore(backend, persist_directory="./store_dir")
        store.build(embedded_files)  # auto-persists
        
        # Later, just load it:
        store = VectorStore(backend, persist_directory="./store_dir")
        results = store.search("query")
    """

    def __init__(
        self, 
        backend: EmbeddingBackend, 
        persist_directory: Optional[str] = None
    ) -> None:
        """
        Args:
            backend: The EmbeddingBackend used to create query embeddings.
            persist_directory: If provided, data is saved to and loaded from this path.
                               If None, an in-memory database is used.
        """
        self.backend = backend
        self.persist_directory = persist_directory

        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self.client = chromadb.EphemeralClient()

        # hnsw:space = "cosine" aligns with our sentence-transformer embeddings
        collection_name = "repo_chunks" if self.persist_directory else f"repo_chunks_{uuid.uuid4().hex}"
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    @staticmethod
    def _make_chunk_id(indexed_file, chunk_index: int, namespace: str = "file") -> str:
        """
        Generate a deterministic, unique ID for a chunk.

        Uses a SHA-256 hash of the namespace + relative path + chunk index so that:
          - The same file re-indexed produces the same IDs (enables upsert).
          - IDs are safe for ChromaDB (no special chars, fixed length).
          - Different namespaces (e.g. source vs semantic) don't collide.
          - If the document is a SemanticDocument, its symbol name, type, and line
            are included to prevent collisions.
        """
        if hasattr(indexed_file, "symbol_name"):
            raw = f"{namespace}::{indexed_file.relative_path}::{indexed_file.symbol_name}::{indexed_file.symbol_type}::{getattr(indexed_file, 'symbol_line', 0)}::{chunk_index}"
        else:
            raw = f"{namespace}::{indexed_file.relative_path}::{chunk_index}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def build(self, embedded_files: List[EmbeddedFile], namespace: str = "file") -> None:
        """
        Populates the ChromaDB collection with chunks from embedded_files.

        Uses upsert() so that re-indexing the same repository is idempotent
        rather than raising DuplicateIDError.

        Args:
            embedded_files: Output of FileEmbedder.embed().
            namespace:      Logical separation to prevent ID collisions between 
                            different types of documents (e.g. "source" vs "semantic").

        Raises:
            ValueError: If embedded_files contains no chunks to index.
        """
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for ef in embedded_files:
            for chunk in ef.chunks:
                chunk_id = self._make_chunk_id(
                    ef.indexed_file, chunk.chunk_index, namespace
                )
                ids.append(chunk_id)
                
                embeddings.append(chunk.embedding)
                documents.append(chunk.content)
                
                # ChromaDB metadata can only store flat dicts of str/int/float/bool.
                # We flatten the IndexedFile + EmbeddedChunk attributes.
                meta_entry = {
                    "path": ef.indexed_file.path,
                    "relative_path": ef.indexed_file.relative_path,
                    "extension": ef.indexed_file.extension,
                    "line_count": ef.indexed_file.line_count,
                    "size_bytes": ef.indexed_file.size_bytes,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "model_name": ef.model_name,
                    "namespace": namespace,
                }
                
                if hasattr(ef.indexed_file, "symbol_name"):
                    meta_entry["symbol_name"] = ef.indexed_file.symbol_name
                    meta_entry["symbol_type"] = ef.indexed_file.symbol_type
                    meta_entry["symbol_line"] = getattr(ef.indexed_file, "symbol_line", 0)
                    
                metadatas.append(meta_entry)

        if not ids:
            raise ValueError(
                "No chunks found in embedded_files. "
                "Ensure FileIndexer and FileEmbedder ran successfully."
            )

        # Use upsert instead of add — idempotent on re-index, no DuplicateIDError.
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(
            f"[store] Upserted {len(ids)} chunk(s) (namespace: {namespace}) to ChromaDB. "
            f"Total size: {self.collection.count()} chunks."
        )

    # ------------------------------------------------------------------
    # Searching the index
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Finds the top_k chunks most semantically similar to query.

        Args:
            query:  Natural language or code query string.
            top_k:  Maximum number of results to return.

        Returns:
            List of SearchResult objects sorted by score descending.

        Raises:
            RuntimeError: If collection is empty.
        """
        if self.collection.count() == 0:
            raise RuntimeError(
                "VectorStore is empty. Call .build(embedded_files) first."
            )

        # Embed the query
        raw = self.backend.embed_batch([query])
        query_vec = raw[0]

        # Query ChromaDB
        # We request N results. ChromaDB handles capping if top_k > count().
        # n_results must be <= collection.count()
        k = min(top_k, self.collection.count())
        
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["embeddings", "documents", "metadatas", "distances"]
        )

        search_results: List[SearchResult] = []
        
        # ChromaDB query results are lists of lists (one per query vector).
        # We only sent 1 query vector, so we take the [0] element of each.
        if not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        embs = results["embeddings"][0] if results.get("embeddings") else None

        from repo_assistant.indexer.models import IndexedFile, SemanticDocument

        for i in range(len(ids)):
            meta = metas[i]
            
            # Reconstruct the IndexedFile
            if "symbol_name" in meta:
                indexed_file = SemanticDocument(
                    path=str(meta["path"]),
                    relative_path=str(meta["relative_path"]),
                    extension=str(meta["extension"]),
                    content="", # We don't store full original file content in DB to save space
                    line_count=int(meta["line_count"]),
                    size_bytes=int(meta["size_bytes"]),
                    symbol_name=str(meta["symbol_name"]),
                    symbol_type=str(meta["symbol_type"]),
                    symbol_line=int(meta.get("symbol_line", 0))
                )
            else:
                indexed_file = IndexedFile(
                    path=str(meta["path"]),
                    relative_path=str(meta["relative_path"]),
                    extension=str(meta["extension"]),
                    content="", # We don't store full original file content in DB to save space
                    line_count=int(meta["line_count"]),
                    size_bytes=int(meta["size_bytes"])
                )
            
            # Reconstruct the EmbeddedChunk
            chunk = EmbeddedChunk(
                content=docs[i],
                embedding=embs[i] if embs is not None else [],
                chunk_index=int(meta["chunk_index"]),
                start_char=int(meta["start_char"]),
            )
            
            # ChromaDB returns 'distances' based on the metric.
            # For cosine, distance = 1.0 - cosine_similarity.
            # We convert back to cosine similarity score.
            score = 1.0 - distances[i]

            search_results.append(
                SearchResult(
                    chunk=chunk,
                    indexed_file=indexed_file,
                    score=float(score),
                    rank=i + 1,
                )
            )

        return search_results

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total number of chunks currently indexed."""
        return self.collection.count()
