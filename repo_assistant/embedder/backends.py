"""
backends.py

Defines the EmbeddingBackend abstraction and provides a concrete
implementation using sentence-transformers (local, free, no API key).

Design pattern: Strategy
  FileEmbedder depends on EmbeddingBackend (the abstract interface), not on
  any specific provider. Swapping from sentence-transformers to OpenAI is a
  one-line change at the call site — FileEmbedder itself never changes.

  This is the "Dependency Inversion Principle": high-level modules (FileEmbedder)
  should depend on abstractions (EmbeddingBackend), not concrete implementations.
"""

from abc import ABC, abstractmethod
from typing import List


# =============================================================================
# Abstract interface
# =============================================================================

class EmbeddingBackend(ABC):
    """
    Abstract base class that every embedding provider must implement.

    Why ABC (Abstract Base Class) instead of a plain class or Protocol?
      - ABC makes it impossible to instantiate EmbeddingBackend directly.
        You must subclass it and implement every @abstractmethod.
      - This gives a clear compile-time-style error if someone forgets to
        implement embed_batch or model_name.
      - Protocol (typing.Protocol) is an alternative for structural subtyping,
        but ABC is more explicit about the inheritance contract.
    """

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of text strings and return their vectors.

        Why batching instead of one text at a time?
          Embedding models (especially GPU-backed ones) are far more efficient
          when processing many texts at once. Calling the model N times for N
          files is N× slower than one call with N texts.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of float vectors, one per input text, in the same order.
            All vectors have the same length (the model's embedding dimension).
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        The identifier of the model being used.
        Stored on EmbeddedFile so downstream code always knows the vector's origin.
        """
        ...


# =============================================================================
# sentence-transformers backend (local)
# =============================================================================

class SentenceTransformerBackend(EmbeddingBackend):
    """
    Embedding backend using the sentence-transformers library.

    Runs entirely locally — no API key, no cost, no internet after first load.
    The model is downloaded once and cached by the library.

    Recommended model: "all-MiniLM-L6-v2"
      - 384-dimensional vectors (compact, fast)
      - Good general-purpose semantic similarity
      - ~90MB download, runs on CPU in milliseconds per chunk

    Example:
        backend = SentenceTransformerBackend()
        backend = SentenceTransformerBackend("all-mpnet-base-v2")  # larger, more accurate
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        """
        Args:
            model: HuggingFace model name or local path.
                   First call will download the model (~90MB) and cache it.
        """
        # Why import inside __init__ instead of at the top of the file?
        #
        # sentence_transformers is an optional, heavy dependency (~500MB with torch).
        # If we import it at module level, importing ANY part of repo_assistant
        # would trigger the import — even if the user only wants RepoCloner.
        # Lazy imports keep startup time fast and the dependency optional.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerBackend.\n"
                "Install it with: pip install sentence-transformers"
            ) from exc

        self._model_name = model
        print(f"[backend] Loading model {model!r} (downloads on first use) ...")
        self._model = SentenceTransformer(model)
        print(f"[backend] Model ready.")

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Args:
            texts: List of text strings to embed.

        Returns:
            List of float vectors (one per text), as plain Python lists.
        """
        # SentenceTransformer.encode() returns a numpy ndarray.
        # We call .tolist() to convert to plain Python floats so callers
        # don't need numpy as a dependency just to read the vectors.
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,   # suppress tqdm output in non-interactive runs
            convert_to_numpy=True,
        )
        return embeddings.tolist()
