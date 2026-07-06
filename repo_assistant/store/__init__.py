"""
store sub-package

Exposes two public components:
  - VectorStore  : builds a FAISS index, searches it, saves/loads it
  - SearchResult : dataclass returned by VectorStore.search()
"""

from .store import VectorStore
from .models import SearchResult

__all__ = ["VectorStore", "SearchResult"]
