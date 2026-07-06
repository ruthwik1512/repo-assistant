"""
indexer sub-package

Exposes two public components:
  - FileIndexer  : reads source files and returns IndexedFile objects
  - IndexedFile  : dataclass representing a single indexed source file
"""

from .indexer import FileIndexer
from .models import IndexedFile

__all__ = ["FileIndexer", "IndexedFile"]
