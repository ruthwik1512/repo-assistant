"""
models.py

Defines the data structure that represents a single indexed source file.

Why a dataclass?
  - @dataclass auto-generates __init__, __repr__, and __eq__ from the fields.
  - Fields are annotated with types, making the data contract explicit.
  - It's lightweight — no ORM, no schema, just a plain Python object.

Why not a plain dict?
  - Dicts are untyped: file["contnent"] typo fails silently at runtime.
  - Dataclasses fail loudly at definition time and support IDE autocomplete.

Why not a NamedTuple?
  - NamedTuples are immutable and don't support default values cleanly.
  - Dataclasses are more Pythonic for structured data in modern Python (3.7+).
"""

from dataclasses import dataclass


@dataclass
class IndexedFile:
    """
    Represents a single source file that has been read from disk.

    This object is the unit of data that flows from the Indexer to all
    downstream components (embedder, chat, etc.). Keeping it as a dataclass
    means every consumer knows exactly what fields to expect.

    Attributes:
        path:          Absolute path to the file on disk.
        relative_path: Path relative to the repository root.
                       e.g. "src/flask/app.py" instead of "C:/repos/flask/src/flask/app.py"
                       This is what gets shown to the LLM — shorter and more meaningful.
        extension:     File extension including the dot. e.g. ".py"
        content:       Full text content of the file.
        line_count:    Number of lines in the file.
        size_bytes:    Size of the file on disk in bytes.
        metadata:      Optional dictionary of extra properties (e.g. symbol_name).
    """

    path: str
    relative_path: str
    extension: str
    content: str
    line_count: int
    size_bytes: int

    def __repr__(self) -> str:
        # Override the default __repr__ to show a concise summary.
        # The default would print the entire file content which is unreadable.
        return (
            f"IndexedFile("
            f"relative_path={self.relative_path!r}, "
            f"extension={self.extension!r}, "
            f"lines={self.line_count}, "
            f"bytes={self.size_bytes}"
            f")"
        )


@dataclass
class SemanticDocument(IndexedFile):
    """
    Represents a semantic abstraction (like a class or function signature)
    derived from a source file. 
    
    It subclasses IndexedFile so it can seamlessly pass through the embedder,
    but carries its own symbol identity for unique chunk IDs.
    """
    symbol_name: str
    symbol_type: str
    symbol_line: int

    def __repr__(self) -> str:
        return (
            f"SemanticDocument("
            f"relative_path={self.relative_path!r}, "
            f"symbol={self.symbol_name!r} ({self.symbol_type} @ L{self.symbol_line})"
            f")"
        )
