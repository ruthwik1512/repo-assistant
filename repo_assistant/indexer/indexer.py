"""
indexer.py

Responsibility: Read a list of source file paths from disk and return
a list of IndexedFile objects with their content and metadata.

Handles:
  - Files that exceed a configurable size limit (skipped with a warning)
  - Files with non-UTF-8 encoding (bad characters replaced, never crash)
  - Files that cannot be read due to permissions (skipped with a warning)
"""

import os
from typing import List, Optional

from .models import IndexedFile


class FileIndexer:
    """
    Reads source files from disk and produces IndexedFile objects.

    Design: max_file_bytes is a constructor argument so the caller can
    tune it for their use case (e.g., lower limit when using a small
    context-window model, higher limit for detailed analysis).

    Example:
        indexer = FileIndexer()
        indexed = indexer.index(file_paths, repo_root="/path/to/repo")
    """

    # 1 MB default. Most source files are well under this.
    # A 1MB Python file would be ~25,000 lines — likely generated, not handwritten.
    DEFAULT_MAX_BYTES: int = 1_000_000

    def __init__(self, max_file_bytes: int = DEFAULT_MAX_BYTES) -> None:
        """
        Args:
            max_file_bytes: Files larger than this (in bytes) are skipped.
                            Defaults to 1MB (1_000_000 bytes).
        """
        self.max_file_bytes = max_file_bytes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, file_paths: List[str], repo_root: str) -> List[IndexedFile]:
        """
        Reads each file in file_paths and returns a list of IndexedFile objects.

        Files that are too large, unreadable, or have permission errors are
        skipped with a printed warning — the rest of the index is unaffected.

        Args:
            file_paths: List of absolute file paths (output of RepoWalker.walk).
            repo_root:  Absolute path to the repository root, used to compute
                        relative_path for each IndexedFile.

        Returns:
            A list of IndexedFile objects, one per successfully read file.

        Raises:
            ValueError: If repo_root is not an existing directory.
        """
        if not os.path.isdir(repo_root):
            raise ValueError(
                f"repo_root must be an existing directory. Got: {repo_root!r}"
            )

        indexed: List[IndexedFile] = []
        skipped = 0

        for path in file_paths:
            content = self._read_file(path)

            if content is None:
                # _read_file already printed a warning explaining why it was skipped
                skipped += 1
                continue

            _, ext = os.path.splitext(path)

            # os.path.relpath computes the path of `path` relative to `repo_root`.
            # e.g. relpath("/repos/flask/src/app.py", "/repos/flask") → "src/app.py"
            # We then normalise separators to forward slashes for consistency
            # across OS (Windows uses backslashes, Linux/macOS use forward slashes).
            relative_path = os.path.relpath(path, repo_root).replace("\\", "/")

            indexed_file = IndexedFile(
                path=path,
                relative_path=relative_path,
                extension=ext,
                content=content,
                line_count=len(content.splitlines()),
                size_bytes=os.path.getsize(path),
            )
            indexed.append(indexed_file)

        print(
            f"[indexer] Indexed {len(indexed)} file(s). "
            f"Skipped {skipped} file(s)."
        )
        return indexed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_file(self, path: str) -> Optional[str]:
        """
        Reads a single file from disk and returns its content as a string.

        Returns None (instead of raising) so that one unreadable file
        doesn't abort the entire indexing run. The caller decides what
        to do with None (currently: skip + warn).

        Args:
            path: Absolute path to the file.

        Returns:
            File content as a string, or None if the file should be skipped.
        """
        # --- Guard: check size before reading ---
        #
        # We check size BEFORE opening the file. Reading a 200MB minified JS
        # file into memory would be slow and wasteful. os.path.getsize() just
        # reads the filesystem metadata — it does not open the file.
        try:
            size = os.path.getsize(path)
        except OSError as exc:
            print(f"[indexer] WARNING: Cannot stat {path!r}: {exc}")
            return None

        if size > self.max_file_bytes:
            print(
                f"[indexer] WARNING: Skipping {path!r} "
                f"({size:,} bytes exceeds limit of {self.max_file_bytes:,} bytes)."
            )
            return None

        # --- Read the file ---
        #
        # encoding="utf-8": almost all source code is UTF-8.
        # errors="replace":  if a byte can't be decoded (e.g. a binary artifact
        #                    with a .py extension), replace it with the Unicode
        #                    replacement character (?) instead of raising UnicodeDecodeError.
        #                    This keeps the indexer running — the LLM will just see
        #                    a ? where the bad byte was.
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError as exc:
            print(f"[indexer] WARNING: Cannot read {path!r}: {exc}")
            return None
