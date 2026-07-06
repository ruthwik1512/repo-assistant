"""
tests/test_indexer.py

Unit tests for the indexer sub-package (FileIndexer + IndexedFile).

Testing strategy:
  - IndexedFile: verify field assignment and custom __repr__
  - FileIndexer._read_file: use real temp files for encoding and size tests;
    mock os.path.getsize to test the size-limit guard cheaply
  - FileIndexer.index: use real temp file trees; mock _read_file where needed
    to isolate index() logic from file-reading logic
"""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from repo_assistant.indexer import FileIndexer, IndexedFile


# =============================================================================
# Helpers
# =============================================================================

def write_file(directory: str, relative_path: str, content: str = "") -> str:
    """
    Creates a file at <directory>/<relative_path> with the given content.
    Parent directories are created automatically.

    Returns the absolute path to the created file.
    """
    full_path = os.path.join(directory, relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    return full_path


# =============================================================================
# IndexedFile tests
# =============================================================================

class TestIndexedFile:
    """Tests for the IndexedFile dataclass."""

    def _make_indexed_file(self, **overrides) -> IndexedFile:
        """Build an IndexedFile with sensible defaults, allowing field overrides."""
        defaults = dict(
            path="/repo/src/main.py",
            relative_path="src/main.py",
            extension=".py",
            content="def hello():\n    pass\n",
            line_count=2,
            size_bytes=24,
        )
        defaults.update(overrides)
        return IndexedFile(**defaults)

    def test_fields_are_set_correctly(self):
        """All constructor arguments should be stored as-is."""
        f = self._make_indexed_file()

        assert f.path == "/repo/src/main.py"
        assert f.relative_path == "src/main.py"
        assert f.extension == ".py"
        assert f.content == "def hello():\n    pass\n"
        assert f.line_count == 2
        assert f.size_bytes == 24

    def test_repr_does_not_contain_full_content(self):
        """
        The custom __repr__ should show a concise summary.
        Printing full file content in repr would be unusable for large files.
        """
        f = self._make_indexed_file(content="x" * 5000)
        representation = repr(f)

        # Should show relative_path and line info
        assert "src/main.py" in representation
        # Should NOT dump the entire content
        assert "x" * 50 not in representation

    def test_repr_contains_key_metadata(self):
        f = self._make_indexed_file(line_count=42, size_bytes=1024)
        r = repr(f)

        assert "42" in r       # line count
        assert "1024" in r     # size_bytes
        assert ".py" in r      # extension

    def test_equality_of_identical_instances(self):
        """
        @dataclass auto-generates __eq__ that compares all fields.
        Two instances with the same data should be equal.
        """
        f1 = self._make_indexed_file()
        f2 = self._make_indexed_file()
        assert f1 == f2

    def test_inequality_when_field_differs(self):
        f1 = self._make_indexed_file(line_count=10)
        f2 = self._make_indexed_file(line_count=99)
        assert f1 != f2


# =============================================================================
# FileIndexer._read_file tests
# =============================================================================

class TestReadFile:
    """
    Tests for the private FileIndexer._read_file() method.

    We test _read_file directly because it contains isolated, testable logic:
    size checking, encoding handling, and OS error handling.
    Each test targets exactly one behaviour.
    """

    def setup_method(self):
        self.indexer = FileIndexer()

    def test_returns_content_for_normal_file(self):
        """A standard UTF-8 file should be read and returned as-is."""
        with tempfile.TemporaryDirectory() as tmp:
            path = write_file(tmp, "app.py", content="print('hello')\n")

            result = self.indexer._read_file(path)

        assert result == "print('hello')\n"

    def test_returns_none_when_file_exceeds_size_limit(self):
        """
        Files exceeding max_file_bytes should be skipped (return None).
        We mock os.path.getsize so we don't need to create a real large file.
        """
        small_indexer = FileIndexer(max_file_bytes=10)  # tiny limit

        with tempfile.TemporaryDirectory() as tmp:
            # This file is 15 bytes — over the 10-byte limit
            path = write_file(tmp, "big.py", content="x" * 15)

            result = small_indexer._read_file(path)

        assert result is None

    def test_returns_content_when_file_is_exactly_at_limit(self):
        """A file exactly at the limit should be read (boundary condition)."""
        content = "x" * 10
        exact_indexer = FileIndexer(max_file_bytes=10)

        with tempfile.TemporaryDirectory() as tmp:
            path = write_file(tmp, "exact.py", content=content)
            result = exact_indexer._read_file(path)

        assert result == content

    def test_handles_non_utf8_bytes_without_crashing(self):
        """
        Files with invalid UTF-8 sequences should be read with bad bytes
        replaced (not raise UnicodeDecodeError).

        We write raw bytes directly — we can't use write_file() here because
        it opens in text mode with UTF-8. We need binary mode to inject bad bytes.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad_encoding.cpp")

            # \xff and \xfe are not valid UTF-8 start bytes
            with open(path, "wb") as f:
                f.write(b"int main() {\n    return \xff\xfe0;\n}")

            result = self.indexer._read_file(path)

        # Should return a string (not None) with replacement characters
        assert result is not None
        assert isinstance(result, str)
        # The valid ASCII parts should still be intact
        assert "int main()" in result

    def test_returns_none_when_file_cannot_be_read(self):
        """
        If open() raises an OSError (e.g. permission denied), _read_file
        should return None rather than propagating the exception.

        We mock the built-in open() to raise OSError.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = write_file(tmp, "locked.py", content="secret")

            # patch "builtins.open" only within the indexer module's scope
            with patch("builtins.open", side_effect=OSError("Permission denied")):
                result = self.indexer._read_file(path)

        assert result is None

    def test_returns_none_when_getsize_fails(self):
        """If os.path.getsize raises (e.g. broken symlink), return None."""
        with patch("os.path.getsize", side_effect=OSError("No such file")):
            result = self.indexer._read_file("/nonexistent/path.py")

        assert result is None


# =============================================================================
# FileIndexer.index tests
# =============================================================================

class TestFileIndexerIndex:
    """Tests for FileIndexer.index()."""

    def setup_method(self):
        self.indexer = FileIndexer()

    def test_raises_value_error_if_repo_root_missing(self):
        with pytest.raises(ValueError, match="must be an existing directory"):
            self.indexer.index([], repo_root="/no/such/directory")

    def test_returns_empty_list_for_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.indexer.index([], repo_root=tmp)
        assert result == []

    def test_returns_indexed_file_objects(self):
        """index() should return a list of IndexedFile instances."""
        with tempfile.TemporaryDirectory() as tmp:
            path = write_file(tmp, "main.py", content="x = 1\n")

            result = self.indexer.index([path], repo_root=tmp)

        assert len(result) == 1
        assert isinstance(result[0], IndexedFile)

    def test_content_is_correct(self):
        content = "def foo():\n    return 42\n"

        with tempfile.TemporaryDirectory() as tmp:
            path = write_file(tmp, "foo.py", content=content)
            result = self.indexer.index([path], repo_root=tmp)

        assert result[0].content == content

    def test_line_count_is_correct(self):
        # 3 lines of actual content
        content = "line1\nline2\nline3\n"

        with tempfile.TemporaryDirectory() as tmp:
            path = write_file(tmp, "lines.py", content=content)
            result = self.indexer.index([path], repo_root=tmp)

        # splitlines() on "line1\nline2\nline3\n" → ["line1", "line2", "line3"] = 3
        assert result[0].line_count == 3

    def test_relative_path_is_computed_correctly(self):
        """
        relative_path must be the path from repo_root to the file,
        using forward slashes regardless of OS.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = write_file(tmp, os.path.join("src", "utils.py"), content="# util")
            result = self.indexer.index([path], repo_root=tmp)

        # Forward slashes on all platforms
        assert result[0].relative_path == "src/utils.py"

    def test_extension_is_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_file(tmp, "component.ts", content="export {}")
            result = self.indexer.index([path], repo_root=tmp)

        assert result[0].extension == ".ts"

    def test_size_bytes_matches_actual_file_size(self):
        content = "hello"  # 5 bytes in ASCII/UTF-8

        with tempfile.TemporaryDirectory() as tmp:
            path = write_file(tmp, "small.py", content=content)
            result = self.indexer.index([path], repo_root=tmp)

        assert result[0].size_bytes == 5

    def test_skips_unreadable_files_and_continues(self):
        """
        If one file can't be read, it should be skipped and the rest
        of the index should be unaffected.
        """
        with tempfile.TemporaryDirectory() as tmp:
            good = write_file(tmp, "good.py", content="x = 1")
            bad = write_file(tmp, "bad.py", content="y = 2")

            # Make _read_file return None for the bad file only
            original_read = self.indexer._read_file

            def selective_read(path):
                if path == bad:
                    return None
                return original_read(path)

            self.indexer._read_file = selective_read
            result = self.indexer.index([good, bad], repo_root=tmp)

        assert len(result) == 1
        assert result[0].path == good

    def test_indexes_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [
                write_file(tmp, "a.py", content="a = 1"),
                write_file(tmp, "b.js", content="const b = 2;"),
                write_file(tmp, "c.ts", content="let c: number = 3;"),
            ]
            result = self.indexer.index(paths, repo_root=tmp)

        assert len(result) == 3
        extensions = {f.extension for f in result}
        assert extensions == {".py", ".js", ".ts"}
