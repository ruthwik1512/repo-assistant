"""
tests/test_traversal.py

Unit tests for the traversal sub-package (RepoCloner + RepoWalker).

Testing philosophy applied here:
  - Tests must NEVER make network calls. We mock GitPython entirely.
  - Tests must NEVER leave files on disk. We use tempfile.TemporaryDirectory
    as a context manager — it auto-deletes on exit, even if a test crashes.
  - Each test follows the Arrange → Act → Assert (AAA) pattern: set up
    the scenario, run the code, check the outcome. This keeps tests readable.
  - Test method names follow: test_<method>_<scenario_being_tested>
"""

import os
import tempfile
from unittest.mock import patch

import git
import pytest

from repo_assistant.traversal import RepoCloner, RepoWalker


# =============================================================================
# Helpers
# =============================================================================

def make_files(base_dir: str, relative_paths: list) -> None:
    """
    Creates a set of empty files inside base_dir, making parent dirs as needed.

    This is a module-level helper (not a method) because it is used by
    multiple test classes. Keeping shared utilities outside classes avoids
    coupling unrelated tests together.

    Args:
        base_dir: Root directory to create files under.
        relative_paths: List of paths relative to base_dir.
                        e.g. ["src/main.py", "node_modules/lib.js"]
    """
    for rel_path in relative_paths:
        full_path = os.path.join(base_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        open(full_path, "w").close()  # create an empty file


# =============================================================================
# RepoCloner — _extract_repo_name
# =============================================================================

class TestExtractRepoName:
    """
    Tests for RepoCloner._extract_repo_name().

    Why test a private method directly?
    This helper has clear, well-defined inputs and outputs and is complex
    enough to warrant its own tests. Testing it directly gives us precise
    failure messages. We can still test it through .clone(), but those tests
    would be harder to read and diagnose.
    """

    def setup_method(self):
        # Use a real temp dir so the constructor's os.makedirs() doesn't
        # create a ./repos/ folder in the project during test runs.
        self.tmp = tempfile.mkdtemp()
        self.cloner = RepoCloner(repos_dir=self.tmp)

    def test_plain_url(self):
        # Arrange + Act
        result = self.cloner._extract_repo_name("https://github.com/user/my-repo")
        # Assert
        assert result == "my-repo"

    def test_git_suffix_is_stripped(self):
        result = self.cloner._extract_repo_name("https://github.com/user/my-repo.git")
        assert result == "my-repo"

    def test_trailing_slash_is_stripped(self):
        result = self.cloner._extract_repo_name("https://github.com/user/my-repo/")
        assert result == "my-repo"

    def test_git_suffix_and_trailing_slash_combined(self):
        result = self.cloner._extract_repo_name("https://github.com/user/my-repo.git/")
        assert result == "my-repo"

    def test_empty_url_raises_value_error(self):
        # pytest.raises() is a context manager that asserts the block raises
        # the specified exception. `match=` checks the error message with regex.
        with pytest.raises(ValueError, match="Could not extract"):
            self.cloner._extract_repo_name("")

    def test_slash_only_url_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not extract"):
            self.cloner._extract_repo_name("/")


# =============================================================================
# RepoCloner — clone()
# =============================================================================

class TestRepoCloner:
    """
    Tests for RepoCloner.clone().

    Key technique: unittest.mock.patch
    patch("git.Repo.clone_from") temporarily replaces the real GitPython
    function with a MagicMock for the duration of the `with` block.
    This means no network calls are made, but we can still assert HOW
    the mock was called (arguments, number of times, etc.).
    """

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.cloner = RepoCloner(repos_dir=self.tmp)

    def test_clone_skips_if_repo_already_exists(self):
        """If the destination directory already exists, git must not be called."""
        # Arrange: simulate a previously-cloned repo by creating the dir
        existing_path = os.path.join(self.tmp, "my-repo")
        os.makedirs(existing_path)

        # Act
        with patch("git.Repo.clone_from") as mock_clone:
            result = self.cloner.clone("https://github.com/user/my-repo")

        # Assert: clone was skipped, existing path returned
        mock_clone.assert_not_called()
        assert result == existing_path

    def test_clone_calls_git_with_correct_arguments(self):
        """When the repo doesn't exist, git.Repo.clone_from must be called correctly."""
        url = "https://github.com/user/new-repo"
        expected_dest = os.path.join(self.tmp, "new-repo")

        # Act
        with patch("git.Repo.clone_from") as mock_clone:
            result = self.cloner.clone(url)

        # Assert: called exactly once with the right url and destination
        mock_clone.assert_called_once_with(url, expected_dest)
        assert result == expected_dest

    def test_clone_returns_correct_path_on_success(self):
        """The returned path should point to <repos_dir>/<repo_name>."""
        url = "https://github.com/org/tool-name"
        expected = os.path.join(self.tmp, "tool-name")

        with patch("git.Repo.clone_from"):
            result = self.cloner.clone(url)

        assert result == expected

    def test_clone_wraps_git_error_as_runtime_error(self):
        """A low-level GitCommandError must be re-raised as RuntimeError."""
        url = "https://github.com/user/private-repo"

        # side_effect tells the mock to raise this exception when called
        with patch(
            "git.Repo.clone_from",
            side_effect=git.exc.GitCommandError("clone", 128),
        ):
            with pytest.raises(RuntimeError, match="Failed to clone"):
                self.cloner.clone(url)

    def test_clone_preserves_original_exception(self):
        """The RuntimeError should chain the original GitCommandError."""
        url = "https://github.com/user/bad-repo"
        original_error = git.exc.GitCommandError("clone", 128)

        with patch("git.Repo.clone_from", side_effect=original_error):
            with pytest.raises(RuntimeError) as exc_info:
                self.cloner.clone(url)

        # __cause__ is set by the `raise ... from exc` pattern in cloner.py
        assert exc_info.value.__cause__ is original_error


# =============================================================================
# RepoWalker — walk()
# =============================================================================

class TestRepoWalkerWalk:
    """
    Tests for RepoWalker.walk().

    Key technique: tempfile.TemporaryDirectory
    Creates a real directory tree on disk for the duration of the `with`
    block, then deletes everything automatically — even if the test fails.
    This gives us realistic os.walk() behavior without leaving test
    artifacts on disk.
    """

    def setup_method(self):
        self.walker = RepoWalker()

    def test_raises_value_error_if_path_does_not_exist(self):
        with pytest.raises(ValueError, match="must be an existing directory"):
            self.walker.walk("/this/path/does/not/exist/anywhere")

    def test_collects_supported_files_only(self):
        """Files with unsupported extensions should be silently ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            # Arrange
            make_files(tmp, [
                "src/main.py",      # ✅ supported
                "src/app.js",       # ✅ supported
                "src/types.ts",     # ✅ supported
                "src/Main.java",    # ✅ supported
                "src/main.cpp",     # ✅ supported
                "README.md",        # ❌ not supported
                ".env",             # ❌ not supported (no extension)
                "Makefile",         # ❌ not supported
                "data.json",        # ❌ not supported
            ])

            # Act
            result = self.walker.walk(tmp)

        # Assert
        extensions = {os.path.splitext(f)[1] for f in result}
        assert extensions == {".py", ".js", ".ts", ".java", ".cpp"}
        assert len(result) == 5

    def test_ignores_git_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_files(tmp, [
                "src/main.py",
                ".git/config",
                ".git/HEAD",
            ])
            result = self.walker.walk(tmp)

        assert len(result) == 1
        assert result[0].endswith("main.py")

    def test_ignores_node_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_files(tmp, [
                "index.js",
                "node_modules/lodash/index.js",
                "node_modules/react/index.js",
            ])
            result = self.walker.walk(tmp)

        assert len(result) == 1
        assert result[0].endswith("index.js")

    def test_ignores_all_default_dirs(self):
        """All five default ignored directories should be pruned."""
        with tempfile.TemporaryDirectory() as tmp:
            make_files(tmp, [
                "app.py",                              # ✅ collected
                ".git/config",                         # ❌ ignored dir
                "node_modules/pkg/index.js",           # ❌ ignored dir
                "__pycache__/app.cpython-311.py",      # ❌ ignored dir
                "build/output.js",                     # ❌ ignored dir
                "dist/bundle.js",                      # ❌ ignored dir
            ])
            result = self.walker.walk(tmp)

        assert len(result) == 1
        assert result[0].endswith("app.py")

    def test_returns_absolute_paths(self):
        """Every path in the result must be absolute."""
        with tempfile.TemporaryDirectory() as tmp:
            make_files(tmp, ["app.py", "lib/utils.py"])
            result = self.walker.walk(tmp)

        assert len(result) == 2
        assert all(os.path.isabs(p) for p in result)

    def test_empty_repo_returns_empty_list(self):
        """A directory with no supported files should return an empty list."""
        with tempfile.TemporaryDirectory() as tmp:
            result = self.walker.walk(tmp)

        assert result == []

    def test_custom_ignored_dirs_are_respected(self):
        """RepoWalker should use custom ignored_dirs passed at construction."""
        custom_walker = RepoWalker(ignored_dirs=frozenset({"vendor", "third_party"}))

        with tempfile.TemporaryDirectory() as tmp:
            make_files(tmp, [
                "src/main.py",           # ✅ collected
                "vendor/lib.py",         # ❌ custom ignored
                "third_party/util.py",   # ❌ custom ignored
            ])
            result = custom_walker.walk(tmp)

        assert len(result) == 1
        assert result[0].endswith("main.py")

    def test_custom_extensions_are_respected(self):
        """RepoWalker should use custom supported_extensions passed at construction."""
        custom_walker = RepoWalker(supported_extensions=frozenset({".md", ".txt"}))

        with tempfile.TemporaryDirectory() as tmp:
            make_files(tmp, [
                "README.md",    # ✅ custom supported
                "notes.txt",    # ✅ custom supported
                "app.py",       # ❌ not in custom extensions
            ])
            result = custom_walker.walk(tmp)

        extensions = {os.path.splitext(f)[1] for f in result}
        assert extensions == {".md", ".txt"}


# =============================================================================
# RepoWalker — summarize()
# =============================================================================

class TestRepoWalkerSummarize:
    """Tests for RepoWalker.summarize()."""

    def setup_method(self):
        self.walker = RepoWalker()

    def test_counts_extensions_correctly(self):
        paths = ["/repo/a.py", "/repo/b.py", "/repo/c.js"]
        assert self.walker.summarize(paths) == {".py": 2, ".js": 1}

    def test_single_file(self):
        assert self.walker.summarize(["/repo/main.py"]) == {".py": 1}

    def test_empty_list_returns_empty_dict(self):
        assert self.walker.summarize([]) == {}

    def test_all_same_extension(self):
        paths = ["/a.ts", "/b.ts", "/c.ts"]
        assert self.walker.summarize(paths) == {".ts": 3}
