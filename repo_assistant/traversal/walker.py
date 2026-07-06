"""
walker.py

Responsibility: Traverse a local repository directory using os.walk()
and return a flat list of supported source file paths, skipping
irrelevant directories and unsupported file types.
"""

import os
from typing import List


# --- Module-level constants ---
#
# Why frozenset instead of set or list?
#
#   list  → O(n) membership test  ("is x in list?" scans every element)
#   set   → O(1) membership test, but mutable (someone could accidentally change it)
#   frozenset → O(1) membership test + immutable (safe to use as a default argument
#               and share across instances without risk of mutation)
#
# These are defined at module level (not inside the class) because they are
# project-wide constants, not instance-specific state. Any code that imports
# this module can reference them directly if needed.

IGNORED_DIRS: frozenset = frozenset({
    ".git",
    "node_modules",
    "__pycache__",
    "build",
    "dist",
})

SUPPORTED_EXTENSIONS: frozenset = frozenset({
    ".py",
    ".js",
    ".ts",
    ".java",
    ".cpp",
})


class RepoWalker:
    """
    Traverses a local repository and collects supported source files.

    Design: Accepts ignored_dirs and supported_extensions as constructor
    arguments (with sensible defaults) so callers can customize behavior
    without subclassing. This is the "composition over inheritance" principle.

    Example:
        walker = RepoWalker()
        files = walker.walk("/path/to/repo")
    """

    def __init__(
        self,
        ignored_dirs: frozenset = IGNORED_DIRS,
        supported_extensions: frozenset = SUPPORTED_EXTENSIONS,
    ) -> None:
        """
        Args:
            ignored_dirs: Directory names to skip entirely during traversal.
                          Defaults to the module-level IGNORED_DIRS constant.
            supported_extensions: File extensions to collect (must include the
                                  leading dot, e.g. ".py" not "py").
                                  Defaults to the module-level SUPPORTED_EXTENSIONS.
        """
        self.ignored_dirs = ignored_dirs
        self.supported_extensions = supported_extensions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def walk(self, repo_path: str) -> List[str]:
        """
        Recursively traverses repo_path and collects source file paths.

        Args:
            repo_path: Absolute path to the root of a local repository.

        Returns:
            A flat list of absolute paths to every supported source file found.
            The list is ordered by directory traversal order (top-down).

        Raises:
            ValueError: If repo_path does not exist or is not a directory.
        """
        if not os.path.isdir(repo_path):
            raise ValueError(
                f"repo_path must be an existing directory. Got: {repo_path!r}"
            )

        collected: List[str] = []

        for root, dirs, files in os.walk(repo_path):

            # --- Prune ignored directories (IN-PLACE modification) ---
            #
            # This is the most important line in this file. Read carefully:
            #
            #   dirs[:] = ...   ← slice assignment: modifies the EXISTING list object
            #   dirs    = ...   ← rebinds the LOCAL variable to a NEW list object
            #
            # os.walk() holds a reference to the original `dirs` list and uses
            # it to decide which subdirectories to recurse into NEXT. If we
            # rebind the variable (dirs = ...), os.walk() still sees the original
            # unfiltered list and descends into .git, node_modules, etc. anyway.
            #
            # Slice assignment (dirs[:] = ...) mutates the object that os.walk()
            # is already pointing to, so our filter is respected.
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs]

            # --- Collect files with supported extensions ---
            for filename in files:
                # os.path.splitext("main.py")  → ("main", ".py")
                # os.path.splitext("README")   → ("README", "")
                # os.path.splitext(".env")     → (".env", "")   ← no extension
                _, ext = os.path.splitext(filename)

                if ext in self.supported_extensions:
                    # Always store the full absolute path so callers never
                    # need to worry about what the CWD is when they use it.
                    full_path = os.path.join(root, filename)
                    collected.append(full_path)

        print(f"[walker] Collected {len(collected)} source file(s) in: {repo_path!r}")
        return collected

    def summarize(self, file_paths: List[str]) -> dict:
        """
        Groups a list of file paths by extension and counts them.

        Useful for a quick sanity-check after walking a repository.

        Args:
            file_paths: The list returned by .walk().

        Returns:
            A dict mapping each extension to its file count.
            Example: {".py": 42, ".js": 7}
        """
        summary: dict = {}
        for path in file_paths:
            _, ext = os.path.splitext(path)
            summary[ext] = summary.get(ext, 0) + 1
        return summary
