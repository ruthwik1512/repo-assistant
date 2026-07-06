"""
cloner.py

Responsibility: Clone a public GitHub repository via HTTPS into a
local ./repos/ directory. If the repository already exists locally,
skip cloning and return the existing path.
"""

import os

import git  # GitPython


class RepoCloner:
    """
    Clones a public GitHub repository (HTTPS) into a local repos directory.

    Design: We use a class instead of a bare function so that configuration
    (the repos_dir path) is set once at construction time and reused across
    multiple .clone() calls without repeating yourself.

    Attributes:
        repos_dir (str): Absolute path to the folder that holds cloned repos.
    """

    def __init__(self, repos_dir: str = None) -> None:
        """
        Args:
            repos_dir: Path where repositories will be cloned.
                       Defaults to ./repos/ relative to the current
                       working directory at the time this object is created.
        """
        # Why default=None instead of default=os.path.join(os.getcwd(), "repos")?
        #
        # Default argument values in Python are evaluated ONCE at function
        # definition time (when the module is first imported), not each time
        # the function is called. If the working directory changed between
        # import and instantiation, a hard-coded default would point to the
        # wrong place. Using None and computing inside __init__ guarantees
        # we capture the CWD at the moment the object is created.
        if repos_dir is None:
            repos_dir = os.path.join(os.getcwd(), "repos")

        self.repos_dir = repos_dir

        # exist_ok=True means: don't raise an error if the folder already exists.
        # This makes the constructor safe to call multiple times.
        os.makedirs(self.repos_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helpers (prefixed with _ to signal "internal use only")
    # ------------------------------------------------------------------

    def _extract_repo_name(self, url: str) -> str:
        """
        Derives a safe local folder name from a GitHub URL.

        Examples:
            "https://github.com/user/my-repo"      →  "my-repo"
            "https://github.com/user/my-repo.git"  →  "my-repo"
            "https://github.com/user/my-repo/"     →  "my-repo"

        Args:
            url: A public GitHub HTTPS URL.

        Returns:
            The repository name as a plain string.

        Raises:
            ValueError: If a repo name cannot be extracted from the URL.
        """
        # Step 1: Remove any trailing slash(es) so split("/") works reliably.
        clean = url.rstrip("/")

        # Step 2: GitHub URLs often end in .git — strip it for a clean folder name.
        if clean.endswith(".git"):
            clean = clean[:-4]

        # Step 3: The repo name is always the last segment of the URL path.
        repo_name = clean.split("/")[-1]

        if not repo_name:
            raise ValueError(
                f"Could not extract a repository name from URL: {url!r}"
            )

        return repo_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone(self, url: str) -> str:
        """
        Clones the repository at `url` into <repos_dir>/<repo_name>/.

        If the destination directory already exists, cloning is skipped
        and the existing path is returned immediately.

        Args:
            url: A public GitHub HTTPS URL
                 (e.g. "https://github.com/pallets/flask").

        Returns:
            Absolute path (str) to the cloned or pre-existing repo directory.

        Raises:
            ValueError: If the URL is malformed or the repo name is empty.
            RuntimeError: If GitPython reports a clone failure (e.g. 404,
                          no network, private repo).
        """
        repo_name = self._extract_repo_name(url)
        dest_path = os.path.join(self.repos_dir, repo_name)

        # --- Skip if already cloned ---
        if os.path.exists(dest_path):
            print(f"[cloner] Already exists, skipping clone: {dest_path}")
            return dest_path

        # --- Clone ---
        print(f"[cloner] Cloning {url!r}")
        print(f"[cloner] Destination: {dest_path!r}")

        try:
            git.Repo.clone_from(url, dest_path)
        except git.exc.GitCommandError as exc:
            # Wrap GitPython's low-level error in a friendlier RuntimeError
            # so callers don't need to import git.exc themselves.
            raise RuntimeError(
                f"Failed to clone {url!r}.\n"
                f"Check that the URL is correct and the repo is public.\n"
                f"Git error: {exc}"
            ) from exc

        print(f"[cloner] Done.")
        return dest_path
