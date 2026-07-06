"""
engine.py

Core engine that runs the parser across the indexed files to build the
Code Graph, generates the architecture skeleton, and produces semantic
signatures for the VectorStore.

Also builds a reverse reference index that maps every symbol name to
all the locations where it is used — the backbone of the /refs command.
"""

from collections import defaultdict
from typing import Dict, List

from repo_assistant.indexer.models import IndexedFile, SemanticDocument
from .parser import GraphParser
from .models import FileNode, SymbolReference


class CodeGraphAnalyzer:
    """
    Orchestrates AST-based analysis of a repository.

    Responsibilities:
      1. Parse each file into a FileNode (classes, functions, imports).
      2. Generate semantic signature documents for the VectorStore.
      3. Build a reverse reference index for /refs queries.
      4. Generate an architecture skeleton for LLM context injection.

    The analyzer is stateful — call analyze() once, then query the results
    with find_symbol(), find_references(), and generate_skeleton().
    """

    def __init__(self, parser: GraphParser) -> None:
        self.parser = parser
        # The stored AST graph representing the entire parsed repository
        self.file_nodes: List[FileNode] = []
        # Reverse index: symbol_name → [SymbolReference, ...]
        self._reference_index: Dict[str, List[SymbolReference]] = defaultdict(list)
        # Set of symbol names that are defined (classes, functions, methods)
        # Used to distinguish definitions from usages in output
        self._defined_symbols: set = set()

    def analyze(self, indexed_files: List[IndexedFile]) -> List[IndexedFile]:
        """
        Runs the parser over all compatible files to build the Code Graph
        and the reverse reference index.

        Returns:
            A list of NEW IndexedFile objects containing just the semantic
            signatures of classes and functions. These can be appended to the
            main list of files before embedding, allowing the LLM to search
            directly for class/function definitions (Strategy B).
        """
        semantic_files: List[IndexedFile] = []

        for idx_f in indexed_files:
            # Our current parser only supports Python
            if idx_f.extension != '.py' or not idx_f.content:
                continue

            # --- Phase A: Structural parsing (classes, functions, imports) ---
            file_node = self.parser.parse_file(idx_f.path, idx_f.content)
            if file_node:
                self.file_nodes.append(file_node)

                # Record defined symbols
                for c in file_node.classes:
                    self._defined_symbols.add(c.name)
                    for m in c.methods:
                        self._defined_symbols.add(m.name)
                for f in file_node.functions:
                    self._defined_symbols.add(f.name)

                # Extract Class signatures as semantic documents
                for c in file_node.classes:
                    sig = c.to_signature_str()
                    semantic_files.append(SemanticDocument(
                        path=idx_f.path,
                        relative_path=idx_f.relative_path,
                        extension=".signature",
                        content=sig,
                        line_count=len(sig.splitlines()),
                        size_bytes=len(sig.encode('utf-8')),
                        symbol_name=c.name,
                        symbol_type="class",
                        symbol_line=c.start_line
                    ))

                # Extract standalone Function signatures as semantic documents
                for f in file_node.functions:
                    sig = f.to_signature_str()
                    semantic_files.append(SemanticDocument(
                        path=idx_f.path,
                        relative_path=idx_f.relative_path,
                        extension=".signature",
                        content=sig,
                        line_count=len(sig.splitlines()),
                        size_bytes=len(sig.encode('utf-8')),
                        symbol_name=f.name,
                        symbol_type="function",
                        symbol_line=f.start_line
                    ))

            # --- Phase B: Reference extraction (every symbol usage) ---
            refs = self.parser.extract_references(idx_f.path, idx_f.content)
            for ref in refs:
                self._reference_index[ref.symbol_name].append(ref)

        return semantic_files

    def generate_skeleton(self) -> str:
        """
        Generates a repository-wide architecture summary string (Strategy A).
        """
        lines = ["# Repository Architecture Skeleton"]

        for fn in self.file_nodes:
            lines.append(f"\n## File: {fn.file_path.split('repos')[-1]}")

            if fn.imports:
                # Truncate imports if there are too many to save context space
                imports_preview = ", ".join(fn.imports[:5])
                if len(fn.imports) > 5:
                    imports_preview += f", and {len(fn.imports) - 5} more..."
                lines.append(f"Imports: {imports_preview}")

            for c in fn.classes:
                base_str = f"({', '.join(c.bases)})" if c.bases else ""
                lines.append(f"- class {c.name}{base_str}:")
                for m in c.methods:
                    lines.append(f"    - def {m.name}(...)")

            for f in fn.functions:
                lines.append(f"- def {f.name}(...)")

        return "\n".join(lines)

    def find_symbol(self, symbol_name: str) -> str:
        """
        Searches the AST graph for a class or function matching the symbol_name.
        Returns a formatted string containing its location and signature.
        """
        for fn in self.file_nodes:
            # Check classes
            for c in fn.classes:
                if c.name == symbol_name:
                    return f"Found Class `{c.name}` in `{fn.file_path}`:\n\n{c.to_signature_str()}"
                # Check methods inside classes
                for m in c.methods:
                    if m.name == symbol_name:
                        return f"Found Method `{m.name}` of class `{c.name}` in `{fn.file_path}`:\n\n{m.to_signature_str()}"

            # Check standalone functions
            for f in fn.functions:
                if f.name == symbol_name:
                    return f"Found Function `{f.name}` in `{fn.file_path}`:\n\n{f.to_signature_str()}"

        return f"Symbol '{symbol_name}' not found in the parsed codebase."

    def find_references(self, symbol_name: str) -> List[SymbolReference]:
        """
        Returns all references to a symbol across the entire parsed codebase.

        This is the backend for the /refs CLI command.

        Args:
            symbol_name: The name to look up (e.g. "UserService", "login").

        Returns:
            A list of SymbolReference objects, sorted by file path then line number.
            Returns an empty list if no references are found.
        """
        refs = self._reference_index.get(symbol_name, [])
        # Sort by file then line for stable, readable output
        return sorted(refs, key=lambda r: (r.file_path, r.line))

    def format_references(self, symbol_name: str) -> str:
        """
        Returns a human-readable formatted string of all references
        to a symbol. Designed for CLI display.

        Groups references by file and shows the line number + context
        for each occurrence.
        """
        refs = self.find_references(symbol_name)

        if not refs:
            return f"No references found for '{symbol_name}'."

        # Group by file
        by_file: Dict[str, List[SymbolReference]] = defaultdict(list)
        for ref in refs:
            by_file[ref.file_path].append(ref)

        lines = [f"Found {len(refs)} reference(s) to `{symbol_name}` across {len(by_file)} file(s):\n"]

        for file_path, file_refs in by_file.items():
            lines.append(f"  📄 {file_path}")
            for ref in file_refs:
                lines.append(f"     L{ref.line:>4} [{ref.ref_type:>6}]  {ref.context}")
            lines.append("")  # blank line between files

        return "\n".join(lines)

