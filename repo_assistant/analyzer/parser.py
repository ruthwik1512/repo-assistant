"""
parser.py

Defines the parser interface and the Python AST implementation.
"""

import ast
from abc import ABC, abstractmethod
from typing import List, Optional

from .models import FileNode, ClassNode, FunctionNode, SymbolReference


class GraphParser(ABC):
    """Abstract interface for a language parser."""

    @abstractmethod
    def parse_file(self, file_path: str, content: str) -> Optional[FileNode]:
        """
        Parses source code into a structured FileNode.
        Returns None if the file type is unsupported or cannot be parsed.
        """
        pass

    @abstractmethod
    def extract_references(
        self, file_path: str, content: str
    ) -> List[SymbolReference]:
        """
        Extracts all symbol references (usages) from a source file.
        Returns a list of SymbolReference objects.
        """
        pass


class _ReferenceCollector(ast.NodeVisitor):
    """
    AST visitor that walks the entire tree and collects SymbolReference objects.

    Why a visitor instead of ast.walk()?
      ast.walk() gives us every node but no parent context. We need to know
      whether a Name appears inside an Import statement (ref_type="import"),
      as a function call (ref_type="call"), or as a bare reference
      (ref_type="name"). The visitor pattern lets us track this context
      as we descend into the tree.
    """

    def __init__(self, file_path: str, source_lines: List[str]) -> None:
        self.file_path = file_path
        self.source_lines = source_lines
        self.refs: List[SymbolReference] = []
        # Track ast nodes that we have already processed (e.g. the func Name in a Call)
        self._ignore_nodes: set = set()

    def _get_line_context(self, lineno: int) -> str:
        """Return the source line (1-indexed), stripped. Safe for out-of-range."""
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.refs.append(SymbolReference(
                symbol_name=alias.name,
                file_path=self.file_path,
                line=node.lineno,
                context=self._get_line_context(node.lineno),
                ref_type="import",
            ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            self.refs.append(SymbolReference(
                symbol_name=f"{module}.{alias.name}" if module else alias.name,
                file_path=self.file_path,
                line=node.lineno,
                context=self._get_line_context(node.lineno),
                ref_type="import",
            ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """
        Handles function/constructor calls.

        Patterns we recognise:
          - foo()             → ast.Call(func=ast.Name(id='foo'))
          - obj.method()      → ast.Call(func=ast.Attribute(attr='method'))
          - module.Class()    → same as above
        """
        func = node.func
        if isinstance(func, ast.Name):
            self.refs.append(SymbolReference(
                symbol_name=func.id,
                file_path=self.file_path,
                line=node.lineno,
                context=self._get_line_context(node.lineno),
                ref_type="call",
            ))
            # Don't record this Name node again as a bare name reference
            self._ignore_nodes.add(id(func))
        elif isinstance(func, ast.Attribute):
            self.refs.append(SymbolReference(
                symbol_name=func.attr,
                file_path=self.file_path,
                line=node.lineno,
                context=self._get_line_context(node.lineno),
                ref_type="call",
            ))
        # Continue walking into the call's arguments and children
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """
        Catches bare name references that are NOT already captured as
        imports or calls. This covers type annotations, variable usage,
        base class references, etc.
        """
        if id(node) not in self._ignore_nodes:
            self.refs.append(SymbolReference(
                symbol_name=node.id,
                file_path=self.file_path,
                line=node.lineno,
                context=self._get_line_context(node.lineno),
                ref_type="name",
            ))
        self.generic_visit(node)


class ASTPythonParser(GraphParser):
    """
    Parses Python source code using the native `ast` module.
    Extracts imports, classes, standalone functions, methods, and symbol references.
    """

    def parse_file(self, file_path: str, content: str) -> Optional[FileNode]:
        if not file_path.endswith('.py'):
            return None
            
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Skip files with syntax errors (e.g. Python 2 code)
            return None
            
        file_node = FileNode(file_path=file_path)
        
        # We only want top-level items to avoid extreme recursion for now
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    file_node.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    file_node.imports.append(f"{module}.{alias.name}")
            elif isinstance(node, ast.ClassDef):
                cls_node = self._parse_class(node)
                file_node.classes.append(cls_node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_node = self._parse_function(node)
                file_node.functions.append(func_node)
                
        return file_node

    def extract_references(
        self, file_path: str, content: str
    ) -> List[SymbolReference]:
        """
        Extracts all symbol references from a Python source file.

        Unlike parse_file() which only looks at top-level definitions,
        this method walks the ENTIRE AST to find every usage of every name.
        
        Args:
            file_path: Path to the source file (used in SymbolReference metadata).
            content:   The full source code of the file.

        Returns:
            A list of SymbolReference objects, one per usage found.
        """
        if not file_path.endswith('.py'):
            return []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []

        source_lines = content.splitlines()
        collector = _ReferenceCollector(file_path, source_lines)
        collector.visit(tree)
        return collector.refs

    def _parse_class(self, node: ast.ClassDef) -> ClassNode:
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)
            else:
                bases.append("UnknownBase")
                
        docstring = ast.get_docstring(node)
        
        cls_node = ClassNode(
            name=node.name,
            docstring=docstring,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            bases=bases
        )
        
        # Find all methods inside the class
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_node = self._parse_function(child, is_method=True)
                cls_node.methods.append(method_node)
                
        return cls_node

    def _parse_function(self, node: ast.FunctionDef, is_method: bool = False) -> FunctionNode:
        args = []
        for a in node.args.args:
            args.append(a.arg)
            
        # Handle *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
            
        # Handle **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
            
        docstring = ast.get_docstring(node)
        
        return FunctionNode(
            name=node.name,
            docstring=docstring,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            args=args,
            is_method=is_method
        )
