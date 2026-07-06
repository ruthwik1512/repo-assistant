"""
models.py

Data structures for representing the extracted Code Graph.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CodeNode:
    """Base class for any parsed code entity."""
    name: str
    docstring: Optional[str]
    start_line: int
    end_line: int


@dataclass
class FunctionNode(CodeNode):
    """Represents a parsed function or method."""
    args: List[str]
    is_method: bool = False

    def to_signature_str(self) -> str:
        """Returns a string representation of the function signature."""
        sig = f"def {self.name}({', '.join(self.args)}):"
        if self.docstring:
            sig += f"\n    \"\"\"{self.docstring}\"\"\""
        return sig


@dataclass
class ClassNode(CodeNode):
    """Represents a parsed class, its inheritance, and its methods."""
    bases: List[str]
    methods: List[FunctionNode] = field(default_factory=list)

    def to_signature_str(self) -> str:
        """Returns a string representation of the class signature."""
        base_str = f"({', '.join(self.bases)})" if self.bases else ""
        sig = f"class {self.name}{base_str}:"
        if self.docstring:
            sig += f"\n    \"\"\"{self.docstring}\"\"\""
        for m in self.methods:
            sig += f"\n    # Method: {m.name}"
        return sig


@dataclass
class FileNode:
    """Represents a parsed source file and its structural contents."""
    file_path: str
    imports: List[str] = field(default_factory=list)
    classes: List[ClassNode] = field(default_factory=list)
    functions: List[FunctionNode] = field(default_factory=list)


@dataclass
class SymbolReference:
    """
    A single usage of a symbol found in the codebase.

    This is the data unit behind the /refs command. Each instance records
    one place where a name (class, function, variable, module) appears
    in the source code.

    Attributes:
        symbol_name: The symbol being referenced (e.g. "UserService").
        file_path:   Absolute path of the file containing this reference.
        line:        1-based line number of the reference.
        context:     The actual source line (trimmed), for display.
        ref_type:    Category of usage — one of:
                       "import"        — appears in an import statement
                       "call"          — used as a function/constructor call
                       "name"          — a bare name reference (variable, type hint, etc.)
    """
    symbol_name: str
    file_path: str
    line: int
    context: str
    ref_type: str  # "import", "call", "name"

    def __repr__(self) -> str:
        return (
            f"SymbolReference("
            f"{self.ref_type}: {self.symbol_name} "
            f"at {self.file_path}:{self.line})"
        )
