"""
models.py

Data structures for representing the extracted Code Graph.

Design note on CallEdge vs CallNode:
  CallEdge is an *intermediate* object produced by the parser for a single file.
  CallNode is the *persistent* graph node owned by CodeGraphAnalyzer.
  Keeping them separate makes the parser stateless and the engine the sole
  owner of graph state — a clean Single-Responsibility split.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


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
    # relative_path is injected by the engine after parsing so the parser
    # stays decoupled from filesystem concerns. Default "" keeps backward
    # compatibility with any code that constructs FileNode directly.
    relative_path: str = ""
    imports: List[str] = field(default_factory=list)
    classes: List[ClassNode] = field(default_factory=list)
    functions: List[FunctionNode] = field(default_factory=list)


@dataclass
class SymbolReference:
    """
    A single usage of a symbol found in the codebase.

    Attributes:
        symbol_name: The symbol being referenced (e.g. "UserService").
        file_path:   Absolute path of the file containing this reference.
        line:        1-based line number of the reference.
        context:     The actual source line (trimmed), for display.
        ref_type:    "import", "call", or "name".
    """
    symbol_name: str
    file_path: str
    line: int
    context: str
    ref_type: str

    def __repr__(self) -> str:
        return (
            f"SymbolReference("
            f"{self.ref_type}: {self.symbol_name} "
            f"at {self.file_path}:{self.line})"
        )


@dataclass
class CallEdge:
    """
    A raw call relationship extracted from a single source file by the parser.

    This is an *intermediate* data structure: the parser produces it, the
    engine consumes it to wire the persistent CallNode graph. It is never
    stored long-term.

    Attributes:
        caller_scope: Scope of the calling function WITHOUT the file path prefix.
                      e.g. "Flask.login" for a method, "locate_app" for a top-level func.
        callee_name:  The bare name of the called symbol as it appears in source.
                      e.g. "validate", "Map", "json"
        line:         Line number of the call expression in the source file.
        file_path:    Absolute path of the file (for diagnostics only).
    """
    caller_scope: str
    callee_name: str
    line: int
    file_path: str


@dataclass
class CallNode:
    """
    A persistent node in the call graph, representing one callable symbol.

    Every function and class method parsed from the repository gets one
    CallNode. The graph is stored as two bidirectional adjacency sets:
      - outgoing: what this symbol calls (drives /calls and /graph)
      - incoming: who calls this symbol (drives /callers and /impact)

    FQN Format:
        <relative_path>::<ClassName>.<method_name>   (methods)
        <relative_path>::<function_name>             (top-level functions)

    Examples:
        src/flask/app.py::Flask.full_dispatch_request
        src/flask/cli.py::locate_app

    The to_dict() / from_dict() methods enable JSON serialization so a
    persistent index file can be loaded on startup rather than rebuilding
    the graph from scratch each run.

    Attributes:
        fqn:        Fully-qualified name (unique graph key).
        short_name: Bare symbol name. e.g. "full_dispatch_request"
        file_path:  Absolute path to the file containing this definition.
        line:       Line number of the function/method definition.
        outgoing:   Callee identifiers (FQNs when unambiguous, raw names when
                    multiple candidates exist or callee is external).
        incoming:   Caller FQNs. Always stores full FQNs (never raw names)
                    because callers are always known nodes in the graph.
    """
    fqn: str
    short_name: str
    file_path: str
    line: int
    outgoing: Set[str] = field(default_factory=set)
    incoming: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a JSON-compatible dictionary.

        Sets are serialized as sorted lists for deterministic, diffable output.
        """
        return {
            "fqn": self.fqn,
            "short_name": self.short_name,
            "file_path": self.file_path,
            "line": self.line,
            "outgoing": sorted(self.outgoing),
            "incoming": sorted(self.incoming),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CallNode":
        """Deserialize from a dictionary (e.g. loaded from JSON)."""
        node = cls(
            fqn=data["fqn"],
            short_name=data["short_name"],
            file_path=data["file_path"],
            line=data["line"],
        )
        node.outgoing = set(data.get("outgoing", []))
        node.incoming = set(data.get("incoming", []))
        return node

    def __repr__(self) -> str:
        return (
            f"CallNode({self.fqn!r}, "
            f"out={len(self.outgoing)}, in={len(self.incoming)})"
        )

