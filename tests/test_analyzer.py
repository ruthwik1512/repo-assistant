"""
tests/test_analyzer.py

Unit tests for the Code Graph analyzer and AST parser.
"""

from repo_assistant.analyzer.parser import ASTPythonParser
from repo_assistant.indexer.models import IndexedFile
from repo_assistant.analyzer.engine import CodeGraphAnalyzer


def test_ast_parser_extracts_classes_and_functions():
    code = '''
import os
from collections import defaultdict

def top_level_func(a, b):
    """Adds two numbers."""
    return a + b

class MyClass(BaseClass):
    """My custom class."""
    
    def method_one(self, *args, **kwargs):
        pass
'''
    parser = ASTPythonParser()
    file_node = parser.parse_file("test.py", code)
    
    assert file_node is not None
    assert len(file_node.imports) == 2
    assert "os" in file_node.imports
    assert "collections.defaultdict" in file_node.imports
    
    assert len(file_node.functions) == 1
    assert file_node.functions[0].name == "top_level_func"
    assert file_node.functions[0].args == ["a", "b"]
    
    assert len(file_node.classes) == 1
    assert file_node.classes[0].name == "MyClass"
    assert file_node.classes[0].bases == ["BaseClass"]
    assert len(file_node.classes[0].methods) == 1
    assert file_node.classes[0].methods[0].name == "method_one"
    assert file_node.classes[0].methods[0].args == ["self", "*args", "**kwargs"]


def test_analyzer_engine_generates_semantic_signatures():
    code = '''
class Auth:
    def login(self): pass
'''
    idx_file = IndexedFile(path="test.py", relative_path="test.py", extension=".py", content=code, line_count=3, size_bytes=len(code))
    
    parser = ASTPythonParser()
    analyzer = CodeGraphAnalyzer(parser)
    
    semantic_files = analyzer.analyze([idx_file])
    
    assert len(semantic_files) == 1
    assert semantic_files[0].relative_path == "test.py"
    assert semantic_files[0].symbol_name == "Auth"
    assert semantic_files[0].symbol_type == "class"
    assert semantic_files[0].extension == ".signature"


def test_analyzer_engine_generates_skeleton():
    code = '''
def global_func(): pass
class Data(object): pass
'''
    idx_file = IndexedFile(path="test.py", relative_path="test.py", extension=".py", content=code, line_count=3, size_bytes=len(code))
    
    parser = ASTPythonParser()
    analyzer = CodeGraphAnalyzer(parser)
    analyzer.analyze([idx_file])
    
    skeleton = analyzer.generate_skeleton()
    assert "## File: test.py" in skeleton
    assert "- class Data(object):" in skeleton
    assert "- def global_func(...)" in skeleton


# =============================================================================
# Reference Extraction — Parser Level
# =============================================================================

class TestExtractReferences:
    """Tests for ASTPythonParser.extract_references()"""

    def test_extracts_import_references(self):
        code = "import os\nfrom collections import defaultdict\n"
        parser = ASTPythonParser()
        refs = parser.extract_references("test.py", code)
        
        import_refs = [r for r in refs if r.ref_type == "import"]
        names = [r.symbol_name for r in import_refs]
        assert "os" in names
        assert "collections.defaultdict" in names

    def test_extracts_call_references(self):
        code = "result = foo()\nbar(x, y)\n"
        parser = ASTPythonParser()
        refs = parser.extract_references("test.py", code)
        
        call_refs = [r for r in refs if r.ref_type == "call"]
        names = [r.symbol_name for r in call_refs]
        assert "foo" in names
        assert "bar" in names

    def test_extracts_attribute_call_references(self):
        code = "obj.method()\nmodule.ClassName()\n"
        parser = ASTPythonParser()
        refs = parser.extract_references("test.py", code)
        
        call_refs = [r for r in refs if r.ref_type == "call"]
        names = [r.symbol_name for r in call_refs]
        assert "method" in names
        assert "ClassName" in names

    def test_extracts_bare_name_references(self):
        code = "x = MyClass\nresult = x + value\n"
        parser = ASTPythonParser()
        refs = parser.extract_references("test.py", code)
        
        name_refs = [r for r in refs if r.ref_type == "name"]
        names = [r.symbol_name for r in name_refs]
        assert "MyClass" in names

    def test_records_correct_line_numbers(self):
        code = "import os\nx = 1\nfoo()\n"
        parser = ASTPythonParser()
        refs = parser.extract_references("test.py", code)

        os_ref = [r for r in refs if r.symbol_name == "os"][0]
        assert os_ref.line == 1

        foo_ref = [r for r in refs if r.symbol_name == "foo"][0]
        assert foo_ref.line == 3

    def test_records_source_line_context(self):
        code = "result = UserService.find(user_id)\n"
        parser = ASTPythonParser()
        refs = parser.extract_references("test.py", code)

        find_ref = [r for r in refs if r.symbol_name == "find"][0]
        assert find_ref.context == "result = UserService.find(user_id)"

    def test_returns_empty_for_non_python_files(self):
        parser = ASTPythonParser()
        refs = parser.extract_references("test.js", "const x = 1;")
        assert refs == []

    def test_returns_empty_for_syntax_errors(self):
        parser = ASTPythonParser()
        refs = parser.extract_references("test.py", "def foo(:\n")
        assert refs == []

    def test_imported_names_are_recorded_when_used(self):
        """Usages of imported modules should be recorded as 'name' type references."""
        code = "import os\npath = os.path.join('a', 'b')\n"
        parser = ASTPythonParser()
        refs = parser.extract_references("test.py", code)

        # "os" should appear as both "import" (the import statement) 
        # and "name" (the os.path.join usage)
        os_refs = [r for r in refs if r.symbol_name == "os"]
        types = {r.ref_type for r in os_refs}
        assert "import" in types
        assert "name" in types


# =============================================================================
# Reference Extraction — Engine Level (find_references / format_references)
# =============================================================================

class TestFindReferences:
    """Tests for CodeGraphAnalyzer.find_references() and format_references()"""

    def _build_analyzer(self, files):
        """Helper: create an analyzer and analyze a list of (path, code) tuples."""
        parser = ASTPythonParser()
        analyzer = CodeGraphAnalyzer(parser)
        indexed = [
            IndexedFile(
                path=path, relative_path=path, extension=".py",
                content=code, line_count=len(code.splitlines()),
                size_bytes=len(code)
            )
            for path, code in files
        ]
        analyzer.analyze(indexed)
        return analyzer

    def test_find_references_returns_sorted_results(self):
        analyzer = self._build_analyzer([
            ("b.py", "x = Foo()\n"),
            ("a.py", "y = Foo()\n"),
        ])
        refs = analyzer.find_references("Foo")
        # Should be sorted by file_path
        assert refs[0].file_path == "a.py"
        assert refs[1].file_path == "b.py"

    def test_find_references_across_multiple_files(self):
        analyzer = self._build_analyzer([
            ("auth.py", "from services import UserService\n"),
            ("api.py", "svc = UserService()\n"),
            ("tests.py", "mock = UserService()\n"),
        ])
        refs = analyzer.find_references("UserService")
        assert len(refs) >= 2  # at least api.py and tests.py call it

    def test_find_references_returns_empty_for_unknown_symbol(self):
        analyzer = self._build_analyzer([("a.py", "x = 1\n")])
        refs = analyzer.find_references("NonExistent")
        assert refs == []

    def test_format_references_shows_file_grouping(self):
        analyzer = self._build_analyzer([
            ("auth.py", "x = Foo()\n"),
            ("api.py", "y = Foo()\n"),
        ])
        output = analyzer.format_references("Foo")
        assert "auth.py" in output
        assert "api.py" in output
        assert "call" in output

    def test_format_references_shows_not_found_message(self):
        analyzer = self._build_analyzer([("a.py", "x = 1\n")])
        output = analyzer.format_references("Missing")
        assert "No references found" in output

