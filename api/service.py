"""
service.py

Orchestrates the full Repo Assistant pipeline and holds session state.

This module mirrors the logic in main.py step-for-step, delegating all
work to the existing backend classes (RepoCloner, RepoWalker, FileIndexer,
CodeGraphAnalyzer, FileEmbedder, VectorStore, RepoBot, etc.).

No business logic is reimplemented here — only wiring and state management.
"""

from typing import Optional

from repo_assistant.traversal import RepoCloner, RepoWalker
from repo_assistant.indexer import FileIndexer
from repo_assistant.analyzer import ASTPythonParser, CodeGraphAnalyzer
from repo_assistant.embedder import FileEmbedder, SentenceTransformerBackend
from repo_assistant.store import VectorStore
from repo_assistant.chat import RepoBot, OllamaBackend, CrossEncoderReranker
from repo_assistant.chat.models import ChatResponse

from .schemas import AnalyzeResponse, SourceInfo


class RepoAssistantService:
    """
    Stateful service that runs the analysis pipeline once and exposes
    chat, symbol lookup, and reference queries against the results.
    """

    def __init__(self) -> None:
        self.analyzer: Optional[CodeGraphAnalyzer] = None
        self.bot: Optional[RepoBot] = None
        self.repo_url: Optional[str] = None
        self.repo_path: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return self.analyzer is not None and self.bot is not None

    def _ensure_ready(self) -> None:
        if not self.is_ready:
            raise RuntimeError(
                "Repository not analyzed yet. Call POST /analyze first."
            )

    def analyze(self, url: str, file_limit: int = 50) -> AnalyzeResponse:
        """
        Run the full pipeline from main.py:
          Clone → Walk → Index → AST Analyze → Embed → VectorStore → RepoBot
        """
        cloner = RepoCloner()
        repo_path = cloner.clone(url)

        walker = RepoWalker()
        file_paths = walker.walk(repo_path)

        indexer = FileIndexer()
        indexed_files = indexer.index(file_paths, repo_root=repo_path)
        indexed_files = indexed_files[:file_limit]

        parser = ASTPythonParser()
        analyzer = CodeGraphAnalyzer(parser)
        semantic_files = analyzer.analyze(indexed_files)
        skeleton = analyzer.generate_skeleton()

        embed_backend = SentenceTransformerBackend()
        embedder = FileEmbedder(embed_backend)

        embedded_source_files = embedder.embed(indexed_files)
        embedded_semantic_files = embedder.embed(semantic_files) if semantic_files else []

        store = VectorStore(embed_backend, persist_directory=None)
        store.build(embedded_source_files, namespace="source")
        if embedded_semantic_files:
            store.build(embedded_semantic_files, namespace="semantic")

        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        chat_backend = OllamaBackend(model_name="llama3")
        bot = RepoBot(store, chat_backend, reranker=reranker, skeleton=skeleton)

        self.analyzer = analyzer
        self.bot = bot
        self.repo_url = url
        self.repo_path = repo_path

        return AnalyzeResponse(
            status="ready",
            repo_url=url,
            repo_path=repo_path,
            files_indexed=len(indexed_files),
            semantic_files_count=len(semantic_files),
            message="Repository loaded and indexed.",
        )

    def chat(self, question: str, top_k: int = 3) -> ChatResponse:
        """Delegate to RepoBot.ask()."""
        self._ensure_ready()
        return self.bot.ask(question, top_k=top_k)

    def show_symbol(self, symbol: str) -> tuple[str, bool]:
        """Delegate to CodeGraphAnalyzer.find_symbol()."""
        self._ensure_ready()
        result = self.analyzer.find_symbol(symbol)
        found = f"Symbol '{symbol}' not found" not in result
        return result, found

    def find_refs(self, symbol: str) -> tuple[str, int]:
        """Delegate to CodeGraphAnalyzer.format_references()."""
        self._ensure_ready()
        refs = self.analyzer.find_references(symbol)
        result = self.analyzer.format_references(symbol)
        return result, len(refs)


def serialize_sources(response: ChatResponse) -> list[SourceInfo]:
    """Convert backend SearchResult objects into API-friendly SourceInfo DTOs."""
    return [
        SourceInfo(
            relative_path=src.indexed_file.relative_path,
            score=src.score,
            rank=src.rank,
            content_preview=src.chunk.content[:200],
        )
        for src in response.sources
    ]
