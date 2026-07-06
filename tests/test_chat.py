"""
tests/test_chat.py

Unit tests for the chat and RAG orchestration module.
"""

import pytest
from typing import List

from repo_assistant.chat.models import ChatMessage, ChatResponse
from repo_assistant.chat.backends import LLMBackend
from repo_assistant.chat.bot import RepoBot
from repo_assistant.store.models import SearchResult
from repo_assistant.indexer.models import IndexedFile
from repo_assistant.embedder.models import EmbeddedChunk


# =============================================================================
# Test Doubles
# =============================================================================

class FakeLLMBackend(LLMBackend):
    """
    Returns a deterministic answer and stores the last prompt it received
    so we can assert that the RepoBot constructed the context correctly.
    """
    def __init__(self, fixed_response: str = "Fake answer"):
        self.fixed_response = fixed_response
        self.last_messages: List[ChatMessage] = []

    def generate(self, messages: List[ChatMessage]) -> str:
        self.last_messages = messages
        return self.fixed_response


class FakeVectorStore:
    """
    Simulates ChromaDB search returning dummy SearchResults.
    """
    def __init__(self, mock_results: List[SearchResult]):
        self.mock_results = mock_results
        self.last_query = ""
        self.last_k = 0

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        self.last_query = query
        self.last_k = top_k
        return self.mock_results


class FakeReranker:
    """
    Simulates a CrossEncoderReranker by reversing the order of the results
    (so we can verify that the reranker actually ran and mutated the list)
    and returning only the first top_k items.
    """
    def __init__(self):
        self.last_query = ""

    def rerank(self, query: str, results: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        self.last_query = query
        results.reverse()
        return results[:top_k]

def make_search_result(path: str, chunk_content: str, score: float = 0.9) -> SearchResult:
    f = IndexedFile(path=path, relative_path=path, extension=".py", content="", line_count=10, size_bytes=100)
    c = EmbeddedChunk(content=chunk_content, embedding=[0.0], chunk_index=0, start_char=0)
    return SearchResult(chunk=c, indexed_file=f, score=score, rank=1)


# =============================================================================
# Tests
# =============================================================================

class TestRepoBot:
    def test_ask_queries_store_with_correct_arguments(self):
        store = FakeVectorStore([])
        backend = FakeLLMBackend()
        bot = RepoBot(store, backend)

        bot.ask("where is the router?", top_k=3)

        assert store.last_query == "where is the router?"
        assert store.last_k == 3

    def test_ask_queries_store_with_higher_fetch_k_if_reranker_present(self):
        store = FakeVectorStore([])
        backend = FakeLLMBackend()
        reranker = FakeReranker()
        bot = RepoBot(store, backend, reranker)

        bot.ask("where is the router?", top_k=5)

        # It should fetch 20 from ChromaDB because we have a reranker
        assert store.last_k == 20
        assert reranker.last_query == "where is the router?"

    def test_ask_returns_chat_response_with_sources(self):
        res1 = make_search_result("router.py", "class Router:")
        store = FakeVectorStore([res1])
        backend = FakeLLMBackend("I found it.")
        bot = RepoBot(store, backend)

        response = bot.ask("question")

        assert isinstance(response, ChatResponse)
        assert response.content == "I found it."
        assert len(response.sources) == 1
        assert response.sources[0].indexed_file.relative_path == "router.py"

    def test_ask_constructs_correct_prompt_with_context(self):
        res1 = make_search_result("main.py", "def main(): pass")
        res2 = make_search_result("utils.py", "def helper(): pass")
        
        store = FakeVectorStore([res1, res2])
        backend = FakeLLMBackend()
        bot = RepoBot(store, backend)

        bot.ask("how does it work?")

        # Bot should have sent exactly two messages: system and user
        assert len(backend.last_messages) == 2
        
        system_msg = backend.last_messages[0]
        user_msg = backend.last_messages[1]

        assert system_msg.role == "system"
        assert user_msg.role == "user"

        # Check if the context was injected into the system message
        assert "main.py" in system_msg.content
        assert "def main(): pass" in system_msg.content
        
        assert "utils.py" in system_msg.content
        assert "def helper(): pass" in system_msg.content
        
        # Check if the question was appended as a pure user message
        assert "how does it work?" in user_msg.content

    def test_ask_handles_empty_search_results(self):
        """If no code matches, the prompt should still be valid but empty."""
        store = FakeVectorStore([])
        backend = FakeLLMBackend()
        bot = RepoBot(store, backend)

        bot.ask("where is the flux capacitor?")

        user_msg = backend.last_messages[1]
        assert "where is the flux capacitor?" in user_msg.content
        # Should not crash if there are no context blocks
