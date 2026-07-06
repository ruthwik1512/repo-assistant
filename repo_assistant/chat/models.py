"""
models.py

Data structures for the Chat and RAG interface.
"""

from dataclasses import dataclass
from typing import List

from repo_assistant.store.models import SearchResult


@dataclass
class ChatMessage:
    """A single message in the conversation history."""
    role: str      # typically "system", "user", or "assistant"
    content: str


@dataclass
class ChatResponse:
    """
    The final output returned to the user, containing both the 
    generated answer and the sources used to generate it.
    """
    content: str
    sources: List[SearchResult]
