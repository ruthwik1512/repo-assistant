"""
chat sub-package

Exposes the Chat/RAG pipeline.
"""

from .models import ChatMessage, ChatResponse
from .backends import LLMBackend, OllamaBackend
from .bot import RepoBot
from .reranker import CrossEncoderReranker

__all__ = [
    "ChatMessage", 
    "ChatResponse", 
    "LLMBackend", 
    "OllamaBackend", 
    "RepoBot",
    "CrossEncoderReranker"
]
