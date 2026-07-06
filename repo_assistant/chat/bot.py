"""
bot.py

Orchestrates the Retrieval-Augmented Generation (RAG) pipeline.
"""

from typing import List

from repo_assistant.store.store import VectorStore
from .models import ChatMessage, ChatResponse
from .backends import LLMBackend


class RepoBot:
    """
    The main assistant class.
    Combines ChromaDB vector search with an LLM backend to answer
    questions about the repository.
    """

    def __init__(self, store: VectorStore, backend: LLMBackend, reranker=None, skeleton: str = "") -> None:
        self.store = store
        self.backend = backend
        self.reranker = reranker
        self.skeleton = skeleton
        self.history: List[ChatMessage] = []
        
        self.system_prompt = (
            "You are an expert software engineer and AI repository assistant. "
            "You will be provided with code snippets from the repository that "
            "are semantically related to the user's question.\n\n"
        )
        if self.skeleton:
            self.system_prompt += (
                "--- REPOSITORY ARCHITECTURE SKELETON ---\n"
                f"{self.skeleton}\n"
                "----------------------------------------\n\n"
            )
            
        self.system_prompt += (
            "Rules:\n"
            "1. Answer the question using ONLY the provided context and the skeleton.\n"
            "2. If the answer cannot be deduced from the context, clearly say so.\n"
            "3. Be concise. Provide code examples if they help explain the answer.\n"
            "4. Always cite the file paths you reference."
        )

    def ask(self, question: str, top_k: int = 5) -> ChatResponse:
        """
        Answers a user's question using the repository context.
        
        1. Embeds the question and searches ChromaDB for top_k chunks.
        2. Formats the chunks into a prompt.
        3. Generates an answer via the LLM.
        
        Args:
            question: Natural language question (e.g. "Where are routes defined?")
            top_k: Number of chunks to retrieve from the vector store.
            
        Returns:
            A ChatResponse containing the LLM's answer and the retrieved sources.
        """
        # Append the new question to history
        self.history.append(ChatMessage(role="user", content=question))

        # --- Stage 1: Retrieval ---
        fetch_k = 20 if self.reranker else top_k
        sources = self.store.search(question, top_k=fetch_k)

        if self.reranker and sources:
            sources = self.reranker.rerank(question, sources, top_k=top_k)

        # --- Stage 2: Prompt Formatting ---
        context_blocks = []
        for res in sources:
            block = (
                f"--- File: {res.indexed_file.relative_path} ---\n"
                f"{res.chunk.content}\n"
            )
            context_blocks.append(block)

        context_str = "\n".join(context_blocks)
        
        # Inject context into the system prompt so it doesn't pollute the chat history
        sys_msg = ChatMessage(
            role="system", 
            content=f"{self.system_prompt}\n\n[RETRIEVED CONTEXT]\n{context_str}"
        )

        messages = [sys_msg] + self.history

        # --- Stage 3: Generation ---
        answer = self.backend.generate(messages)

        # Save the assistant's answer to history
        self.history.append(ChatMessage(role="assistant", content=answer))

        return ChatResponse(
            content=answer,
            sources=sources
        )

    def ask_stream(self, question: str, top_k: int = 5):
        """
        Same as ask(), but returns a tuple of (sources, generator) so the 
        caller can stream the response token-by-token to the console.
        """
        self.history.append(ChatMessage(role="user", content=question))
        
        fetch_k = 20 if self.reranker else top_k
        sources = self.store.search(question, top_k=fetch_k)

        if self.reranker and sources:
            sources = self.reranker.rerank(question, sources, top_k=top_k)

        context_blocks = []
        for res in sources:
            block = (
                f"--- File: {res.indexed_file.relative_path} ---\n"
                f"{res.chunk.content}\n"
            )
            context_blocks.append(block)

        context_str = "\n".join(context_blocks)
        
        sys_msg = ChatMessage(
            role="system", 
            content=f"{self.system_prompt}\n\n[RETRIEVED CONTEXT]\n{context_str}"
        )

        messages = [sys_msg] + self.history

        raw_generator = self.backend.generate_stream(messages)
        
        def memory_generator():
            full_answer = []
            for chunk in raw_generator:
                full_answer.append(chunk)
                yield chunk
            # Once stream is done, save the complete answer to history
            self.history.append(ChatMessage(role="assistant", content="".join(full_answer)))

        return sources, memory_generator()
