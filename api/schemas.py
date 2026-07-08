"""
schemas.py

Pydantic request/response models for the REST API.

These are API-only DTOs — they do not replace the backend dataclasses
in repo_assistant/. They translate between HTTP JSON and the existing
service layer.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Body for POST /analyze."""

    url: str = Field(
        default="https://github.com/pallets/flask",
        description="Public GitHub HTTPS URL to clone and index.",
    )
    file_limit: int = Field(
        default=50,
        ge=1,
        description="Maximum number of source files to index (demo cap from main.py).",
    )


class AnalyzeResponse(BaseModel):
    """Result of a successful repository analysis."""

    status: str
    repo_url: str
    repo_path: str
    files_indexed: int
    semantic_files_count: int
    message: str


class ChatRequest(BaseModel):
    """Body for POST /chat."""

    question: str = Field(..., min_length=1, description="Natural-language question about the repo.")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of context chunks to retrieve.")


class SourceInfo(BaseModel):
    """One retrieved context chunk cited in a chat answer."""

    relative_path: str
    score: float
    rank: int
    content_preview: str


class ChatResponse(BaseModel):
    """Answer from the RAG pipeline plus cited sources."""

    content: str
    sources: List[SourceInfo]


class ShowResponse(BaseModel):
    """Symbol lookup result (mirrors the /show CLI command)."""

    symbol: str
    result: str
    found: bool


class RefsResponse(BaseModel):
    """Reference listing for a symbol (mirrors the /refs CLI command)."""

    symbol: str
    result: str
    reference_count: int
