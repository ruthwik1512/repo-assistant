"""
routes.py

REST endpoint definitions that map the main.py CLI commands to HTTP:

  POST /analyze  — clone, index, embed, and build the vector store
  POST /chat     — RAG question answering via RepoBot
  GET  /show/{symbol} — jump to a class/function definition
  GET  /refs/{symbol} — list all references to a symbol
"""

from fastapi import APIRouter, Depends, HTTPException

from .dependencies import get_service
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    RefsResponse,
    ShowResponse,
)
from .service import RepoAssistantService, serialize_sources

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_repo(
    body: AnalyzeRequest,
    service: RepoAssistantService = Depends(get_service),
) -> AnalyzeResponse:
    try:
        return service.analyze(url=body.url, file_limit=body.file_limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    service: RepoAssistantService = Depends(get_service),
) -> ChatResponse:
    try:
        response = service.chat(question=body.question, top_k=body.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        content=response.content,
        sources=serialize_sources(response),
    )


@router.get("/show/{symbol}", response_model=ShowResponse)
def show_symbol(
    symbol: str,
    service: RepoAssistantService = Depends(get_service),
) -> ShowResponse:
    try:
        result, found = service.show_symbol(symbol)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not found:
        raise HTTPException(status_code=404, detail=result)

    return ShowResponse(symbol=symbol, result=result, found=True)


@router.get("/refs/{symbol}", response_model=RefsResponse)
def find_references(
    symbol: str,
    service: RepoAssistantService = Depends(get_service),
) -> RefsResponse:
    try:
        result, count = service.find_refs(symbol)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if count == 0:
        raise HTTPException(status_code=404, detail=result)

    return RefsResponse(symbol=symbol, result=result, reference_count=count)
