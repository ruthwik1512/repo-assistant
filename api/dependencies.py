"""
dependencies.py

FastAPI dependency-injection helpers.

Provides a single shared RepoAssistantService instance per application
lifecycle via app.state, so all routes operate on the same analyzed session.
"""

from fastapi import Request

from .service import RepoAssistantService


def get_service(request: Request) -> RepoAssistantService:
    return request.app.state.service
