"""
app.py

FastAPI application factory.

Creates the app, attaches a shared RepoAssistantService to app.state,
and mounts the route handlers from routes.py.
"""

from fastapi import FastAPI

from .routes import router
from .service import RepoAssistantService


def create_app() -> FastAPI:
    app = FastAPI(
        title="Repo Assistant API",
        description=(
            "REST API for the AI-powered repository assistant. "
            "Analyze a GitHub repo, then chat about it or query symbols."
        ),
        version="1.0.0",
    )

    app.state.service = RepoAssistantService()
    app.include_router(router)

    return app


app = create_app()
