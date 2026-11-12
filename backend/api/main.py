"""FastAPI-приложение."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import auth, evaluate, forecast, hitl, system
from backend.db.session import init_db
from backend.utils.logging import get_logger, setup_logging

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await init_db()
    from backend.retrieval.index_manager import get_retriever

    log.info("API started; RAG docs: %d", get_retriever().size)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="MLF-2026 · Прогноз успеваемости студентов", version="1.0.0", lifespan=lifespan,
                  description="Мультиагентная система (Supervisor + DataAnalyst/ModelSelector/Validator/Critic/Editor) "
                              "с Tool Calling, гибридным RAG и Human-in-the-Loop.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    for r in (system.router, auth.router, forecast.router, hitl.router, evaluate.router):
        app.include_router(r)
    return app


app = create_app()
