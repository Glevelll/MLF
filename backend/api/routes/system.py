"""Системные эндпоинты: health, журналы LLM, агентные метрики, база знаний, переиндексация."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, require_admin
from backend.core.config import get_settings
from backend.db.models import CacheEntry, Glossary, KnowledgeBase, LLMCall, Task, User
from backend.db.session import get_session
from backend.retrieval.index_manager import get_retriever
from backend.worker.dispatcher import dispatch

router = APIRouter(tags=["system"])


@router.get("/healthz")
async def healthz() -> dict:
    s = get_settings()
    return {"status": "ok", "env": s.app_env, "llm_provider": s.llm_provider, "task_backend": s.task_backend,
            "vector_backend": s.vector_backend, "rag_docs": get_retriever().size}


@router.get("/api/v1/logs/llm-calls")
async def llm_calls(request_id: str | None = None, agent: str | None = None, limit: int = Query(100, le=1000),
                    _: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)) -> list[dict]:
    q = select(LLMCall).order_by(LLMCall.created_at.desc()).limit(limit)
    if request_id:
        q = q.where(LLMCall.request_id == request_id)
    if agent:
        q = q.where(LLMCall.agent_name == agent)
    rows = (await session.execute(q)).scalars().all()
    return [{"id": r.id, "request_id": r.request_id, "agent": r.agent_name, "provider": r.provider,
             "model": r.model_name, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
             "tokens": r.tokens_used, "estimated": r.tokens_estimated, "latency_ms": r.latency_ms,
             "status": r.status, "error": r.error, "cached": r.cached, "created_at": r.created_at.isoformat()}
            for r in rows]


@router.get("/api/v1/logs/agent-metrics")
async def agent_metrics(_: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)) -> dict:
    """Агентные метрики по всем задачам: итерации, доля HITL, токены, латентность LLM."""
    tasks = (await session.execute(select(Task).where(Task.task_type != "evaluation"))).scalars().all()
    states = [t.state or {} for t in tasks if t.status in ("completed", "waiting_for_human")]
    n = len(states)
    hitl = sum(1 for s in states if s.get("hitl_reason") or s.get("hitl_resolution"))
    llm = (await session.execute(select(func.count(LLMCall.id), func.coalesce(func.sum(LLMCall.tokens_used), 0),
                                        func.avg(LLMCall.latency_ms)))).one()
    by_agent = (await session.execute(select(LLMCall.agent_name, func.count(LLMCall.id),
                                             func.sum(LLMCall.tokens_used)).group_by(LLMCall.agent_name))).all()
    status_counts = (await session.execute(select(Task.status, func.count()).where(Task.task_type != "evaluation")
                                           .group_by(Task.status))).all()
    return {
        "tasks_finished": n,
        "avg_iterations": (sum(s.get("iteration", 0) for s in states) / n) if n else None,
        "hitl_rate": hitl / n if n else None,
        "avg_tokens_per_task": (sum(s.get("tokens_used", 0) for s in states) / n) if n else None,
        "llm_calls": int(llm[0]), "llm_tokens": int(llm[1]), "avg_llm_latency_ms": float(llm[2] or 0),
        "by_agent": {a: {"calls": int(c), "tokens": int(t or 0)} for a, c, t in by_agent},
        "status_counts": {s: int(c) for s, c in status_counts},
    }


@router.get("/api/v1/kb/stats")
async def kb_stats(_: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)) -> dict:
    kinds = (await session.execute(select(KnowledgeBase.kind, func.count()).group_by(KnowledgeBase.kind))).all()
    gl = (await session.execute(select(func.count(Glossary.id)))).scalar_one()
    cache = (await session.execute(select(func.count(CacheEntry.id), func.coalesce(func.sum(CacheEntry.hits), 0)))).one()
    return {"knowledge_base": {k: int(n) for k, n in kinds}, "glossary": int(gl), "index_docs": get_retriever().size,
            "cache_entries": int(cache[0]), "cache_hits": int(cache[1])}


@router.get("/api/v1/kb/search")
async def kb_search(q: str, kind: str = "student_case", k: int = Query(5, le=20),
                    _: User = Depends(get_current_user)) -> dict:
    r = get_retriever()
    return {"results": [d.brief() for d in r.search(q, kind=kind, k=k)], "glossary": r.glossary_search(q)}


@router.post("/api/v1/admin/reindex", status_code=202)
async def reindex(_: User = Depends(require_admin)) -> dict:
    return {"status": "scheduled", "execution": dispatch("reindex_rag_task")}
