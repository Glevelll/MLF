"""Запуск пайплайна задачи: загрузка State из БД, Supervisor, сохранение результата и HITL-очереди."""
from __future__ import annotations

import traceback
from datetime import UTC, datetime
from typing import Any

from backend.agents.base import AgentContext
from backend.agents.supervisor import Supervisor
from backend.core.config import get_settings
from backend.core.llm.manager import build_llm_manager
from backend.core.llm.tools import default_registry
from backend.core.models import ForecastState, TaskStatus
from backend.data.cache import get_cache
from backend.data.storage import ArtifactStore
from backend.db.models import Task
from backend.db.session import session_scope
from backend.hitl.active_learning import enqueue_students, enqueue_task_quality
from backend.retrieval.index_manager import get_retriever
from backend.utils.logging import get_logger
from backend.utils.tracing import DbRecorder, Recorder, Tracer

log = get_logger("pipeline")


def build_context(request_id: str, recorder: Recorder | None = None, llm=None) -> AgentContext:
    settings = get_settings()
    recorder = recorder or DbRecorder()
    return AgentContext(
        settings=settings,
        llm=llm or build_llm_manager(settings, recorder=recorder, cache=get_cache(), request_id=request_id),
        tracer=Tracer(request_id, recorder),
        registry=default_registry(),
        store=ArtifactStore(),
        retriever=get_retriever(),
    )


async def _set(task_id: str, **fields: Any) -> None:
    async with session_scope() as s:
        task = await s.get(Task, task_id)
        for k, v in fields.items():
            setattr(task, k, v)


async def _persist_result(task_id: str, state: ForecastState) -> None:
    async with session_scope() as s:
        task = await s.get(Task, task_id)
        task.state = state.model_dump(mode="json")
        if state.hitl_requested:
            task.status = TaskStatus.waiting_for_human
            await enqueue_task_quality(s, task_id, state.hitl_reason or "", {
                "model_type": state.model_type, "critic_score": state.critic_score,
                "metrics": (state.metrics or {}).get("cv"),
                "issues": [i.model_dump() for i in state.validation_issues]})
        else:
            task.status = TaskStatus.completed
            task.completed_at = datetime.now(UTC)
            await enqueue_students(s, task_id, state.metadata.get("hitl_students", []))


async def run_forecast(task_id: str, ctx: AgentContext | None = None) -> ForecastState | None:
    async with session_scope() as s:
        task = await s.get(Task, task_id)
        if task is None:
            raise KeyError(task_id)
        state = ForecastState.model_validate(task.state)
    await _set(task_id, status=TaskStatus.running)
    ctx = ctx or build_context(task_id)
    try:
        state = await Supervisor().run(state, ctx)
        await _persist_result(task_id, state)
        return state
    except Exception as e:
        log.error("task %s failed: %s", task_id, e)
        await _set(task_id, status=TaskStatus.failed, error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}")
        return None


async def resume_forecast(task_id: str, resolution: dict[str, Any], ctx: AgentContext | None = None) -> ForecastState | None:
    async with session_scope() as s:
        task = await s.get(Task, task_id)
        state = ForecastState.model_validate(task.state)
    if resolution.get("action") == "reject":
        await _set(task_id, status=TaskStatus.failed, error="отклонено пользователем (HITL)",
                   state={**state.model_dump(mode="json"), "hitl_resolution": resolution})
        return None
    await _set(task_id, status=TaskStatus.running)
    ctx = ctx or build_context(task_id)
    try:
        state = await Supervisor().resume(state, ctx, resolution)
        await _persist_result(task_id, state)
        return state
    except Exception as e:
        log.error("task %s resume failed: %s", task_id, e)
        await _set(task_id, status=TaskStatus.failed, error=f"{type(e).__name__}: {e}")
        return None
