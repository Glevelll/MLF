"""Иерархический Reasoning Log и журнал LLM-вызовов.

`Recorder` — приёмник записей (БД или память). `Tracer` создаёт вложенные спаны
с parent_id, текущий спан хранится в contextvar, поэтому вложенность восстанавливается
автоматически (Supervisor -> Agent -> LLM call / tool).
"""
from __future__ import annotations

import contextvars
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from backend.utils.logging import get_logger

log = get_logger("trace")

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar("current_span", default=None)


def summarize(obj: Any, limit: int = 600) -> str:
    if obj is None:
        return ""
    if not isinstance(obj, str):
        try:
            obj = json.dumps(obj, ensure_ascii=False, default=str)
        except TypeError:
            obj = str(obj)
    return obj if len(obj) <= limit else obj[: limit - 3] + "..."


class Recorder(Protocol):
    async def record_span(self, data: dict[str, Any]) -> None: ...

    async def record_llm_call(self, data: dict[str, Any]) -> None: ...


class MemoryRecorder:
    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []

    async def record_span(self, data: dict[str, Any]) -> None:
        self.spans.append(data)

    async def record_llm_call(self, data: dict[str, Any]) -> None:
        self.llm_calls.append(data)


class DbRecorder:
    """Сохраняет спаны в reasoning_logs и вызовы LLM в llm_calls."""

    async def record_span(self, data: dict[str, Any]) -> None:
        from backend.db.models import ReasoningLog
        from backend.db.session import session_scope

        try:
            async with session_scope() as s:
                await s.merge(ReasoningLog(**data))
        except Exception as e:  # pragma: no cover - логирование не должно ронять пайплайн
            log.warning("failed to persist span: %s", e)

    async def record_llm_call(self, data: dict[str, Any]) -> None:
        from backend.db.models import LLMCall
        from backend.db.session import session_scope

        try:
            async with session_scope() as s:
                s.add(LLMCall(**data))
        except Exception as e:  # pragma: no cover
            log.warning("failed to persist llm call: %s", e)


@dataclass
class Span:
    id: str
    request_id: str
    agent_name: str
    step: str
    parent_id: str | None
    input_summary: str = ""
    output_summary: str = ""
    tool_name: str | None = None
    model_name: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    tokens: int = 0
    status: str = "ok"
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    _t0: float = field(default_factory=time.perf_counter)

    def set_output(self, out: Any) -> None:
        self.output_summary = summarize(out)

    def add_tokens(self, n: int) -> None:
        self.tokens += int(n or 0)


class Tracer:
    def __init__(self, request_id: str, recorder: Recorder | None = None) -> None:
        self.request_id = request_id
        self.recorder = recorder or MemoryRecorder()
        self.total_tokens = 0

    @staticmethod
    def current() -> Span | None:
        return _current_span.get()

    @asynccontextmanager
    async def span(
        self,
        agent_name: str,
        step: str = "process",
        input_summary: Any = None,
        *,
        tool_name: str | None = None,
        model_name: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> AsyncIterator[Span]:
        parent = _current_span.get()
        sp = Span(
            id=uuid.uuid4().hex,
            request_id=self.request_id,
            agent_name=agent_name,
            step=step,
            parent_id=parent.id if parent else None,
            input_summary=summarize(input_summary),
            tool_name=tool_name,
            model_name=model_name,
            parameters=parameters or {},
        )
        token = _current_span.set(sp)
        try:
            yield sp
        except Exception as e:
            sp.status = "error"
            sp.output_summary = summarize(f"{type(e).__name__}: {e}")
            raise
        finally:
            _current_span.reset(token)
            if parent is not None:
                parent.tokens += sp.tokens
            else:
                self.total_tokens += sp.tokens
            await self.recorder.record_span(
                {
                    "id": sp.id,
                    "request_id": sp.request_id,
                    "parent_id": sp.parent_id,
                    "agent_name": sp.agent_name,
                    "step": sp.step,
                    "tool_name": sp.tool_name,
                    "input_summary": sp.input_summary,
                    "output_summary": sp.output_summary,
                    "model_name": sp.model_name,
                    "parameters": json.loads(json.dumps(sp.parameters, default=str)),
                    "tokens": sp.tokens,
                    "started_at": sp.started_at,
                    "finished_at": datetime.now(UTC),
                    "duration_ms": (time.perf_counter() - sp._t0) * 1000,
                    "status": sp.status,
                }
            )


def build_tree(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Восстанавливает дерево рассуждений по parent_id."""
    nodes = {r["id"]: {**r, "children": []} for r in rows}
    roots: list[dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: str(x.get("started_at"))):
        node = nodes[r["id"]]
        pid = r.get("parent_id")
        if pid and pid in nodes:
            nodes[pid]["children"].append(node)
        else:
            roots.append(node)
    for n in nodes.values():
        n["children"].sort(key=lambda x: str(x.get("started_at")))
    return roots
