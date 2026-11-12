"""Базовый класс агента и контекст выполнения."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from backend.core.config import Settings
from backend.core.llm.manager import LLMManager
from backend.core.llm.tools import ToolRegistry
from backend.core.models import ForecastState
from backend.data.storage import ArtifactStore
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.utils.tracing import Tracer


@dataclass
class AgentContext:
    settings: Settings
    llm: LLMManager
    tracer: Tracer
    registry: ToolRegistry
    store: ArtifactStore
    retriever: HybridRetriever | None = None
    cache: dict[str, Any] = field(default_factory=dict)  # кэш тяжёлых объектов в пределах одного запуска


def context_block(data: dict[str, Any]) -> str:
    """JSON-контекст для промпта (модель видит данные в структурированном виде)."""
    return "<context>" + json.dumps(data, ensure_ascii=False, default=str) + "</context>"


class Agent(ABC):
    name: ClassVar[str] = "agent"

    @abstractmethod
    async def process(self, state: ForecastState, ctx: AgentContext) -> ForecastState: ...

    def summarize_input(self, state: ForecastState) -> dict[str, Any]:
        return {"iteration": state.iteration, "model_type": state.model_type}

    def summarize_output(self, state: ForecastState) -> dict[str, Any]:
        return {"model_type": state.model_type, "validation_passed": state.validation_passed,
                "critic_score": state.critic_score}

    async def __call__(self, state: ForecastState, ctx: AgentContext) -> ForecastState:
        async with ctx.tracer.span(self.name, "process", self.summarize_input(state),
                                   model_name=ctx.llm.model_name) as span:
            new_state = await self.process(state, ctx)
            span.set_output(self.summarize_output(new_state))
            return new_state
