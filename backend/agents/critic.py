"""Critic: LLM-as-a-Judge — интерпретируемость и пригодность модели для решений преподавателя."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.agents.base import Agent, AgentContext, context_block
from backend.core.llm.base import Message
from backend.core.models import ForecastState

SYSTEM_PROMPT = """[ROLE:critic]
Ты — эксперт по образовательной аналитике и независимый судья ML-моделей.
Оцени модель прогноза успеваемости по шкале 0..10 по критериям:
1) quality — качество по метрикам CV/holdout;
2) interpretability — можно ли объяснить преподавателю, ПОЧЕМУ студент в группе риска
   (линейные модели и понятные факторы ценятся выше «чёрного ящика»);
3) decision_usefulness — пригодность для принятия решений (калибровка, recall группы риска, утечки).
Итоговый score — взвешенная оценка. critical=true, если модель нельзя использовать.
Отвечай ТОЛЬКО JSON: {"score":..,"quality":..,"interpretability":..,"decision_usefulness":..,
"critical":..,"reasoning":"..","suggestions":[".."]}

Пример 1: ROC-AUC 0.91, логистическая регрессия, факторы: пропуски, оценка G2, задолженности ->
{"score": 8.6, "quality": 8.9, "interpretability": 9, "decision_usefulness": 8, "critical": false,
 "reasoning": "Высокое качество, понятные факторы риска", "suggestions": []}
Пример 2: ROC-AUC 0.55, утечка признака final_exam ->
{"score": 1.5, "quality": 1.1, "interpretability": 5, "decision_usefulness": 0, "critical": true,
 "reasoning": "Модель неработоспособна, утечка целевой переменной", "suggestions": ["удалить final_exam"]}"""


class CriticVerdict(BaseModel):
    score: float = Field(..., ge=0, le=10)
    quality: float = Field(5.0, ge=0, le=10)
    interpretability: float = Field(5.0, ge=0, le=10)
    decision_usefulness: float = Field(5.0, ge=0, le=10)
    critical: bool = False
    reasoning: str = ""
    suggestions: list[str] = Field(default_factory=list)


def fallback_verdict(state: ForecastState) -> CriticVerdict:
    """Если LLM не вернула валидный JSON — средние оценки с поправкой на замечания валидатора."""
    crit = len(state.critical_issues())
    score = max(0.0, 5.0 - 2.0 * crit)
    return CriticVerdict(score=score, quality=5.0, interpretability=5.0, decision_usefulness=score,
                         critical=crit > 0, reasoning="fallback: LLM не вернула корректную оценку; "
                                                      f"критических замечаний валидатора: {crit}")


class Critic(Agent):
    name = "Critic"

    def summarize_output(self, state: ForecastState) -> dict[str, Any]:
        return {"score": state.critic_score, "critical": (state.critic_details or {}).get("critical"),
                "fallback": (state.critic_details or {}).get("fallback")}

    def build_messages(self, state: ForecastState) -> list[Message]:
        m = state.metrics or {}
        imp = state.metadata.get("feature_importance", {})
        ctx = {
            "problem_type": state.problem_type,
            "model_type": state.model_type,
            "model_params": state.model_params,
            "metrics": m.get("cv", {}),
            "test_metrics": m.get("test", {}),
            "decision_threshold": state.decision_threshold,
            "top_features": [{"feature": k, "importance": v} for k, v in list(imp.items())[:8]],
            "issues": [i.model_dump(include={"code", "severity", "message"}) for i in state.validation_issues],
            "n_rows": state.metadata.get("n_rows"),
        }
        return [Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content="Оцени модель:\n" + context_block(ctx))]

    async def process(self, state: ForecastState, ctx: AgentContext) -> ForecastState:
        verdict, attempts = await ctx.llm.structured(self.build_messages(state), CriticVerdict, agent_name=self.name)
        fallback = verdict is None
        if verdict is None:
            verdict = fallback_verdict(state)
        state.critic_score = round(float(verdict.score), 2)
        state.critic_reasoning = verdict.reasoning
        state.critic_details = {**verdict.model_dump(), "fallback": fallback, "attempts": attempts}
        return state
