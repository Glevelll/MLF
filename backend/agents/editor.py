"""Editor: предлагает и применяет правки (признаки, порог, гиперпараметры, смена модели) через Tool Calling."""
from __future__ import annotations

from typing import Any

from backend.agents.base import Agent, AgentContext, context_block
from backend.core.llm.base import Message, ToolCall
from backend.core.models import EditorAction, ForecastState
from backend.tabular.features import AVAILABLE_OPS

SYSTEM_PROMPT = """[ROLE:editor]
Ты — редактор ML-пайплайна. По замечаниям валидатора и критика предложи правки, вызывая инструменты:
add_features (динамика оценок, агрегаты активности, отношения...), drop_features (утечки),
change_threshold (порог группы риска, например по максимуму F1), change_hyperparams, switch_model.
Можно вызвать несколько инструментов. Если правки не нужны — верни {"tool_calls": []}. Не пиши код."""


class Editor(Agent):
    name = "Editor"

    def summarize_output(self, state: ForecastState) -> dict[str, Any]:
        return {"changed": state.editor_changed,
                "actions": [a.tool for a in state.editor_actions if a.reason != "hitl"][-5:]}

    def build_messages(self, state: ForecastState, ctx: AgentContext) -> list[Message]:
        leak = next((i for i in state.validation_issues if i.code == "leakage"), None)
        context = {
            "problem_type": state.problem_type,
            "model_type": state.model_type,
            "model_params": state.model_params,
            "metrics": (state.metrics or {}).get("cv", {}),
            "best_f1_threshold": (state.metrics or {}).get("best_f1_threshold"),
            "decision_threshold": state.decision_threshold,
            "issues": [i.model_dump(include={"code", "severity", "message"}) for i in state.validation_issues],
            "leaking_features": list(leak.details.get("features", {})) if leak else [],
            "critic_score": state.critic_score,
            "critic_threshold": ctx.settings.critic_threshold,
            "critic_suggestions": (state.critic_details or {}).get("suggestions", []),
            "feature_ops": state.feature_ops,
            "available_ops": list(AVAILABLE_OPS),
            "available_models": ctx.registry.names(kind="model", task_type=state.task_type,
                                                   problem_type=state.problem_type),
            "iteration": state.iteration,
        }
        return [Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content="Предложи правки:\n" + context_block(context))]

    def apply(self, state: ForecastState, call: ToolCall, ctx: AgentContext, reason: str = "") -> bool:
        a = call.arguments
        changed = False
        if call.name == "add_features":
            new = [o for o in a["ops"] if o not in state.feature_ops]
            if new:
                state.feature_ops = state.feature_ops + new
                changed = True
        elif call.name == "drop_features":
            new = [c for c in a["columns"] if c not in state.dropped_features and c in state.feature_columns
                   + state.metadata.get("working_features", [])]
            if new:
                state.dropped_features = state.dropped_features + new
                changed = True
        elif call.name == "change_threshold":
            if abs(a["threshold"] - state.decision_threshold) > 1e-6:
                state.decision_threshold = float(a["threshold"])
                changed = True
        elif call.name == "change_hyperparams" and state.model_type:
            tool = ctx.registry.get(state.model_type)
            try:
                merged = tool.validate_args({**(state.model_params or {}), **a["params"]})
            except Exception:
                return False
            if merged != state.model_params:
                state.model_params = merged
                changed = True
        elif call.name == "switch_model" and not state.requested_model:
            mt = a["model_type"]
            allowed = ctx.registry.names(kind="model", task_type=state.task_type, problem_type=state.problem_type)
            if mt in allowed and mt != state.model_type:
                state.forced_model = mt
                changed = True
        if changed:
            state.editor_actions.append(EditorAction(tool=call.name, arguments=a, reason=reason))
        return changed

    async def process(self, state: ForecastState, ctx: AgentContext) -> ForecastState:
        tools = ctx.registry.list(kind="editor")
        if state.requested_model:
            tools = [tool for tool in tools if tool.name != "switch_model"]
        res = await ctx.llm.call_tool(self.build_messages(state, ctx), tools, agent_name=self.name, multi=True)
        changed = False
        for call in res.calls:
            async with ctx.tracer.span(self.name, "tool_call", call.arguments, tool_name=call.name,
                                       parameters=call.arguments) as sp:
                ok = self.apply(state, call, ctx, reason=f"llm:{res.mode}")
                sp.set_output({"applied": ok})
                changed |= ok
        if not changed:
            changed = self._rule_based(state, ctx)
        state.editor_changed = changed
        return state

    def _rule_based(self, state: ForecastState, ctx: AgentContext) -> bool:
        """Детерминированный fallback, если LLM ничего не предложила."""
        leak = next((i for i in state.validation_issues if i.code == "leakage"), None)
        if leak:
            cols = list(leak.details.get("features", {}))
            if self.apply(state, ToolCall(name="drop_features", arguments={"columns": cols, "reason": "leakage"}),
                          ctx, "rule"):
                return True
        for op in ("score_dynamics", "activity_aggregates", "ratios"):
            if op not in state.feature_ops:
                return self.apply(state, ToolCall(name="add_features", arguments={"ops": [op]}), ctx, "rule")
        return False
