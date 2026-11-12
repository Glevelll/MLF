"""Supervisor: оркестрация цикла агентов, иерархический Reasoning Log, HITL, финализация результата."""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pandas as pd

from backend.agents.base import Agent, AgentContext
from backend.agents.common import load_working_data
from backend.core.models import ForecastState
from backend.data.knowledge_base import (
    canonical_profile,
    describe_student,
    profile_vector,
    recommendations_for,
)
from backend.tabular.datasets import read_table
from backend.tabular.training import occlusion_drivers

MAX_EXPLAINED = 300
MAX_HITL_STUDENTS = 100


class Supervisor:
    name = "Supervisor"

    def __init__(self, agents: dict[str, Agent] | None = None) -> None:
        self._agents: dict[str, Agent] = dict(agents or {})

    def _agent(self, key: str) -> Agent:
        """Ленивая инициализация агентов."""
        if key not in self._agents:
            if key == "analyst":
                from backend.agents.data_analyst import DataAnalyst as cls
            elif key == "selector":
                from backend.agents.model_selector import ModelSelector as cls
            elif key == "validator":
                from backend.agents.validator import Validator as cls
            elif key == "critic":
                from backend.agents.critic import Critic as cls
            elif key == "editor":
                from backend.agents.editor import Editor as cls
            else:
                raise KeyError(key)
            self._agents[key] = cls()
        return self._agents[key]

    @staticmethod
    def converged(state: ForecastState, ctx: AgentContext) -> bool:
        crit = bool((state.critic_details or {}).get("critical"))
        return (state.validation_passed and not crit and state.critic_score is not None
                and state.critic_score >= ctx.settings.critic_threshold)

    def needs_hitl(self, state: ForecastState, ctx: AgentContext) -> str | None:
        if state.critical_issues():
            return "критические замечания валидатора: " + ", ".join(i.code for i in state.critical_issues())
        if (state.critic_details or {}).get("critical"):
            return "критик признал модель непригодной"
        if not self.converged(state, ctx):
            return f"цикл не сошёлся за {state.max_iterations} итераций (score={state.critic_score})"
        return None

    async def run(self, state: ForecastState, ctx: AgentContext) -> ForecastState:
        state.log_id = ctx.tracer.request_id
        async with ctx.tracer.span(self.name, "run", {"task_id": state.task_id, "max_iter": state.max_iterations}) as sp:
            if not state.processed_path:
                state = await self._agent("analyst")(state, ctx)
            state = await self._loop(state, ctx)
            reason = self.needs_hitl(state, ctx)
            if reason:
                state.hitl_requested, state.hitl_reason = True, reason
                sp.set_output({"status": "waiting_for_human", "reason": reason})
            else:
                state = await self.finalize(state, ctx)
                sp.set_output({"status": "completed", "iterations": state.iteration, "score": state.critic_score})
        state.tokens_used = ctx.tracer.total_tokens
        return state

    async def _loop(self, state: ForecastState, ctx: AgentContext) -> ForecastState:
        while state.iteration < state.max_iterations:
            async with ctx.tracer.span(self.name, f"iteration_{state.iteration + 1}",
                                       {"feature_ops": state.feature_ops, "threshold": state.decision_threshold}) as sp:
                state = await self._agent("selector")(state, ctx)
                state = await self._agent("validator")(state, ctx)
                state = await self._agent("critic")(state, ctx)
                state.iteration += 1
                if self.converged(state, ctx):
                    sp.set_output({"converged": True, "score": state.critic_score})
                    break
                if state.iteration >= state.max_iterations:
                    sp.set_output({"converged": False, "reason": "max_iterations"})
                    break
                state = await self._agent("editor")(state, ctx)
                sp.set_output({"converged": False, "editor_changed": state.editor_changed})
                if not state.editor_changed:
                    break
        return state

    async def resume(self, state: ForecastState, ctx: AgentContext, resolution: dict[str, Any]) -> ForecastState:
        """Продолжение после решения человека (HITL)."""
        state.hitl_resolution = resolution
        state.hitl_requested = False
        action = resolution.get("action", "accept")
        async with ctx.tracer.span(self.name, "hitl_resume", resolution) as sp:
            if action in ("choose_model", "change_threshold", "change_horizon"):
                editor = self._agent("editor")
                from backend.core.llm.base import ToolCall

                if action == "choose_model":
                    state.forced_model = resolution["model_type"]
                    if resolution.get("params"):
                        state.model_type = resolution["model_type"]
                        state.model_params = resolution["params"]
                elif action == "change_threshold":
                    editor.apply(state, ToolCall(name="change_threshold",
                                                 arguments={"threshold": float(resolution["threshold"])}), ctx, "hitl")
                else:
                    state.horizon = int(resolution["horizon"])
                state.max_iterations = state.iteration + 1
                state = await self._agent("selector")(state, ctx)
                state = await self._agent("validator")(state, ctx)
                state = await self._agent("critic")(state, ctx)
                state.iteration += 1
            state = await self.finalize(state, ctx)
            sp.set_output({"action": action, "score": state.critic_score})
        state.tokens_used += ctx.tracer.total_tokens
        return state

    async def finalize(self, state: ForecastState, ctx: AgentContext) -> ForecastState:
        async with ctx.tracer.span(self.name, "finalize", {"threshold": state.decision_threshold}) as sp:
            if state.task_type == "timeseries":
                state.predictions_path = str(ctx.store.path(state.task_id, "forecast.parquet"))
            else:
                await asyncio.to_thread(self._finalize_tabular, state, ctx)
            state.summary_path = ctx.store.save_json(state.task_id, "summary.json", self.summary(state))
            sp.set_output({"predictions": state.predictions_path,
                           "hitl_students": len(state.metadata.get("hitl_students", []))})
        return state

    @staticmethod
    def summary(state: ForecastState) -> dict[str, Any]:
        return {
            "task_id": state.task_id, "problem_type": state.problem_type, "model_type": state.model_type,
            "model_params": state.model_params, "selection_mode": state.selection_mode,
            "decision_threshold": state.decision_threshold, "metrics": state.metrics,
            "feature_importance": state.metadata.get("feature_importance"), "feature_ops": state.feature_ops,
            "dropped_features": state.dropped_features, "critic_score": state.critic_score,
            "critic_reasoning": state.critic_reasoning, "iterations": state.iteration,
            "issues": [i.model_dump() for i in state.validation_issues],
            "risk_counts": state.metadata.get("risk_counts"), "rag_cases": state.rag_context,
            "editor_actions": [a.model_dump() for a in state.editor_actions],
        }

    def _finalize_tabular(self, state: ForecastState, ctx: AgentContext) -> None:
        s = ctx.settings
        data = load_working_data(state, ctx)
        pipe = ctx.store.load_model(state.model_path)
        oof = pd.read_parquet(state.oof_path)
        frames = [pd.DataFrame({"id": data.ids.astype(str).values, "split": "labeled",
                                "actual": data.y.values, "score": oof["oof"].values})]
        X_all = [data.X]
        pred_path = ctx.store.path(state.task_id, "predict_output.npy")
        if data.X_predict is not None and pred_path.exists():
            frames.append(pd.DataFrame({"id": data.predict_ids.astype(str).values, "split": "new",
                                        "actual": np.nan, "score": np.load(pred_path)}))
            X_all.append(data.X_predict)
        out = pd.concat(frames, ignore_index=True)
        X_all = pd.concat(X_all, ignore_index=True)

        if state.problem_type == "classification":
            out["at_risk"] = out["score"] >= state.decision_threshold
            out["risk_level"] = np.where(out["score"] >= s.high_risk_threshold, "high",
                                         np.where(out["at_risk"], "medium", "low"))
            order = out["score"].values
        else:
            pass_mark = state.metadata.get("pass_mark")
            if pass_mark is None:
                pass_mark = float(np.percentile(data.y, 25))
                state.metadata["pass_mark"] = pass_mark
            spread = float(np.std(data.y)) or 1.0
            out["at_risk"] = out["score"] < pass_mark
            out["risk_level"] = np.where(out["score"] < pass_mark - 0.5 * spread, "high",
                                         np.where(out["at_risk"], "medium", "low"))
            order = -out["score"].values

        raw = read_table(state.data_path)
        id_cols = state.metadata.get("id_columns") or []
        raw_ids = raw[id_cols[0]].astype(str) if id_cols else pd.Series([f"row_{i}" for i in range(len(raw))])
        raw_by_id = dict(zip(raw_ids, raw.to_dict("records"), strict=True))

        drivers_col, rec_col, sim_col = [""] * len(out), [""] * len(out), [""] * len(out)
        hitl: list[dict[str, Any]] = []
        risky = [i for i in np.argsort(-order) if out.at[i, "at_risk"]][:MAX_EXPLAINED]
        if not state.metadata.get("explain", True):
            risky = []
        for i in risky:
            row = X_all.iloc[[i]]
            drivers = occlusion_drivers(pipe, row, data.X, state.problem_type)
            raw_row = raw_by_id.get(out.at[i, "id"], {})
            profile = canonical_profile(raw_row)
            recs = recommendations_for(profile, True)
            similar = []
            if ctx.retriever is not None and ctx.retriever.size:
                docs = ctx.retriever.search(describe_student(profile, raw_row), kind="student_case",
                                            numeric=profile_vector(profile), k=3)
                similar = [{"id": d.id, "status": d.payload.get("status"), "outcome": d.payload.get("outcome")}
                           for d in docs]
                for d in docs:
                    for r in d.payload.get("recommendations", [])[:2]:
                        if r not in recs:
                            recs.append(r)
            recs = recs[:4]
            drivers_col[i] = "; ".join(f"{d['feature']}={d['value']} (+{d['effect']:.3f})" for d in drivers)
            rec_col[i] = " | ".join(recs)
            sim_col[i] = ", ".join(f"{d['id']}:{d['status']}" for d in similar)
            if out.at[i, "risk_level"] == "high" and len(hitl) < MAX_HITL_STUDENTS:
                hitl.append({"student_id": out.at[i, "id"], "score": round(float(out.at[i, "score"]), 4),
                             "split": out.at[i, "split"], "drivers": drivers, "recommendations": recs,
                             "similar_cases": similar, "profile": {k: v for k, v in profile.items() if v is not None},
                             "description": describe_student(profile, raw_row)})
        out["risk_drivers"], out["recommendations"], out["similar_cases"] = drivers_col, rec_col, sim_col
        score_name = "risk_probability" if state.problem_type == "classification" else "predicted_score"
        out = out.rename(columns={"score": score_name})
        path = ctx.store.path(state.task_id, "predictions.csv")
        out.to_csv(path, index=False, encoding="utf-8")
        state.predictions_path = str(path)
        state.metadata["risk_counts"] = {k: int(v) for k, v in out["risk_level"].value_counts().items()}
        state.metadata["hitl_students"] = hitl
