"""ModelSelector: RAG-поиск похожих кейсов + выбор модели через Tool Calling (без генерации кода)."""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from backend.agents.base import Agent, AgentContext, context_block
from backend.agents.common import load_working_data, oof_frame
from backend.core.llm.base import Message
from backend.core.llm.tools import DEFAULT_MODEL, Tool
from backend.core.metrics import primary_metric
from backend.core.models import ForecastState
from backend.data.embeddings import get_extractor
from backend.data.knowledge_base import describe_dataset_case
from backend.tabular.training import fit_and_evaluate

SYSTEM_PROMPT = """[ROLE:model_selector]
Ты — ML-инженер, выбирающий модель для прогноза успеваемости студентов.
Правила:
- Выбери РОВНО ОДИН инструмент из переданного списка и задай гиперпараметры в его JSON-схеме.
- Никогда не пиши код. Отвечай только вызовом инструмента.
- Учитывай: размер выборки, долю категориальных признаков, пропуски, дисбаланс классов,
  а также модели, которые лучше всего сработали на похожих кейсах из базы знаний (rag_cases).
- Для малых выборок (<400 строк) предпочитай интерпретируемые линейные модели;
  при выраженных нелинейностях и большом объёме — ансамбли деревьев.
Пример: {"name": "fit_random_forest", "arguments": {"n_estimators": 300, "min_samples_leaf": 3}}"""


class ModelSelector(Agent):
    name = "ModelSelector"

    def summarize_input(self, state: ForecastState) -> dict[str, Any]:
        return {"iteration": state.iteration, "forced_model": state.forced_model, "feature_ops": state.feature_ops,
                "dropped": state.dropped_features}

    def summarize_output(self, state: ForecastState) -> dict[str, Any]:
        m = state.metrics or {}
        return {"model_type": state.model_type, "params": state.model_params, "mode": state.selection_mode,
                "cv": m.get("cv"), "rag_cases": len(state.rag_context)}

    async def retrieve(self, state: ForecastState, ctx: AgentContext) -> list[dict[str, Any]]:
        if ctx.retriever is None or ctx.retriever.size == 0:
            return []
        meta = state.metadata.get("meta_features", {})
        query = describe_dataset_case(meta, "", {}, "запрос пользователя")
        numeric = get_extractor(state.task_type).vector(meta) if state.task_type == "tabular" else None
        async with ctx.tracer.span(self.name, "rag_search", {"query": query}) as sp:
            docs = ctx.retriever.search(query, kind="dataset_case", numeric=numeric, k=ctx.settings.rag_top_k,
                                        filters={"problem_type": state.problem_type, "task_type": state.task_type})
            out = [{"id": d.id, "best_model": d.payload.get("best_model"), "hyperparams": d.payload.get("hyperparams"),
                    "metrics": d.payload.get("metrics"), "num_rows": d.payload.get("num_rows"),
                    "rrf": round(d.score, 5), "ranks": d.ranks} for d in docs]
            sp.set_output([(o["id"], o["best_model"]) for o in out])
        return out

    def build_messages(self, state: ForecastState, tools: list[Tool], rag: list[dict[str, Any]]) -> list[Message]:
        eda = (state.diagnostics or {}).get("eda", {})
        ctx = {
            "problem_type": state.problem_type,
            "meta": {k: state.metadata.get("meta_features", {}).get(k) for k in
                     ("n_rows", "n_features", "n_categorical", "missing_rate", "minority_ratio", "max_abs_corr",
                      "mean_abs_corr", "has_score_columns")},
            "class_balance": eda.get("class_balance"),
            "top_correlations": dict(list(eda.get("target_correlations", {}).items())[:5]),
            "rag_cases": [{k: c[k] for k in ("best_model", "hyperparams", "metrics", "num_rows")} for c in rag],
            "available_tools": [t.name for t in tools],
        }
        user = ("Выбери модель для набора данных. Описание набора и похожие кейсы:\n" + context_block(ctx))
        return [Message(role="system", content=SYSTEM_PROMPT), Message(role="user", content=user)]

    async def choose(self, state: ForecastState, ctx: AgentContext, tools: list[Tool]) -> tuple[str, dict, str]:
        names = {t.name for t in tools}
        if state.forced_model:
            if state.forced_model not in names:
                raise ValueError(f"модель {state.forced_model} недоступна для задачи {state.problem_type}")
            params = state.model_params if state.model_type == state.forced_model else {}
            return state.forced_model, params or {}, "forced"
        if state.model_type and state.iteration > 0 and state.model_type in names:
            return state.model_type, state.model_params or {}, "carried"
        rag = await self.retrieve(state, ctx)
        state.rag_context = rag
        res = await ctx.llm.call_tool(self.build_messages(state, tools, rag), tools, agent_name=self.name,
                                      max_attempts=3)
        if res.call is not None:
            return res.call.name, res.call.arguments, res.mode
        # после 3 неудачных попыток — правило по умолчанию
        ptype = "forecast" if state.task_type == "timeseries" else state.problem_type
        return DEFAULT_MODEL[ptype], {}, "default"

    async def process(self, state: ForecastState, ctx: AgentContext) -> ForecastState:
        ptype = "forecast" if state.task_type == "timeseries" else state.problem_type
        tools = ctx.registry.list(kind="model", task_type=state.task_type, problem_type=ptype)
        name, args, mode = await self.choose(state, ctx, tools)
        tool = ctx.registry.get(name)
        try:
            args = tool.validate_args(args)
        except Exception:
            args = tool.validate_args({})
        async with ctx.tracer.span(self.name, "tool_call", {"tool": name, "mode": mode}, tool_name=name,
                                   parameters=args) as sp:
            if state.task_type == "timeseries":
                metrics = await asyncio.to_thread(self._run_ts, state, ctx, name, args)
            else:
                metrics = await asyncio.to_thread(self._run_tabular, state, ctx, tool, args)
            sp.set_output(metrics.get("cv"))
        state.model_type, state.model_params, state.selection_mode = name, args, mode
        state.forced_model = None
        state.metrics = metrics
        return state

    def _run_tabular(self, state: ForecastState, ctx: AgentContext, tool: Tool, args: dict) -> dict[str, Any]:
        data = load_working_data(state, ctx)
        est = tool.builder(args, state.problem_type, ctx.settings.random_seed)
        fit = fit_and_evaluate(est, data, threshold=state.decision_threshold, seed=ctx.settings.random_seed,
                               scale=tool.scale)
        state.model_path = ctx.store.save_model(state.task_id, fit.pipeline)
        oof_path = ctx.store.path(state.task_id, "oof.parquet")
        oof_frame(state, data, fit.oof).to_parquet(oof_path, index=False)
        state.oof_path = str(oof_path)
        if fit.predict_output is not None:
            np.save(ctx.store.path(state.task_id, "predict_output.npy"), fit.predict_output)
        state.metadata["feature_importance"] = dict(list(fit.feature_importance.items())[:20])
        state.metadata["fit_seconds"] = round(fit.fit_seconds, 3)
        state.metadata["working_features"] = data.feature_columns
        metrics = fit.metrics_dict()
        pm = primary_metric(state.problem_type)
        metrics["primary"] = pm
        metrics["primary_value"] = metrics["cv"].get(pm)
        return metrics

    def _run_ts(self, state: ForecastState, ctx: AgentContext, name: str, args: dict) -> dict[str, Any]:
        import pandas as pd

        from backend.core.metrics import ljung_box, mase, regression_metrics, shapiro_wilk
        from backend.timeseries.classical import fit_arima, fit_ets

        y = pd.read_parquet(state.processed_path)["y"]
        h = max(1, state.horizon or 1)
        train, test = y.iloc[:-h], y.iloc[-h:]
        fn = fit_arima if name == "fit_arima" else fit_ets
        back = fn(train, h, **args)
        final = fn(y, h, **args)
        pd.DataFrame({"forecast": final.forecast}).to_parquet(ctx.store.path(state.task_id, "forecast.parquet"))
        np.save(ctx.store.path(state.task_id, "residuals.npy"), final.residuals)
        test_m = regression_metrics(test.values, back.forecast)
        test_m["mase"] = mase(test.values, back.forecast, train.values)
        return {"cv": test_m, "test": test_m, "train": {}, "residuals": {**ljung_box(final.residuals),
                                                                     **shapiro_wilk(final.residuals)},
                "primary": "mae", "primary_value": test_m["mae"], "forecast_len": len(final.forecast)}
