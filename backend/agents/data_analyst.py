"""DataAnalyst: подготовка данных и EDA (табличные данные; ветка для временных рядов)."""
from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from backend.agents.base import Agent, AgentContext
from backend.core.models import ForecastState, Issue
from backend.data.embeddings import get_extractor
from backend.tabular.datasets import read_table
from backend.tabular.preprocessing import build_preprocessor, eda_summary, prepare_dataframe, save_prepared


class DataAnalyst(Agent):
    name = "DataAnalyst"

    def summarize_input(self, state: ForecastState) -> dict[str, Any]:
        return {"data_path": state.data_path, "target": state.target_column, "task_type": state.task_type}

    def summarize_output(self, state: ForecastState) -> dict[str, Any]:
        m = state.metadata
        return {"problem_type": state.problem_type, "n_rows": m.get("n_rows"), "n_features": m.get("n_features"),
                "n_predict": m.get("n_predict"), "issues": [i.code for i in state.validation_issues]}

    async def process(self, state: ForecastState, ctx: AgentContext) -> ForecastState:
        if not state.data_path:
            raise ValueError("data_path не задан")
        if state.task_type == "timeseries":
            return await asyncio.to_thread(self._timeseries, state, ctx)
        return await asyncio.to_thread(self._tabular, state, ctx)

    def _tabular(self, state: ForecastState, ctx: AgentContext) -> ForecastState:
        if not state.target_column:
            raise ValueError("для табличной задачи нужен target_column")
        df = read_table(state.data_path)
        data = prepare_dataframe(df, state.target_column, state.problem_type, state.binarize_threshold,
                                 state.positive_label, state.id_column)
        eda = eda_summary(data.X, data.y, data.problem_type, data.numeric_cols, data.categorical_cols)
        # демонстрация импутации/OHE/масштабирования: матрица сохраняется для EDA;
        # при обучении тот же препроцессор подгоняется внутри фолдов CV (без утечек)
        prep = build_preprocessor(data.numeric_cols, data.categorical_cols)
        mat = prep.fit_transform(data.X)
        processed = pd.DataFrame(mat, columns=list(prep.get_feature_names_out()))
        processed_path = ctx.store.path(state.task_id, "processed_matrix.parquet")
        processed.to_parquet(processed_path, index=False)
        prepared_path = ctx.store.path(state.task_id, "prepared.parquet")
        save_prepared(data, prepared_path)

        meta = get_extractor("tabular").extract(X=data.X, y=data.y, problem_type=data.problem_type,
                                                numeric_cols=data.numeric_cols, categorical_cols=data.categorical_cols)
        state.problem_type = data.problem_type  # type: ignore[assignment]
        state.processed_path = str(prepared_path)
        state.feature_columns = data.feature_columns
        state.metadata.update({
            "numeric_cols": data.numeric_cols, "categorical_cols": data.categorical_cols,
            "n_rows": int(len(data.X)), "n_features": int(data.X.shape[1]), "n_features_encoded": int(mat.shape[1]),
            "n_predict": int(len(data.X_predict)) if data.X_predict is not None else 0,
            "target_encoding": data.info["target_encoding"], "id_columns": data.info["id_columns"],
            "dropped_constant": data.info["dropped_constant"], "processed_matrix_path": str(processed_path),
            "meta_features": meta,
        })
        state.diagnostics = {**(state.diagnostics or {}), "eda": eda}
        issues = []
        if len(data.X) < 60:
            issues.append(Issue(code="small_sample", severity="warning",
                                message=f"мало размеченных строк ({len(data.X)}) — оценки метрик нестабильны"))
        if eda.get("imbalanced"):
            issues.append(Issue(code="class_imbalance", severity="info",
                                message=f"дисбаланс классов: доля миноритарного {eda['minority_ratio']:.2f}"))
        if eda["missing_rate"] > 0.3:
            issues.append(Issue(code="high_missing", severity="warning", message="более 30% пропусков"))
        state.validation_issues = issues
        return state

    def _timeseries(self, state: ForecastState, ctx: AgentContext) -> ForecastState:
        from backend.timeseries.preprocessing import (
            adf_test,
            infer_freq,
            is_regular,
            resample,
            stl_decompose,
            to_series,
        )

        df = read_table(state.data_path)
        s = to_series(df, value_col=state.target_column)
        freq = state.freq or infer_freq(s)
        regular = is_regular(s)
        if not regular and freq:
            s = resample(s, freq)
        path = ctx.store.path(state.task_id, "series.parquet")
        s.to_frame("y").to_parquet(path)
        state.freq = freq
        state.processed_path = str(path)
        state.problem_type = "regression"
        diag = {"regular": regular, "freq": freq, **adf_test(s), **stl_decompose(s)}
        state.diagnostics = {**(state.diagnostics or {}), "ts": diag}
        state.metadata.update({"n_rows": len(s), "meta_features": get_extractor("timeseries").extract(series=s)})
        return state
