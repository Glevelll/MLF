"""Общие операции агентов: загрузка рабочего набора с учётом правок Editor-а, обучение."""
from __future__ import annotations

import pandas as pd

from backend.agents.base import AgentContext
from backend.core.models import ForecastState
from backend.tabular.features import apply_feature_ops
from backend.tabular.preprocessing import PreparedData, load_prepared


def load_working_data(state: ForecastState, ctx: AgentContext) -> PreparedData:
    """Базовые данные DataAnalyst + feature_ops − dropped_features (кэшируется в пределах запуска)."""
    key = f"data:{state.processed_path}:{','.join(state.feature_ops)}:{','.join(sorted(state.dropped_features))}"
    if key in ctx.cache:
        return ctx.cache[key]
    m = state.metadata
    data = load_prepared(state.processed_path, m["numeric_cols"], m["categorical_cols"], state.problem_type)
    if state.feature_ops:
        X, created = apply_feature_ops(data.X, state.feature_ops)
        data.X = X
        if data.X_predict is not None:
            data.X_predict, _ = apply_feature_ops(data.X_predict, state.feature_ops)
        data.numeric_cols = data.numeric_cols + [c for c in created if c not in data.numeric_cols]
    if state.dropped_features:
        drop = set(state.dropped_features)
        data.numeric_cols = [c for c in data.numeric_cols if c not in drop]
        data.categorical_cols = [c for c in data.categorical_cols if c not in drop]
    cols = data.feature_columns
    data.X = data.X[cols]
    if data.X_predict is not None:
        data.X_predict = data.X_predict[cols]
    ctx.cache[key] = data
    return data


def oof_frame(state: ForecastState, data: PreparedData, oof) -> pd.DataFrame:
    return pd.DataFrame({"id": data.ids.astype(str).values, "y": data.y.values, "oof": oof})
