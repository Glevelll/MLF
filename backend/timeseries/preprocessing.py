"""Временные ряды (точка расширения архитектуры): DatetimeIndex, ресемплинг, ADF, STL."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def to_series(df: pd.DataFrame, date_col: str | None = None, value_col: str | None = None) -> pd.Series:
    date_col = date_col or next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), df.columns[0])
    value_col = value_col or next(c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c]))
    s = pd.Series(df[value_col].astype(float).values, index=pd.to_datetime(df[date_col]), name=value_col)
    return s.sort_index()


def infer_freq(s: pd.Series) -> str | None:
    if len(s) < 3:
        return None
    f = pd.infer_freq(s.index)
    if f:
        return f
    step = pd.Series(s.index).diff().median()
    return pd.tseries.frequencies.to_offset(step).freqstr if pd.notna(step) else None


def is_regular(s: pd.Series) -> bool:
    d = pd.Series(s.index).diff().dropna()
    return bool(len(d) == 0 or d.nunique() == 1)


def resample(s: pd.Series, freq: str, how: str = "mean") -> pd.Series:
    r = s.resample(freq)
    out = r.sum() if how == "sum" else r.mean()
    return out.interpolate(limit_direction="both")


def adf_test(s: pd.Series) -> dict[str, Any]:
    from statsmodels.tsa.stattools import adfuller

    stat, p, *_ = adfuller(s.dropna().values, autolag="AIC")
    return {"adf_stat": float(stat), "adf_pvalue": float(p), "stationary": bool(p < 0.05)}


def stl_decompose(s: pd.Series, period: int | None = None) -> dict[str, float]:
    from statsmodels.tsa.seasonal import STL

    period = period or 7
    if len(s) < 2 * period + 1:
        return {"trend_strength": float("nan"), "seasonal_strength": float("nan"), "period": period}
    res = STL(s.values, period=period, robust=True).fit()
    var_r = np.var(res.resid)
    trend = max(0.0, 1 - var_r / (np.var(res.trend + res.resid) + 1e-12))
    seas = max(0.0, 1 - var_r / (np.var(res.seasonal + res.resid) + 1e-12))
    return {"trend_strength": float(trend), "seasonal_strength": float(seas), "period": period}
