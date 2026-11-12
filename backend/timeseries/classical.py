"""Классические модели рядов для реестра инструментов (ARIMA, ETS)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TSFitResult:
    forecast: np.ndarray
    residuals: np.ndarray
    fitted_params: dict


def fit_arima(y: pd.Series, horizon: int, order: tuple[int, int, int] = (1, 1, 1),
              seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)) -> TSFitResult:
    from statsmodels.tsa.arima.model import ARIMA

    res = ARIMA(y.values, order=order, seasonal_order=seasonal_order).fit()
    return TSFitResult(np.asarray(res.forecast(horizon)), np.asarray(res.resid),
                       {"order": list(order), "seasonal_order": list(seasonal_order), "aic": float(res.aic)})


def fit_ets(y: pd.Series, horizon: int, trend: str | None = "add", seasonal: str | None = None,
            seasonal_periods: int | None = None) -> TSFitResult:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    res = ExponentialSmoothing(y.values, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods).fit()
    return TSFitResult(np.asarray(res.forecast(horizon)), np.asarray(y.values - res.fittedvalues),
                       {"trend": trend, "seasonal": seasonal})
