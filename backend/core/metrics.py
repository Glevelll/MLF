"""Метрики качества: регрессия, классификация, калибровка, диагностика остатков."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats
from sklearn import metrics as skm


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


# ---------------------------------------------------------------- регрессия / временные ряды
def mae(y, p) -> float:
    return float(np.mean(np.abs(_arr(y) - _arr(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((_arr(y) - _arr(p)) ** 2)))


def mape(y, p, eps: float = 1e-8) -> float:
    y, p = _arr(y), _arr(p)
    mask = np.abs(y) > eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)


def wape(y, p) -> float:
    y, p = _arr(y), _arr(p)
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - p)) / denom * 100) if denom > 0 else float("nan")


def mase(y, p, y_train, season: int = 1) -> float:
    y_train = _arr(y_train)
    if len(y_train) <= season:
        return float("nan")
    scale = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    return float(mae(y, p) / scale) if scale > 0 else float("nan")


def r2(y, p) -> float:
    return float(skm.r2_score(_arr(y), _arr(p)))


def regression_metrics(y, p) -> dict[str, float]:
    return {"mae": mae(y, p), "rmse": rmse(y, p), "r2": r2(y, p), "mape": mape(y, p), "wape": wape(y, p)}


# ---------------------------------------------------------------- классификация
def expected_calibration_error(y, proba, n_bins: int = 10) -> float:
    y, proba = _arr(y), _arr(proba)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(proba[m].mean() - y[m].mean())
    return float(ece)


def classification_metrics(y, proba, threshold: float = 0.5) -> dict[str, float]:
    y, proba = _arr(y).astype(int), _arr(proba)
    pred = (proba >= threshold).astype(int)
    out: dict[str, float] = {
        "accuracy": float(skm.accuracy_score(y, pred)),
        "f1": float(skm.f1_score(y, pred, zero_division=0)),
        "precision": float(skm.precision_score(y, pred, zero_division=0)),
        "recall": float(skm.recall_score(y, pred, zero_division=0)),
        "brier": float(skm.brier_score_loss(y, np.clip(proba, 0, 1))),
        "ece": expected_calibration_error(y, proba),
    }
    out["roc_auc"] = float(skm.roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan")
    return out


def best_f1_threshold(y, proba) -> tuple[float, float]:
    """Порог, максимизирующий F1 (по out-of-fold вероятностям)."""
    y, proba = _arr(y).astype(int), _arr(proba)
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = skm.f1_score(y, (proba >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), float(f)
    return round(best_t, 2), best_f


def primary_metric(problem_type: str) -> str:
    return "roc_auc" if problem_type == "classification" else "r2"


def compute_metrics(problem_type: str, y, pred, threshold: float = 0.5) -> dict[str, float]:
    if problem_type == "classification":
        return classification_metrics(y, pred, threshold)
    return regression_metrics(y, pred)


# ---------------------------------------------------------------- диагностика остатков
def ljung_box(residuals, lags: int = 10) -> dict[str, float]:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    r = _arr(residuals)
    lags = max(1, min(lags, len(r) // 2 - 1))
    res = acorr_ljungbox(r, lags=[lags], return_df=True)
    return {"lb_stat": float(res["lb_stat"].iloc[0]), "lb_pvalue": float(res["lb_pvalue"].iloc[0]), "lags": lags}


def shapiro_wilk(residuals) -> dict[str, float]:
    r = _arr(residuals)
    if len(r) > 5000:
        r = np.random.default_rng(0).choice(r, 5000, replace=False)
    if len(r) < 3:
        return {"sw_stat": float("nan"), "sw_pvalue": float("nan")}
    stat, p = stats.shapiro(r)
    return {"sw_stat": float(stat), "sw_pvalue": float(p)}


def clean(d: dict[str, Any]) -> dict[str, Any]:
    """Заменяет NaN/inf на None для JSON-сериализации."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = clean(v)
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        elif isinstance(v, np.floating):
            fv = float(v)
            out[k] = None if math.isnan(fv) or math.isinf(fv) else fv
        elif isinstance(v, np.integer):
            out[k] = int(v)
        else:
            out[k] = v
    return out
