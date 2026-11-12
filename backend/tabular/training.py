"""Обучение и оценка: CV без утечек (препроцессинг внутри Pipeline), holdout, OOF, важность признаков."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from backend.core.metrics import best_f1_threshold, clean, compute_metrics
from backend.tabular.preprocessing import PreparedData, build_preprocessor


@dataclass
class FitResult:
    pipeline: Pipeline
    problem_type: str
    cv_metrics: dict[str, float]
    test_metrics: dict[str, float]
    train_metrics: dict[str, float]
    oof: np.ndarray
    test_index: np.ndarray
    test_pred: np.ndarray
    feature_importance: dict[str, float] = field(default_factory=dict)
    predict_output: np.ndarray | None = None
    best_threshold: float | None = None
    fit_seconds: float = 0.0

    def metrics_dict(self) -> dict[str, Any]:
        return clean({"cv": self.cv_metrics, "test": self.test_metrics, "train": self.train_metrics,
                      "best_f1_threshold": self.best_threshold})


def make_pipeline(estimator: BaseEstimator, data: PreparedData, scale: bool = True) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(data.numeric_cols, data.categorical_cols, scale=scale)),
                     ("model", estimator)])


def _predict(pipe: Pipeline, X: pd.DataFrame, problem_type: str) -> np.ndarray:
    if problem_type == "classification":
        return pipe.predict_proba(X)[:, 1]
    return pipe.predict(X)


def _splitter(problem_type: str, folds: int, seed: int):
    if problem_type == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def fit_and_evaluate(estimator: BaseEstimator, data: PreparedData, *, threshold: float = 0.5, cv_folds: int = 5,
                     test_size: float = 0.2, seed: int = 42, scale: bool = True,
                     compute_importance: bool = True, refit_full: bool = True) -> FitResult:
    import time

    t0 = time.perf_counter()
    X, y, ptype = data.X, data.y, data.problem_type
    if ptype == "classification":
        min_class = int(y.value_counts().min())
        cv_folds = max(2, min(cv_folds, min_class))
    pipe = make_pipeline(estimator, data, scale)
    method = "predict_proba" if ptype == "classification" else "predict"
    oof = cross_val_predict(clone(pipe), X, y, cv=_splitter(ptype, cv_folds, seed), method=method)
    if ptype == "classification":
        oof = oof[:, 1]
    best_t = best_f1_threshold(y, oof)[0] if ptype == "classification" else None
    cv_metrics = compute_metrics(ptype, y, oof, threshold)

    idx = np.arange(len(X))
    strat = y if ptype == "classification" else None
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)
    holdout = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
    test_pred = _predict(holdout, X.iloc[te], ptype)
    test_metrics = compute_metrics(ptype, y.iloc[te], test_pred, threshold)
    train_metrics = compute_metrics(ptype, y.iloc[tr], _predict(holdout, X.iloc[tr], ptype), threshold)

    importance: dict[str, float] = {}
    if compute_importance:
        scoring = "roc_auc" if ptype == "classification" else "r2"
        pi = permutation_importance(holdout, X.iloc[te], y.iloc[te], n_repeats=5, random_state=seed,
                                    scoring=scoring, n_jobs=1)
        importance = {c: round(float(v), 5) for c, v in zip(X.columns, pi.importances_mean, strict=True)}
        importance = dict(sorted(importance.items(), key=lambda kv: -kv[1]))

    final = clone(pipe).fit(X, y) if refit_full else holdout
    pred_out = None
    if data.X_predict is not None and len(data.X_predict):
        pred_out = _predict(final, data.X_predict, ptype)
    return FitResult(pipeline=final, problem_type=ptype, cv_metrics=cv_metrics, test_metrics=test_metrics,
                     train_metrics=train_metrics, oof=np.asarray(oof), test_index=te, test_pred=test_pred,
                     feature_importance=importance, predict_output=pred_out, best_threshold=best_t,
                     fit_seconds=time.perf_counter() - t0)


def occlusion_drivers(pipe: Pipeline, X_row: pd.DataFrame, reference: pd.DataFrame, problem_type: str,
                      top_k: int = 3) -> list[dict[str, Any]]:
    """Локальное объяснение: как меняется прогноз, если признак заменить на «типичное» значение.

    Положительный effect — признак увеличивает риск (или снижает оценку для регрессии).
    """
    base = float(_predict(pipe, X_row, problem_type)[0])
    rows = []
    typical = {}
    for c in X_row.columns:
        col = reference[c]
        typical[c] = col.median() if pd.api.types.is_numeric_dtype(col) else (col.mode().iloc[0] if col.notna().any() else None)
    batch = pd.concat([X_row] * len(X_row.columns), ignore_index=True)
    for i, c in enumerate(X_row.columns):
        batch.at[i, c] = typical[c]
    preds = _predict(pipe, batch, problem_type)
    for i, c in enumerate(X_row.columns):
        delta = base - float(preds[i])
        effect = delta if problem_type == "classification" else -delta
        rows.append({"feature": c, "value": X_row.iloc[0][c], "typical": typical[c], "effect": round(effect, 4)})
    rows.sort(key=lambda r: -r["effect"])
    out = [r for r in rows if r["effect"] > 0][:top_k]
    for r in out:
        for k in ("value", "typical"):
            v = r[k]
            r[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else (float(v) if isinstance(v, (int, float, np.number)) else str(v))
    return out
