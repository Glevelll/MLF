"""Validator: формальные проверки прогнозов, утечки, переобучение, калибровка, остатки."""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backend.agents.base import Agent, AgentContext
from backend.agents.common import load_working_data
from backend.core.metrics import ljung_box, shapiro_wilk
from backend.core.models import ForecastState, Issue

BLOCKING = {"low_quality", "leakage", "invalid_predictions"}


def single_feature_leakage(X: pd.DataFrame, y: pd.Series, problem_type: str, numeric_cols: list[str],
                           threshold: float = 0.97) -> dict[str, float]:
    """Признаки, которые в одиночку почти идеально предсказывают цель — подозрение на утечку."""
    out = {}
    yv = y.values
    for c in numeric_cols:
        v = X[c].astype(float)
        if v.isna().all() or v.std() == 0:
            continue
        v = v.fillna(v.median()).values
        if problem_type == "classification":
            if len(np.unique(yv)) < 2:
                continue
            auc = roc_auc_score(yv, v)
            score = max(auc, 1 - auc)
        else:
            score = abs(np.corrcoef(v, yv.astype(float))[0, 1])
        if np.isfinite(score) and score >= threshold:
            out[c] = round(float(score), 4)
    return out


class Validator(Agent):
    name = "Validator"

    def summarize_output(self, state: ForecastState) -> dict[str, Any]:
        return {"passed": state.validation_passed,
                "issues": [f"{i.severity}:{i.code}" for i in state.validation_issues]}

    async def process(self, state: ForecastState, ctx: AgentContext) -> ForecastState:
        if state.task_type == "timeseries":
            issues = self._timeseries(state, ctx)
        else:
            issues = await asyncio.to_thread(self._tabular, state, ctx)
        state.validation_issues = issues
        state.validation_passed = not any(i.severity == "critical" or i.code in BLOCKING for i in issues)
        state.diagnostics = {**(state.diagnostics or {}), "validation": [i.model_dump() for i in issues]}
        return state

    def _tabular(self, state: ForecastState, ctx: AgentContext) -> list[Issue]:
        s = ctx.settings
        issues: list[Issue] = []
        m = state.metrics or {}
        cv, test, train = m.get("cv", {}), m.get("test", {}), m.get("train", {})
        data = load_working_data(state, ctx)
        oof = pd.read_parquet(state.oof_path)["oof"].values

        if len(oof) != len(data.X) or np.isnan(oof).any():
            issues.append(Issue(code="invalid_predictions", severity="critical",
                                message="форма прогнозов не совпадает с данными или есть NaN"))
        if state.problem_type == "classification" and (oof.min() < -1e-9 or oof.max() > 1 + 1e-9):
            issues.append(Issue(code="invalid_probabilities", severity="critical",
                                message="вероятности вне диапазона [0, 1]"))

        leaks = single_feature_leakage(data.X, data.y, state.problem_type, data.numeric_cols)
        if leaks:
            issues.append(Issue(code="leakage", severity="critical",
                                message=f"признаки почти идеально предсказывают цель: {list(leaks)}",
                                details={"features": leaks}))
        imp = state.metadata.get("feature_importance", {})
        pos = {k: v for k, v in imp.items() if v > 0}
        if pos:
            top, val = max(pos.items(), key=lambda kv: kv[1])
            share = val / sum(pos.values())
            if share > 0.6 and len(pos) > 2:
                issues.append(Issue(code="dominant_feature", severity="warning",
                                    message=f"прогноз определяется в основном признаком {top} ({share:.0%})",
                                    details={"feature": top, "share": round(share, 3)}))

        pm = m.get("primary", "roc_auc")
        cv_v, test_v, train_v = cv.get(pm), test.get(pm), train.get(pm)
        if state.problem_type == "classification":
            if cv_v is None or cv_v < s.min_auc:
                sev = "critical" if (cv_v or 0) < 0.6 else "warning"
                issues.append(Issue(code="low_quality", severity=sev, message=f"ROC-AUC на CV = {cv_v} < {s.min_auc}"))
            ece = cv.get("ece")
            if ece is not None and ece > 0.1:
                issues.append(Issue(code="poor_calibration", severity="warning",
                                    message=f"плохая калибровка вероятностей (ECE={ece:.3f})"))
            rec = cv.get("recall")
            if rec is not None and rec < 0.5:
                issues.append(Issue(code="low_recall", severity="warning",
                                    message=f"модель пропускает студентов из группы риска (recall={rec:.2f}); "
                                            "рассмотрите снижение порога"))
        else:
            if cv_v is None or cv_v < s.min_r2:
                sev = "critical" if (cv_v or 0) < 0.1 else "warning"
                issues.append(Issue(code="low_quality", severity=sev, message=f"R² на CV = {cv_v} < {s.min_r2}"))
            resid = data.y.values - oof
            sw = shapiro_wilk(resid)
            if sw["sw_pvalue"] is not None and sw["sw_pvalue"] < 0.01:
                issues.append(Issue(code="non_normal_residuals", severity="info",
                                    message="остатки не нормальны (Shapiro-Wilk p<0.01): возможны выбросы/нули",
                                    details=sw))
        if train_v is not None and test_v is not None and train_v - test_v > 0.15:
            issues.append(Issue(code="overfitting", severity="warning",
                                message=f"разрыв train/test по {pm}: {train_v:.3f} vs {test_v:.3f}"))
        if cv_v is not None and test_v is not None and abs(cv_v - test_v) > 0.12:
            issues.append(Issue(code="unstable_estimate", severity="info",
                                message=f"оценки CV и holdout расходятся ({cv_v:.3f} vs {test_v:.3f})"))
        return issues

    def _timeseries(self, state: ForecastState, ctx: AgentContext) -> list[Issue]:
        issues: list[Issue] = []
        m = state.metrics or {}
        if m.get("forecast_len") != max(1, state.horizon):
            issues.append(Issue(code="invalid_predictions", severity="critical", message="длина прогноза ≠ горизонту"))
        res = m.get("residuals", {})
        if res.get("lb_pvalue") is not None and res["lb_pvalue"] < 0.05:
            issues.append(Issue(code="residual_autocorrelation", severity="warning",
                                message=f"Ljung-Box p={res['lb_pvalue']:.3f}: остатки автокоррелированы"))
        resid_path = ctx.store.path(state.task_id, "residuals.npy")
        if resid_path.exists():
            lb = ljung_box(np.load(resid_path))
            state.diagnostics = {**(state.diagnostics or {}), "ljung_box": lb}
        return issues
