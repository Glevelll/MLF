"""Статистические тесты для сравнения моделей/режимов."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                 stat=np.mean) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)])
    return {"mean": float(stat(v)), "low": float(np.quantile(boots, alpha / 2)),
            "high": float(np.quantile(boots, 1 - alpha / 2)), "n": int(len(v))}


def paired_ttest(a, b) -> dict[str, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2 or np.allclose(a[m], b[m]):
        return {"t": 0.0, "pvalue": 1.0, "n": int(m.sum())}
    t, p = stats.ttest_rel(a[m], b[m])
    return {"t": float(t), "pvalue": float(p), "n": int(m.sum())}


def wilcoxon(a, b) -> dict[str, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if len(d) < 2 or np.allclose(d, 0):
        return {"w": 0.0, "pvalue": 1.0, "n": int(len(d))}
    w, p = stats.wilcoxon(a[m], b[m], zero_method="zsplit")
    return {"w": float(w), "pvalue": float(p), "n": int(len(d))}


def diebold_mariano(loss_a, loss_b, h: int = 1) -> dict[str, float]:
    """DM-тест: H0 — равная ожидаемая ошибка. d_t = L_a - L_b; HAC-дисперсия с h-1 лагами,
    поправка Harvey-Leybourne-Newbold для малых выборок. stat<0 => модель A точнее."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3 or np.allclose(d, 0):
        return {"dm": 0.0, "pvalue": 1.0, "n": n}
    mean = d.mean()
    gamma = [np.mean((d[k:] - mean) * (d[: n - k] - mean)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return {"dm": 0.0, "pvalue": 1.0, "n": n}
    dm = mean / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"dm": float(stat), "pvalue": float(p), "n": n}


def compare_paired(a, b, name_a: str, name_b: str, higher_is_better: bool = True) -> dict[str, Any]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b if higher_is_better else b - a
    return {"a": name_a, "b": name_b, "mean_a": float(np.nanmean(a)), "mean_b": float(np.nanmean(b)),
            "mean_improvement": float(np.nanmean(diff)), "ci_improvement": bootstrap_ci(diff),
            "wilcoxon": wilcoxon(a, b), "ttest": paired_ttest(a, b),
            "win_rate": float(np.nanmean(diff > 0)) if len(diff) else float("nan")}
