"""Автоматический feature engineering (инструмент Editor-а, бонус задания).

Все операции детерминированы и НЕ используют целевую переменную — это исключает утечку
при последующей кросс-валидации.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

SCORE_PAT = re.compile(r"^(g\d+|test\d*_?score|test\d+|quiz\d+|score_?\d+|exam\d+|midterm\d*|assessment\d+)$", re.I)
ACTIVITY_PAT = re.compile(r"(attend|absen|hw_|homework|lms|forum|login|click|late|submission|visit)", re.I)

AVAILABLE_OPS = {
    "score_dynamics": "динамика промежуточных оценок: разница последний-первый, средняя, наклон тренда",
    "activity_aggregates": "агрегаты активности: z-среднее посещаемости/ДЗ/LMS/форума",
    "ratios": "отношения: оценка на час LMS, доля поздних сдач",
    "interactions": "попарные произведения 3 наиболее вариативных числовых признаков",
    "polynomial": "квадраты 3 наиболее вариативных числовых признаков",
    "log_transform": "log1p для неотрицательных признаков с сильной асимметрией",
}


def score_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if SCORE_PAT.match(c) and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(cols, key=lambda c: [int(x) for x in re.findall(r"\d+", c)] or [0])


def activity_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if ACTIVITY_PAT.search(c) and pd.api.types.is_numeric_dtype(df[c])
            and not c.startswith("fe_")]


def _top_var_numeric(df: pd.DataFrame, k: int = 3) -> list[str]:
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith("fe_")]
    cv = {c: float(df[c].std() / (abs(df[c].mean()) + 1e-6)) for c in num if df[c].std() > 0}
    return [c for c, _ in sorted(cv.items(), key=lambda kv: -kv[1])[:k]]


def apply_feature_ops(df: pd.DataFrame, ops: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Возвращает новый DataFrame и список созданных признаков."""
    df = df.copy()
    created: list[str] = []
    for op in ops:
        if op == "score_dynamics":
            sc = score_columns(df)
            if len(sc) >= 2:
                vals = df[sc].astype(float)
                df["fe_score_delta"] = vals[sc[-1]] - vals[sc[0]]
                df["fe_score_mean"] = vals.mean(axis=1)
                x = np.arange(len(sc)) - (len(sc) - 1) / 2
                df["fe_score_slope"] = (vals.fillna(vals.mean()) * x).sum(axis=1) / (x**2).sum()
                created += ["fe_score_delta", "fe_score_mean", "fe_score_slope"]
        elif op == "activity_aggregates":
            ac = activity_columns(df)
            if ac:
                z = df[ac].astype(float)
                z = (z - z.mean()) / (z.std() + 1e-9)
                sign = np.array([-1.0 if re.search(r"(absen|late)", c, re.I) else 1.0 for c in ac])
                df["fe_engagement"] = (z * sign).mean(axis=1)
                created.append("fe_engagement")
                if len(ac) >= 2:
                    df["fe_activity_std"] = z.std(axis=1)
                    created.append("fe_activity_std")
        elif op == "ratios":
            sc = score_columns(df)
            lms = [c for c in df.columns if re.search(r"lms|hours|studytime", c, re.I)
                   and pd.api.types.is_numeric_dtype(df[c])]
            if sc and lms:
                df["fe_score_per_effort"] = df[sc].astype(float).mean(axis=1) / (df[lms[0]].astype(float) + 1)
                created.append("fe_score_per_effort")
            late = [c for c in df.columns if re.search(r"late|absen", c, re.I) and pd.api.types.is_numeric_dtype(df[c])]
            if late:
                df["fe_late_log"] = np.log1p(df[late[0]].clip(lower=0).astype(float))
                created.append("fe_late_log")
        elif op == "interactions":
            top = _top_var_numeric(df)
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    name = f"fe_{top[i]}_x_{top[j]}"
                    df[name] = df[top[i]].astype(float) * df[top[j]].astype(float)
                    created.append(name)
        elif op == "polynomial":
            for c in _top_var_numeric(df):
                df[f"fe_{c}_sq"] = df[c].astype(float) ** 2
                created.append(f"fe_{c}_sq")
        elif op == "log_transform":
            for c in [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith("fe_")]:
                s = df[c].astype(float)
                if s.min() >= 0 and s.skew() > 1.5:
                    df[f"fe_log_{c}"] = np.log1p(s)
                    created.append(f"fe_log_{c}")
        else:
            raise ValueError(f"неизвестная операция feature engineering: {op}")
    return df, list(dict.fromkeys(created))
