"""Предобработка табличных данных: типы, цель, импутация, OHE, масштабирование, EDA."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "__target__"
ROW_ID = "__id__"


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    problem_type: str
    numeric_cols: list[str]
    categorical_cols: list[str]
    ids: pd.Series
    X_predict: pd.DataFrame | None = None
    predict_ids: pd.Series | None = None
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_cols + self.categorical_cols


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Строки с числами -> числа, булевы -> категории ('yes'/'no')."""
    df = df.copy()
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            df[c] = s.map({True: "yes", False: "no"})
        elif not pd.api.types.is_numeric_dtype(s):
            conv = pd.to_numeric(s, errors="coerce")
            if conv.notna().sum() >= 0.95 * s.notna().sum() and s.notna().any():
                df[c] = conv
    return df


def detect_id_columns(df: pd.DataFrame, explicit: str | None = None) -> list[str]:
    ids = [explicit] if explicit and explicit in df.columns else []
    for c in df.columns:
        if c in ids:
            continue
        lc = c.lower()
        unique_ratio = df[c].nunique(dropna=True) / max(len(df), 1)
        if (lc in ("id", "student_id", "uid") or lc.endswith("_id")) and unique_ratio > 0.9:
            ids.append(c)
        elif not pd.api.types.is_numeric_dtype(df[c]) and unique_ratio > 0.98 and len(df) > 50:
            ids.append(c)
    return ids


def infer_column_types(df: pd.DataFrame, max_cat_numeric: int = 0) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > max_cat_numeric:
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


def prepare_target(s: pd.Series, problem_type: str = "auto", binarize_threshold: float | None = None,
                   positive_label: str | None = None) -> tuple[pd.Series, str, dict[str, Any]]:
    """Возвращает y (для классификации 1 = «в группе риска»), тип задачи и описание кодировки."""
    info: dict[str, Any] = {}
    numeric = pd.api.types.is_numeric_dtype(s)
    if problem_type == "auto":
        if binarize_threshold is not None:
            problem_type = "classification"
        elif numeric and s.nunique(dropna=True) > 15:
            problem_type = "regression"
        else:
            problem_type = "classification"
    if problem_type == "regression":
        if not numeric:
            raise ValueError("для регрессии целевая переменная должна быть числовой")
        return s.astype(float), problem_type, {"encoding": "identity"}
    if binarize_threshold is not None:
        if not numeric:
            raise ValueError("binarize_threshold требует числовую целевую переменную")
        y = (s < binarize_threshold).astype(float).where(s.notna())
        info = {"encoding": "threshold", "threshold": binarize_threshold, "positive": f"< {binarize_threshold}"}
        return y, "classification", info
    values = s.dropna().unique().tolist()
    if len(values) != 2:
        raise ValueError(f"бинарная классификация требует 2 класса, найдено {len(values)}; "
                         "укажите binarize_threshold")
    if positive_label is None:
        counts = s.value_counts()
        pos = counts.idxmin()
        if set(map(str, values)) == {"0", "1"} or set(values) == {0, 1}:
            pos = 1 if 1 in values else "1"
    else:
        pos = next((v for v in values if str(v) == str(positive_label)), None)
        if pos is None:
            raise ValueError(f"positive_label={positive_label} не найден среди {values}")
    y = (s == pos).astype(float).where(s.notna())
    return y, "classification", {"encoding": "label", "positive": str(pos)}


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str], scale: bool = True) -> ColumnTransformer:
    num_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5))])
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(num_steps), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat, categorical_cols))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def to_object_categoricals(X: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


def eda_summary(X: pd.DataFrame, y: pd.Series, problem_type: str, numeric_cols: list[str],
                categorical_cols: list[str]) -> dict[str, Any]:
    missing = X.isna().mean()
    out: dict[str, Any] = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "missing_rate": float(missing.mean()) if len(missing) else 0.0,
        "missing_by_column": {k: round(float(v), 4) for k, v in missing.items() if v > 0},
        "constant_columns": [c for c in X.columns if X[c].nunique(dropna=True) <= 1],
        "cardinality": {c: int(X[c].nunique()) for c in categorical_cols},
    }
    if problem_type == "classification":
        vc = y.value_counts(normalize=True)
        out["class_balance"] = {str(int(k)): round(float(v), 4) for k, v in vc.items()}
        out["minority_ratio"] = float(vc.min()) if len(vc) else 0.0
        out["imbalanced"] = bool(out["minority_ratio"] < 0.2)
    else:
        out["target_stats"] = {"mean": float(y.mean()), "std": float(y.std()), "min": float(y.min()),
                               "max": float(y.max()), "skew": float(y.skew())}
    corr = {}
    for c in numeric_cols:
        v = X[c].astype(float)
        if v.std() > 0:
            r = np.corrcoef(v.fillna(v.median()), y.astype(float))[0, 1]
            if np.isfinite(r):
                corr[c] = round(float(r), 4)
    out["target_correlations"] = dict(sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:15])
    return out


def save_prepared(data: PreparedData, path: str | Path) -> None:
    frames = [data.X.assign(**{TARGET: data.y.values, ROW_ID: data.ids.values})]
    if data.X_predict is not None and len(data.X_predict):
        frames.append(data.X_predict.assign(**{TARGET: np.nan, ROW_ID: data.predict_ids.values}))
    df = pd.concat(frames, ignore_index=True)
    for c in data.categorical_cols:
        df[c] = df[c].astype("string")
    df[ROW_ID] = df[ROW_ID].astype(str)
    df.to_parquet(path, index=False)


def load_prepared(path: str | Path, numeric_cols: list[str], categorical_cols: list[str],
                  problem_type: str) -> PreparedData:
    df = pd.read_parquet(path)
    cols = numeric_cols + categorical_cols
    lab = df[df[TARGET].notna()].reset_index(drop=True)
    unl = df[df[TARGET].isna()].reset_index(drop=True)
    X = to_object_categoricals(lab[cols], categorical_cols)
    y = lab[TARGET].astype(int if problem_type == "classification" else float)
    Xp = to_object_categoricals(unl[cols], categorical_cols) if len(unl) else None
    return PreparedData(X=X, y=y, problem_type=problem_type, numeric_cols=list(numeric_cols),
                        categorical_cols=list(categorical_cols), ids=lab[ROW_ID],
                        X_predict=Xp, predict_ids=unl[ROW_ID] if len(unl) else None)


def prepare_dataframe(df: pd.DataFrame, target: str, problem_type: str = "auto",
                      binarize_threshold: float | None = None, positive_label: str | None = None,
                      id_column: str | None = None, drop_columns: list[str] | None = None) -> PreparedData:
    if target not in df.columns:
        raise ValueError(f"целевая колонка '{target}' отсутствует; есть: {list(df.columns)[:30]}")
    df = coerce_types(df)
    id_cols = detect_id_columns(df.drop(columns=[target]), id_column)
    ids = df[id_cols[0]].astype(str) if id_cols else pd.Series([f"row_{i}" for i in range(len(df))])
    y_all, ptype, enc = prepare_target(df[target], problem_type, binarize_threshold, positive_label)
    X_all = df.drop(columns=[target, *id_cols, *(drop_columns or [])], errors="ignore")
    const = [c for c in X_all.columns if X_all[c].nunique(dropna=True) <= 1]
    X_all = X_all.drop(columns=const)
    numeric, categorical = infer_column_types(X_all)
    X_all = to_object_categoricals(X_all, categorical)
    lab = y_all.notna().values
    data = PreparedData(
        X=X_all[lab].reset_index(drop=True),
        y=y_all[lab].reset_index(drop=True).astype(int if ptype == "classification" else float),
        problem_type=ptype, numeric_cols=numeric, categorical_cols=categorical,
        ids=ids[lab].reset_index(drop=True),
        X_predict=X_all[~lab].reset_index(drop=True) if (~lab).any() else None,
        predict_ids=ids[~lab].reset_index(drop=True) if (~lab).any() else None,
        info={"target_encoding": enc, "id_columns": id_cols, "dropped_constant": const},
    )
    return data
