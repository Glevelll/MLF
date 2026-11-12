"""Эмбеддинги: стратегия FeatureExtractor (tabular / timeseries) + текстовый эмбеддер.

Вектор кейса (128 dims) = [стандартизованные мета-признаки (32)] ⊕ [LSA/ST-эмбеддинг
контекстуализированного описания (96)]. Мета-часть отражает «статистическую похожесть»
данных, текстовая — смысловую.
"""
from __future__ import annotations

import threading
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

NUM_DIM = 32
TEXT_DIM = 96
DIM = NUM_DIM + TEXT_DIM

TABULAR_KEYS = ["log_rows", "n_features", "n_numeric", "n_categorical", "cat_ratio", "missing_rate",
                "minority_ratio", "is_classification", "target_skew", "mean_abs_corr", "max_abs_corr",
                "n_strong_corr", "mean_skew", "mean_kurt", "mean_cardinality", "log_rows_per_feature",
                "has_score_columns", "n_score_columns"]
TS_KEYS = ["log_length", "trend_strength", "seasonal_strength", "acf1", "spectral_entropy", "cv", "adf_pvalue"]


class FeatureExtractor(ABC):
    keys: list[str]

    @abstractmethod
    def extract(self, **kwargs: Any) -> dict[str, float]: ...

    def vector(self, feats: dict[str, float]) -> np.ndarray:
        return np.array([float(feats.get(k, 0.0) or 0.0) for k in self.keys], dtype=float)


class TabularFeatureExtractor(FeatureExtractor):
    keys = TABULAR_KEYS

    def extract(self, *, X: pd.DataFrame, y: pd.Series, problem_type: str, numeric_cols: list[str],
                categorical_cols: list[str], **_: Any) -> dict[str, float]:
        from backend.tabular.features import score_columns

        n, p = X.shape
        feats: dict[str, float] = {
            "n_rows": float(n), "log_rows": float(np.log10(max(n, 1))), "n_features": float(p),
            "n_numeric": float(len(numeric_cols)), "n_categorical": float(len(categorical_cols)),
            "cat_ratio": len(categorical_cols) / max(p, 1), "missing_rate": float(X.isna().mean().mean()) if p else 0.0,
            "is_classification": 1.0 if problem_type == "classification" else 0.0,
        }
        if problem_type == "classification":
            feats["minority_ratio"] = float(y.value_counts(normalize=True).min())
            feats["target_skew"] = 0.0
        else:
            feats["minority_ratio"] = 0.5
            feats["target_skew"] = float(np.nan_to_num(y.skew()))
        corrs, skews, kurts = [], [], []
        yv = y.astype(float).values
        for c in numeric_cols:
            v = X[c].astype(float)
            v = v.fillna(v.median())
            if v.std() > 0:
                r = np.corrcoef(v, yv)[0, 1]
                if np.isfinite(r):
                    corrs.append(abs(r))
                skews.append(float(np.nan_to_num(v.skew())))
                kurts.append(float(np.nan_to_num(v.kurt())))
        feats["mean_abs_corr"] = float(np.mean(corrs)) if corrs else 0.0
        feats["max_abs_corr"] = float(np.max(corrs)) if corrs else 0.0
        feats["n_strong_corr"] = float(sum(c > 0.3 for c in corrs))
        feats["mean_skew"] = float(np.clip(np.mean(skews), -10, 10)) if skews else 0.0
        feats["mean_kurt"] = float(np.clip(np.mean(kurts), -10, 50)) if kurts else 0.0
        feats["mean_cardinality"] = float(np.mean([X[c].nunique() for c in categorical_cols])) if categorical_cols else 0.0
        feats["log_rows_per_feature"] = float(np.log10(max(n, 1) / max(p, 1)))
        sc = score_columns(X)
        feats["has_score_columns"] = 1.0 if len(sc) >= 2 else 0.0
        feats["n_score_columns"] = float(len(sc))
        return {k: round(float(v), 6) for k, v in feats.items()}


class TimeSeriesFeatureExtractor(FeatureExtractor):
    keys = TS_KEYS

    def extract(self, *, series: pd.Series, period: int | None = None, **_: Any) -> dict[str, float]:
        from backend.timeseries.preprocessing import adf_test, stl_decompose

        v = series.dropna().astype(float).values
        feats: dict[str, float] = {"log_length": float(np.log10(max(len(v), 1)))}
        feats.update({k: v_ for k, v_ in stl_decompose(series.dropna(), period).items() if k != "period"})
        feats["acf1"] = float(np.corrcoef(v[:-1], v[1:])[0, 1]) if len(v) > 2 else 0.0
        spec = np.abs(np.fft.rfft(v - v.mean())) ** 2
        p = spec / (spec.sum() + 1e-12)
        feats["spectral_entropy"] = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p) + 1e-12)) if len(p) > 1 else 0.0
        feats["cv"] = float(v.std() / (abs(v.mean()) + 1e-9))
        try:
            feats["adf_pvalue"] = adf_test(series)["adf_pvalue"]
        except Exception:
            feats["adf_pvalue"] = 1.0
        return {k: float(np.nan_to_num(x)) for k, x in feats.items()}


def get_extractor(task_type: str) -> FeatureExtractor:
    return TimeSeriesFeatureExtractor() if task_type == "timeseries" else TabularFeatureExtractor()


class TextEmbedder:
    """LSA (char n-gram TF-IDF + SVD) — лёгкий локальный семантический эмбеддер без GPU.
    При text_embedder=sentence_transformers используется multilingual MiniLM + проекция PCA."""

    def __init__(self, backend: str = "lsa") -> None:
        self.backend = backend
        self.vectorizer = None
        self.svd = None
        self.st_model = None
        self.st_pca = None

    def fit(self, corpus: list[str]) -> TextEmbedder:
        if self.backend == "sentence_transformers":  # pragma: no cover - тяжёлая зависимость
            from sentence_transformers import SentenceTransformer
            from sklearn.decomposition import PCA

            self.st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            emb = self.st_model.encode(corpus, batch_size=64, normalize_embeddings=True)
            self.st_pca = PCA(n_components=min(TEXT_DIM, len(corpus) - 1)).fit(emb)
            return self
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1 if len(corpus) < 50 else 2,
                                          max_features=60000, sublinear_tf=True)
        m = self.vectorizer.fit_transform(corpus)
        k = max(1, min(TEXT_DIM, m.shape[0] - 1, m.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(m)
        return self

    @property
    def fitted(self) -> bool:
        return self.svd is not None or self.st_pca is not None

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self.fitted:
            return np.zeros((len(texts), TEXT_DIM))
        if self.st_model is not None:  # pragma: no cover
            z = self.st_pca.transform(self.st_model.encode(texts, normalize_embeddings=True))
        else:
            z = self.svd.transform(self.vectorizer.transform(texts))
        if z.shape[1] < TEXT_DIM:
            z = np.hstack([z, np.zeros((z.shape[0], TEXT_DIM - z.shape[1]))])
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


class CaseEmbedder:
    """Комбинированный эмбеддер кейсов. Скейлеры мета-признаков обучаются отдельно на каждом kind."""

    def __init__(self, text_backend: str = "lsa", num_weight: float = 0.6) -> None:
        self.text = TextEmbedder(text_backend)
        self.scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.num_weight = num_weight

    def fit(self, corpus: list[str], numeric_by_kind: dict[str, np.ndarray]) -> CaseEmbedder:
        self.text.fit(corpus)
        for kind, mat in numeric_by_kind.items():
            mat = np.atleast_2d(np.asarray(mat, dtype=float))
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mu = np.nan_to_num(np.nanmean(mat, axis=0))
                sd = np.nan_to_num(np.nanstd(mat, axis=0))
            self.scalers[kind] = (mu, sd + 1e-9)
        return self

    def numeric_part(self, kind: str, vec: np.ndarray) -> np.ndarray:
        """Стандартизация; пропуск = среднее по базе (нулевой вклад)."""
        vec = np.asarray(vec, dtype=float)
        if kind in self.scalers and len(self.scalers[kind][0]) == len(vec):
            mu, sd = self.scalers[kind]
            vec = (vec - mu) / sd
        vec = np.clip(np.nan_to_num(vec), -5, 5)
        out = np.zeros(NUM_DIM)
        out[: min(NUM_DIM, len(vec))] = vec[:NUM_DIM]
        n = np.linalg.norm(out)
        return out / n if n > 0 else out

    def embed(self, kind: str, numeric: np.ndarray | None, texts: list[str]) -> np.ndarray:
        t = self.text.transform(texts)
        if numeric is None:
            num = np.zeros((len(texts), NUM_DIM))
        else:
            numeric = np.atleast_2d(numeric)
            num = np.vstack([self.numeric_part(kind, r) for r in numeric])
        z = np.hstack([self.num_weight * num, (1 - self.num_weight) * t])
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> CaseEmbedder:
        return joblib.load(path)


_lock = threading.Lock()
_instance: CaseEmbedder | None = None


def get_embedder(path: Path | None = None, text_backend: str = "lsa") -> CaseEmbedder:
    """Синглтон эмбеддера (загружается из индекса, если он построен)."""
    global _instance
    with _lock:
        if _instance is None:
            _instance = CaseEmbedder.load(path) if path is not None and path.exists() else CaseEmbedder(text_backend)
        return _instance


def set_embedder(e: CaseEmbedder | None) -> None:
    global _instance
    with _lock:
        _instance = e
