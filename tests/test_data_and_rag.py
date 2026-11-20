"""Предобработка, feature engineering, обучение, временные ряды, эмбеддинги, RAG (BM25/вектор/RRF/Qdrant)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from backend.data.embeddings import (
    DIM,
    CaseEmbedder,
    TabularFeatureExtractor,
    TextEmbedder,
    TimeSeriesFeatureExtractor,
    get_embedder,
    get_extractor,
    set_embedder,
)
from backend.data.knowledge_base import (
    canonical_profile,
    contextualize,
    describe_dataset_case,
    describe_student,
    profile_vector,
    recommendations_for,
)
from backend.retrieval.bm25_index import BM25Index, tokenize
from backend.retrieval.filters import match
from backend.retrieval.hybrid_retriever import HybridRetriever, rrf
from backend.retrieval.index_manager import build_retriever, get_retriever, load_retriever, set_retriever
from backend.retrieval.vector_index import InMemoryVectorIndex, QdrantVectorIndex
from backend.tabular.datasets import generate_synthetic, load_uci, read_table
from backend.tabular.features import AVAILABLE_OPS, apply_feature_ops, score_columns
from backend.tabular.preprocessing import (
    STUDENT_GROUP,
    coerce_types,
    detect_id_columns,
    eda_summary,
    load_prepared,
    prepare_dataframe,
    prepare_target,
    save_prepared,
)
from backend.tabular.training import cv_splits, fit_and_evaluate, holdout_indices, occlusion_drivers
from tests.conftest import needs_uci


# ------------------------------------------------------------------ предобработка
def test_coerce_and_ids():
    df = pd.DataFrame({"student_id": [f"s{i}" for i in range(60)], "g": [str(i) for i in range(60)],
                       "flag": [True, False] * 30, "cat": ["a", "b", "c"] * 20})
    c = coerce_types(df)
    assert pd.api.types.is_numeric_dtype(c["g"]) and set(c["flag"]) == {"yes", "no"}
    assert detect_id_columns(c) == ["student_id"]
    assert detect_id_columns(c, "cat")[0] == "cat"


def test_prepare_target_variants():
    s = pd.Series([1, 5, 12, 15, 9, 11, 2, 18, 3, 4, 6, 7, 8, 13, 14, 16, 17, np.nan])
    y, pt, info = prepare_target(s, "auto")
    assert pt == "regression"
    y, pt, info = prepare_target(s, "auto", binarize_threshold=10)
    assert pt == "classification" and y.iloc[0] == 1 and np.isnan(y.iloc[-1]) and info["encoding"] == "threshold"
    y, pt, info = prepare_target(pd.Series(["yes", "no", "no"]), "auto")
    assert info["positive"] == "yes" and y.tolist() == [1, 0, 0]
    y, _, info = prepare_target(pd.Series([0, 1, 1, 1]), "classification")
    assert info["positive"] == "1"
    y, _, _ = prepare_target(pd.Series(["pass", "fail"]), "classification", positive_label="fail")
    assert y.tolist() == [0, 1]
    for bad in [dict(s=pd.Series(["a", "b"]), problem_type="regression"),
                dict(s=pd.Series(["a", "b", "c"]), problem_type="classification"),
                dict(s=pd.Series(["a", "b"]), problem_type="auto", binarize_threshold=1),
                dict(s=pd.Series(["a", "b"]), problem_type="auto", positive_label="z")]:
        with pytest.raises(ValueError):
            prepare_target(**bad)


def test_prepare_dataframe_and_roundtrip(tmp_path):
    df = generate_synthetic(200, seed=1).drop(columns=["final_score"])
    df["const"] = 1
    df.loc[:4, "dropout"] = np.nan
    data = prepare_dataframe(df, "dropout")
    assert data.problem_type == "classification" and len(data.X) == 195 and len(data.X_predict) == 5
    assert "const" not in data.feature_columns and "student_id" not in data.feature_columns
    eda = eda_summary(data.X, data.y, data.problem_type, data.numeric_cols, data.categorical_cols)
    assert eda["n_rows"] == 195 and "class_balance" in eda and eda["target_correlations"]
    p = tmp_path / "prep.parquet"
    save_prepared(data, p)
    back = load_prepared(p, data.numeric_cols, data.categorical_cols, "classification")
    assert len(back.X) == 195 and len(back.X_predict) == 5 and back.ids.iloc[0] == data.ids.iloc[0]
    with pytest.raises(ValueError):
        prepare_dataframe(df, "missing_col")
    reg = prepare_dataframe(generate_synthetic(100, seed=2).drop(columns=["dropout"]), "final_score")
    assert reg.problem_type == "regression"
    assert "target_stats" in eda_summary(reg.X, reg.y, "regression", reg.numeric_cols, reg.categorical_cols)


def test_read_table_semicolon(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("a;b\n1;2\n", encoding="utf-8")
    assert list(read_table(p).columns) == ["a", "b"]
    q = tmp_path / "a.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(q)
    assert read_table(q).x.iloc[0] == 1


@needs_uci
def test_load_uci():
    df = load_uci("mat")
    assert len(df) == 395 and pd.api.types.is_numeric_dtype(df["G1"]) and df["student_id"].is_unique
    assert len(load_uci("both")) == 1044


@needs_uci
def test_uci_student_groups_survive_preprocessing_and_stay_in_one_fold(tmp_path):
    df = load_uci("both")
    mat = df.loc[df.subject == "mat", STUDENT_GROUP].value_counts()
    por = df.loc[df.subject == "por", STUDENT_GROUP].value_counts()
    assert (mat * por).sum() == 382
    data = prepare_dataframe(df, "G3", "classification", binarize_threshold=10)
    assert STUDENT_GROUP not in data.feature_columns
    path = tmp_path / "grouped.parquet"
    save_prepared(data, path)
    prepared = load_prepared(path, data.numeric_cols, data.categorical_cols, data.problem_type)
    assert prepared.groups.equals(data.groups)
    splits = [holdout_indices(prepared, 0.2, seed=42), *cv_splits(prepared, 5, seed=42)]
    for train, test in splits:
        assert set(prepared.groups.iloc[train]).isdisjoint(prepared.groups.iloc[test])
        assert prepared.y.iloc[train].nunique() == prepared.y.iloc[test].nunique() == 2


def test_load_uci_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_uci("mat", raw_dir=tmp_path)


# ------------------------------------------------------------------ feature engineering
def test_feature_ops():
    df = generate_synthetic(100, seed=3)
    assert score_columns(df) == ["test1_score", "test2_score", "test3_score"]
    out, created = apply_feature_ops(df, list(AVAILABLE_OPS))
    assert {"fe_score_delta", "fe_score_slope", "fe_engagement", "fe_score_per_effort"} <= set(created)
    assert all(c in out.columns for c in created)
    uci_like = pd.DataFrame({"G1": [5, 10, 15], "G2": [6, 9, 16], "absences": [1, 2, 30], "studytime": [1, 2, 3]})
    out, created = apply_feature_ops(uci_like, ["score_dynamics", "activity_aggregates"])
    assert out["fe_score_delta"].tolist() == [1, -1, 1] and "fe_activity_std" not in created
    with pytest.raises(ValueError):
        apply_feature_ops(df, ["magic"])


# ------------------------------------------------------------------ обучение
def test_fit_and_evaluate_classification():
    df = generate_synthetic(300, seed=4).drop(columns=["final_score"])
    df.loc[:9, "dropout"] = np.nan
    data = prepare_dataframe(df, "dropout")
    fit = fit_and_evaluate(LogisticRegression(max_iter=2000), data, cv_folds=3)
    assert fit.cv_metrics["roc_auc"] > 0.7 and len(fit.oof) == len(data.X)
    assert fit.predict_output is not None and len(fit.predict_output) == 10
    assert fit.feature_importance and 0.05 <= fit.best_threshold <= 0.95
    md = fit.metrics_dict()
    assert set(md) == {"cv", "test", "train", "best_f1_threshold"}
    drivers = occlusion_drivers(fit.pipeline, data.X.iloc[[0]], data.X, "classification")
    assert all(d["effect"] > 0 for d in drivers) and len(drivers) <= 3


def test_fit_and_evaluate_regression():
    from sklearn.linear_model import Ridge

    data = prepare_dataframe(generate_synthetic(250, seed=5).drop(columns=["dropout"]), "final_score")
    fit = fit_and_evaluate(Ridge(), data, compute_importance=False, refit_full=False)
    assert fit.cv_metrics["r2"] > 0.3 and fit.best_threshold is None and fit.feature_importance == {}
    occlusion_drivers(fit.pipeline, data.X.iloc[[1]], data.X, "regression")


# ------------------------------------------------------------------ временные ряды
def test_timeseries_domain():
    from backend.timeseries.classical import fit_arima, fit_ets
    from backend.timeseries.preprocessing import (
        adf_test,
        infer_freq,
        is_regular,
        resample,
        stl_decompose,
        to_series,
    )

    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    vals = 10 + np.sin(np.arange(120) * 2 * np.pi / 7) + np.random.default_rng(0).normal(0, 0.2, 120)
    s = to_series(pd.DataFrame({"date": idx, "value": vals}))
    assert infer_freq(s) == "D" and is_regular(s)
    assert len(resample(s, "W")) < 120 and len(resample(s, "W", "sum")) < 120
    assert "adf_pvalue" in adf_test(s)
    assert stl_decompose(s, 7)["seasonal_strength"] > 0.5
    assert np.isnan(stl_decompose(s.iloc[:5], 7)["trend_strength"])
    assert len(fit_arima(s, 5).forecast) == 5 and len(fit_ets(s, 3, trend=None).forecast) == 3
    feats = TimeSeriesFeatureExtractor().extract(series=s, period=7)
    assert set(TimeSeriesFeatureExtractor.keys) <= set(feats)
    assert isinstance(get_extractor("timeseries"), TimeSeriesFeatureExtractor)


# ------------------------------------------------------------------ база знаний и эмбеддинги
def test_profiles_and_descriptions():
    uci_row = {"failures": 2, "absences": 20, "studytime": 1, "G1": 8, "G2": 6, "schoolsup": "no", "famsup": "no",
               "higher": "no", "Dalc": 4, "Walc": 5, "health": 3, "goout": 5, "internet": "no", "sex": "M", "age": 18}
    p = canonical_profile(uci_row)
    assert p["prior_failures"] == 2 and p["score_trend"] < 0 and p["support"] == 0.0
    txt = describe_student(p, uci_row)
    assert "снижаются" in txt and "алкоголя" in txt and "интернету" in txt
    recs = recommendations_for(p, True)
    assert len(recs) >= 4
    syn = canonical_profile(generate_synthetic(1, seed=0).iloc[0].to_dict())
    assert syn["absence"] is not None and syn["score_first"] is not None
    assert describe_student({}) == "Студент: профиль без деталей."
    assert "Поддерживать" in recommendations_for({}, False)[0]
    assert np.isnan(profile_vector({})).all()
    meta = {"n_rows": 300, "n_features": 20, "n_categorical": 5, "missing_rate": 0.05, "is_classification": 1,
            "minority_ratio": 0.1, "max_abs_corr": 0.8, "has_score_columns": 1}
    d = describe_dataset_case(meta, "fit_random_forest", {"roc_auc": 0.9}, "syn")
    assert "случайный лес" in d and "дисбаланс" in d
    assert "Лучшая" not in describe_dataset_case({"mean_abs_corr": 0.01}, "", {}, "q")
    assert contextualize("glossary", "x", {}).startswith("[glossary]")


def test_embedders(tmp_path):
    te = TextEmbedder()
    assert te.transform(["a"]).shape == (1, 96) and not te.fitted
    te.fit(["студент пропускает занятия", "высокие оценки", "низкая активность в LMS"])
    z = te.transform(["пропуски занятий"])
    assert z.shape == (1, 96) and np.isclose(np.linalg.norm(z), 1)
    ce = CaseEmbedder().fit(["a b", "c d", "e f"], {"k": np.array([[1, 2], [3, 4], [5, 6]])})
    v = ce.embed("k", np.array([[1, 2]]), ["a b"])
    assert v.shape == (1, DIM) and np.isclose(np.linalg.norm(v), 1)
    assert ce.embed("other", None, ["x"]).shape == (1, DIM)
    ce.save(tmp_path / "e.joblib")
    assert CaseEmbedder.load(tmp_path / "e.joblib").scalers.keys() == {"k"}
    set_embedder(None)
    assert get_embedder(tmp_path / "missing.joblib") is not None
    set_embedder(None)


def test_tabular_extractor():
    data = prepare_dataframe(generate_synthetic(150, seed=6).drop(columns=["dropout"]), "final_score")
    f = TabularFeatureExtractor().extract(X=data.X, y=data.y, problem_type="regression",
                                          numeric_cols=data.numeric_cols, categorical_cols=data.categorical_cols)
    assert f["is_classification"] == 0 and f["has_score_columns"] == 1 and len(TabularFeatureExtractor().vector(f)) == 18


# ------------------------------------------------------------------ RAG
def test_filters():
    p = {"a": 1, "b": "x", "n": 500}
    assert match(p, None) and match(p, {"a": 1}) and not match(p, {"a": 2})
    assert match(p, {"n": {"gte": 100, "lte": 1000}}) and not match(p, {"n": {"gte": 600}})
    assert match(p, {"b": {"in": ["x", "y"]}}) and not match(p, {"b": {"ne": "x"}}) and not match(p, {"b": {"eq": "y"}})
    assert not match(p, {"missing": {"lte": 1}}) and not match(p, {"n": {"lte": 100}}) and not match(p, {"b": {"in": ["z"]}})


def test_bm25():
    idx = BM25Index()
    assert idx.search("x") == []
    idx.add(["1", "2", "3"], ["студент часто пропускает занятия", "отличные оценки по математике",
                              "пропуски и низкие оценки"], [{"k": "a"}, {"k": "a"}, {"k": "b"}])
    res = idx.search("пропуски занятий", k=2)
    assert res[0][0] in ("1", "3") and len(res) <= 2
    assert all(r[2]["k"] == "a" for r in idx.search("оценки", filters={"k": "a"}))
    assert idx.text_of("2").startswith("отличные") and idx.text_of("zzz") == ""
    assert "и" not in tokenize("и в на студент")


def test_vector_index_and_rrf(tmp_path):
    vi = InMemoryVectorIndex(3)
    assert vi.search([1, 0, 0]) == []
    vi.add(["a", "b"], np.array([[1, 0, 0], [0, 1, 0]]), [{"t": 1}, {"t": 2}])
    assert vi.search([1, 0.1, 0], k=1)[0][0] == "a" and vi.search([1, 0, 0], filters={"t": 2})[0][0] == "b"
    with pytest.raises(ValueError):
        vi.add(["c"], np.array([[1, 0]]), [{}])
    vi.save(tmp_path / "v")
    assert InMemoryVectorIndex.load(tmp_path / "v").count() == 2
    fused = rrf({"bm25": [("a", 1, {}), ("b", 0.5, {})], "vector": [("b", 0.9, {}), ("c", 0.1, {})]})
    assert fused[0].id == "b" and fused[0].ranks == {"bm25": 2, "vector": 1}
    assert HybridRetriever(BM25Index(), None).glossary_search("x") == []


def test_qdrant_local_mode():
    from qdrant_client import QdrantClient

    q = QdrantVectorIndex(QdrantClient(":memory:"), "t", 3, recreate=True)
    q.add(["a", "b", "c"], np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]]),
          [{"kind": "x", "n": 10}, {"kind": "y", "n": 20}, {"kind": "x", "n": 30}])
    assert q.count() == 3
    assert q.search([1, 0, 0], k=1)[0][0] == "a"
    assert {r[0] for r in q.search([1, 0, 0], filters={"kind": "x", "n": {"gte": 20}})} == {"c"}
    assert {r[0] for r in q.search([1, 0, 0], filters={"kind": {"in": ["y"]}})} == {"b"}
    assert "a" not in {r[0] for r in q.search([1, 0, 0], filters={"n": {"ne": 10}, "kind": {"eq": "x"}})}
    QdrantVectorIndex(q.client, "t", 3, recreate=True)


def test_hybrid_retriever(retriever, settings):
    docs = retriever.search("большой набор, много строк, XGBoost", kind="dataset_case", k=4,
                            filters={"problem_type": "classification"})
    assert docs and all(d.payload["problem_type"] == "classification" for d in docs)
    assert set(docs[0].ranks) <= {"bm25", "vector"} and "text" in docs[0].payload
    st = retriever.search("пропуски снижаются оценки", kind="student_case", k=3,
                          numeric=profile_vector({"absence": 0.5}))
    assert len(st) == 3 and all(d.payload["kind"] == "student_case" for d in st)
    assert retriever.glossary_search("ROC-AUC")[0]["term"] == "ROC-AUC"
    assert st[0].brief()["id"] == st[0].id


def test_index_persist_and_load(small_kb, settings):
    cases, gl = small_kb
    r = build_retriever(cases, gl, settings, persist=True)
    loaded = load_retriever(settings)
    assert loaded.size == r.size and loaded.vector.count() == len(cases)
    set_retriever(None)
    assert get_retriever().size == r.size
    empty = build_retriever([], [], settings, persist=False)
    assert empty.size == 0 and empty.search("x", kind="dataset_case") == []
