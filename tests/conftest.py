"""Общие фикстуры: изолированный data_dir, SQLite во временной папке, mock LLM, inline-задачи."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="mlf_test_"))
os.environ.update({
    "MLF_APP_ENV": "test",
    "MLF_DATA_DIR": str(_TMP),
    "MLF_LLM_PROVIDER": "mock",
    "MLF_TASK_BACKEND": "inline",
    "MLF_VECTOR_BACKEND": "memory",
    "MLF_LLM_CACHE_ENABLED": "true",
    "MLF_JWT_SECRET": "test-secret",
})
_REAL_RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci_student"
if _REAL_RAW.exists():
    shutil.copytree(_REAL_RAW, _TMP / "raw" / "uci_student", dirs_exist_ok=True)

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from backend.agents.base import AgentContext  # noqa: E402
from backend.core.config import get_settings  # noqa: E402
from backend.core.llm.manager import LLMManager  # noqa: E402
from backend.core.llm.providers.mock import MockProvider  # noqa: E402
from backend.core.llm.tools import default_registry  # noqa: E402
from backend.data.storage import ArtifactStore  # noqa: E402
from backend.tabular.datasets import generate_synthetic  # noqa: E402
from backend.utils.tracing import MemoryRecorder, Tracer  # noqa: E402

HAS_UCI = (_TMP / "raw" / "uci_student" / "student-mat.csv").exists()
needs_uci = pytest.mark.skipif(not HAS_UCI, reason="UCI данные не загружены")


@pytest.fixture(scope="session")
def settings():
    s = get_settings()
    s.ensure_dirs()
    return s


@pytest.fixture
def recorder():
    return MemoryRecorder()


@pytest.fixture
def make_ctx(settings, recorder):
    def _make(provider=None, retriever=None, request_id="test-req"):
        llm = LLMManager([provider or MockProvider()], recorder=recorder, request_id=request_id, backoff=0)
        return AgentContext(settings=settings, llm=llm, tracer=Tracer(request_id, recorder),
                            registry=default_registry(), store=ArtifactStore(), retriever=retriever)
    return _make


@pytest.fixture
def synthetic_csv(tmp_path) -> Path:
    df = generate_synthetic(320, seed=11)
    df = df.drop(columns=["final_score"])
    df.loc[:9, "dropout"] = None  # 10 «новых» студентов без метки
    p = tmp_path / "students.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def regression_csv(tmp_path) -> Path:
    df = generate_synthetic(300, seed=12).drop(columns=["dropout"])
    p = tmp_path / "students_reg.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def small_kb():
    """Маленькая база знаний для RAG-тестов (без тяжёлой генерации)."""
    from backend.data.embeddings import TabularFeatureExtractor
    from backend.data.knowledge_base import (
        canonical_profile,
        describe_dataset_case,
        describe_student,
        recommendations_for,
    )
    from backend.tabular.preprocessing import prepare_dataframe

    cases = []
    for i, (n, target, model) in enumerate([(300, "dropout", "fit_logistic_regression"),
                                            (2000, "dropout", "fit_tabular_xgboost"),
                                            (800, "dropout", "fit_random_forest"),
                                            (500, "final_score", "fit_ridge_regression")]):
        df = generate_synthetic(n, seed=100 + i)
        df = df.drop(columns=["final_score" if target == "dropout" else "dropout"])
        data = prepare_dataframe(df, target)
        meta = TabularFeatureExtractor().extract(X=data.X, y=data.y, problem_type=data.problem_type,
                                                 numeric_cols=data.numeric_cols, categorical_cols=data.categorical_cols)
        metrics = {"roc_auc": 0.85} if data.problem_type == "classification" else {"r2": 0.6}
        cases.append({"source_id": f"dataset_t{i}", "kind": "dataset_case", "task_type": "tabular",
                      "problem_type": data.problem_type, "target_column": target, "series_features": meta,
                      "best_model": model, "hyperparams": {}, "metrics": metrics,
                      "description": describe_dataset_case(meta, model, metrics, "synthetic"), "meta": {"source": "t"}})
    df = generate_synthetic(60, seed=200)
    for row in df.to_dict("records"):
        prof = canonical_profile(row)
        risk = bool(row["dropout"])
        cases.append({"source_id": f"student_{row['student_id']}", "kind": "student_case", "task_type": "tabular",
                      "problem_type": "classification", "target_column": "dropout", "series_features": prof,
                      "best_model": None, "hyperparams": {}, "metrics": {},
                      "description": describe_student(prof, row),
                      "meta": {"source": "synthetic", "status": "at_risk" if risk else "ok",
                               "outcome": row["final_score"], "recommendations": recommendations_for(prof, risk)}})
    glossary = [{"term": "ROC-AUC", "definition": "площадь под ROC-кривой", "category": "metrics"},
                {"term": "Утечка данных", "definition": "использование будущей информации", "category": "ml"}]
    return cases, glossary


@pytest.fixture
def retriever(small_kb, settings):
    from backend.retrieval.index_manager import build_retriever

    cases, glossary = small_kb
    return build_retriever(cases, glossary, settings, persist=False)


def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path)
