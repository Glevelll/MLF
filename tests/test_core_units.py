"""Юнит-тесты: метрики, парсинг ответов LLM, реестр инструментов, статистика, трассировка."""
from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from backend.core import metrics as M
from backend.core.llm.parsing import extract_json, parse_tool_calls
from backend.core.llm.tokens import count_tokens
from backend.core.llm.tools import DEFAULT_MODEL, Tool, ToolRegistry, default_registry
from backend.evaluation import stats
from backend.utils.tracing import MemoryRecorder, Tracer, build_tree, summarize


# ------------------------------------------------------------------ метрики
def test_regression_metrics():
    y, p = [1, 2, 3, 4], [1, 2, 4, 4]
    m = M.regression_metrics(y, p)
    assert m["mae"] == pytest.approx(0.25)
    assert m["rmse"] == pytest.approx(0.5)
    assert m["r2"] == pytest.approx(0.8)
    assert M.wape(y, p) == pytest.approx(10.0)
    assert math.isnan(M.mape([0, 0], [1, 1]))
    assert M.mase([2, 3], [2, 3], [1, 2, 3, 4]) == 0.0
    assert math.isnan(M.mase([1], [1], [1]))


def test_classification_metrics_and_threshold():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    m = M.classification_metrics(y, p, 0.5)
    assert 0 <= m["ece"] <= 1 and m["roc_auc"] == pytest.approx(8 / 9)
    t, f = M.best_f1_threshold(y, p)
    assert 0.05 <= t <= 0.95 and f >= m["f1"]
    assert math.isnan(M.classification_metrics([1, 1], [0.2, 0.9])["roc_auc"])
    assert M.primary_metric("classification") == "roc_auc" and M.primary_metric("regression") == "r2"
    assert "mae" in M.compute_metrics("regression", [1, 2], [1, 2])


def test_residual_tests_and_clean():
    rng = np.random.default_rng(0)
    lb = M.ljung_box(rng.normal(size=200))
    assert lb["lb_pvalue"] > 0.01
    sw = M.shapiro_wilk(rng.normal(size=100))
    assert sw["sw_pvalue"] > 0.01
    assert math.isnan(M.shapiro_wilk([1.0])["sw_stat"])
    c = M.clean({"a": float("nan"), "b": np.float64(1.5), "c": {"d": np.inf}, "e": np.int64(3), "f": "x"})
    assert c == {"a": None, "b": 1.5, "c": {"d": None}, "e": 3, "f": "x"}


# ------------------------------------------------------------------ парсинг ответов LLM
@pytest.mark.parametrize("text,name", [
    ('{"name": "fit_random_forest", "arguments": {"n_estimators": 100}}', "fit_random_forest"),
    ('Конечно! ```json\n{"tool": "fit_ridge_regression", "args": {"alpha": 2}}\n```', "fit_ridge_regression"),
    ('{"function": {"name": "fit_tabular_xgboost", "arguments": "{\\"max_depth\\": 3}"}}', "fit_tabular_xgboost"),
    ('Выбираю: {"tool_calls": [{"name": "add_features", "arguments": {"ops": ["ratios"]}}]} готово', "add_features"),
])
def test_parse_tool_calls(text, name):
    calls = parse_tool_calls(text)
    assert calls and calls[0].name == name and isinstance(calls[0].arguments, dict)


def test_extract_json_edge_cases():
    assert extract_json("") is None
    assert extract_json("нет json") is None
    assert extract_json('текст {"a": "x}y"} хвост') == {"a": "x}y"}
    assert extract_json("[1, 2]") == [1, 2]
    assert parse_tool_calls('{"name": 5}') == []
    assert parse_tool_calls('[1, {"name": "x", "arguments": "bad"}]')[0].arguments == {}


def test_count_tokens():
    assert count_tokens("") == 0
    assert count_tokens("Привет, мир! Hello world") > 2


# ------------------------------------------------------------------ реестр инструментов
def test_registry_filters_and_schemas():
    r = default_registry()
    clf = r.names(kind="model", task_type="tabular", problem_type="classification")
    reg = r.names(kind="model", task_type="tabular", problem_type="regression")
    assert {"fit_logistic_regression", "fit_random_forest", "fit_tabular_xgboost"} <= set(clf)
    assert "fit_logistic_regression" not in reg and "fit_ridge_regression" in reg
    assert set(r.names(kind="model", task_type="timeseries", problem_type="forecast")) == {"fit_arima", "fit_ets"}
    assert len(r.list(kind="editor")) == 5
    sch = r.get("fit_tabular_xgboost").schema()
    assert sch["function"]["name"] == "fit_tabular_xgboost" and "max_depth" in sch["function"]["parameters"]["properties"]
    assert DEFAULT_MODEL["classification"] in r
    with pytest.raises(KeyError):
        r.get("fit_magic")
    with pytest.raises(ValueError):
        r.register(r.get("fit_ets"))


def test_tool_validation_and_builders():
    r = default_registry()
    lr = r.get("fit_logistic_regression")
    assert lr.validate_args({})["C"] == 1.0
    with pytest.raises(ValidationError):
        lr.validate_args({"C": -1})
    with pytest.raises(ValidationError):
        lr.validate_args({"unknown": 1})
    est = lr.builder(lr.validate_args({"l1_ratio": 0.5}), "classification", 0)
    assert est.solver == "saga"
    for name, ptype in [("fit_random_forest", "regression"), ("fit_random_forest", "classification"),
                        ("fit_tabular_xgboost", "regression"), ("fit_tabular_xgboost", "classification"),
                        ("fit_ridge_regression", "regression")]:
        t = r.get(name)
        assert t.builder(t.validate_args({}), ptype, 0) is not None
    with pytest.raises(ValidationError):
        r.get("add_features").validate_args({"ops": ["magic"]})


def test_custom_registry():
    from pydantic import BaseModel

    class A(BaseModel):
        x: int = 1

    reg = ToolRegistry()
    reg.register(Tool("t", "desc", A, kind="editor"))
    assert "t" in reg and reg.names(kind="editor") == ["t"]


# ------------------------------------------------------------------ статистика
def test_stat_tests():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.02, 30)
    b = a - 0.05 + rng.normal(0, 0.01, 30)
    assert stats.wilcoxon(a, b)["pvalue"] < 0.01
    assert stats.paired_ttest(a, b)["pvalue"] < 0.01
    ci = stats.bootstrap_ci(a - b, n_boot=300)
    assert ci["low"] > 0
    la, lb = rng.random(200) * 0.5, rng.random(200)
    dm = stats.diebold_mariano(la, lb, h=2)
    assert dm["dm"] < 0 and dm["pvalue"] < 0.05
    assert stats.diebold_mariano([1, 1], [1, 1])["pvalue"] == 1.0
    assert stats.wilcoxon([1, 2], [1, 2])["pvalue"] == 1.0
    assert stats.paired_ttest([1], [2])["pvalue"] == 1.0
    assert stats.bootstrap_ci([])["n"] == 0
    cmp_ = stats.compare_paired(a, b, "a", "b")
    assert cmp_["win_rate"] > 0.9 and cmp_["mean_improvement"] > 0


# ------------------------------------------------------------------ трассировка
async def test_tracer_hierarchy_and_errors():
    rec = MemoryRecorder()
    tr = Tracer("r1", rec)
    async with tr.span("Supervisor", "run") as root:
        async with tr.span("Agent", "process") as child:
            child.add_tokens(10)
            async with tr.span("Agent", "tool_call", tool_name="fit_x"):
                pass
        with pytest.raises(RuntimeError):
            async with tr.span("Agent", "boom"):
                raise RuntimeError("x")
        root.set_output({"ok": True})
    assert tr.total_tokens == 10
    tree = build_tree(rec.spans)
    assert len(tree) == 1 and len(tree[0]["children"]) == 2
    assert tree[0]["children"][0]["children"][0]["tool_name"] == "fit_x"
    assert any(s["status"] == "error" for s in rec.spans)
    assert summarize("x" * 1000, 50).endswith("...") and summarize(None) == ""
