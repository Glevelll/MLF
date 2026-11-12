"""Юнит-тесты агентов и Supervisor (LLM — мок-провайдер)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.agents.critic import Critic, fallback_verdict
from backend.agents.data_analyst import DataAnalyst
from backend.agents.editor import Editor
from backend.agents.model_selector import ModelSelector
from backend.agents.supervisor import Supervisor
from backend.agents.validator import Validator, single_feature_leakage
from backend.core.llm.base import ToolCall
from backend.core.llm.providers.mock import MockProvider
from backend.core.models import ForecastState, Issue
from backend.tabular.datasets import generate_synthetic
from backend.utils.tracing import build_tree


def make_state(path, task_id="t1", **kw) -> ForecastState:
    return ForecastState(task_id=task_id, series_id="test", data_path=str(path), **kw)


async def analysed(ctx, path, task_id="t1", **kw):
    st = make_state(path, task_id, **kw)
    return await DataAnalyst()(st, ctx)


async def test_data_analyst_tabular(make_ctx, synthetic_csv):
    ctx = make_ctx()
    st = await analysed(ctx, synthetic_csv, target_column="dropout")
    m = st.metadata
    assert st.problem_type == "classification" and m["n_rows"] == 310 and m["n_predict"] == 10
    assert m["id_columns"] == ["student_id"] and m["n_features_encoded"] > m["n_features"]
    assert "eda" in st.diagnostics and m["meta_features"]["n_rows"] == 310
    assert pd.read_parquet(m["processed_matrix_path"]).isna().sum().sum() == 0  # импутация выполнена


async def test_data_analyst_errors_and_small(make_ctx, tmp_path):
    ctx = make_ctx()
    with pytest.raises(ValueError):
        await DataAnalyst().process(ForecastState(task_id="e", series_id="x"), ctx)
    p = tmp_path / "s.csv"
    df = generate_synthetic(50, seed=1).drop(columns=["final_score"])
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        await DataAnalyst().process(make_state(p, "e2"), ctx)
    st = await analysed(ctx, p, "e3", target_column="dropout")
    assert "small_sample" in {i.code for i in st.validation_issues}


async def test_data_analyst_and_pipeline_timeseries(make_ctx, tmp_path):
    idx = pd.date_range("2024-01-01", periods=90, freq="D")
    vals = 20 + np.sin(np.arange(90) * 2 * np.pi / 7) * 3 + np.random.default_rng(1).normal(0, 0.3, 90)
    p = tmp_path / "ts.csv"
    pd.DataFrame({"date": idx, "attendance": vals}).to_csv(p, index=False)
    ctx = make_ctx()
    st = make_state(p, "ts1", task_type="timeseries", target_column="attendance", horizon=7, max_iterations=1)
    st = await Supervisor().run(st, ctx)
    assert st.freq == "D" and st.model_type in ("fit_arima", "fit_ets")
    assert st.metrics["forecast_len"] == 7 and "ts" in st.diagnostics
    assert st.hitl_requested or st.predictions_path.endswith("forecast.parquet")


async def test_model_selector_native_with_rag(make_ctx, synthetic_csv, retriever):
    ctx = make_ctx(retriever=retriever)
    st = await analysed(ctx, synthetic_csv, target_column="dropout")
    st = await ModelSelector()(st, ctx)
    assert st.selection_mode == "native" and st.rag_context and st.model_type in ctx.registry
    assert st.metrics["primary"] == "roc_auc" and st.metrics["cv"]["roc_auc"] > 0.6
    assert st.model_path and st.oof_path and st.metadata["feature_importance"]
    spans = ctx.tracer.recorder.spans
    assert any(s["step"] == "tool_call" and s["tool_name"] == st.model_type for s in spans)
    assert any(s["step"] == "rag_search" for s in spans)


async def test_model_selector_fallback_default_after_3_failures(make_ctx, synthetic_csv):
    ctx = make_ctx(provider=MockProvider(fail_times=10))
    st = await analysed(ctx, synthetic_csv, target_column="dropout")
    st = await ModelSelector()(st, ctx)
    assert st.selection_mode == "default" and st.model_type == "fit_logistic_regression"
    assert len(ctx.tracer.recorder.llm_calls) == 3


async def test_model_selector_forced_and_invalid(make_ctx, synthetic_csv):
    ctx = make_ctx()
    st = await analysed(ctx, synthetic_csv, target_column="dropout")
    st.forced_model = "fit_random_forest"
    st.model_params = {"n_estimators": 9999}  # другой модели → игнорируется
    st = await ModelSelector()(st, ctx)
    assert st.selection_mode == "forced" and st.model_type == "fit_random_forest" and st.forced_model is None
    st.forced_model = "fit_ridge_regression"
    with pytest.raises(ValueError):
        await ModelSelector().process(st, ctx)


async def test_validator_detects_leakage_and_low_quality(make_ctx, tmp_path):
    rng = np.random.default_rng(0)
    df = generate_synthetic(300, seed=3).drop(columns=["final_score"])
    df["leak"] = df["dropout"] * 10 + rng.normal(0, 0.01, len(df))
    p = tmp_path / "leak.csv"
    df.to_csv(p, index=False)
    ctx = make_ctx()
    st = await analysed(ctx, p, "leak", target_column="dropout")
    st = await ModelSelector()(st, ctx)
    st = await Validator()(st, ctx)
    codes = {i.code for i in st.validation_issues}
    assert "leakage" in codes and not st.validation_passed
    st2 = await Editor()(await Critic()(st, ctx), ctx)
    assert "leak" in st2.dropped_features and st2.editor_changed

    noise = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "y": rng.integers(0, 2, 200)})
    p2 = tmp_path / "noise.csv"
    noise.to_csv(p2, index=False)
    st = await analysed(ctx, p2, "noise", target_column="y")
    st = await ModelSelector()(st, ctx)
    st = await Validator()(st, ctx)
    assert any(i.code == "low_quality" and i.severity == "critical" for i in st.validation_issues)


def test_single_feature_leakage_regression():
    X = pd.DataFrame({"a": np.arange(50.0), "b": np.random.default_rng(0).normal(size=50)})
    y = pd.Series(np.arange(50.0) * 2)
    assert list(single_feature_leakage(X, y, "regression", ["a", "b"])) == ["a"]


async def test_validator_regression(make_ctx, regression_csv):
    ctx = make_ctx()
    st = await analysed(ctx, regression_csv, "reg", target_column="final_score")
    assert st.problem_type == "regression"
    st = await ModelSelector()(st, ctx)
    st = await Validator()(st, ctx)
    assert st.metrics["primary"] == "r2" and isinstance(st.validation_passed, bool)


async def test_critic_and_fallback(make_ctx, synthetic_csv):
    ctx = make_ctx()
    st = await analysed(ctx, synthetic_csv, target_column="dropout")
    st = await ModelSelector()(st, ctx)
    st = await Critic()(st, ctx)
    assert 0 <= st.critic_score <= 10 and not st.critic_details["fallback"] and st.critic_reasoning
    ctx_bad = make_ctx(provider=MockProvider(fail_times=10))
    st.validation_issues = [Issue(code="leakage", severity="critical", message="x")]
    st = await Critic()(st, ctx_bad)
    assert st.critic_details["fallback"] and st.critic_score == 3.0 and st.critic_details["critical"]
    assert fallback_verdict(ForecastState(task_id="a", series_id="b")).score == 5.0


async def test_editor_apply_all_tools(make_ctx, synthetic_csv):
    ctx = make_ctx()
    st = await analysed(ctx, synthetic_csv, target_column="dropout")
    st.model_type, st.model_params = "fit_random_forest", ctx.registry.get("fit_random_forest").validate_args({})
    ed = Editor()
    assert ed.apply(st, ToolCall(name="add_features", arguments={"ops": ["ratios"]}), ctx)
    assert not ed.apply(st, ToolCall(name="add_features", arguments={"ops": ["ratios"]}), ctx)
    assert ed.apply(st, ToolCall(name="drop_features", arguments={"columns": ["region"]}), ctx)
    assert not ed.apply(st, ToolCall(name="drop_features", arguments={"columns": ["nope"]}), ctx)
    assert ed.apply(st, ToolCall(name="change_threshold", arguments={"threshold": 0.3}), ctx)
    assert ed.apply(st, ToolCall(name="change_hyperparams", arguments={"params": {"max_depth": 5}}), ctx)
    assert st.model_params["max_depth"] == 5
    assert not ed.apply(st, ToolCall(name="change_hyperparams", arguments={"params": {"max_depth": -1}}), ctx)
    assert ed.apply(st, ToolCall(name="switch_model", arguments={"model_type": "fit_tabular_xgboost"}), ctx)
    assert not ed.apply(st, ToolCall(name="switch_model", arguments={"model_type": "fit_ridge_regression"}), ctx)
    assert st.forced_model == "fit_tabular_xgboost" and len(st.editor_actions) == 5
    # рабочие данные учитывают правки
    from backend.agents.common import load_working_data

    data = load_working_data(st, ctx)
    assert "region" not in data.feature_columns and any(c.startswith("fe_") for c in data.feature_columns)


async def test_editor_rule_based_fallback(make_ctx, synthetic_csv):
    ctx = make_ctx(provider=MockProvider(fail_times=10))
    st = await analysed(ctx, synthetic_csv, target_column="dropout")
    st.critic_score = 1.0
    st = await Editor()(st, ctx)
    assert st.editor_changed and st.feature_ops == ["score_dynamics"]
    st.feature_ops = ["score_dynamics", "activity_aggregates", "ratios"]
    st = await Editor()(st, ctx)
    assert not st.editor_changed


async def test_supervisor_converges_and_finalizes(make_ctx, synthetic_csv, retriever):
    ctx = make_ctx(retriever=retriever)
    st = await Supervisor().run(make_state(synthetic_csv, target_column="dropout"), ctx)
    assert not st.hitl_requested and st.iteration >= 1 and st.tokens_used > 0
    preds = pd.read_csv(st.predictions_path)
    assert len(preds) == 320 and set(preds["split"]) == {"labeled", "new"}
    assert {"risk_probability", "risk_level", "risk_drivers", "recommendations"} <= set(preds.columns)
    risky = preds[preds["at_risk"]]
    assert (risky["recommendations"].fillna("") != "").any()
    assert st.summary_path and st.metadata["risk_counts"]
    tree = build_tree(ctx.tracer.recorder.spans)
    assert len(tree) == 1 and tree[0]["agent_name"] == "Supervisor"
    agents = {c["agent_name"] for it in tree[0]["children"] for c in it["children"]}
    assert {"ModelSelector", "Validator", "Critic"} <= agents
    for h in st.metadata["hitl_students"]:
        assert h["score"] >= ctx.settings.high_risk_threshold


async def test_supervisor_hitl_and_resume(make_ctx, tmp_path):
    rng = np.random.default_rng(0)
    noise = pd.DataFrame({"a": rng.normal(size=150), "b": rng.normal(size=150), "c": rng.normal(size=150),
                          "y": rng.integers(0, 2, 150)})
    p = tmp_path / "noise.csv"
    noise.to_csv(p, index=False)
    ctx = make_ctx()
    st = await Supervisor().run(make_state(p, "h1", target_column="y", max_iterations=2), ctx)
    assert st.hitl_requested and st.hitl_reason and st.iteration <= 2
    st = await Supervisor().resume(st, ctx, {"action": "choose_model", "model_type": "fit_random_forest",
                                              "params": {"n_estimators": 60}})
    assert st.model_type == "fit_random_forest" and st.model_params["n_estimators"] == 60 and st.predictions_path
    st2 = await Supervisor().resume(st, ctx, {"action": "change_threshold", "threshold": 0.4})
    assert st2.decision_threshold == 0.4
    st3 = await Supervisor().resume(st2, ctx, {"action": "accept"})
    assert st3.hitl_resolution["action"] == "accept" and not st3.hitl_requested


async def test_supervisor_regression_pass_mark(make_ctx, regression_csv):
    ctx = make_ctx()
    st = await Supervisor().run(make_state(regression_csv, "r1", target_column="final_score", max_iterations=1,
                                           metadata={"explain": False}), ctx)
    if st.hitl_requested:
        st = await Supervisor().resume(st, ctx, {"action": "accept"})
    preds = pd.read_csv(st.predictions_path)
    assert "predicted_score" in preds.columns and st.metadata["pass_mark"] is not None
