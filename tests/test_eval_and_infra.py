"""Оценка (runner/specs), кэш, задачи воркера, скрипты генерации и импорта."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from backend.data.cache import DbCache, MemoryCache, cleanup_expired, get_cache, make_key
from backend.db.session import init_db
from backend.evaluation.runner import (
    compare_models,
    evaluate_spec,
    per_sample_loss,
    rag_choice,
    split_holdout,
    subsample,
    summarize,
)
from backend.evaluation.specs import DatasetSpec, build_dataset, make_specs, prepare_kwargs
from backend.tabular.datasets import generate_synthetic
from backend.tabular.preprocessing import prepare_dataframe
from tests.conftest import needs_uci

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_specs_disjoint_and_build():
    kb, ev = make_specs(20, "kb"), make_specs(20, "eval")
    assert {s.seed for s in kb}.isdisjoint({s.seed for s in ev})
    syn = DatasetSpec("x", "synthetic", "dropout", "classification", n=150, seed=1, drop_cols=["test2_score"])
    df = build_dataset(syn)
    assert "final_score" not in df and "test2_score" not in df and len(df) == 150
    assert prepare_kwargs(syn)["target"] == "dropout"
    assert DatasetSpec(**syn.to_dict()) == syn


def test_split_subsample_loss():
    data = prepare_dataframe(generate_synthetic(200, seed=1).drop(columns=["final_score"]), "dropout")
    tr, te = split_holdout(data, 0.25, 0)
    assert len(te.X) == 50 and abs(tr.y.mean() - te.y.mean()) < 0.1
    sub = subsample(tr, 0.2, 0)
    assert len(sub.X) == 30 and subsample(tr, 1.0, 0) is tr
    assert per_sample_loss("classification", [1, 0], [0.5, 0.5]).tolist() == [0.25, 0.25]


def test_rag_choice(retriever):
    data = prepare_dataframe(generate_synthetic(300, seed=2).drop(columns=["final_score"]), "dropout")
    model, args, ids = rag_choice(retriever, data)
    assert model in ("fit_logistic_regression", "fit_random_forest", "fit_tabular_xgboost") and ids
    assert rag_choice(None, data)[0] == "fit_logistic_regression"
    assert rag_choice(retriever, data, available=["fit_magic"])[0] == "fit_logistic_regression"


async def test_evaluate_spec_and_summarize(retriever, settings):
    specs = [DatasetSpec("e1", "synthetic", "dropout", "classification", n=260, seed=5),
             DatasetSpec("e2", "synthetic", "dropout", "classification", n=240, seed=6),
             DatasetSpec("e3", "synthetic", "final_score", "regression", n=220, seed=7)]
    rows = []
    for s in specs:
        rows += await evaluate_spec(s, modes=["baseline", "+rag", "+agents", "+foundation"], fractions=[0.5, 1.0],
                                    holdouts=[0.25], retriever=retriever, settings=settings)
    assert {r["mode"] for r in rows} == {"baseline", "+rag", "+agents"}  # foundation (TabPFN) не установлен
    ag = [r for r in rows if r["mode"] == "+agents"]
    assert all(r["a_iterations"] >= 1 and r["a_tokens"] > 0 for r in ag)
    summary = summarize(rows + [{"mode": "x", "error": "boom"}])
    assert summary["groups"] and summary["tests"] and summary["dm"] and summary["errors"] == 1
    assert summary["agents"]["runs"] == len(ag) and 0 <= summary["agents"]["hitl_rate"] <= 1
    assert summarize([])["groups"] == []


async def test_evaluate_spec_bad_mode_recorded(settings):
    rows = await evaluate_spec(DatasetSpec("e4", "synthetic", "dropout", "classification", n=150, seed=8),
                               modes=["bogus"], fractions=[1.0], holdouts=[0.25], retriever=None, settings=settings)
    assert rows[0]["error"]


def test_compare_models():
    data = prepare_dataframe(generate_synthetic(300, seed=3).drop(columns=["dropout"]), "final_score")
    res = compare_models(data, ["fit_ridge_regression", "fit_random_forest"], sample_size=200, folds=3)
    assert res["metric"] == "r2" and res["sample_size"] == 200 and len(res["results"]) == 2
    assert res["tests"][0]["diebold_mariano"]["n"] == 200


async def test_caches():
    await init_db()
    mc = MemoryCache()
    assert await mc.get("k") is None
    await mc.set("k", {"a": 1})
    assert (await mc.get("k")) == {"a": 1} and mc.hits["k"] == 1
    dc = DbCache()
    key = make_key("test", {"x": 1})
    assert key.startswith("test:") and await dc.get(key) is None
    await dc.set(key, {"v": 1})
    await dc.set(key, {"v": 2})
    assert (await dc.get(key)) == {"v": 2}
    await dc.set(make_key("test", 2), {"v": 3}, ttl_seconds=-10)
    assert await dc.get(make_key("test", 2)) is None
    assert await cleanup_expired() >= 1
    assert isinstance(get_cache(), DbCache)


async def test_worker_jobs():
    from backend.worker import jobs

    await init_db()
    assert await jobs.cleanup_cache_task() >= 0
    assert await jobs.cleanup_artifacts_task(max_age_days=0) >= 0
    with pytest.raises(KeyError):
        from backend.worker.pipeline import run_forecast

        await run_forecast("missing-task")


def test_generate_glossary():
    import generate_glossary

    items = generate_glossary.build()
    assert len(items) >= 500 and len({i["term"] for i in items}) == len(items)


@needs_uci
def test_kb_generation_helpers():
    import generate_knowledge_base as g

    cases = g.student_cases(9)
    assert len(cases) == 1044 + 9 and all(c["kind"] == "student_case" for c in cases)
    spec = DatasetSpec("t", "synthetic", "dropout", "classification", n=200, seed=1)
    c = g.dataset_case(spec.to_dict())
    assert c["best_model"] in {n for n, _ in g.GRID["classification"]} and len(c["meta"]["candidates"]) == 6
    tiny = DatasetSpec("t2", "synthetic", "dropout", "classification", n=30, seed=1, dropout_shift=-5)
    assert g.dataset_case(tiny.to_dict()) is None


async def test_import_data(settings, small_kb):
    import import_data

    kb = settings.kb_dir
    cases, gl = small_kb
    (kb / "student_cases.jsonl").write_text("\n".join(json.dumps(c, default=str) for c in cases if c["kind"] ==
                                                      "student_case"), encoding="utf-8")
    (kb / "dataset_cases.jsonl").write_text("\n".join(json.dumps(c, default=str) for c in cases if c["kind"] ==
                                                      "dataset_case"), encoding="utf-8")
    (kb / "glossary.jsonl").write_text("\n".join(json.dumps(g) for g in gl), encoding="utf-8")
    stats = await import_data.import_all()
    assert stats["total_cases"] >= len(cases) and stats["total_terms"] >= 2
    again = await import_data.import_all()
    assert again["added_cases"] == 0 and again["added_terms"] == 0
    assert import_data.read_jsonl(kb / "missing.jsonl") == []


async def test_run_eval_script(retriever, monkeypatch, tmp_path):
    import run_eval

    monkeypatch.setattr(run_eval, "ROOT", tmp_path)
    monkeypatch.setattr(run_eval, "get_retriever", lambda: retriever)
    args = run_eval.parse_args(["--num-series", "2", "--modes", "baseline", "+rag", "--horizons", "25"])
    assert args.holdouts == [0.25]
    out = await run_eval.main_async(args)
    assert (out / "summary.json").exists() and (out / "report.md").read_text(encoding="utf-8").startswith("# Отчёт")
    assert run_eval.check_data(DatasetSpec("a", "s", "t", "classification"), 10, 5, "classification", 2)


async def test_seed_demo_user():
    import seed_demo_user

    assert "создан" in await seed_demo_user.seed("seeduser", "pass12345", "editor")
    assert "уже существует" in await seed_demo_user.seed("seeduser", "pass12345", "admin")
